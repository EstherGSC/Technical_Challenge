import os
import sys
import gc
import json
import time
import hashlib
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.prompt import build_prompt


DEFAULT_REFERENCE_MODEL = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-RSFT"
DEFAULT_DATASET = "/root/Qwen/dpo/sampling-v1/dpo_dataset.jsonl"
DEFAULT_CACHE = "/root/Qwen/dpo/cache/reference_logps.pt"

MAX_LENGTH = 1536
BATCH_SIZE = 1
LOGPROB_CHUNK_SIZE = 128


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_question(item):
    for key in ("question", "problem", "prompt"):
        if key in item:
            return item[key]
    raise KeyError("DPO item has no question/problem/prompt field")


def get_response(item, side):
    for key in (
        side,
        f"{side}_response",
        f"{side}_answer",
        f"{side}_completion",
    ):
        if key in item:
            return item[key]
    raise KeyError(f"DPO item has no {side} response field")


def tokenize_response_pair(tokenizer, item, max_length):
    question = get_question(item)
    chosen = get_response(item, "chosen")
    rejected = get_response(item, "rejected")

    prompt = build_prompt(question)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]

    def make_example(response_ids):
        available = max_length - len(prompt_ids)
        if available <= 0:
            return None

        ids = prompt_ids + response_ids[:available]
        response_len = len(ids) - len(prompt_ids)
        if response_len <= 0:
            return None

        mask = [0] * len(prompt_ids) + [1] * response_len
        return {
            "input_ids": ids,
            "response_mask": mask,
            "response_tokens": response_len,
            "truncated": response_len < len(response_ids),
        }

    chosen_ex = make_example(chosen_ids)
    rejected_ex = make_example(rejected_ids)

    if chosen_ex is None or rejected_ex is None:
        return None

    return {"chosen": chosen_ex, "rejected": rejected_ex}


class PairDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.examples = []
        self.original_indices = []
        self.truncated_chosen = 0
        self.truncated_rejected = 0
        self.dropped = 0

        for idx, item in enumerate(data):
            pair = tokenize_response_pair(tokenizer, item, max_length)
            if pair is None:
                self.dropped += 1
                continue

            if pair["chosen"]["truncated"]:
                self.truncated_chosen += 1
            if pair["rejected"]["truncated"]:
                self.truncated_rejected += 1

            self.examples.append(pair)
            self.original_indices.append(idx)

        print(
            f"[DATA] raw={len(data)} valid={len(self.examples)} "
            f"dropped={self.dropped} "
            f"chosen_truncated={self.truncated_chosen} "
            f"rejected_truncated={self.truncated_rejected}",
            flush=True,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_single_pair(batch, pad_token_id):
    assert len(batch) == 1

    item = batch[0]

    def pad_side(side):
        ex = item[side]
        ids = torch.tensor([ex["input_ids"]], dtype=torch.long)
        mask = torch.tensor([ex["response_mask"]], dtype=torch.float32)
        attn = torch.ones_like(ids, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "response_mask": mask,
        }

    return {
        "chosen": pad_side("chosen"),
        "rejected": pad_side("rejected"),
    }


def move_to_device(side, device):
    return {
        "input_ids": side["input_ids"].to(device, non_blocking=True),
        "attention_mask": side["attention_mask"].to(device, non_blocking=True),
        "response_mask": side["response_mask"].to(device, non_blocking=True),
    }


@torch.inference_mode()
def sequence_logprob(model, input_ids, attention_mask, response_mask,
                     chunk_size=LOGPROB_CHUNK_SIZE):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]

    total = torch.zeros(
        input_ids.size(0), dtype=torch.float32, device=input_ids.device
    )

    for start in range(0, shift_logits.size(1), chunk_size):
        end = min(start + chunk_size, shift_logits.size(1))

        chunk = shift_logits[:, start:end, :].float()
        labels = shift_labels[:, start:end]
        mask = shift_mask[:, start:end]

        target_logits = torch.gather(
            chunk, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        log_norm = torch.logsumexp(chunk, dim=-1)
        token_logp = target_logits - log_norm
        total = total + (token_logp * mask).sum(dim=-1)

        del chunk, target_logits, log_norm, token_logp

    return total


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reference_model", default=DEFAULT_REFERENCE_MODEL)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--output", default=DEFAULT_CACHE)
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--limit", type=int, default=None,
                   help="For sanity check, compute only the first N valid pairs.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing cache.")
    return p.parse_args()


def main():
    args = build_args()

    if args.batch_size != 1:
        raise ValueError("Use batch_size=1 for the 4090D memory-safe reference pass.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DPO REFERENCE LOGP PRECOMPUTATION")
    print("=" * 80)
    print(f"reference_model = {args.reference_model}")
    print(f"dataset         = {args.dataset}")
    print(f"output          = {args.output}")
    print(f"max_length      = {args.max_length}")
    print(f"batch_size      = {args.batch_size}")
    print("=" * 80)

    dataset_hash = file_sha256(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        args.reference_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_jsonl(args.dataset)
    pair_dataset = PairDataset(data, tokenizer, args.max_length)

    if args.limit is not None:
        # Keep the first N valid pairs, not N raw lines.
        pair_dataset.examples = pair_dataset.examples[:args.limit]
        pair_dataset.original_indices = pair_dataset.original_indices[:args.limit]

    print(f"[DATA] computing reference logp for {len(pair_dataset)} pairs")

    if os.path.exists(args.output) and not args.force:
        try:
            old = torch.load(args.output, map_location="cpu", weights_only=False)
            meta = old.get("metadata", {})
            if (
                meta.get("dataset_sha256") == dataset_hash
                and meta.get("reference_model") == os.path.abspath(args.reference_model)
                and meta.get("max_length") == args.max_length
                and len(old.get("sample_indices", [])) == len(pair_dataset)
            ):
                print(f"[CACHE] matching cache already exists: {args.output}")
                return
        except Exception as e:
            print(f"[CACHE] existing cache cannot be reused: {e}")

    loader = DataLoader(
        pair_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_single_pair(b, tokenizer.pad_token_id),
        pin_memory=True,
        num_workers=0,
    )

    print("[MODEL] loading RSFT reference model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.reference_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = False
    model.eval()

    # No gradient checkpointing is necessary for inference.
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    print("[MODEL] reference model loaded.", flush=True)
    print(
        f"[GPU] allocated={torch.cuda.memory_allocated()/1024**3:.2f}GB "
        f"reserved={torch.cuda.memory_reserved()/1024**3:.2f}GB",
        flush=True,
    )

    chosen_values = []
    rejected_values = []

    start_time = time.time()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            chosen = move_to_device(batch["chosen"], device)
            rejected = move_to_device(batch["rejected"], device)

            c = sequence_logprob(
                model,
                chosen["input_ids"],
                chosen["attention_mask"],
                chosen["response_mask"],
            )
            r = sequence_logprob(
                model,
                rejected["input_ids"],
                rejected["attention_mask"],
                rejected["response_mask"],
            )

            chosen_values.append(float(c[0].cpu()))
            rejected_values.append(float(r[0].cpu()))

            if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
                elapsed = time.time() - start_time
                print(
                    f"[REF] {batch_idx+1}/{len(loader)} "
                    f"({100*(batch_idx+1)/len(loader):.1f}%) | "
                    f"chosen_ref_logp={chosen_values[-1]:.3f} | "
                    f"rejected_ref_logp={rejected_values[-1]:.3f} | "
                    f"time={elapsed/60:.1f}m",
                    flush=True,
                )

            del chosen, rejected, c, r

    chosen_ref = torch.tensor(chosen_values, dtype=torch.float32)
    rejected_ref = torch.tensor(rejected_values, dtype=torch.float32)
    sample_indices = torch.tensor(
        pair_dataset.original_indices, dtype=torch.long
    )

    metadata = {
        "dataset": os.path.abspath(args.dataset),
        "dataset_sha256": dataset_hash,
        "reference_model": os.path.abspath(args.reference_model),
        "max_length": args.max_length,
        "num_raw_pairs": len(data),
        "num_valid_pairs": len(pair_dataset),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dtype": "float32_cpu_cache",
        "logprob_definition": "sum log P(response | prompt)",
    }

    payload = {
        "chosen_ref_logp": chosen_ref,
        "rejected_ref_logp": rejected_ref,
        "sample_indices": sample_indices,
        "metadata": metadata,
    }

    tmp = args.output + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, args.output)

    print("=" * 80)
    print(f"[DONE] saved reference cache: {args.output}")
    print(f"[DONE] pairs: {len(chosen_ref)}")
    print(
        f"[STATS] chosen_ref_logp mean={chosen_ref.mean():.4f} "
        f"std={chosen_ref.std():.4f}"
    )
    print(
        f"[STATS] rejected_ref_logp mean={rejected_ref.mean():.4f} "
        f"std={rejected_ref.std():.4f}"
    )
    print("=" * 80)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
