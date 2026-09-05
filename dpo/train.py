import os
import sys
import gc
import json
import math
import time
import random
import hashlib
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baseline.prompt import build_prompt


DEFAULT_POLICY_MODEL = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-RSFT"
DEFAULT_DATASET = "/root/Qwen/dpo/sampling-v1/dpo_dataset.jsonl"
DEFAULT_OUTPUT = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-DPO"
DEFAULT_TB = "/root/Qwen/dpo/tensorboard"
DEFAULT_CACHE = "/root/Qwen/dpo/cache/reference_logps.pt"

BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_LENGTH = 1536
LR = 2e-6
BETA = 0.1
WEIGHT_DECAY = 0.0
EPOCHS = 1
SEED = 42

# Limiting this chunk keeps the temporary float32 vocabulary tensor small.
LOGPROB_CHUNK_SIZE = 128


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_mem(prefix=""):
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[GPU] {prefix} allocated={alloc:.2f}GB "
        f"reserved={reserved:.2f}GB peak={peak:.2f}GB",
        flush=True,
    )


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
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
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
    """
    Keep chosen/rejected as one aligned pair.

    The prompt and response are tokenized separately so the response mask is exact.
    Right truncation is used, matching the usual causal-LM convention.
    """
    question = get_question(item)
    chosen = get_response(item, "chosen")
    rejected = get_response(item, "rejected")

    prompt = build_prompt(question)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]

    # Reserve space for the response. We preserve the prompt prefix and truncate
    # the response on the right if necessary.
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

    c = make_example(chosen_ids)
    r = make_example(rejected_ids)

    # A pair is valid only when both sides have at least one response token.
    if c is None or r is None:
        return None

    return {"chosen": c, "rejected": r}


def collate_single_pair(batch, pad_token_id):
    assert len(batch) == 1, "This implementation intentionally uses batch_size=1."
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
            "response_tokens": ex["response_tokens"],
            "truncated": ex["truncated"],
        }

    return {
        "chosen": pad_side("chosen"),
        "rejected": pad_side("rejected"),
    }


class PairDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.examples = []
        self.original_indices = []
        dropped = 0

        for idx, item in enumerate(data):
            pair = tokenize_response_pair(tokenizer, item, max_length)
            if pair is None:
                dropped += 1
                continue
            self.examples.append(pair)
            self.original_indices.append(idx)

        self.dropped = dropped
        print(
            f"[DATA] total={len(data)} valid_pairs={len(self.examples)} "
            f"dropped={dropped} max_length={max_length}",
            flush=True,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def move_side_to_device(side, device):
    return {
        "input_ids": side["input_ids"].to(device, non_blocking=True),
        "attention_mask": side["attention_mask"].to(device, non_blocking=True),
        "response_mask": side["response_mask"].to(device, non_blocking=True),
    }


def sequence_logprob(model, input_ids, attention_mask, response_mask,
                     chunk_size=LOGPROB_CHUNK_SIZE):
    """
    Sum log P(response | prompt) for each sequence.

    Memory optimization:
      Instead of materializing log_softmax(logits) for all [B,L,V],
      compute target logits - logsumexp(logits) in small sequence chunks.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits

    # Causal shift.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]

    total = torch.zeros(
        input_ids.size(0), dtype=torch.float32, device=input_ids.device
    )

    for start in range(0, shift_logits.size(1), chunk_size):
        end = min(start + chunk_size, shift_logits.size(1))

        # Only this chunk is promoted to float32.
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


def save_model(model, tokenizer, output_dir, tag=None):
    out = Path(output_dir)
    if tag:
        out = out / tag
    out.mkdir(parents=True, exist_ok=True)

    # Save in HF format. The model remains BF16.
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print(f"[SAVE] {out}", flush=True)


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_model", default=DEFAULT_POLICY_MODEL)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--reference_cache", default=DEFAULT_CACHE)
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    p.add_argument("--tensorboard_dir", default=DEFAULT_TB)

    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--gradient_accumulation_steps", type=int, default=GRAD_ACCUM)
    p.add_argument("--max_length", type=int, default=MAX_LENGTH)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--beta", type=float, default=BETA)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--seed", type=int, default=SEED)

    p.add_argument("--limit", type=int, default=None,
                   help="Use only the first N valid pairs for sanity checking.")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--resume", default=None,
                   help="Optional HF model directory to resume policy weights from.")
    return p.parse_args()


def load_reference_cache(cache_path, dataset_path, reference_model,
                         max_length, limit):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Reference cache not found: {cache_path}\n"
            f"Run precompute_ref.py first."
        )

    cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    required = {
        "chosen_ref_logp",
        "rejected_ref_logp",
        "sample_indices",
        "metadata",
    }
    missing = required - set(cache.keys())
    if missing:
        raise RuntimeError(f"Reference cache missing keys: {sorted(missing)}")

    meta = cache["metadata"]
    expected_hash = file_sha256(dataset_path)

    checks = {
        "dataset_sha256": expected_hash,
        "reference_model": os.path.abspath(reference_model),
        "max_length": max_length,
    }

    for k, expected in checks.items():
        if meta.get(k) != expected:
            raise RuntimeError(
                f"Reference cache mismatch for {k}: "
                f"cache={meta.get(k)!r}, expected={expected!r}\n"
                f"Do not reuse this cache; rerun precompute_ref.py."
            )

    sample_indices = list(cache["sample_indices"])
    chosen_ref = cache["chosen_ref_logp"].float()
    rejected_ref = cache["rejected_ref_logp"].float()

    if len(sample_indices) != len(chosen_ref) or len(sample_indices) != len(rejected_ref):
        raise RuntimeError("Reference cache tensor/index lengths do not match.")

    if limit is not None:
        # Sanity cache may be shorter; for a full cache, use the first N pairs.
        sample_indices = sample_indices[:limit]
        chosen_ref = chosen_ref[:limit]
        rejected_ref = rejected_ref[:limit]

    print(
        f"[REF CACHE] loaded {len(sample_indices)} pairs from {cache_path}",
        flush=True,
    )
    return chosen_ref, rejected_ref, sample_indices


def main():
    args = build_args()

    if args.batch_size != 1:
        raise ValueError("This DPO implementation is intentionally fixed to batch_size=1.")

    if args.gradient_accumulation_steps != 8:
        raise ValueError(
            "This run is configured for gradient_accumulation_steps=8."
        )

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 80)
    print("DPO TRAINING")
    print("=" * 80)
    print(f"policy_model       = {args.policy_model}")
    print(f"dataset            = {args.dataset}")
    print(f"reference_cache    = {args.reference_cache}")
    print(f"output_dir         = {args.output_dir}")
    print(f"batch_size         = {args.batch_size}")
    print(f"grad_accum         = {args.gradient_accumulation_steps}")
    print(f"max_length         = {args.max_length}")
    print(f"lr                 = {args.lr}")
    print(f"beta               = {args.beta}")
    print(f"epochs             = {args.epochs}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data = load_jsonl(args.dataset)
    dataset_hash = file_sha256(args.dataset)
    print(f"[DATA] loaded {len(data)} raw pairs")
    print(f"[DATA] sha256={dataset_hash}")

    chosen_ref, rejected_ref, sample_indices = load_reference_cache(
        args.reference_cache,
        args.dataset,
        DEFAULT_POLICY_MODEL,
        args.max_length,
        args.limit,
    )

    selected_data = [data[i] for i in sample_indices]

    # Tokenize only the selected pairs. This must produce exactly the same
    # valid-pair ordering used by precompute_ref.py.
    pair_dataset = PairDataset(
        selected_data, tokenizer, args.max_length
    )

    if len(pair_dataset) != len(sample_indices):
        raise RuntimeError(
            "Tokenized training pair count does not match reference cache. "
            "The tokenizer/prompt/max_length configuration differs."
        )

    loader = DataLoader(
        pair_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_single_pair(b, tokenizer.pad_token_id),
        pin_memory=True,
        num_workers=0,
    )

    print("[MODEL] loading policy model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.resume or args.policy_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] trainable={trainable:,} / total={total:,}", flush=True)
    if trainable != total:
        raise RuntimeError("DPO requires full-parameter training here.")

    cuda_mem("after policy load")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    writer = SummaryWriter(args.tensorboard_dir)
    global_step = 0

    # A true initial sanity point: policy and reference start from the same
    # RSFT checkpoint, so before any update the DPO margin should be ~0 and
    # the loss should be close to log(2)=0.6931.
    print("[SANITY] The expected pre-update DPO loss is approximately 0.6931.", flush=True)

    total_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    running = {
        "loss": 0.0,
        "margin": 0.0,
        "acc": 0.0,
        "chosen_logp": 0.0,
        "rejected_logp": 0.0,
        "grad_norm": 0.0,
        "n": 0,
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_margin = 0.0
        epoch_acc = 0.0
        epoch_n = 0

        for batch_idx, batch in enumerate(loader):
            chosen = move_side_to_device(batch["chosen"], device)
            rejected = move_side_to_device(batch["rejected"], device)

            # Policy chosen forward.
            chosen_logp = sequence_logprob(
                model,
                chosen["input_ids"],
                chosen["attention_mask"],
                chosen["response_mask"],
            )

            # Policy rejected forward.
            rejected_logp = sequence_logprob(
                model,
                rejected["input_ids"],
                rejected["attention_mask"],
                rejected["response_mask"],
            )

            ref_c = chosen_ref[batch_idx].to(device, non_blocking=True)
            ref_r = rejected_ref[batch_idx].to(device, non_blocking=True)

            policy_logratio = chosen_logp - rejected_logp
            reference_logratio = ref_c - ref_r
            reward_margin = policy_logratio - reference_logratio

            loss = -F.logsigmoid(args.beta * reward_margin).mean()
            pref_acc = (reward_margin > 0).float().mean()

            # Correct gradient accumulation scaling.
            (loss / args.gradient_accumulation_steps).backward()

            loss_value = float(loss.detach().cpu())
            margin_value = float(reward_margin.detach().mean().cpu())
            acc_value = float(pref_acc.detach().mean().cpu())

            epoch_loss += loss_value
            epoch_margin += margin_value
            epoch_acc += acc_value
            epoch_n += 1

            running["loss"] += loss_value
            running["margin"] += margin_value
            running["acc"] += acc_value
            running["chosen_logp"] += float(chosen_logp.detach().mean().cpu())
            running["rejected_logp"] += float(rejected_logp.detach().mean().cpu())
            running["n"] += 1

            do_step = (
                (batch_idx + 1) % args.gradient_accumulation_steps == 0
                or (batch_idx + 1) == total_batches
            )

            if do_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                grad_value = float(grad_norm.detach().cpu())
                running["grad_norm"] += grad_value

                if global_step % args.log_every == 0:
                    n = max(running["n"], 1)
                    avg_loss = running["loss"] / n
                    avg_margin = running["margin"] / n
                    avg_acc = running["acc"] / n
                    avg_c = running["chosen_logp"] / n
                    avg_r = running["rejected_logp"] / n

                    elapsed = time.time() - start_time
                    print(
                        f"[Epoch {epoch+1}/{args.epochs}] "
                        f"step {global_step} | batch {batch_idx+1}/{total_batches} "
                        f"({100*(batch_idx+1)/total_batches:.1f}%) | "
                        f"loss {avg_loss:.6f} | "
                        f"margin {avg_margin:.6f} | "
                        f"pref_acc {avg_acc:.4f} | "
                        f"chosen_logp {avg_c:.3f} | "
                        f"rejected_logp {avg_r:.3f} | "
                        f"grad_norm {grad_value:.3f} | "
                        f"time {elapsed/60:.1f}m",
                        flush=True,
                    )

                    writer.add_scalar("dpo/loss", avg_loss, global_step)
                    writer.add_scalar("dpo/reward_margin", avg_margin, global_step)
                    writer.add_scalar("dpo/preference_accuracy", avg_acc, global_step)
                    writer.add_scalar("dpo/chosen_logp", avg_c, global_step)
                    writer.add_scalar("dpo/rejected_logp", avg_r, global_step)
                    writer.add_scalar("train/grad_norm", grad_value, global_step)
                    writer.add_scalar(
                        "train/progress",
                        (batch_idx + 1) / total_batches,
                        global_step,
                    )

                    cuda_mem(f"step {global_step}")

                    running = {
                        "loss": 0.0,
                        "margin": 0.0,
                        "acc": 0.0,
                        "chosen_logp": 0.0,
                        "rejected_logp": 0.0,
                        "grad_norm": 0.0,
                        "n": 0,
                    }

            del chosen, rejected, ref_c, ref_r
            del chosen_logp, rejected_logp, policy_logratio
            del reference_logratio, reward_margin, loss, pref_acc

        avg_epoch_loss = epoch_loss / max(epoch_n, 1)
        avg_epoch_margin = epoch_margin / max(epoch_n, 1)
        avg_epoch_acc = epoch_acc / max(epoch_n, 1)

        print(
            f"[EPOCH {epoch+1}] "
            f"loss={avg_epoch_loss:.6f} "
            f"margin={avg_epoch_margin:.6f} "
            f"pref_acc={avg_epoch_acc:.4f}",
            flush=True,
        )

        writer.add_scalar("epoch/loss", avg_epoch_loss, epoch + 1)
        writer.add_scalar("epoch/reward_margin", avg_epoch_margin, epoch + 1)
        writer.add_scalar("epoch/preference_accuracy", avg_epoch_acc, epoch + 1)

        if args.save_every_epoch:
            save_model(model, tokenizer, args.output_dir, f"epoch-{epoch+1}")

        gc.collect()
        torch.cuda.empty_cache()

    save_model(model, tokenizer, args.output_dir)

    writer.close()
    print("[DONE] DPO training finished.", flush=True)
    cuda_mem("final")


if __name__ == "__main__":
    main()
