import os
import sys
import json
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

# Allow:
# from baseline.prompt import build_prompt
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baseline.prompt import build_prompt


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rsft_data(path: str, limit=None):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_no}: {e}"
                ) from e

            question = item.get("question")
            response = item.get("response")

            if not question:
                raise ValueError(
                    f"Missing question at {path}:{line_no}"
                )
            if not response:
                raise ValueError(
                    f"Missing response at {path}:{line_no}"
                )

            data.append(item)

            if limit is not None and len(data) >= limit:
                break

    if not data:
        raise ValueError(f"No usable samples found in {path}")

    return data


class RSFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.examples = []
        self.stats = {
            "total": 0,
            "kept": 0,
            "truncated": 0,
            "empty_response": 0,
        }

        self._build()

    def _build(self):
        for item in self.data:
            self.stats["total"] += 1

            question = item["question"]
            response = item["response"]

            if not response.strip():
                self.stats["empty_response"] += 1
                continue

            prompt = build_prompt(question)

            # The prompt already ends with:
            # Assistant: <think>
            #
            # response is the exact vLLM continuation after that prefix.
            full_text = prompt + response

            prompt_ids = self.tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            full_ids = self.tokenizer(
                full_text,
                add_special_tokens=False,
            )["input_ids"]

            prompt_len = len(prompt_ids)

            # Safety check: the concatenated tokenization should normally
            # preserve the prompt prefix.
            if prompt_len >= len(full_ids):
                self.stats["empty_response"] += 1
                continue

            was_truncated = len(full_ids) > self.max_length

            input_ids = full_ids[:self.max_length]

            # Response-only loss:
            # labels before the response are -100.
            labels = input_ids.copy()

            response_start = min(prompt_len, len(labels))
            for i in range(response_start):
                labels[i] = -100

            # If truncation removed the whole response, discard the sample.
            if all(x == -100 for x in labels):
                self.stats["truncated"] += 1
                continue

            if was_truncated:
                self.stats["truncated"] += 1

            attention_mask = [1] * len(input_ids)

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "question": question,
                    "response": response,
                    "answer": item.get("answer"),
                    "id": item.get("id"),
                    "config": item.get("config"),
                }
            )

            self.stats["kept"] += 1


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

        if self.pad_token_id is None:
            raise ValueError(
                "Tokenizer has no pad_token_id. "
                "Set one before training."
            )

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for x in batch:
            pad_len = max_len - len(x["input_ids"])

            input_ids.append(
                x["input_ids"] + [self.pad_token_id] * pad_len
            )
            attention_mask.append(
                x["attention_mask"] + [0] * pad_len
            )
            labels.append(
                x["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long
            ),
            "labels": torch.tensor(
                labels, dtype=torch.long
            ),
        }


def compute_response_entropy(logits, labels):
    """
    Mean token entropy over response positions only.

    logits:
        [B, T, V]
    labels:
        [B, T]

    For causal LM, logits[:, :-1] predict labels[:, 1:].
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels != -100

    if not mask.any():
        return torch.tensor(
            0.0,
            device=logits.device,
            dtype=logits.dtype,
        )

    selected_logits = shift_logits[mask]

    log_probs = F.log_softmax(
        selected_logits.float(), dim=-1
    )
    probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=-1)

    return entropy.mean().to(logits.dtype)


def compute_response_loss(logits, labels):
    """
    Causal LM cross entropy, response tokens only.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    return loss


def save_checkpoint(
    model,
    tokenizer,
    output_dir,
    epoch_name,
):
    ckpt_dir = os.path.join(output_dir, epoch_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    model.save_pretrained(
        ckpt_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(ckpt_dir)

    return ckpt_dir



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/models/Qwen2.5-Math-1.5B-SFT-v2",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/Qwen/rsft/sampling-v1/rsft_dataset.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/models/Qwen2.5-Math-1.5B-RSFT",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/root/Qwen/rsft/tensorboard-v1",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--save_every_epoch",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of RSFT samples for a small sanity run.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    device = torch.device("cuda")

    print("=" * 80)
    print("RSFT TRAINING")
    print("=" * 80)
    print(f"Model path                  : {args.model_path}")
    print(f"Data path                   : {args.data_path}")
    print(f"Output dir                  : {args.output_dir}")
    print(f"Epochs                      : {args.epochs}")
    print(f"Batch size                  : {args.batch_size}")
    print(f"Gradient accumulation      : {args.gradient_accumulation_steps}")
    print(f"Learning rate               : {args.learning_rate}")
    print(f"Weight decay                : {args.weight_decay}")
    print(f"Max length                  : {args.max_length}")
    print(f"Seed                        : {args.seed}")
    print("=" * 80)

    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"pad_token_id = {tokenizer.pad_token_id}")
    print(f"eos_token_id = {tokenizer.eos_token_id}")

    print("\n[2/6] Loading RSFT dataset...")
    raw_data = load_rsft_data(
        args.data_path,
        limit=args.limit,
    )

    print(f"Raw RSFT samples: {len(raw_data)}")

    print("\n[3/6] Tokenizing dataset...")
    dataset = RSFTDataset(
        raw_data,
        tokenizer,
        max_length=args.max_length,
    )

    print(f"Dataset stats:")
    for key, value in dataset.stats.items():
        print(f"  {key:16s}: {value}")

    if len(dataset) == 0:
        raise RuntimeError("No usable training samples remain.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=Collator(tokenizer),
        drop_last=False,
    )

    print(f"Training batches per epoch: {len(dataloader)}")

    print("\n[4/6] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.to(device)

    # Full-parameter fine-tuning.
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : {total_params:,}"
    )
    print(
        f"Trainable parameters : {trainable_params:,}"
    )
    print(
        f"Trainable ratio      : "
        f"{trainable_params / total_params:.6f}"
    )

    if trainable_params != total_params:
        raise RuntimeError(
            "Not all parameters are trainable. "
            "RSFT requires full-parameter fine-tuning."
        )

    # Memory optimization for 4090D.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    writer = SummaryWriter(args.log_dir)

    steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    total_optimizer_steps = steps_per_epoch * args.epochs

    print("\n[5/6] Training configuration")
    print(f"Steps per epoch        : {steps_per_epoch}")
    print(f"Total optimizer steps  : {total_optimizer_steps}")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.0
        epoch_entropy = 0.0
        epoch_batches = 0
        epoch_optimizer_steps = 0

        num_batches = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                loss = compute_response_loss(
                    outputs.logits,
                    labels,
                )

                entropy = compute_response_entropy(
                    outputs.logits,
                    labels,
                )

            # Correct scaling for the final incomplete accumulation group.
            group_start = (
                batch_idx
                // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps

            group_end = min(
                group_start + args.gradient_accumulation_steps,
                num_batches,
            )

            actual_group_size = group_end - group_start

            scaled_loss = loss / actual_group_size
            scaled_loss.backward()

            epoch_loss += loss.item()
            epoch_entropy += entropy.item()
            epoch_batches += 1

            is_last_batch = (
                batch_idx == num_batches - 1
            )
            is_accumulation_boundary = (
                (batch_idx + 1)
                % args.gradient_accumulation_steps
                == 0
            )

            if is_accumulation_boundary or is_last_batch:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                epoch_optimizer_steps += 1

                writer.add_scalar(
                    "train/loss",
                    loss.item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/response_entropy",
                    entropy.item(),
                    global_step,
                )

                if (
                    global_step % args.log_every == 0
                    or is_last_batch
                ):
                    avg_loss = (
                        epoch_loss / epoch_batches
                    )
                    avg_entropy = (
                        epoch_entropy / epoch_batches
                    )

                    progress = (
                        (batch_idx + 1)
                        / num_batches
                    )

                    print(
                        f"[Epoch {epoch}/{args.epochs}] "
                        f"batch {batch_idx + 1}/{num_batches} "
                        f"({progress:.1%}) | "
                        f"step {global_step} | "
                        f"loss {loss.item():.6f} | "
                        f"avg_loss {avg_loss:.6f} | "
                        f"entropy {entropy.item():.6f} | "
                        f"avg_entropy {avg_entropy:.6f}"
                    )

        avg_epoch_loss = (
            epoch_loss / max(epoch_batches, 1)
        )
        avg_epoch_entropy = (
            epoch_entropy / max(epoch_batches, 1)
        )

        writer.add_scalar(
            "epoch/loss",
            avg_epoch_loss,
            epoch,
        )
        writer.add_scalar(
            "epoch/response_entropy",
            avg_epoch_entropy,
            epoch,
        )

        print("\n" + "-" * 80)
        print(f"Epoch {epoch} complete")
        print(f"Average loss              : {avg_epoch_loss:.6f}")
        print(
            f"Average response entropy : "
            f"{avg_epoch_entropy:.6f}"
        )
        print(
            f"Optimizer steps          : "
            f"{epoch_optimizer_steps}"
        )
        print("-" * 80)

        # Save epoch checkpoint by default in order to make the experiment
        # reproducible and allow rollback/comparison.
        ckpt_dir = save_checkpoint(
            model,
            tokenizer,
            args.output_dir,
            f"epoch-{epoch}",
        )
        print(f"Saved checkpoint: {ckpt_dir}")

    print("\n[6/6] Saving final model...")

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
    )
    tokenizer.save_pretrained(args.output_dir)

    writer.close()

    print("=" * 80)
    print("RSFT TRAINING COMPLETE")
    print("=" * 80)
    print(f"Final model : {args.output_dir}")
    print(f"TensorBoard : {args.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()