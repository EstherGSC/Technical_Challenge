import os
import json
import math
import random
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
DATA_ROOT = "/root/autodl-tmp/datasets/MATH"

CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

SYSTEM_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""


def extract_boxed_answer(solution):
    """
    Extract the final \\boxed{...} from a MATH solution.
    Supports nested braces such as:
        \\boxed{\\frac{1}{2}}
        \\boxed{x^{2}+1}
    """
    matches = []
    start = 0

    while True:
        pos = solution.find(r"\boxed{", start)

        if pos == -1:
            break

        i = pos + len(r"\boxed{")
        depth = 1

        while i < len(solution) and depth > 0:
            if solution[i] == "{":
                depth += 1
            elif solution[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            content = solution[pos + len(r"\boxed{"):i - 1]
            matches.append(content)

        start = i

    if not matches:
        return None

    return matches[-1].strip()


def build_sample(question, solution, tokenizer, max_length):
    answer = extract_boxed_answer(solution)

    if answer is None:
        return None

    prompt = SYSTEM_PROMPT.format(question=question)

    response = (
        solution.strip()
        + f"\n</think> <answer>{answer}</answer>"
    )

    full_text = prompt + response

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    full_ids = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    # If the sample is too long, truncate from the left so that
    # the end of the reasoning and answer are retained.
    if len(full_ids) > max_length:
        full_ids = full_ids[-max_length:]

        # Since prompt occupies the beginning, response starts
        # after the prompt. If truncation removes prompt tokens,
        # all retained tokens are response tokens.
        prompt_length = max(0, len(prompt_ids) - (len(tokenizer(full_text)["input_ids"]) - max_length))
        prompt_length = min(prompt_length, max_length)
    else:
        prompt_length = min(len(prompt_ids), len(full_ids))

    attention_mask = [1] * len(full_ids)

    # Response mask: prompt = 0, response = 1.
    response_mask = [0] * prompt_length + [1] * (len(full_ids) - prompt_length)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "answer": answer,
        "prompt": prompt,
        "response": response,
    }


class MathSFTDataset(Dataset):
    def __init__(self, tokenizer, max_length):
        self.samples = []
        self.num_missing_answer = 0
        self.num_truncated = 0

        for config in CONFIGS:
            path = os.path.join(DATA_ROOT, config, "train")
            ds = load_from_disk(path)

            print(f"Loading {config}: {len(ds)} examples")

            for item in ds:
                question = item["problem"]
                solution = item["solution"]

                answer = extract_boxed_answer(solution)

                if answer is None:
                    self.num_missing_answer += 1
                    continue

                prompt = SYSTEM_PROMPT.format(question=question)
                response = (
                    solution.strip()
                    + f"\n</think> <answer>{answer}</answer>"
                )

                raw_ids = tokenizer(
                    prompt + response,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]

                if len(raw_ids) > max_length:
                    self.num_truncated += 1

                sample = build_sample(
                    question,
                    solution,
                    tokenizer,
                    max_length,
                )

                if sample is not None:
                    self.samples.append(sample)

        print()
        print("Dataset statistics")
        print("------------------")
        print("usable samples:", len(self.samples))
        print("missing boxed answer:", self.num_missing_answer)
        print("truncated samples:", self.num_truncated)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention_masks = []
    response_masks = []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])

        input_ids.append(
            x["input_ids"] + [pad_token_id] * pad_len
        )

        attention_masks.append(
            x["attention_mask"] + [0] * pad_len
        )

        response_masks.append(
            x["response_mask"] + [0] * pad_len
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "response_mask": torch.tensor(response_masks, dtype=torch.float32),
    }


def compute_response_loss(model, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    # Next-token prediction.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_response_mask = response_mask[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    # Only response tokens contribute to loss.
    token_loss = token_loss * shift_response_mask

    loss = token_loss.sum() / shift_response_mask.sum().clamp_min(1.0)

    return loss


def evaluate_training_loss(model, loader, device, max_batches=20):
    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            loss = compute_response_loss(model, batch)

            total_loss += loss.item()
            count += 1

    model.train()

    if count == 0:
        return 0.0

    return total_loss / count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/models/Qwen2.5-Math-1.5B-SFT",
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")

    print("=" * 80)
    print("Loading tokenizer")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("pad_token_id:", tokenizer.pad_token_id)
    print("eos_token_id:", tokenizer.eos_token_id)

    print()
    print("=" * 80)
    print("Preparing MATH training dataset")
    print("=" * 80)

    dataset = MathSFTDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    if args.limit is not None:
        dataset.samples = dataset.samples[:args.limit]
        print(f"Limited training dataset to {len(dataset.samples)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(
            x,
            tokenizer.pad_token_id,
        ),
    )

    print()
    print("=" * 80)
    print("Loading Qwen2.5-Math-1.5B")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.to(device)
    model.train()

    # Full-parameter SFT:
    # all parameters remain trainable.
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    trainable_count = sum(
        p.numel() for p in trainable_params
    )

    total_count = sum(
        p.numel() for p in model.parameters()
    )

    print("Total parameters:", total_count)
    print("Trainable parameters:", trainable_count)
    print(
        "Trainable ratio:",
        trainable_count / total_count,
    )

    if trainable_count != total_count:
        raise RuntimeError(
            "This SFT implementation must be full-parameter fine-tuning."
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Simple linear warmup.
    total_steps = math.ceil(
        len(loader) * args.epochs
        / args.gradient_accumulation_steps
    )

    warmup_steps = max(1, int(total_steps * 0.03))

    print("Total optimizer steps:", total_steps)
    print("Warmup steps:", warmup_steps)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    log_path = os.path.join(
        args.output_dir,
        "training_log.jsonl",
    )

    with open(log_path, "w") as log_file:
        for epoch in range(args.epochs):

            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, batch in enumerate(loader):

                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }

                loss = compute_response_loss(
                    model,
                    batch,
                )

                loss_for_backward = (
                    loss / args.gradient_accumulation_steps
                )

                loss_for_backward.backward()

                epoch_loss += loss.item()
                epoch_batches += 1

                if (
                    (batch_idx + 1)
                    % args.gradient_accumulation_steps == 0
                    or (batch_idx + 1) == len(loader)
                ):
                    global_step += 1

                    # Linear warmup.
                    if global_step <= warmup_steps:
                        lr_scale = global_step / warmup_steps
                    else:
                        lr_scale = 1.0

                    for group in optimizer.param_groups:
                        group["lr"] = (
                            args.learning_rate * lr_scale
                        )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        1.0,
                    )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    record = {
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "global_step": global_step,
                        "loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }

                    log_file.write(
                        json.dumps(record) + "\n"
                    )
                    log_file.flush()

                    if global_step % 10 == 0:
                        print(
                            f"epoch={epoch + 1} "
                            f"batch={batch_idx + 1}/{len(loader)} "
                            f"step={global_step} "
                            f"loss={loss.item():.6f} "
                            f"lr={optimizer.param_groups[0]['lr']:.3e}"
                        )

            avg_epoch_loss = (
                epoch_loss / max(epoch_batches, 1)
            )

            print()
            print(
                f"Epoch {epoch + 1} finished. "
                f"Average loss: {avg_epoch_loss:.6f}"
            )

            # Save checkpoint after each epoch.
            epoch_dir = os.path.join(
                args.output_dir,
                f"epoch-{epoch + 1}",
            )

            os.makedirs(epoch_dir, exist_ok=True)

            model.save_pretrained(
                epoch_dir,
                safe_serialization=True,
            )

            tokenizer.save_pretrained(epoch_dir)

            print("Saved:", epoch_dir)
            print()

    # Save final model.
    print("=" * 80)
    print("Saving final SFT model")
    print("=" * 80)

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(args.output_dir)

    print("Saved:", args.output_dir)
    print("Training log:", log_path)


if __name__ == "__main__":
    main()
