import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

# Reuse the official prompt template used by the baseline.
from baseline.prompt import build_prompt


# ============================================================
# Configuration
# ============================================================

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


# ============================================================
# Utility functions
# ============================================================

def extract_boxed_answer(solution):
    """
    Extract the final \\boxed{...} from a MATH solution.

    Supports nested braces, e.g.
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
            content = solution[
                pos + len(r"\boxed{"):
                i - 1
            ]
            matches.append(content)

        start = i

    if not matches:
        return None

    return matches[-1].strip()

# ============================================================
# Sample construction
# ============================================================

def build_sample(question, solution, tokenizer, max_length):
    """
    Construct one SFT sample.

    Prompt:
        Official baseline prompt template.

    Response:
        MATH solution followed by
        </think> <answer>...</answer>

    The prompt and response are tokenized separately so that
    response_mask can be constructed exactly.

    response_mask:
        prompt tokens   -> 0
        response tokens -> 1
        padding         -> 0

    If the total sequence exceeds max_length, tokens are removed
    from the beginning of the response while preserving the
    final answer whenever possible.
    """

    answer = extract_boxed_answer(solution)

    if answer is None:
        return None

    # --------------------------------------------------------
    # Official prompt
    # --------------------------------------------------------

    prompt = build_prompt(question)

    # --------------------------------------------------------
    # SFT response
    #
    # IMPORTANT:
    # The official grader expects the exact substring
    #
    #     </think> <answer>
    #
    # Therefore there is a SPACE between </think> and <answer>.
    # --------------------------------------------------------

    response = (
        solution.strip()
        + f"\n</think> <answer>{answer}</answer>"
    )

    # --------------------------------------------------------
    # Tokenize prompt and response separately
    # --------------------------------------------------------

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    original_length = (
        len(prompt_ids) + len(response_ids)
    )

    # --------------------------------------------------------
    # Truncation
    # --------------------------------------------------------

    truncated = False

    if original_length > max_length:

        truncated = True

        available_response_length = (
            max_length - len(prompt_ids)
        )

        if available_response_length > 0:

            # Keep the END of the response because the final
            # answer and </answer> are located there.
            response_ids = response_ids[
                -available_response_length:
            ]

        else:

            # Extremely long prompt.
            #
            # Keep the end of the prompt as a fallback.
            # There will be no response tokens in this case.
            prompt_ids = prompt_ids[-max_length:]
            response_ids = []

    input_ids = prompt_ids + response_ids

    attention_mask = [1] * len(input_ids)

    response_mask = (
        [0] * len(prompt_ids)
        + [1] * len(response_ids)
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "answer": answer,
        "prompt": prompt,
        "response": response,
        "original_length": original_length,
        "truncated": truncated,
    }


# ============================================================
# Dataset
# ============================================================

class MathSFTDataset(Dataset):

    def __init__(self, tokenizer, max_length):

        self.samples = []

        self.num_missing_answer = 0
        self.num_truncated = 0
        self.num_empty_response = 0

        for config in CONFIGS:

            path = os.path.join(
                DATA_ROOT,
                config,
                "train",
            )

            ds = load_from_disk(path)

            print(
                f"Loading {config}: "
                f"{len(ds)} examples"
            )

            for item in ds:

                question = item["problem"]
                solution = item["solution"]

                sample = build_sample(
                    question,
                    solution,
                    tokenizer,
                    max_length,
                )

                if sample is None:

                    self.num_missing_answer += 1
                    continue

                if sample["truncated"]:
                    self.num_truncated += 1

                if len(sample["response"]) == 0:
                    self.num_empty_response += 1

                self.samples.append(sample)

        print()
        print("Dataset statistics")
        print("------------------")
        print(
            "usable samples:",
            len(self.samples),
        )
        print(
            "missing boxed answer:",
            self.num_missing_answer,
        )
        print(
            "truncated samples:",
            self.num_truncated,
        )
        print(
            "empty response samples:",
            self.num_empty_response,
        )

        if len(self.samples) == 0:
            raise RuntimeError(
                "No usable training samples found."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# Collator
# ============================================================

def collate_fn(batch, pad_token_id):

    max_len = max(
        len(x["input_ids"])
        for x in batch
    )

    input_ids = []
    attention_masks = []
    response_masks = []

    for x in batch:

        pad_len = (
            max_len - len(x["input_ids"])
        )

        input_ids.append(
            x["input_ids"]
            + [pad_token_id] * pad_len
        )

        attention_masks.append(
            x["attention_mask"]
            + [0] * pad_len
        )

        response_masks.append(
            x["response_mask"]
            + [0] * pad_len
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_masks,
            dtype=torch.long,
        ),
        "response_mask": torch.tensor(
            response_masks,
            dtype=torch.float32,
        ),
    }


# ============================================================
# Loss
# ============================================================

def compute_response_loss(model, batch):
    """
    Compute causal LM loss only on response tokens.

    Also returns the average entropy of the model's token
    distribution over response positions.
    """

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    # --------------------------------------------------------
    # Next-token prediction
    # --------------------------------------------------------

    shift_logits = (
        logits[:, :-1, :]
        .contiguous()
    )

    shift_labels = (
        input_ids[:, 1:]
        .contiguous()
    )

    shift_response_mask = (
        response_mask[:, 1:]
        .contiguous()
    )

    vocab_size = shift_logits.size(-1)

    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    # Only response tokens contribute to the loss.
    token_loss = (
        token_loss
        * shift_response_mask
    )

    valid_tokens = (
        shift_response_mask.sum()
        .clamp_min(1.0)
    )

    loss = (
        token_loss.sum()
        / valid_tokens
    )

    # --------------------------------------------------------
    # Response entropy
    # --------------------------------------------------------

    with torch.no_grad():

        float_logits = shift_logits.float()

        log_probs = F.log_softmax(
            float_logits,
            dim=-1,
        )

        probs = log_probs.exp()

        entropy = -(
            probs * log_probs
        ).sum(dim=-1)

        entropy = (
            entropy
            * shift_response_mask
        ).sum() / valid_tokens

    return loss, entropy


# ============================================================
# Evaluation loss
# ============================================================

def evaluate_training_loss(
    model,
    loader,
    device,
    max_batches=20,
):

    model.eval()

    total_loss = 0.0
    total_entropy = 0.0
    count = 0

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            if batch_idx >= max_batches:
                break

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            loss, entropy = compute_response_loss(
                model,
                batch,
            )

            total_loss += loss.item()
            total_entropy += entropy.item()
            count += 1

    model.train()

    if count == 0:
        return 0.0, 0.0

    return (
        total_loss / count,
        total_entropy / count,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default=(
            "/root/autodl-tmp/models/"
            "Qwen2.5-Math-1.5B-SFT-v2"
        ),
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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--sample_interval",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    tensorboard_dir = os.path.join(
        args.output_dir,
        "tensorboard",
    )

    os.makedirs(
        tensorboard_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Random seeds
    # --------------------------------------------------------

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device("cuda")

    print("=" * 80)
    print("Device")
    print("=" * 80)
    print(torch.cuda.get_device_name(0))
    print()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("=" * 80)
    print("Loading tokenizer")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        "pad_token_id:",
        tokenizer.pad_token_id,
    )

    print(
        "eos_token_id:",
        tokenizer.eos_token_id,
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("=" * 80)
    print("Preparing MATH training dataset")
    print("=" * 80)

    dataset = MathSFTDataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    if args.limit is not None:

        dataset.samples = (
            dataset.samples[:args.limit]
        )

        print(
            f"Limited training dataset to "
            f"{len(dataset.samples)} samples"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(
            x,
            tokenizer.pad_token_id,
        ),
        pin_memory=True,
    )

    print(
        "Training batches:",
        len(loader),
    )

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("=" * 80)
    print("Loading Qwen2.5-Math-1.5B")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Gradient checkpointing reduces activation memory.
    # This does NOT change full-parameter training.
    model.gradient_checkpointing_enable()

    # Required when using gradient checkpointing with
    # decoder-only causal language models.
    model.config.use_cache = False

    model.to(device)
    model.train()

    # --------------------------------------------------------
    # Full parameter verification
    # --------------------------------------------------------

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    trainable_count = sum(
        p.numel()
        for p in trainable_params
    )

    total_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Total parameters:",
        total_count,
    )

    print(
        "Trainable parameters:",
        trainable_count,
    )

    print(
        "Trainable ratio:",
        trainable_count / total_count,
    )

    if trainable_count != total_count:
        raise RuntimeError(
            "This SFT implementation must be "
            "full-parameter fine-tuning."
        )

    print()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # --------------------------------------------------------
    # Training steps
    # --------------------------------------------------------

    total_batches = (
        len(loader) * args.epochs
    )

    total_steps = math.ceil(
        total_batches
        / args.gradient_accumulation_steps
    )

    warmup_steps = max(
        1,
        int(total_steps * 0.03),
    )

    print(
        "Total optimizer steps:",
        total_steps,
    )

    print(
        "Warmup steps:",
        warmup_steps,
    )

    print()

    # --------------------------------------------------------
    # TensorBoard
    # --------------------------------------------------------

    writer = SummaryWriter(
        log_dir=tensorboard_dir
    )

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------

    training_log_path = os.path.join(
        args.output_dir,
        "training_log.jsonl",
    )

    samples_log_path = os.path.join(
        args.output_dir,
        "samples.jsonl",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    global_step = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    with open(
        training_log_path,
        "w",
        encoding="utf-8",
    ) as log_file, open(
        samples_log_path,
        "w",
        encoding="utf-8",
    ) as sample_log_file:

        for epoch in range(args.epochs):

            epoch_loss = 0.0
            epoch_entropy = 0.0
            epoch_batches = 0

            print("=" * 80)
            print(
                f"Starting epoch "
                f"{epoch + 1}/{args.epochs}"
            )
            print("=" * 80)

            for batch_idx, batch in enumerate(loader):

                batch = {
                    k: v.to(
                        device,
                        non_blocking=True,
                    )
                    for k, v in batch.items()
                }

                loss, entropy = (
                    compute_response_loss(
                        model,
                        batch,
                    )
                )

                # ------------------------------------------------
                # Numerical stability check
                # ------------------------------------------------

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss detected: "
                        f"{loss.item()}"
                    )

                epoch_loss += loss.item()
                epoch_entropy += entropy.item()
                epoch_batches += 1

                # ------------------------------------------------
                # Determine actual accumulation size
                #
                # This is important for the final incomplete
                # accumulation group.
                # ------------------------------------------------

                group_start = (
                    batch_idx
                    // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps

                group_end = min(
                    group_start
                    + args.gradient_accumulation_steps,
                    len(loader),
                )

                current_accumulation_steps = (
                    group_end - group_start
                )

                # ------------------------------------------------
                # Gradient accumulation
                # ------------------------------------------------

                loss_for_backward = (
                    loss
                    / current_accumulation_steps
                )

                loss_for_backward.backward()

                # ------------------------------------------------
                # Optimizer update
                #
                # Update after a complete accumulation group
                # or at the final batch.
                # ------------------------------------------------

                should_step = (
                    (batch_idx + 1)
                    % args.gradient_accumulation_steps
                    == 0
                    or
                    (batch_idx + 1)
                    == len(loader)
                )

                if should_step:

                    global_step += 1

                    # ------------------------------------------------
                    # Linear warmup
                    # ------------------------------------------------

                    if global_step <= warmup_steps:

                        lr_scale = (
                            global_step
                            / warmup_steps
                        )

                    else:

                        lr_scale = 1.0

                    current_lr = (
                        args.learning_rate
                        * lr_scale
                    )

                    for group in optimizer.param_groups:
                        group["lr"] = current_lr

                    # ------------------------------------------------
                    # Gradient clipping
                    # ------------------------------------------------

                    grad_norm = (
                        torch.nn.utils
                        .clip_grad_norm_(
                            model.parameters(),
                            1.0,
                        )
                    )

                    # ------------------------------------------------
                    # Optimizer
                    # ------------------------------------------------

                    optimizer.step()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    # ------------------------------------------------
                    # TensorBoard
                    # ------------------------------------------------

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

                    writer.add_scalar(
                        "train/learning_rate",
                        current_lr,
                        global_step,
                    )

                    writer.add_scalar(
                        "train/grad_norm",
                        grad_norm.item(),
                        global_step,
                    )

                    # ------------------------------------------------
                    # JSONL training log
                    # ------------------------------------------------

                    record = {
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "global_step": global_step,
                        "loss": loss.item(),
                        "response_entropy": entropy.item(),
                        "learning_rate": current_lr,
                        "grad_norm": grad_norm.item(),
                        "accumulation_steps": current_accumulation_steps,
                    }

                    log_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    log_file.flush()

                    # ------------------------------------------------
                    # Console output
                    # ------------------------------------------------

                    if (
                        global_step
                        % args.log_interval
                        == 0
                    ):

                        print(
                            f"epoch={epoch + 1} "
                            f"batch={batch_idx + 1}/"
                            f"{len(loader)} "
                            f"step={global_step} "
                            f"loss={loss.item():.6f} "
                            f"entropy={entropy.item():.6f} "
                            f"lr={current_lr:.3e} "
                            f"grad_norm={grad_norm.item():.4f} "
                            f"accum={current_accumulation_steps}"
                        )

                    # ------------------------------------------------
                    # Record training sample
                    # ------------------------------------------------

                    if (
                        global_step
                        % args.sample_interval
                        == 0
                    ):

                        sample_idx = random.randrange(
                            len(dataset)
                        )

                        sample = dataset[
                            sample_idx
                        ]

                        sample_record = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "prompt": sample["prompt"],
                            "response": sample["response"],
                            "answer": sample["answer"],
                            "truncated": sample["truncated"],
                            "original_length": sample[
                                "original_length"
                            ],
                        }

                        sample_log_file.write(
                            json.dumps(
                                sample_record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        sample_log_file.flush()

                        writer.add_text(
                            "samples/prompt",
                            sample["prompt"],
                            global_step,
                        )

                        writer.add_text(
                            "samples/response",
                            sample["response"],
                            global_step,
                        )

                        writer.add_text(
                            "samples/answer",
                            sample["answer"],
                            global_step,
                        )

            # --------------------------------------------------------
            # Epoch statistics
            # --------------------------------------------------------

            avg_epoch_loss = (
                epoch_loss
                / max(epoch_batches, 1)
            )

            avg_epoch_entropy = (
                epoch_entropy
                / max(epoch_batches, 1)
            )

            writer.add_scalar(
                "epoch/loss",
                avg_epoch_loss,
                epoch + 1,
            )

            writer.add_scalar(
                "epoch/response_entropy",
                avg_epoch_entropy,
                epoch + 1,
            )

            print()
            print(
                f"Epoch {epoch + 1} finished."
            )

            print(
                f"Average loss: "
                f"{avg_epoch_loss:.6f}"
            )

            print(
                f"Average response entropy: "
                f"{avg_epoch_entropy:.6f}"
            )

            # --------------------------------------------------------
            # Save epoch checkpoint
            # --------------------------------------------------------

            epoch_dir = os.path.join(
                args.output_dir,
                f"epoch-{epoch + 1}",
            )

            os.makedirs(
                epoch_dir,
                exist_ok=True,
            )

            model.save_pretrained(
                epoch_dir,
                safe_serialization=True,
            )

            tokenizer.save_pretrained(
                epoch_dir
            )

            print(
                "Saved:",
                epoch_dir,
            )

            print()

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    print("=" * 80)
    print("Saving final SFT model")
    print("=" * 80)

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        args.output_dir
    )

    writer.close()

    print(
        "Saved:",
        args.output_dir,
    )

    print(
        "Training log:",
        training_log_path,
    )

    print(
        "Samples log:",
        samples_log_path,
    )

    print(
        "TensorBoard:",
        tensorboard_dir,
    )


if __name__ == "__main__":
    main()
