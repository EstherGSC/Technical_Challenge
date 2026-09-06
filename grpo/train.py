import os
import sys
import gc
import json
import math
import time
import random
import hashlib
import shutil
import argparse
import re
from datasets import load_from_disk
from pathlib import Path
import multiprocessing as mp


import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.prompt import build_prompt
from baseline.grader import r1_zero_reward_fn

# vLLM V1 EngineCore must use spawn because CUDA has already
# been initialized in the main process.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_MODEL = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-DPO"
DEFAULT_OUTPUT = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-GRPO"
DEFAULT_TB = "/root/autodl-tmp/grpo/tensorboard"
DEFAULT_ROLLOUT_DIR = "/root/autodl-tmp/grpo/rollouts"
DEFAULT_SYNC_DIR = "/root/autodl-tmp/grpo/sync_model"

# Assignment hyperparameters
N_GRPO_STEPS = 200
LEARNING_RATE = 1e-5

ADVANTAGE_EPS = 1e-6

ROLLOUT_BATCH_SIZE = 32
GROUP_SIZE = 8

SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_MIN_TOKENS = 4
SAMPLING_MAX_TOKENS = 1024

EPOCHS_PER_ROLLOUT_BATCH = 2

TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4

GPU_MEMORY_UTILIZATION = 0.85

WEIGHT_DECAY = 0.0
BETA1 = 0.9
BETA2 = 0.95

# PPO / GRPO clipping epsilon.
# The assignment formula contains epsilon but the provided
# hyperparameter block does not specify it. 0.2 is the standard choice.
CLIP_EPS = 0.2

MAX_TRAIN_LENGTH = 1536

SEED = 42

# Same stop convention as the previous stages.
STOP_STR = "</answer>"

DATASET_ROOT = "/root/autodl-tmp/datasets/MATH"
MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def extract_boxed_answer(solution):
    """
    Extract the last nested \boxed{...} answer from a solution.
    """
    if not solution:
        return None

    matches = []
    start = 0
    marker = r"\boxed{"

    while True:
        position = solution.find(marker, start)

        if position == -1:
            break

        index = position + len(marker)
        depth = 1

        while index < len(solution) and depth > 0:
            if solution[index] == "{":
                depth += 1
            elif solution[index] == "}":
                depth -= 1

            index += 1

        if depth == 0:
            matches.append(
                solution[
                    position + len(marker):index - 1
                ].strip()
            )

        start = index

    if not matches:
        return None

    return matches[-1]


def get_ground_truth(item):
    """
    Match the RSFT/DPO answer-loading rule.

    The official solution is authoritative because some MATH records
    contain no answer field at all.
    """
    solution = (
        item.get("solution")
        or item.get("rationale")
        or item.get("completion")
        or ""
    )

    boxed_answer = extract_boxed_answer(solution)

    if boxed_answer is not None:
        return boxed_answer

    return (
        item.get("answer")
        or item.get("ground_truth")
        or item.get("target")
    )


def set_seed(seed):
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
        f"[GPU] {prefix} "
        f"allocated={alloc:.2f}GB "
        f"reserved={reserved:.2f}GB "
        f"peak={peak:.2f}GB",
        flush=True,
    )


def load_math_train():
    """
    Load all MATH train splits and filter out questions whose
    rollout prompt is too long for the training context.

    The prompt length is measured using the same build_prompt()
    function that is used during vLLM rollout.
    """
    print("[DATA] loading MATH train ...")

    all_samples = []

    # ---------------------------------------------------------
    # Tokenizer used for prompt-length filtering
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL,
        trust_remote_code=True,
    )

    num_filtered = 0
    max_prompt_length = 0
    longest_question = None

    for config in MATH_CONFIGS:
        path = os.path.join(DATASET_ROOT, config, "train")

        print(f"[DATA] loading {config}: {path}")

        ds = load_from_disk(path)

        print(f"[DATA]   {config}: {len(ds)} examples")

        for idx, item in enumerate(ds):
            # MATH datasets used in the previous SFT/RSFT/DPO stages
            # may use slightly different field names.
            question = (
                item.get("problem")
                or item.get("question")
                or item.get("instruction")
            )

            ground_truth = get_ground_truth(item)

            if not question:
                print(
                    f"[WARN] {config}[{idx}] has no question/problem, skip"
                )
                continue

            if ground_truth is None:
                print(
                    f"[WARN] {config}[{idx}] has no ground truth, skip"
                )
                continue

            question = str(question)
            ground_truth = str(ground_truth)

            # -------------------------------------------------
            # Check the actual rollout prompt length.
            #
            # IMPORTANT:
            # Use < MAX_TRAIN_LENGTH rather than <= because
            # vLLM must have room for generated tokens.
            # -------------------------------------------------
            prompt = build_prompt(question)

            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]

            prompt_length = len(prompt_ids)

            if prompt_length > max_prompt_length:
                max_prompt_length = prompt_length
                longest_question = (
                    config,
                    idx,
                    question,
                )

            if prompt_length >= MAX_TRAIN_LENGTH:
                num_filtered += 1
                continue

            all_samples.append({
                "config": config,
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
            })

    print(
        f"[DATA] loaded {len(all_samples)} valid MATH train examples"
    )

    print(
        f"[DATA] filtered {num_filtered} examples "
        f"with prompt length >= {MAX_TRAIN_LENGTH}"
    )

    print(
        f"[DATA] maximum prompt length among all examples: "
        f"{max_prompt_length}"
    )

    if longest_question is not None:
        config, idx, _ = longest_question
        print(
            f"[DATA] longest prompt: "
            f"{config}[{idx}] -> {max_prompt_length} tokens"
        )

    return all_samples

def load_math_train_fallback():
    """
    Fallback loader.

    This intentionally uses the same actual directory structure:
        MATH/<config>/train
    """
    print("[DATA] using fallback MATH train loader ...")

    all_samples = []

    for config in MATH_CONFIGS:
        path = os.path.join(DATASET_ROOT, config, "train")

        if not os.path.isdir(path):
            print(f"[WARN] missing directory: {path}")
            continue

        ds = load_from_disk(path)

        print(f"[DATA] {config}: {len(ds)} examples")

        for idx, item in enumerate(ds):
            question = (
                item.get("problem")
                or item.get("question")
                or item.get("instruction")
            )

            ground_truth = get_ground_truth(item)

            if not question or ground_truth is None:
                continue

            all_samples.append({
                "config": config,
                "index": idx,
                "question": str(question),
                "ground_truth": str(ground_truth),
            })

    print(f"[DATA] fallback loaded {len(all_samples)} examples")

    return all_samples

def get_rollout_questions(all_samples, start_idx, batch_size):
    end_idx = min(start_idx + batch_size, len(all_samples))
    return all_samples[start_idx:end_idx]


# ---------------------------------------------------------------------
# vLLM rollout
# ---------------------------------------------------------------------

def build_vllm(model_path):
    """
    Construct a fresh vLLM engine for the current policy checkpoint.

    vLLM is used only during rollout. The HF policy/optimizer are offloaded
    to CPU before this function is called, so vLLM does not compete with them
    for GPU memory.
    """
    from vllm import LLM

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_TRAIN_LENGTH + SAMPLING_MAX_TOKENS,
        enforce_eager=True,
    )

    print("[vLLM] loaded successfully")

    return llm


def rollout_questions(llm, questions, seed):
    """
    Generate GROUP_SIZE responses for every question.

    SamplingParams.n == GROUP_SIZE, exactly matching the assignment.
    """
    from vllm import SamplingParams

    prompts = [
        build_prompt(item["question"])
        for item in questions
    ]

    sampling_params = SamplingParams(
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        max_tokens=SAMPLING_MAX_TOKENS,
        min_tokens=SAMPLING_MIN_TOKENS,
        n=GROUP_SIZE,
        stop=[STOP_STR],
        include_stop_str_in_output=True,
        seed=seed,
    )

    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    return outputs


# ---------------------------------------------------------------------
# Reward / advantage
# ---------------------------------------------------------------------

def build_rollout_records(questions, outputs):
    """
    Convert vLLM outputs into flat trajectory records.

    outputs[i].outputs contains GROUP_SIZE sampled responses for
    questions[i].
    """
    records = []

    total_reward = 0.0
    total_format = 0.0
    total_answer = 0.0

    group_all_correct = 0
    group_all_wrong = 0
    group_mixed = 0

    for q_idx, (question, output) in enumerate(zip(questions, outputs)):
        candidates = []

        for candidate_idx, candidate in enumerate(output.outputs):
            response = candidate.text

            reward_result = r1_zero_reward_fn(
                response,
                question["ground_truth"],
                fast=True,
            )

            format_reward = reward_result["format_reward"]
            answer_reward = reward_result["answer_reward"]
            reward = reward_result["reward"]

            candidates.append(
                {
                    "response": response,
                    "reward": float(reward),
                    "format_reward": float(format_reward),
                    "answer_reward": float(answer_reward),
                    "candidate_index": candidate_idx,
                }
            )

            total_reward += float(reward)
            total_format += float(format_reward)
            total_answer += float(answer_reward)

        rewards = torch.tensor(
            [x["reward"] for x in candidates],
            dtype=torch.float32,
        )

        mean_reward = rewards.mean()

        # Assignment formula:
        #
        # A_i = r_i - mean(r_1,...,r_G)
        #
        advantages = rewards - mean_reward

        # Numerical safety only.
        # We DO NOT divide by std: the assignment explicitly defines
        # advantage as reward minus group mean.
        if torch.isnan(advantages).any():
            raise RuntimeError("NaN detected in group advantages.")

        reward_sum = float(rewards.sum())

        if reward_sum == 0:
            group_all_wrong += 1
        elif reward_sum == GROUP_SIZE:
            group_all_correct += 1
        else:
            group_mixed += 1

        for candidate, advantage in zip(candidates, advantages.tolist()):
            records.append(
                {
                    "question": question["question"],
                    "ground_truth": question["ground_truth"],
                    "config": question["config"],
                    "dataset_index": question["index"],
                    "response": candidate["response"],
                    "reward": candidate["reward"],
                    "format_reward": candidate["format_reward"],
                    "answer_reward": candidate["answer_reward"],
                    "advantage": float(advantage),
                    "candidate_index": candidate["candidate_index"],
                }
            )

    n_questions = len(questions)
    n_trajectories = len(records)

    stats = {
        "questions": n_questions,
        "trajectories": n_trajectories,
        "reward_mean": total_reward / max(n_trajectories, 1),
        "format_mean": total_format / max(n_trajectories, 1),
        "answer_mean": total_answer / max(n_trajectories, 1),
        "groups_all_correct": group_all_correct,
        "groups_all_wrong": group_all_wrong,
        "groups_mixed": group_mixed,
    }

    return records, stats


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

def tokenize_trajectory(tokenizer, record, max_length):
    """
    Tokenize:
        prompt + response

    The response-only mask is exactly aligned with the generated response.

    Important:
      - prompt tokens are mask=0
      - response tokens are mask=1
      - right truncate response if necessary
    """
    prompt = build_prompt(record["question"])
    response = record["response"]

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]

    available = max_length - len(prompt_ids)

    if available <= 0:
        return None

    response_ids = response_ids[:available]

    if len(response_ids) == 0:
        return None

    input_ids = prompt_ids + response_ids

    response_mask = (
        [0] * len(prompt_ids)
        + [1] * len(response_ids)
    )

    return {
        "input_ids": input_ids,
        "response_mask": response_mask,
        "response_tokens": len(response_ids),
        "truncated": len(response_ids) < len(
            tokenizer(
                response,
                add_special_tokens=False,
            )["input_ids"]
        ),
    }


def make_training_examples(tokenizer, records, max_length):
    examples = []

    dropped = 0
    truncated = 0

    for record in records:
        ex = tokenize_trajectory(
            tokenizer,
            record,
            max_length,
        )

        if ex is None:
            dropped += 1
            continue

        if ex["truncated"]:
            truncated += 1

        ex["record"] = record
        examples.append(ex)

    return examples, dropped, truncated


# ---------------------------------------------------------------------
# Log probability
# ---------------------------------------------------------------------

LOGPROB_CHUNK_SIZE = 128


def sequence_token_logprob(
    model,
    input_ids,
    attention_mask,
    response_mask,
):
    """
    Return per-response-token log probabilities.

    Output:
        [batch, sequence-1]

    Non-response positions are zeroed.

    Uses:
        log p(token) = target_logit - logsumexp(logits)
    instead of materializing full log_softmax tensor in FP32.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]

    token_logp = torch.zeros(
        shift_labels.shape,
        dtype=torch.float32,
        device=input_ids.device,
    )

    for start in range(
        0,
        shift_logits.size(1),
        LOGPROB_CHUNK_SIZE,
    ):
        end = min(
            start + LOGPROB_CHUNK_SIZE,
            shift_logits.size(1),
        )

        chunk = shift_logits[:, start:end, :].float()
        labels = shift_labels[:, start:end]

        target_logits = torch.gather(
            chunk,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        log_norm = torch.logsumexp(
            chunk,
            dim=-1,
        )

        chunk_logp = target_logits - log_norm

        token_logp[:, start:end] = chunk_logp

        del chunk
        del labels
        del target_logits
        del log_norm
        del chunk_logp

    token_logp = token_logp * shift_mask

    return token_logp


# ---------------------------------------------------------------------
# Training batch
# ---------------------------------------------------------------------

def prepare_batch(examples, start, end, device):
    batch = examples[start:end]

    max_len = max(
        len(x["input_ids"])
        for x in batch
    )

    batch_size = len(batch)

    input_ids = torch.full(
        (batch_size, max_len),
        fill_value=0,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    response_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.float32,
    )

    advantages = torch.zeros(
        batch_size,
        dtype=torch.float32,
    )

    rewards = torch.zeros(
        batch_size,
        dtype=torch.float32,
    )

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        mask = item["response_mask"]

        length = len(ids)

        input_ids[i, :length] = torch.tensor(
            ids,
            dtype=torch.long,
        )

        attention_mask[i, :length] = 1

        response_mask[i, :length] = torch.tensor(
            mask,
            dtype=torch.float32,
        )

        advantages[i] = item["record"]["advantage"]
        rewards[i] = item["record"]["reward"]

    return {
        "input_ids": input_ids.to(
            device,
            non_blocking=True,
        ),
        "attention_mask": attention_mask.to(
            device,
            non_blocking=True,
        ),
        "response_mask": response_mask.to(
            device,
            non_blocking=True,
        ),
        "advantages": advantages.to(
            device,
            non_blocking=True,
        ),
        "rewards": rewards.to(
            device,
            non_blocking=True,
        ),
    }


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_model(model, tokenizer, output_dir, tag=None):
    out = Path(output_dir)

    if tag is not None:
        out = out / tag

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        out,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(out)

    print(
        f"[SAVE] {out}",
        flush=True,
    )


# ---------------------------------------------------------------------
# vLLM lifecycle
# ---------------------------------------------------------------------

def release_vllm(llm):
    """
    Explicitly release vLLM GPU memory before loading the HF training model.
    """
    if llm is None:
        return

    try:
        del llm
    except Exception:
        pass

    gc.collect()
    torch.cuda.empty_cache()

    # A second collection is useful because vLLM owns worker/process
    # resources that can otherwise be released one GC cycle later.
    gc.collect()
    torch.cuda.empty_cache()

    cuda_mem("after vLLM release")


def move_optimizer_to_cpu(optimizer):
    """
    Move all optimizer state tensors to CPU while preserving the optimizer
    object and AdamW momentum/variance state across GRPO steps.
    """
    if optimizer is None:
        return

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.detach().cpu()


def move_optimizer_to_device(optimizer, device):
    """
    Move optimizer state tensors back to the training device.
    """
    if optimizer is None:
        return

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def offload_policy_for_vllm(model, optimizer=None):
    """
    Move the HF policy and AdamW state off GPU before vLLM rollout.

    We intentionally preserve the optimizer object/state on CPU instead of
    deleting it. This keeps AdamW's exp_avg/exp_avg_sq momentum across GRPO
    steps while freeing GPU memory for vLLM.
    """
    if model is not None:
        model.zero_grad(set_to_none=True)
        model.to("cpu")

    move_optimizer_to_cpu(optimizer)

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    cuda_mem("after HF policy CPU offload")


def load_policy(model_path, device):
    """
    Load a fresh HF policy on GPU for GRPO training.
    """
    print(
        f"[MODEL] loading policy from {model_path}",
        flush=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
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

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"[MODEL] total      = {total_params:,}",
        flush=True,
    )
    print(
        f"[MODEL] trainable   = {trainable_params:,}",
        flush=True,
    )

    if total_params != trainable_params:
        raise RuntimeError(
            "GRPO requires full-parameter training."
        )

    cuda_mem("after policy load")

    return model


def build_optimizer(model, lr):
    """
    Build a fresh AdamW optimizer for the currently loaded HF policy.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(BETA1, BETA2),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )


def prepare_sync_model(model, tokenizer, sync_dir):
    """
    Save the current policy to a temporary directory used by the next
    vLLM rollout.

    The directory is replaced atomically at the filesystem level as far
    as practical: write to sync_dir.tmp, then replace sync_dir.
    """
    sync_path = Path(sync_dir)
    tmp_path = Path(str(sync_dir) + ".tmp")

    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    tmp_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        tmp_path,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(tmp_path)

    if sync_path.exists():
        shutil.rmtree(sync_path)

    tmp_path.rename(sync_path)

    print(
        f"[SYNC] policy checkpoint prepared: {sync_path}",
        flush=True,
    )


# ---------------------------------------------------------------------
# Rollout persistence
# ---------------------------------------------------------------------

def save_rollout(records, path, step):
    out = Path(path)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = out / f"step-{step:04d}.jsonl"

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[ROLLOUT SAVE] {file_path}",
        flush=True,
    )


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def build_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--tensorboard_dir",
        default=DEFAULT_TB,
    )

    parser.add_argument(
        "--rollout_dir",
        default=DEFAULT_ROLLOUT_DIR,
    )

    parser.add_argument(
        "--sync_dir",
        default=DEFAULT_SYNC_DIR,
    )

    parser.add_argument(
        "--n_grpo_steps",
        type=int,
        default=N_GRPO_STEPS,
    )

    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=ROLLOUT_BATCH_SIZE,
    )

    parser.add_argument(
        "--group_size",
        type=int,
        default=GROUP_SIZE,
    )

    parser.add_argument(
        "--epochs_per_rollout_batch",
        type=int,
        default=EPOCHS_PER_ROLLOUT_BATCH,
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=TRAIN_BATCH_SIZE,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--clip_eps",
        type=float,
        default=CLIP_EPS,
    )

    parser.add_argument(
        "--max_train_length",
        type=int,
        default=MAX_TRAIN_LENGTH,
    )

    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=SAMPLING_TEMPERATURE,
    )

    parser.add_argument(
        "--sampling_max_tokens",
        type=int,
        default=SAMPLING_MAX_TOKENS,
    )

    parser.add_argument(
        "--limit_questions",
        type=int,
        default=None,
        help="For sanity checking only.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a HF model directory.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = build_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if args.group_size != 8:
        raise ValueError(
            "This implementation follows the assignment with group_size=8."
        )

    if args.rollout_batch_size % args.train_batch_size != 0:
        raise ValueError(
            "rollout_batch_size must be divisible by train_batch_size."
        )

    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError(
            "train_batch_size must be divisible by gradient_accumulation_steps."
        )

    set_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")

    print("=" * 80)
    print("GRPO TRAINING")
    print("=" * 80)
    print(f"initial_model              = {args.model}")
    print(f"output_dir                 = {args.output_dir}")
    print(f"rollout_batch_size        = {args.rollout_batch_size}")
    print(f"group_size                = {args.group_size}")
    print(f"n_grpo_steps              = {args.n_grpo_steps}")
    print(f"epochs_per_rollout_batch  = {args.epochs_per_rollout_batch}")
    print(f"train_batch_size          = {args.train_batch_size}")
    print(f"gradient_accumulation     = {args.gradient_accumulation_steps}")
    print(f"learning_rate             = {args.lr}")
    print(f"clip_eps                  = {args.clip_eps}")
    print(f"sampling_temperature      = {args.sampling_temperature}")
    print(f"sampling_max_tokens       = {args.sampling_max_tokens}")
    print(f"max_train_length          = {args.max_train_length}")
    print("=" * 80)

    # -----------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------

    print("[DATA] loading MATH train ...", flush=True)


    all_samples = load_math_train()


    print(
        f"[DATA] total usable questions = {len(all_samples)}",
        flush=True,
    )

    if args.limit_questions is not None:
        all_samples = all_samples[
            :args.limit_questions
        ]

        print(
            f"[DATA] sanity limit = {len(all_samples)}",
            flush=True,
        )

    # -----------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.resume or args.model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------------------
    # Initial policy / optimizer
    # -----------------------------------------------------------------

    # We keep the current policy on CPU initially. The first rollout uses
    # the saved input checkpoint directly, so there is no reason to occupy
    # GPU memory with the HF policy before vLLM starts.
    policy_path = args.resume or args.model

    writer = SummaryWriter(
        args.tensorboard_dir
    )

    print(
        "[SYNC] preparing initial policy for vLLM rollout ...",
        flush=True,
    )

    # Make the initial sync checkpoint available to vLLM.
    if Path(args.sync_dir).resolve() != Path(policy_path).resolve():
        if Path(args.sync_dir).exists():
            shutil.rmtree(args.sync_dir)
        shutil.copytree(policy_path, args.sync_dir)

    cuda_mem("before first vLLM")

    # -----------------------------------------------------------------
    # GRPO loop
    # -----------------------------------------------------------------

    global_optimizer_step = 0

    total_questions = len(all_samples)

    question_cursor = 0

    training_start = time.time()

    for grpo_step in range(
        1,
        args.n_grpo_steps + 1,
    ):

        step_start = time.time()

        print()
        print("=" * 80)
        print(
            f"[GRPO STEP {grpo_step}/{args.n_grpo_steps}]"
        )
        print("=" * 80)

        # -------------------------------------------------------------
        # Select rollout questions
        # -------------------------------------------------------------

        questions = get_rollout_questions(
            all_samples,
            question_cursor,
            args.rollout_batch_size,
        )

        # Cycle through the training set.
        if len(questions) < args.rollout_batch_size:
            first_part = questions

            remaining = (
                args.rollout_batch_size
                - len(first_part)
            )

            second_part = all_samples[
                :remaining
            ]

            questions = (
                first_part
                + second_part
            )

            question_cursor = remaining
        else:
            question_cursor += args.rollout_batch_size

        if question_cursor >= total_questions:
            question_cursor = 0

        print(
            f"[ROLLOUT] questions={len(questions)} "
            f"trajectories={len(questions) * args.group_size}",
            flush=True,
        )

        # -------------------------------------------------------------
        # Rollout with vLLM
        # -------------------------------------------------------------

        print(
            "[ROLLOUT] starting vLLM ...",
            flush=True,
        )

        # IMPORTANT:
        # The HF policy and AdamW optimizer must NOT be resident on GPU here.
        # Keep them on CPU between training and rollout.
        if "model" in locals() and model is not None:
            if next(model.parameters()).device.type != "cpu":
                offload_policy_for_vllm(model, optimizer)

        cuda_mem("before vLLM")

        llm = build_vllm(
            args.sync_dir
        )

        cuda_mem("after vLLM load")

        outputs = rollout_questions(
            llm,
            questions,
            seed=args.seed + grpo_step,
        )

        # -------------------------------------------------------------
        # Reward + advantage
        # -------------------------------------------------------------

        records, rollout_stats = build_rollout_records(
            questions,
            outputs,
        )

        print(
            f"[ROLLOUT] reward_mean       = "
            f"{rollout_stats['reward_mean']:.6f}",
            flush=True,
        )

        print(
            f"[ROLLOUT] format_mean       = "
            f"{rollout_stats['format_mean']:.6f}",
            flush=True,
        )

        print(
            f"[ROLLOUT] answer_mean       = "
            f"{rollout_stats['answer_mean']:.6f}",
            flush=True,
        )

        print(
            f"[ROLLOUT] all_wrong_groups  = "
            f"{rollout_stats['groups_all_wrong']}",
            flush=True,
        )

        print(
            f"[ROLLOUT] all_correct_groups = "
            f"{rollout_stats['groups_all_correct']}",
            flush=True,
        )

        print(
            f"[ROLLOUT] mixed_groups      = "
            f"{rollout_stats['groups_mixed']}",
            flush=True,
        )

        # Save rollout before freeing vLLM.
        save_rollout(
            records,
            args.rollout_dir,
            grpo_step,
        )

        # -------------------------------------------------------------
        # Release vLLM
        # -------------------------------------------------------------

        release_vllm(llm)
        llm = None

        # -------------------------------------------------------------
        # Load HF policy + optimizer for training
        # -------------------------------------------------------------

        # The policy used for this update is exactly the policy that
        # generated the rollout (args.sync_dir).
        #
        # Loading it only AFTER vLLM has been destroyed guarantees that
        # rollout and HF training never compete for GPU memory.
        if "model" not in locals() or model is None:
            model = load_policy(
                args.sync_dir,
                device,
            )

            optimizer = build_optimizer(
                model,
                args.lr,
            )
        else:
            # The policy was kept on CPU during rollout. Bring only the
            # training policy and its AdamW states back to GPU.
            model.to(device)
            move_optimizer_to_device(optimizer, device)
            model.train()
            cuda_mem("after HF policy GPU reload")

        # -------------------------------------------------------------
        # Tokenize trajectories
        # -------------------------------------------------------------

        examples, dropped, truncated = (
            make_training_examples(
                tokenizer,
                records,
                args.max_train_length,
            )
        )

        print(
            f"[TRAIN DATA] trajectories={len(records)} "
            f"kept={len(examples)} "
            f"dropped={dropped} "
            f"truncated={truncated}",
            flush=True,
        )

        if len(examples) == 0:
            raise RuntimeError(
                "No valid GRPO training trajectories."
            )

        # -------------------------------------------------------------
        # Advantage statistics
        # -------------------------------------------------------------

        advantage_tensor = torch.tensor(
            [
                x["record"]["advantage"]
                for x in examples
            ],
            dtype=torch.float32,
        )

        reward_tensor = torch.tensor(
            [
                x["record"]["reward"]
                for x in examples
            ],
            dtype=torch.float32,
        )

        print(
            f"[ADV] mean={advantage_tensor.mean():.6f} "
            f"std={advantage_tensor.std(unbiased=False):.6f} "
            f"min={advantage_tensor.min():.6f} "
            f"max={advantage_tensor.max():.6f}",
            flush=True,
        )

        # The group-centered advantage should have mean ~0 within
        # each group. Small deviations here are only due to truncation
        # filtering if some trajectories were removed.
        #
        # We do NOT normalize by std.

        # -------------------------------------------------------------
        # OLD LOGPROB
        # -------------------------------------------------------------

        print(
            "[OLD LOGP] computing rollout-policy log probabilities ...",
            flush=True,
        )

        model.eval()

        old_logps = []

        with torch.no_grad():
            for start in range(
                0,
                len(examples),
                args.train_batch_size,
            ):
                end = min(
                    start + args.train_batch_size,
                    len(examples),
                )

                batch = prepare_batch(
                    examples,
                    start,
                    end,
                    device,
                )

                token_logp = sequence_token_logprob(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["response_mask"],
                )

                for row_index, example_index in enumerate(
                    range(start, end)
                ):
                    sequence_length = len(
                        examples[example_index]["input_ids"]
                    )

                    old_logp = token_logp[
                        row_index,
                        :sequence_length - 1,
                    ].detach().cpu()

                    examples[example_index][
                        "old_logp"
                    ] = old_logp
                    old_logps.append(old_logp)

                del batch
                del token_logp

                torch.cuda.empty_cache()

        print(
            f"[OLD LOGP] trajectories={len(old_logps)} "
            f"min_length={min(x.numel() for x in old_logps)} "
            f"max_length={max(x.numel() for x in old_logps)} "
            f"mean={torch.cat(old_logps).mean().item():.6f}",
            flush=True,
        )

        # -------------------------------------------------------------
        # GRPO UPDATE
        # -------------------------------------------------------------

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        n_examples = len(examples)

        n_micro_batches = math.ceil(
            n_examples /
            args.train_batch_size
        )

        optimizer_steps_this_rollout = 0

        epoch_losses = []
        epoch_ratios = []
        epoch_clip_fractions = []
        epoch_advantages = []

        for local_epoch in range(
            args.epochs_per_rollout_batch
        ):

            print(
                f"[TRAIN] rollout_epoch "
                f"{local_epoch + 1}/"
                f"{args.epochs_per_rollout_batch}",
                flush=True,
            )

            epoch_loss = 0.0
            epoch_count = 0

            for mb_idx, start in enumerate(
                range(
                    0,
                    n_examples,
                    args.train_batch_size,
                )
            ):

                end = min(
                    start + args.train_batch_size,
                    n_examples,
                )

                batch = prepare_batch(
                    examples,
                    start,
                    end,
                    device,
                )

                current_batch_size = end - start

                # -----------------------------------------------------
                # Current policy log probability
                # -----------------------------------------------------

                token_logp = sequence_token_logprob(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["response_mask"],
                )

                batch_old_logps = [
                    x["old_logp"]
                    for x in examples[start:end]
                ]

                old_token_logp = torch.nn.utils.rnn.pad_sequence(
                    batch_old_logps,
                    batch_first=True,
                    padding_value=0.0,
                ).to(
                    device,
                    non_blocking=True,
                )

                old_logp_mask = torch.nn.utils.rnn.pad_sequence(
                    [
                        torch.ones_like(
                            x,
                            dtype=torch.bool,
                        )
                        for x in batch_old_logps
                    ],
                    batch_first=True,
                    padding_value=False,
                ).to(
                    device,
                    non_blocking=True,
                )

                # -----------------------------------------------------
                # Token-level ratio
                # -----------------------------------------------------

                log_ratio = (
                    token_logp
                    - old_token_logp
                )

                # Important numerical protection.
                log_ratio = torch.clamp(
                    log_ratio,
                    min=-20.0,
                    max=20.0,
                )

                ratio = torch.exp(
                    log_ratio
                )

                # -----------------------------------------------------
                # Advantage broadcast
                # -----------------------------------------------------

                advantages = batch[
                    "advantages"
                ].unsqueeze(-1)

                response_mask = batch[
                    "response_mask"
                ][:, 1:]
                response_mask = (
                    response_mask
                    * old_logp_mask.to(
                        response_mask.dtype
                    )
                )

                # -----------------------------------------------------
                # GRPO / PPO clipped objective
                # -----------------------------------------------------

                unclipped = (
                    ratio * advantages
                )

                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - args.clip_eps,
                    1.0 + args.clip_eps,
                )

                clipped = (
                    clipped_ratio
                    * advantages
                )

                surrogate = torch.minimum(
                    unclipped,
                    clipped,
                )

                # Only response tokens participate.
                token_objective = (
                    surrogate
                    * response_mask
                )

                valid_tokens = (
                    response_mask.sum()
                    .clamp_min(1.0)
                )

                loss = (
                    -token_objective.sum()
                    / valid_tokens
                )

                # -----------------------------------------------------
                # Statistics
                # -----------------------------------------------------

                with torch.no_grad():
                    active_ratio = ratio[
                        response_mask > 0
                    ]

                    active_log_ratio = log_ratio[
                        response_mask > 0
                    ]

                    active_adv = advantages.expand_as(
                        ratio
                    )[
                        response_mask > 0
                    ]

                    clip_fraction = (
                        (
                            torch.abs(
                                ratio - 1.0
                            )
                            > args.clip_eps
                        )
                        & (response_mask > 0)
                    ).float().sum() / valid_tokens

                    ratio_mean = (
                        active_ratio.mean()
                        if active_ratio.numel()
                        else torch.tensor(
                            1.0,
                            device=device,
                        )
                    )

                    log_ratio_mean = (
                        active_log_ratio.mean()
                        if active_log_ratio.numel()
                        else torch.tensor(
                            0.0,
                            device=device,
                        )
                    )

                    adv_mean = (
                        active_adv.mean()
                        if active_adv.numel()
                        else torch.tensor(
                            0.0,
                            device=device,
                        )

                    )

                # -----------------------------------------------------
                # Gradient accumulation
                # -----------------------------------------------------

                # The configured effective batch is 8 * 4 = 32.
                #
                # For the last incomplete accumulation group, use the
                # actual number of micro-batches so its gradient scale
                # remains correct.
                group_start = (
                    mb_idx
                    // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps

                group_end = min(
                    group_start
                    + args.gradient_accumulation_steps,
                    n_micro_batches,
                )

                actual_accum = (
                    group_end - group_start
                )

                scaled_loss = (
                    loss
                    / actual_accum
                )

                scaled_loss.backward()

                epoch_loss += float(
                    loss.detach().cpu()
                )
                epoch_count += 1

                epoch_ratios.append(
                    float(
                        ratio_mean.detach().cpu()
                    )
                )

                epoch_clip_fractions.append(
                    float(
                        clip_fraction.detach().cpu()
                    )
                )

                epoch_advantages.append(
                    float(
                        adv_mean.detach().cpu()
                    )
                )

                do_optimizer_step = (
                    mb_idx + 1
                ) % args.gradient_accumulation_steps == 0

                if (
                    mb_idx + 1 == n_micro_batches
                ):
                    do_optimizer_step = True

                if do_optimizer_step:
                    grad_norm = (
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=1.0,
                        )
                    )

                    optimizer.step()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    global_optimizer_step += 1
                    optimizer_steps_this_rollout += 1

                    elapsed = (
                        time.time()
                        - training_start
                    )

                    print(
                        f"[TRAIN] "
                        f"grpo_step={grpo_step} "
                        f"epoch={local_epoch + 1}/"
                        f"{args.epochs_per_rollout_batch} "
                        f"mb={mb_idx + 1}/"
                        f"{n_micro_batches} "
                        f"loss={float(loss.detach().cpu()):.6f} "
                        f"ratio={float(ratio_mean.detach().cpu()):.6f} "
                        f"clip={float(clip_fraction.detach().cpu()):.6f} "
                        f"grad={float(grad_norm.detach().cpu()):.4f} "
                        f"time={elapsed/60:.1f}m",
                        flush=True,
                    )

                    writer.add_scalar(
                        "grpo/loss",
                        float(
                            loss.detach().cpu()
                        ),
                        global_optimizer_step,
                    )

                    writer.add_scalar(
                        "grpo/ratio_mean",
                        float(
                            ratio_mean.detach().cpu()
                        ),
                        global_optimizer_step,
                    )

                    writer.add_scalar(
                        "grpo/log_ratio_mean",
                        float(
                            log_ratio_mean.detach().cpu()
                        ),
                        global_optimizer_step,
                    )

                    writer.add_scalar(
                        "grpo/clip_fraction",
                        float(
                            clip_fraction.detach().cpu()
                        ),
                        global_optimizer_step,
                    )

                    writer.add_scalar(
                        "grpo/advantage_mean",
                        float(
                            adv_mean.detach().cpu()
                        ),
                        global_optimizer_step,
                    )

                    writer.add_scalar(
                        "train/grad_norm",
                        float(
                            grad_norm.detach().cpu()
                        ),
                        global_optimizer_step,
                    )

                    writer.add_scalar(
                        "train/reward_mean",
                        rollout_stats[
                            "reward_mean"
                        ],
                        global_optimizer_step,
                    )

                    writer.add_scalar(
                        "train/format_mean",
                        rollout_stats[
                            "format_mean"
                        ],
                        global_optimizer_step,
                    )

                    writer.add_scalar(
                        "train/answer_mean",
                        rollout_stats[
                            "answer_mean"
                        ],
                        global_optimizer_step,
                    )

                del batch
                del token_logp
                del old_token_logp
                del old_logp_mask
                del batch_old_logps
                del log_ratio
                del ratio
                del advantages
                del response_mask
                del unclipped
                del clipped_ratio
                del clipped
                del surrogate
                del token_objective
                del loss

                torch.cuda.empty_cache()

            mean_epoch_loss = (
                epoch_loss
                / max(epoch_count, 1)
            )

            mean_ratio = (
                sum(epoch_ratios)
                / max(len(epoch_ratios), 1)
            )

            mean_clip = (
                sum(epoch_clip_fractions)
                / max(
                    len(epoch_clip_fractions),
                    1,
                )
            )

            print(
                f"[TRAIN EPOCH] "
                f"epoch={local_epoch + 1} "
                f"loss={mean_epoch_loss:.6f} "
                f"ratio={mean_ratio:.6f} "
                f"clip_fraction={mean_clip:.6f}",
                flush=True,
            )

        # -------------------------------------------------------------
        # Save current policy for next rollout
        # -------------------------------------------------------------

        print(
            "[SYNC] preparing updated policy for next rollout ...",
            flush=True,
        )

        model.eval()

        prepare_sync_model(
            model,
            tokenizer,
            args.sync_dir,
        )

        model.train()

        # -------------------------------------------------------------
        # Periodic checkpoint
        # -------------------------------------------------------------

        if (
            grpo_step % args.save_every == 0
            or grpo_step == args.n_grpo_steps
        ):
            save_model(
                model,
                tokenizer,
                args.output_dir,
                f"step-{grpo_step}",
            )

        # -------------------------------------------------------------
        # Offload HF policy + optimizer BEFORE the next rollout
        # -------------------------------------------------------------

        # Keep the model/optimizer alive on CPU so AdamW momentum is preserved,
        # but free their GPU memory before the next vLLM startup.
        offload_policy_for_vllm(
            model,
            optimizer,
        )

        # -------------------------------------------------------------
        # Step-level statistics
        # -------------------------------------------------------------

        step_time = (
            time.time()
            - step_start
        )

        advantage_std = float(
            advantage_tensor.std(
                unbiased=False
            )
        )

        mean_ratio = (
            sum(epoch_ratios)
            / max(len(epoch_ratios), 1)
        )

        mean_clip = (
            sum(epoch_clip_fractions)
            / max(
                len(epoch_clip_fractions),
                1,
            )
        )

        print("-" * 80)
        print(
            f"[GRPO STEP {grpo_step}] COMPLETE"
        )
        print(
            f"reward_mean          = "
            f"{rollout_stats['reward_mean']:.6f}"
        )
        print(
            f"format_mean          = "
            f"{rollout_stats['format_mean']:.6f}"
        )
        print(
            f"answer_mean          = "
            f"{rollout_stats['answer_mean']:.6f}"
        )
        print(
            f"advantage_mean       = "
            f"{advantage_tensor.mean().item():.6f}"
        )
        print(
            f"advantage_std        = "
            f"{advantage_std:.6f}"
        )
        print(
            f"mixed_groups         = "
            f"{rollout_stats['groups_mixed']}"
        )
        print(
            f"all_wrong_groups     = "
            f"{rollout_stats['groups_all_wrong']}"
        )
        print(
            f"all_correct_groups   = "
            f"{rollout_stats['groups_all_correct']}"
        )
        print(
            f"optimizer_steps      = "
            f"{optimizer_steps_this_rollout}"
        )
        print(
            f"ratio_mean           = "
            f"{mean_ratio:.6f}"
        )
        print(
            f"clip_fraction        = "
            f"{mean_clip:.6f}"
        )
        print(
            f"step_time            = "
            f"{step_time/60:.2f} min"
        )
        print("-" * 80)

        writer.add_scalar(
            "rollout/reward_mean",
            rollout_stats["reward_mean"],
            grpo_step,
        )

        writer.add_scalar(
            "rollout/format_mean",
            rollout_stats["format_mean"],
            grpo_step,
        )

        writer.add_scalar(
            "rollout/answer_mean",
            rollout_stats["answer_mean"],
            grpo_step,
        )

        writer.add_scalar(
            "rollout/advantage_mean",
            float(
                advantage_tensor.mean()
            ),
            grpo_step,
        )

        writer.add_scalar(
            "rollout/advantage_std",
            advantage_std,
            grpo_step,
        )

        writer.add_scalar(
            "rollout/mixed_groups",
            rollout_stats["groups_mixed"],
            grpo_step,
        )

        writer.add_scalar(
            "rollout/all_wrong_groups",
            rollout_stats["groups_all_wrong"],
            grpo_step,
        )

        writer.add_scalar(
            "rollout/all_correct_groups",
            rollout_stats["groups_all_correct"],
            grpo_step,
        )

        writer.add_scalar(
            "rollout/ratio_mean",
            mean_ratio,
            grpo_step,
        )

        writer.add_scalar(
            "rollout/clip_fraction",
            mean_clip,
            grpo_step,
        )


    # -----------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------

    print("=" * 80)
    print("GRPO TRAINING COMPLETE")
    print("=" * 80)

    # The final policy remains on CPU after the last offload. Save it directly
    # without reloading another copy onto GPU.
    model.eval()

    save_model(
        model,
        tokenizer,
        args.output_dir,
    )

    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    writer.close()

    cuda_mem("final")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()