import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import random
import argparse
from pathlib import Path

from datasets import load_from_disk, DatasetDict
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baseline.prompt import build_prompt
from baseline.grader import r1_zero_reward_fn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)


def extract_boxed_answer(text: str):
    """
    Extract the last \\boxed{...} from a solution.

    This is mainly used as a fallback when loading MATH data.
    """
    if not text:
        return None

    marker = r"\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    i = start + len(marker)
    depth = 1

    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                return text[start + len(marker):i]

        i += 1

    return None


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def get_split_from_loaded_dataset(obj):
    """
    Normalize Dataset / DatasetDict into a train split.
    """
    if isinstance(obj, DatasetDict):
        if "train" in obj:
            return obj["train"]

        # Fallback: first available split.
        first_key = next(iter(obj.keys()))
        return obj[first_key]

    return obj


def load_math_train(data_root: str):
    """
    Load all MATH train examples from the seven configs.

    Supports layouts such as:

        DATA_ROOT/algebra/train
        DATA_ROOT/algebra

    and DatasetDict / Dataset saved by datasets.save_to_disk().
    """

    all_data = []

    for config in CONFIGS:
        candidates = [
            os.path.join(data_root, config, "train"),
            os.path.join(data_root, config),
        ]

        dataset = None
        selected_path = None

        for path in candidates:
            if os.path.exists(path):
                try:
                    loaded = load_from_disk(path)
                    dataset = get_split_from_loaded_dataset(loaded)
                    selected_path = path
                    break
                except Exception:
                    continue

        if dataset is None:
            raise FileNotFoundError(
                f"Cannot load config '{config}'. "
                f"Tried:\n"
                + "\n".join(candidates)
            )

        print(
            f"[Dataset] {config:30s} "
            f"N={len(dataset):5d} "
            f"path={selected_path}"
        )

        for idx, sample in enumerate(dataset):
            question = (
                sample.get("question")
                or sample.get("problem")
                or sample.get("prompt")
            )

            solution = (
                sample.get("solution")
                or sample.get("rationale")
                or sample.get("completion")
                or ""
            )

            answer = (
                sample.get("answer")
                or sample.get("ground_truth")
            )

            # Prefer the final boxed answer from the official solution.
            boxed = extract_boxed_answer(solution)

            if boxed is not None:
                answer = boxed

            if not question:
                raise ValueError(
                    f"Missing question in config={config}, idx={idx}"
                )

            if answer is None:
                print(
                    f"[Skip] Missing ground-truth answer "
                    f"in config={config}, idx={idx}"
                )
                continue

            sample_id = (
                sample.get("id")
                or sample.get("problem_id")
                or sample.get("uid")
                or f"{config}_{idx}"
            )

            all_data.append(
                {
                    "id": str(sample_id),
                    "config": config,
                    "question": str(question),
                    "answer": str(answer),
                }
            )

    return all_data


# ---------------------------------------------------------------------------
# Reward calculation
# ---------------------------------------------------------------------------

def grade_response(response: str, ground_truth: str):
    """
    Return the assignment reward tuple.

    Expected:
        format_reward
        answer_reward
        reward

    r1_zero_reward_fn already implements the exact assignment grading rule.
    """
    result = r1_zero_reward_fn(
        response,
        ground_truth,
        fast=True,
    )

    if isinstance(result, dict):
        format_reward = result.get("format_reward", 0)
        answer_reward = result.get("answer_reward", 0)
        reward = result.get("reward", 0)

        return (
            float(format_reward),
            float(answer_reward),
            float(reward),
        )

    if isinstance(result, (tuple, list)):
        if len(result) >= 3:
            return (
                float(result[0]),
                float(result[1]),
                float(result[2]),
            )

    raise RuntimeError(
        "Unexpected return value from r1_zero_reward_fn: "
        f"{result!r}"
    )


# ---------------------------------------------------------------------------
# Preference selection
# ---------------------------------------------------------------------------

def choose_preference_pair(
    candidates,
    prefer_format_correct_rejected=True,
):
    """
    candidates:
        [
            {
                "response": str,
                "format_reward": float,
                "answer_reward": float,
                "reward": float,
                "candidate_index": int,
            },
            ...
        ]

    Selection policy:

    chosen:
        any reward-positive response.

    rejected:
        preferably format-correct but answer-wrong:
            format_reward == 1
            answer_reward == 0
            reward == 0

        If unavailable, fall back to any reward=0 response.

    Returns:
        chosen, rejected
        or
        None, None
    """

    positives = [
        x for x in candidates
        if x["reward"] >= 1.0
    ]

    if not positives:
        return None, None

    # We deliberately prefer a mathematically wrong but correctly
    # formatted rejected response.
    format_wrong_answer = [
        x for x in candidates
        if (
            x["format_reward"] >= 1.0
            and x["answer_reward"] <= 0.0
            and x["reward"] <= 0.0
        )
    ]

    all_negatives = [
        x for x in candidates
        if x["reward"] <= 0.0
    ]

    # Deterministic selection:
    # first positive candidate and first suitable negative candidate.
    chosen = positives[0]

    if (
        prefer_format_correct_rejected
        and format_wrong_answer
    ):
        rejected = format_wrong_answer[0]
    elif all_negatives:
        rejected = all_negatives[0]
    else:
        # All G samples are positive.
        return None, None

    return chosen, rejected


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def append_jsonl(path, item):
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


def load_existing_ids(path):
    """
    Load IDs of questions already processed into dpo_dataset.jsonl.

    This allows the script to resume after interruption.
    """
    ids = set()

    if not os.path.exists(path):
        return ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if item.get("id") is not None:
                ids.add(str(item["id"]))

    return ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Sample DPO preference pairs from RSFT model."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=(
            "/root/autodl-tmp/models/"
            "Qwen2.5-Math-1.5B-RSFT"
        ),
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-tmp/datasets/MATH",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/Qwen/dpo/sampling-v1",
    )

    parser.add_argument(
        "--G",
        type=int,
        default=8,
        help="Number of responses sampled per question.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of prompts sent to vLLM per batch.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--min_tokens",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
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
        help="Only process the first N questions.",
    )

    parser.add_argument(
        "--max_new_samples",
        type=int,
        default=None,
        help=(
            "Stop after writing this many DPO pairs. "
            "Useful for debugging."
        ),
    )

    parser.add_argument(
        "--allow_format_wrong_rejected",
        action="store_true",
        help=(
            "Allow format-wrong responses as rejected if no "
            "format-correct/wrong-answer response exists."
        ),
    )

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    dpo_path = os.path.join(
        args.output_dir,
        "dpo_dataset.jsonl",
    )

    results_path = os.path.join(
        args.output_dir,
        "sampling_results.jsonl",
    )

    summary_path = os.path.join(
        args.output_dir,
        "sampling_summary.json",
    )

    print("=" * 80)
    print("DPO PREFERENCE SAMPLING")
    print("=" * 80)
    print(f"Model path                 : {args.model_path}")
    print(f"Data root                  : {args.data_root}")
    print(f"Output dir                 : {args.output_dir}")
    print(f"G                          : {args.G}")
    print(f"Batch size                 : {args.batch_size}")
    print(f"Temperature                : {args.temperature}")
    print(f"Top-p                      : {args.top_p}")
    print(f"Max tokens                 : {args.max_tokens}")
    print(f"Min tokens                 : {args.min_tokens}")
    print(f"GPU memory utilization     : {args.gpu_memory_utilization}")
    print(f"Seed                       : {args.seed}")
    print(f"Limit                      : {args.limit}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Load MATH train
    # -----------------------------------------------------------------------

    print("\n[1/4] Loading MATH train dataset...")

    data = load_math_train(args.data_root)

    print(f"\nTotal train questions: {len(data)}")

    if args.limit is not None:
        data = data[:args.limit]
        print(
            f"Applying limit={args.limit}, "
            f"questions to process: {len(data)}"
        )

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------

    existing_ids = load_existing_ids(dpo_path)

    if existing_ids:
        print(
            f"\nExisting DPO pairs found: "
            f"{len(existing_ids)}"
        )
        print("These question IDs will be skipped.")

    pending = [
        item
        for item in data
        if item["id"] not in existing_ids
    ]

    print(f"Pending questions: {len(pending)}")

    if not pending:
        print("No pending questions. Nothing to sample.")
        return

    # -----------------------------------------------------------------------
    # Load vLLM
    # -----------------------------------------------------------------------

    print("\n[2/4] Loading RSFT model with vLLM...")

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        n=args.G,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    stats = {
        "total_questions": len(data),
        "pending_questions": len(pending),
        "G": args.G,
        "total_generated_responses": 0,

        "format_correct_responses": 0,
        "answer_correct_responses": 0,
        "reward_positive_responses": 0,

        "questions_with_positive": 0,
        "questions_without_positive": 0,

        "questions_with_pair": 0,
        "questions_without_rejected": 0,

        "pairs_with_format_correct_rejected": 0,
        "pairs_with_format_wrong_rejected": 0,

        "accepted_pairs": len(existing_ids),

        "by_config": {
            config: {
                "questions": 0,
                "positive_questions": 0,
                "pairs": 0,
            }
            for config in CONFIGS
        },
    }

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    print("\n[3/4] Sampling preference pairs...")

    for start in range(
        0,
        len(pending),
        args.batch_size,
    ):
        batch_items = pending[
            start:start + args.batch_size
        ]

        prompts = [
            build_prompt(item["question"])
            for item in batch_items
        ]

        outputs = llm.generate(
            prompts,
            sampling_params,
        )

        for item, request_output in zip(
            batch_items,
            outputs,
        ):

            config = item["config"]

            stats["by_config"][config][
                "questions"
            ] += 1

            candidates = []

            for candidate_index, candidate in enumerate(
                request_output.outputs
            ):

                response = candidate.text

                format_reward, answer_reward, reward = (
                    grade_response(
                        response,
                        item["answer"],
                    )
                )

                stats[
                    "total_generated_responses"
                ] += 1

                if format_reward >= 1.0:
                    stats[
                        "format_correct_responses"
                    ] += 1

                if answer_reward >= 1.0:
                    stats[
                        "answer_correct_responses"
                    ] += 1

                if reward >= 1.0:
                    stats[
                        "reward_positive_responses"
                    ] += 1

                candidates.append(
                    {
                        "candidate_index": candidate_index,
                        "response": response,
                        "format_reward": format_reward,
                        "answer_reward": answer_reward,
                        "reward": reward,
                    }
                )

            positive_candidates = [
                x for x in candidates
                if x["reward"] >= 1.0
            ]

            if positive_candidates:
                stats[
                    "questions_with_positive"
                ] += 1

                stats["by_config"][config][
                    "positive_questions"
                ] += 1

            else:
                stats[
                    "questions_without_positive"
                ] += 1

            chosen, rejected = choose_preference_pair(
                candidates,
                prefer_format_correct_rejected=True,
            )

            # No positive OR no negative => cannot form a DPO pair.
            if chosen is None or rejected is None:

                if chosen is not None and rejected is None:
                    stats[
                        "questions_without_rejected"
                    ] += 1

                # Save per-question result anyway.
                append_jsonl(
                    results_path,
                    {
                        "id": item["id"],
                        "config": config,
                        "question": item["question"],
                        "ground_truth": item["answer"],
                        "num_candidates": len(candidates),
                        "num_positive": len(
                            positive_candidates
                        ),
                        "num_format_correct": sum(
                            x["format_reward"] >= 1.0
                            for x in candidates
                        ),
                        "num_answer_correct": sum(
                            x["answer_reward"] >= 1.0
                            for x in candidates
                        ),
                        "has_pair": False,
                        "chosen_candidate_index": None,
                        "rejected_candidate_index": None,
                    },
                )

                continue

            # If a format-correct rejected exists, record that.
            if (
                rejected["format_reward"] >= 1.0
                and rejected["answer_reward"] <= 0.0
            ):
                stats[
                    "pairs_with_format_correct_rejected"
                ] += 1

            else:
                stats[
                    "pairs_with_format_wrong_rejected"
                ] += 1

            pair = {
                "id": item["id"],
                "config": config,
                "question": item["question"],

                # Exact sampled continuations.
                "chosen": chosen["response"],
                "rejected": rejected["response"],

                "chosen_reward": chosen["reward"],
                "rejected_reward": rejected["reward"],

                "chosen_format_reward": (
                    chosen["format_reward"]
                ),
                "chosen_answer_reward": (
                    chosen["answer_reward"]
                ),

                "rejected_format_reward": (
                    rejected["format_reward"]
                ),
                "rejected_answer_reward": (
                    rejected["answer_reward"]
                ),
            }

            append_jsonl(
                dpo_path,
                pair,
            )

            append_jsonl(
                results_path,
                {
                    "id": item["id"],
                    "config": config,
                    "question": item["question"],
                    "ground_truth": item["answer"],
                    "num_candidates": len(candidates),
                    "num_positive": len(
                        positive_candidates
                    ),
                    "num_format_correct": sum(
                        x["format_reward"] >= 1.0
                        for x in candidates
                    ),
                    "num_answer_correct": sum(
                        x["answer_reward"] >= 1.0
                        for x in candidates
                    ),
                    "has_pair": True,
                    "chosen_candidate_index": (
                        chosen["candidate_index"]
                    ),
                    "rejected_candidate_index": (
                        rejected["candidate_index"]
                    ),
                    "chosen_reward": chosen["reward"],
                    "rejected_reward": rejected["reward"],
                },
            )

            stats["questions_with_pair"] += 1
            stats["accepted_pairs"] += 1

            stats["by_config"][config][
                "pairs"
            ] += 1

            # Optional stopping condition.
            if (
                args.max_new_samples is not None
                and stats["accepted_pairs"]
                - len(existing_ids)
                >= args.max_new_samples
            ):
                print(
                    "\nReached "
                    f"--max_new_samples="
                    f"{args.max_new_samples}"
                )

                break

        # Stop outer loop too.
        if (
            args.max_new_samples is not None
            and stats["accepted_pairs"]
            - len(existing_ids)
            >= args.max_new_samples
        ):
            break

        processed = min(
            start + args.batch_size,
            len(pending),
        )

        pairs_now = (
            stats["accepted_pairs"]
            - len(existing_ids)
        )

        print(
            f"[Progress] "
            f"{processed}/{len(pending)} questions | "
            f"new pairs={pairs_now} | "
            f"generated={stats['total_generated_responses']} | "
            f"positive="
            f"{stats['reward_positive_responses']}"
        )

    # -----------------------------------------------------------------------
    # Final statistics
    # -----------------------------------------------------------------------

    total_generated = stats[
        "total_generated_responses"
    ]

    format_correct = stats[
        "format_correct_responses"
    ]

    answer_correct = stats[
        "answer_correct_responses"
    ]

    positive = stats[
        "reward_positive_responses"
    ]

    questions_with_positive = stats[
        "questions_with_positive"
    ]

    questions_without_positive = stats[
        "questions_without_positive"
    ]

    pairs = stats["accepted_pairs"]

    new_pairs = pairs - len(existing_ids)

    if total_generated > 0:
        format_accuracy = (
            format_correct / total_generated
        )
        answer_accuracy = (
            answer_correct / total_generated
        )
        reward_positive_rate = (
            positive / total_generated
        )
    else:
        format_accuracy = 0.0
        answer_accuracy = 0.0
        reward_positive_rate = 0.0

    if len(pending) > 0:
        positive_coverage = (
            questions_with_positive
            / len(pending)
        )
    else:
        positive_coverage = 0.0

    if len(pending) > 0:
        pair_coverage = (
            stats["questions_with_pair"]
            / len(pending)
        )
    else:
        pair_coverage = 0.0

    stats["format_accuracy"] = format_accuracy
    stats["answer_accuracy"] = answer_accuracy
    stats["reward_positive_rate"] = (
        reward_positive_rate
    )
    stats["positive_coverage"] = positive_coverage
    stats["pair_coverage"] = pair_coverage
    stats["new_pairs"] = new_pairs

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("DPO PREFERENCE SAMPLING COMPLETE")
    print("=" * 80)

    print(
        f"Total questions             : "
        f"{stats['total_questions']}"
    )
    print(
        f"Pending questions           : "
        f"{stats['pending_questions']}"
    )
    print(
        f"G                           : "
        f"{stats['G']}"
    )
    print(
        f"Total generated responses   : "
        f"{stats['total_generated_responses']}"
    )
    print(
        f"Format-correct responses    : "
        f"{stats['format_correct_responses']}"
    )
    print(
        f"Answer-correct responses    : "
        f"{stats['answer_correct_responses']}"
    )
    print(
        f"Reward-positive responses   : "
        f"{stats['reward_positive_responses']}"
    )

    print(
        f"Format accuracy             : "
        f"{format_accuracy:.4f}"
    )
    print(
        f"Answer accuracy             : "
        f"{answer_accuracy:.4f}"
    )
    print(
        f"Reward-positive rate        : "
        f"{reward_positive_rate:.4f}"
    )

    print(
        f"Questions with >=1 positive  : "
        f"{questions_with_positive}"
    )
    print(
        f"Questions without positive  : "
        f"{questions_without_positive}"
    )
    print(
        f"Positive coverage           : "
        f"{positive_coverage:.4f}"
    )

    print(
        f"DPO pairs                   : "
        f"{pairs}"
    )
    print(
        f"New DPO pairs               : "
        f"{new_pairs}"
    )
    print(
        f"Pair coverage               : "
        f"{pair_coverage:.4f}"
    )

    print(
        f"Pairs with format-correct "
        f"rejected                   : "
        f"{stats['pairs_with_format_correct_rejected']}"
    )
    print(
        f"Pairs with format-wrong "
        f"rejected                   : "
        f"{stats['pairs_with_format_wrong_rejected']}"
    )

    print("-" * 80)
    print("DPO pairs by config:")

    for config in CONFIGS:
        info = stats["by_config"][config]

        print(
            f"{config:30s} "
            f"questions={info['questions']:5d} "
            f"positive={info['positive_questions']:5d} "
            f"pairs={info['pairs']:5d}"
        )

    print("-" * 80)
    print(f"DPO dataset   : {dpo_path}")
    print(f"Results       : {results_path}")
    print(f"Summary       : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()