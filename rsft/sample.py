import os
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from baseline.prompt import build_prompt
from baseline.grader import r1_zero_reward_fn


# ============================================================
# Default paths
# ============================================================

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-SFT-v2"
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
# Utilities
# ============================================================

def extract_boxed_answer(solution):
    """
    Extract the content of the last \\boxed{...} from a MATH solution.

    Supports nested braces, e.g.
        \\boxed{\\frac{1}{\\sqrt{2}}}
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


def get_field(sample, names, default=None):
    """
    Return the first existing field among names.
    """
    for name in names:
        if name in sample:
            value = sample[name]

            if value is not None:
                return value

    return default

#加载数据集并且清洗格式
def load_math_train(data_root, configs):
    """
    Load all MATH train configurations.

    The function supports several common load_from_disk layouts:
        DATA_ROOT/config/train
        DATA_ROOT/config
    """

    all_samples = []

    print("=" * 80)
    print("Loading MATH train dataset")
    print("=" * 80)

    for config in configs:

        dataset_path = Path(data_root) / config / "train"

        print(f"[{config}] loading from {dataset_path}")

        dataset = load_from_disk(str(dataset_path))

        # DatasetDict: use train split if available
        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset = dataset["train"]
            else:
                # fallback: first split
                split_name = list(dataset.keys())[0]
                dataset = dataset[split_name]

        print(f"[{config}] examples = {len(dataset)}")

        for idx, sample in enumerate(dataset):
            question = get_field(
                sample,
                ["question", "problem", "prompt"],
            )

            solution = get_field(
                sample,
                ["solution", "rationale"],
            )

            ground_truth = get_field(
                sample,
                ["answer", "ground_truth"],
            )

            if question is None:
                raise KeyError(
                    f"Cannot find question field in config={config}, "
                    f"index={idx}. Available fields: {list(sample.keys())}"
                )

            if solution is None:
                raise KeyError(
                    f"Cannot find solution field in config={config}, "
                    f"index={idx}. Available fields: {list(sample.keys())}"
                )

            # Prefer extracting the answer from the official solution.
            # This is consistent with the SFT data construction.
            extracted_answer = extract_boxed_answer(solution)

            if extracted_answer is not None:
                ground_truth = extracted_answer

            if ground_truth is None:
                print(
                    f"[WARNING] no ground truth for "
                    f"{config}_{idx}; skipping"
                )
                continue

            sample_id = get_field(
                sample,
                ["id", "problem_id", "uid"],
                default=f"{config}_{idx}",
            )

            all_samples.append(
                {
                    "id": str(sample_id),
                    "config": config,
                    "index": idx,
                    "question": str(question),
                    "solution": str(solution),
                    "ground_truth": str(ground_truth),
                }
            )

    print("-" * 80)
    print(f"Total train examples: {len(all_samples)}")
    print("=" * 80)

    return all_samples

#跳过已经采样过的样本
def load_existing_ids(output_path):
    """
    Read previously accepted RSFT samples.

    This enables basic resume behavior:
    if an example has already produced an accepted response,
    it will not be sampled again.
    """

    if not output_path.exists():
        return set()

    existing_ids = set()

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "id" in item:
                existing_ids.add(str(item["id"]))

    return existing_ids


def append_jsonl(path, item):
    """
    Append one JSON object to a JSONL file.
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="RSFT rejection sampling with vLLM"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/rsft/sampling",
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
        help="For debugging. Example: --limit 16",
    )

    parser.add_argument(
        "--max_new_samples",
        type=int,
        default=None,
        help=(
            "Maximum number of accepted RSFT samples. "
            "Useful for debugging."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Random seeds
    # --------------------------------------------------------

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted_path = output_dir / "rsft_dataset.jsonl"
    all_results_path = output_dir / "sampling_results.jsonl"
    summary_path = output_dir / "sampling_summary.json"

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    samples = load_math_train(
        args.data_root,
        CONFIGS,
    )

    if args.limit is not None:
        samples = samples[:args.limit]

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    existing_ids = load_existing_ids(accepted_path)

    if existing_ids:
        print(
            f"[Resume] Found {len(existing_ids)} already accepted samples."
        )

    pending_samples = [
        sample
        for sample in samples
        if sample["id"] not in existing_ids
    ]

    print(
        f"Total examples       : {len(samples)}"
    )
    print(
        f"Already accepted     : {len(existing_ids)}"
    )
    print(
        f"Pending examples     : {len(pending_samples)}"
    )

    if not pending_samples:
        print("Nothing to sample.")
        return

    # --------------------------------------------------------
    # Initialize vLLM
    # --------------------------------------------------------

    print("=" * 80)
    print("Loading model with vLLM")
    print("=" * 80)

    print(f"Model path           : {args.model_path}")
    print(f"G                    : {args.G}")
    print(f"Temperature          : {args.temperature}")
    print(f"Top-p                : {args.top_p}")
    print(f"Max tokens           : {args.max_tokens}")
    print(f"Min tokens           : {args.min_tokens}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Seed                 : {args.seed}")
    print(
        f"GPU memory util.     : "
        f"{args.gpu_memory_utilization}"
    )

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

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_questions = len(samples)
    total_pending = len(pending_samples)

    total_generated = 0
    total_format_correct = 0
    total_answer_correct = 0
    total_reward_positive = 0

    questions_with_positive = 0
    questions_without_positive = 0

    accepted_samples = 0

    accepted_by_config = {
        config: 0
        for config in CONFIGS
    }

    questions_with_positive_by_config = {
        config: 0
        for config in CONFIGS
    }

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    for batch_start in range(
        0,
        len(pending_samples),
        args.batch_size,
    ):
        batch = pending_samples[
            batch_start:
            batch_start + args.batch_size
        ]

        prompts = [
            build_prompt(sample["question"])
            for sample in batch
        ]

        print(
            f"\n"
            f"[Batch "
            f"{batch_start // args.batch_size + 1}"
            f"/"
            f"{(len(pending_samples) + args.batch_size - 1) // args.batch_size}"
            f"] "
            f"examples "
            f"{batch_start + 1}-"
            f"{min(batch_start + args.batch_size, len(pending_samples))}"
        )

        # ----------------------------------------------------
        # Generate G responses per prompt
        # ----------------------------------------------------

        outputs = llm.generate(
            prompts,
            sampling_params,
        )

        # vLLM returns one RequestOutput per prompt.
        # Each RequestOutput contains G output candidates.
        for sample, request_output in zip(batch, outputs):

            candidates = request_output.outputs

            if len(candidates) != args.G:
                print(
                    f"[WARNING] {sample['id']}: "
                    f"expected {args.G} outputs, "
                    f"got {len(candidates)}"
                )

            question_positive = False
            question_best = None

            for sample_index, candidate in enumerate(candidates):

                response = candidate.text

                result = r1_zero_reward_fn(
                    response,
                    sample["ground_truth"],
                    fast=True,
                )

                format_reward = result["format_reward"]
                answer_reward = result["answer_reward"]
                reward = result["reward"]

                total_generated += 1

                total_format_correct += int(
                    format_reward > 0
                )

                total_answer_correct += int(
                    answer_reward > 0
                )

                total_reward_positive += int(
                    reward > 0
                )

                # ------------------------------------------------
                # First correct response wins.
                # ------------------------------------------------

                if reward > 0 and not question_positive:

                    question_positive = True

                    question_best = {
                        "id": sample["id"],
                        "config": sample["config"],
                        "index": sample["index"],
                        "question": sample["question"],
                        "solution": sample["solution"],
                        "ground_truth": sample["ground_truth"],
                        "response": response,
                        "format_reward": float(format_reward),
                        "answer_reward": float(answer_reward),
                        "reward": float(reward),
                        "sample_index": sample_index,
                    }

            # ----------------------------------------------------
            # Save the first positive sample only
            # ----------------------------------------------------

            if question_best is not None:

                append_jsonl(
                    accepted_path,
                    question_best,
                )

                accepted_samples += 1
                questions_with_positive += 1

                config = sample["config"]

                accepted_by_config[config] += 1
                questions_with_positive_by_config[config] += 1

                # Save a detailed event as well.
                append_jsonl(
                    all_results_path,
                    {
                        "id": sample["id"],
                        "config": sample["config"],
                        "question": sample["question"],
                        "ground_truth": sample["ground_truth"],
                        "accepted": True,
                        "accepted_sample_index": question_best[
                            "sample_index"
                        ],
                        "accepted_response": question_best[
                            "response"
                        ],
                    },
                )

            else:

                questions_without_positive += 1

                append_jsonl(
                    all_results_path,
                    {
                        "id": sample["id"],
                        "config": sample["config"],
                        "question": sample["question"],
                        "ground_truth": sample["ground_truth"],
                        "accepted": False,
                        "accepted_sample_index": None,
                        "accepted_response": None,
                    },
                )

            # ------------------------------------------------
            # Optional accepted-sample limit
            # ------------------------------------------------

            if (
                args.max_new_samples is not None
                and accepted_samples >= args.max_new_samples
            ):
                print(
                    f"\nReached --max_new_samples="
                    f"{args.max_new_samples}"
                )
                break

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        print(
            f"Generated responses : {total_generated}"
        )

        print(
            f"Reward=1 responses  : "
            f"{total_reward_positive}"
        )

        print(
            f"Accepted questions  : "
            f"{accepted_samples}"
        )

        if total_generated > 0:
            print(
                f"Current reward rate : "
                f"{total_reward_positive / total_generated:.4f}"
            )

        if total_pending > 0:
            print(
                f"Progress            : "
                f"{min(batch_start + args.batch_size, total_pending)}"
                f"/{total_pending}"
            )

        # ----------------------------------------------------
        # Save intermediate summary
        # ----------------------------------------------------

        current_summary = {
            "model": args.model_path,
            "data_root": args.data_root,
            "total_questions": total_questions,
            "pending_questions": total_pending,
            "G": args.G,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "min_tokens": args.min_tokens,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "total_generated": total_generated,
            "format_correct": total_format_correct,
            "answer_correct": total_answer_correct,
            "reward_positive": total_reward_positive,
            "questions_with_positive": questions_with_positive,
            "questions_without_positive": questions_without_positive,
            "accepted_samples": accepted_samples,
            "accepted_by_config": accepted_by_config,
            "questions_with_positive_by_config": (
                questions_with_positive_by_config
            ),
        }

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                current_summary,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Stop outer loop too.
        if (
            args.max_new_samples is not None
            and accepted_samples >= args.max_new_samples
        ):
            break

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RSFT REJECTION SAMPLING COMPLETE")
    print("=" * 80)

    print(
        f"Total questions             : "
        f"{total_questions}"
    )

    print(
        f"Pending questions           : "
        f"{total_pending}"
    )

    print(
        f"G                           : "
        f"{args.G}"
    )

    print(
        f"Total generated responses   : "
        f"{total_generated}"
    )

    print(
        f"Format-correct responses    : "
        f"{total_format_correct}"
    )

    print(
        f"Answer-correct responses    : "
        f"{total_answer_correct}"
    )

    print(
        f"Reward-positive responses   : "
        f"{total_reward_positive}"
    )

    if total_generated > 0:
        print(
            f"Format accuracy             : "
            f"{total_format_correct / total_generated:.4f}"
        )

        print(
            f"Answer accuracy             : "
            f"{total_answer_correct / total_generated:.4f}"
        )

        print(
            f"Reward-positive rate        : "
            f"{total_reward_positive / total_generated:.4f}"
        )

    print(
        f"Questions with >=1 positive  : "
        f"{questions_with_positive}"
    )

    print(
        f"Questions without positive  : "
        f"{questions_without_positive}"
    )

    if total_pending > 0:
        print(
            f"Coverage of pending         : "
            f"{questions_with_positive / total_pending:.4f}"
        )

    print(
        f"Accepted RSFT samples       : "
        f"{accepted_samples}"
    )

    print("-" * 80)
    print("Accepted samples by config:")

    for config in CONFIGS:
        print(
            f"{config:30s} "
            f"{accepted_by_config[config]:6d}"
        )

    print("-" * 80)

    print(
        f"Accepted dataset : "
        f"{accepted_path}"
    )

    print(
        f"Sampling results : "
        f"{all_results_path}"
    )

    print(
        f"Summary          : "
        f"{summary_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()