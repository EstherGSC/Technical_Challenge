import json
import argparse
from pathlib import Path

from datasets import load_from_disk
from vllm import LLM, SamplingParams

from prompt import build_prompt
from grader import r1_zero_reward_fn


# ============================================================
# Default configuration
# ============================================================

DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
)

DATASET_ROOT = Path(
    "/root/autodl-tmp/datasets/MATH"
)

DEFAULT_OUTPUT_DIR = Path(
    "/root/Qwen/evaluation/base"
)

CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

BATCH_SIZE = 32


# ============================================================
# Dataset
# ============================================================

def load_all_test_data():
    """
    Load all MATH test examples from the seven configurations.
    """

    samples = []

    for config in CONFIGS:

        dataset_path = (
            DATASET_ROOT
            / config
            / "test"
        )

        dataset = load_from_disk(
            str(dataset_path)
        )

        print(
            f"{config:30s}: "
            f"{len(dataset)} test examples"
        )

        for idx, item in enumerate(dataset):

            samples.append({
                "id": f"{config}_{idx}",
                "config": config,
                "index": idx,
                "question": item["problem"],
                "ground_truth": item["solution"],
            })

    return samples


# ============================================================
# Existing results
# ============================================================

def load_existing_results(result_file):
    """
    Load already completed predictions.

    This allows the evaluation to resume after interruption.
    """

    completed = {}

    if not result_file.exists():
        return completed

    with result_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            result = json.loads(line)

            completed[result["id"]] = result

    return completed


def save_result(result_file, result):

    with result_file.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


# ============================================================
# Statistics
# ============================================================

def calculate_statistics(results):

    if not results:
        return {
            "num_examples": 0,
            "format_accuracy": 0.0,
            "answer_accuracy": 0.0,
            "reward": 0.0,
        }

    values = list(results.values())

    n = len(values)

    format_accuracy = sum(
        x["format_reward"]
        for x in values
    ) / n

    answer_accuracy = sum(
        x["answer_reward"]
        for x in values
    ) / n

    reward_average = sum(
        x["reward"]
        for x in values
    ) / n

    return {
        "num_examples": n,
        "format_accuracy": format_accuracy,
        "answer_accuracy": answer_accuracy,
        "reward": reward_average,
    }


def calculate_config_statistics(
    samples,
    results,
):

    statistics = {}

    for config in CONFIGS:

        config_results = {
            sample["id"]: results[sample["id"]]
            for sample in samples
            if (
                sample["config"] == config
                and sample["id"] in results
            )
        }

        statistics[config] = (
            calculate_statistics(
                config_results
            )
        )

    return statistics


# ============================================================
# Summary
# ============================================================

def write_summary(
    summary_file,
    model_path,
    samples,
    results,
):

    overall = calculate_statistics(
        results
    )

    by_config = calculate_config_statistics(
        samples,
        results,
    )

    summary = {
        "model": str(model_path),
        "dataset": "MATH",
        "split": "test",
        "expected_examples": 5000,
        "num_examples": overall["num_examples"],
        "format_accuracy": overall[
            "format_accuracy"
        ],
        "answer_accuracy": overall[
            "answer_accuracy"
        ],
        "reward": overall["reward"],
        "by_config": by_config,
    }

    with summary_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


def print_summary(results):

    if not results:
        print("No evaluated examples.")
        return

    overall = calculate_statistics(
        results
    )

    print()
    print("=" * 80)
    print("Evaluation summary")
    print("=" * 80)

    print(
        f"Evaluated examples : "
        f"{overall['num_examples']}"
    )

    print(
        f"Format accuracy    : "
        f"{overall['format_accuracy']:.4f}"
    )

    print(
        f"Answer accuracy    : "
        f"{overall['answer_accuracy']:.4f}"
    )

    print(
        f"Average reward     : "
        f"{overall['reward']:.4f}"
    )

    print("-" * 80)

    # --------------------------------------------------------
    # Per-config statistics
    # --------------------------------------------------------

    configs = {}

    for result in results.values():

        config = result["config"]

        configs.setdefault(
            config,
            [],
        ).append(result)

    for config in CONFIGS:

        if config not in configs:
            continue

        stats = calculate_statistics(
            {
                x["id"]: x
                for x in configs[config]
            }
        )

        print(
            f"{config:30s} "
            f"N={stats['num_examples']:4d} "
            f"format={stats['format_accuracy']:.4f} "
            f"answer={stats['answer_accuracy']:.4f} "
            f"reward={stats['reward']:.4f}"
        )

    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a causal LM on the "
            "MATH test set."
        )
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model to evaluate.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N examples. "
            "If omitted, evaluate all 5000."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Number of prompts passed to vLLM at once.",
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization.",
    )

    args = parser.parse_args()

    model_path = Path(
        args.model_path
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_file = (
        output_dir
        / "predictions.jsonl"
    )

    summary_file = (
        output_dir
        / "summary.json"
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 80)
    print("MATH Evaluation")
    print("=" * 80)

    print(
        "Model:",
        model_path,
    )

    print(
        "Output:",
        output_dir,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "GPU memory utilization:",
        args.gpu_memory_utilization,
    )

    print()

    # ========================================================
    # Load test dataset
    # ========================================================

    print("=" * 80)
    print("Loading MATH test dataset")
    print("=" * 80)

    samples = load_all_test_data()

    print("-" * 80)

    print(
        "Total test examples:",
        len(samples),
    )

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be positive."
            )

        samples = samples[
            :args.limit
        ]

        print(
            "Limited evaluation examples:",
            len(samples),
        )

    else:

        if len(samples) != 5000:
            raise RuntimeError(
                "Expected 5000 test examples, "
                f"but got {len(samples)}."
            )

    print("-" * 80)

    # ========================================================
    # Resume
    # ========================================================

    completed = load_existing_results(
        result_file
    )

    # Only consider results belonging to the
    # currently requested evaluation subset.

    current_ids = {
        sample["id"]
        for sample in samples
    }

    completed = {
        key: value
        for key, value in completed.items()
        if key in current_ids
    }

    remaining = [
        sample
        for sample in samples
        if sample["id"]
        not in completed
    ]

    print(
        "Already completed:",
        len(completed),
    )

    print(
        "Remaining:",
        len(remaining),
    )

    print()

    # ========================================================
    # Nothing left
    # ========================================================

    if not remaining:

        print(
            "All requested predictions already exist."
        )

        write_summary(
            summary_file,
            model_path,
            samples,
            completed,
        )

        print_summary(
            completed
        )

        return

    # ========================================================
    # Load vLLM
    # ========================================================

    print("=" * 80)
    print("Loading model with vLLM")
    print("=" * 80)

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    print()
    print(
        "Sampling parameters:"
    )
    print(
        "temperature = 1.0"
    )
    print(
        "top_p       = 1.0"
    )
    print(
        "max_tokens  = 1024"
    )
    print(
        "stop        = </answer>"
    )
    print()

    # ========================================================
    # Generation
    # ========================================================

    total = len(remaining)

    for start in range(
        0,
        total,
        args.batch_size,
    ):

        batch = remaining[
            start:
            start + args.batch_size
        ]

        prompts = [
            build_prompt(
                sample["question"]
            )
            for sample in batch
        ]

        end = min(
            start + args.batch_size,
            total,
        )

        print()
        print(
            "=" * 80
        )
        print(
            f"Generating "
            f"{start + 1}-{end} "
            f"/ {total}"
        )
        print(
            "=" * 80
        )

        outputs = llm.generate(
            prompts,
            sampling_params,
        )

        for sample, output in zip(
            batch,
            outputs,
        ):

            response = (
                output.outputs[0].text
            )

            reward = r1_zero_reward_fn(
                response=response,
                ground_truth=(
                    sample["ground_truth"]
                ),
                fast=True,
            )

            result = {
                "id": sample["id"],
                "config": sample["config"],
                "index": sample["index"],
                "question": sample["question"],
                "ground_truth": (
                    sample["ground_truth"]
                ),
                "prompt": build_prompt(
                    sample["question"]
                ),
                "response": response,
                "format_reward": (
                    reward["format_reward"]
                ),
                "answer_reward": (
                    reward["answer_reward"]
                ),
                "reward": reward["reward"],
            }

            save_result(
                result_file,
                result,
            )

            completed[
                sample["id"]
            ] = result

        # ----------------------------------------------------
        # Update summary after every batch.
        # ----------------------------------------------------

        write_summary(
            summary_file,
            model_path,
            samples,
            completed,
        )

        print_summary(
            completed
        )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print(
        "Predictions:",
        result_file,
    )

    print(
        "Summary:",
        summary_file,
    )

    print_summary(
        completed
    )


if __name__ == "__main__":
    main()
