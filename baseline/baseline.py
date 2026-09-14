import json
import os
import argparse
from pathlib import Path

from datasets import load_from_disk
from vllm import LLM, SamplingParams

from prompt import build_prompt
from grader import r1_zero_reward_fn


MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
DATASET_ROOT = Path("/root/autodl-tmp/datasets/MATH")

RESULT_DIR = Path("/root/Qwen/baseline/results")
RESULT_FILE = RESULT_DIR / "predictions.jsonl"
SUMMARY_FILE = RESULT_DIR / "summary.json"


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

# 按照特定的格式加载所有测试数据，返回samples列表
def load_all_test_data():
    samples = []

    for config in CONFIGS:
        dataset_path = DATASET_ROOT / config / "test"
        dataset = load_from_disk(str(dataset_path))

        print(f"{config:30s}: {len(dataset)} test examples")

        for idx, item in enumerate(dataset):
            samples.append({
                "id": f"{config}_{idx}",
                "config": config,
                "index": idx,
                "question": item["problem"],
                "ground_truth": item["solution"],
            })

    return samples

# 加载已经完成的预测结果，返回completed字典
def load_existing_results():
    completed = {}

    if not RESULT_FILE.exists():
        return completed

    with RESULT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            result = json.loads(line)
            completed[result["id"]] = result

    return completed

# 保存预测结果到文件
def save_result(result):
    with RESULT_FILE.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                result,
                ensure_ascii=False,
            )
            + "\n"
        )


def main():
    #增加了限制参数，只评估前N个例子
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N examples.",
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Loading MATH test dataset")
    print("=" * 80)

    samples = load_all_test_data()

    print("-" * 80)
    print("Total test examples:", len(samples))

    if args.limit is not None:
        samples = samples[:args.limit]
        print("Limited evaluation examples:", len(samples))
    print("Expected:", 5000)
    print("-" * 80)

    if args.limit is None and len(samples) != 5000:
        raise RuntimeError(
            f"Expected 5000 test examples, but got {len(samples)}"
        )

    completed = load_existing_results()

    remaining = [
        sample
        for sample in samples
        if sample["id"] not in completed
    ]

    print("Already completed:", len(completed))
    print("Remaining:", len(remaining))

    if not remaining:
        print("All predictions already exist.")
        write_summary(samples, completed)
        return

    print("\n" + "=" * 80)
    print("Loading model")
    print("=" * 80)

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    total = len(remaining)

    for start in range(0, total, BATCH_SIZE):
        batch = remaining[start:start + BATCH_SIZE]
        #每个问题的prompt构建
        prompts = [
            build_prompt(sample["question"])
            for sample in batch
        ]
        #打印当前处理的样本数量
        print(
            f"\nGenerating "
            f"{start + 1}-{min(start + BATCH_SIZE, total)} "
            f"/ {total}"
        )
        #批量生成答案
        outputs = llm.generate(
            prompts,
            sampling_params,
        )

        for sample, output in zip(batch, outputs):
            response = output.outputs[0].text

            reward = r1_zero_reward_fn(
                response=response,
                ground_truth=sample["ground_truth"],
                fast=True,
            )

            result = {
                "id": sample["id"],
                "config": sample["config"],
                "index": sample["index"],
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "response": response,
                "format_reward": reward["format_reward"],
                "answer_reward": reward["answer_reward"],
                "reward": reward["reward"],
            }

            save_result(result)
            completed[sample["id"]] = result

        # 每个 batch 完成后更新 summary
        write_summary(samples, completed)

        print_summary(completed)

    print("\n" + "=" * 80)
    print("BASELINE COMPLETE")
    print("=" * 80)

    write_summary(samples, completed)
    print_summary(completed)


def write_summary(samples, results):
    evaluated = [
        results[sample["id"]]
        for sample in samples
        if sample["id"] in results
    ]

    if not evaluated:
        return

    n = len(evaluated)

    format_accuracy = sum(
        x["format_reward"] for x in evaluated
    ) / n

    answer_accuracy = sum(
        x["answer_reward"] for x in evaluated
    ) / n

    reward_average = sum(
        x["reward"] for x in evaluated
    ) / n

    summary = {
        "model": MODEL_PATH,
        "dataset": "EleutherAI/hendrycks_math",
        "split": "test",
        "num_examples": n,
        "expected_examples": 5000,
        "format_reward": format_accuracy,
        "answer_reward": answer_accuracy,
        "reward": reward_average,
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


def print_summary(results):
    evaluated = list(results.values())

    if not evaluated:
        return

    n = len(evaluated)

    format_accuracy = sum(
        x["format_reward"] for x in evaluated
    ) / n

    answer_accuracy = sum(
        x["answer_reward"] for x in evaluated
    ) / n

    reward_average = sum(
        x["reward"] for x in evaluated
    ) / n

    print(
        f"Evaluated: {n:4d} | "
        f"Format: {format_accuracy:.4f} | "
        f"Answer: {answer_accuracy:.4f} | "
        f"Reward: {reward_average:.4f}"
    )


if __name__ == "__main__":
    main()
