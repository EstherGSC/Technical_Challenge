import json
from pathlib import Path

from datasets import load_from_disk
from vllm import LLM, SamplingParams

from prompt import build_prompt
from grader import r1_zero_reward_fn


MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
DATASET_ROOT = Path("/root/autodl-tmp/datasets/MATH")
OUTPUT_PATH = Path("/root/Qwen/baseline/results/smoke_test.jsonl")


CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def main():
    samples = []

    # 从七个 config 各取一题
    for config in CONFIGS:
        dataset = load_from_disk(
            str(DATASET_ROOT / config / "test")
        )

        samples.append({
            "config": config,
            "question": dataset[0]["problem"],
            "solution": dataset[0]["solution"],
        })

    prompts = [build_prompt(x["question"]) for x in samples]

    print("=" * 80)
    print("Running baseline smoke test")
    print("Number of questions:", len(prompts))
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

    outputs = llm.generate(prompts, sampling_params)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for sample, output in zip(samples, outputs):
            response = output.outputs[0].text

            # grader 使用 solution 作为 ground truth。
            reward = r1_zero_reward_fn(
                response=response,
                ground_truth=sample["solution"],
                fast=True,
            )

            result = {
                "config": sample["config"],
                "question": sample["question"],
                "ground_truth": sample["solution"],
                "response": response,
                **reward,
            }

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print("\n" + "=" * 80)
            print("CONFIG:", sample["config"])
            print("QUESTION:")
            print(sample["question"])
            print("\nRESPONSE:")
            print(response)
            print("\nREWARD:")
            print(reward)

    print("\n" + "=" * 80)
    print("Saved to:", OUTPUT_PATH)
    print("=" * 80)


if __name__ == "__main__":
    main()
