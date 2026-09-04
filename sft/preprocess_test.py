import re
from datasets import load_from_disk
from transformers import AutoTokenizer

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
DATA_ROOT = "/root/autodl-tmp/datasets/MATH"

MAX_LENGTH = 2048


def extract_boxed_answer(solution):
    """
    提取 solution 中最后一个 \\boxed{...} 的内容。
    支持简单嵌套花括号。
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


def build_response(solution):
    answer = extract_boxed_answer(solution)

    if answer is None:
        return None

    return f"<think>\n{solution.strip()}\n</think> <answer>{answer}</answer>"


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    ds = load_from_disk(DATA_ROOT + "/algebra/train")

    for idx in range(3):
        sample = ds[idx]
        problem = sample["problem"]
        solution = sample["solution"]

        response = build_response(solution)

        print("=" * 100)
        print(f"SAMPLE {idx}")
        print("-" * 100)
        print("QUESTION:")
        print(problem)
        print()
        print("EXTRACTED ANSWER:")
        print(extract_boxed_answer(solution))
        print()
        print("RESPONSE:")
        print(response)
        print()

        # 按课程 baseline 的 prompt 构造方式
        prompt = f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {problem}
Assistant: <think>
"""

        full_text = prompt + solution.strip() + f"\n</think> <answer>{extract_boxed_answer(solution)}</answer>"

        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors=None,
        )

        input_ids = tokens["input_ids"]

        print("TOKEN COUNT:", len(input_ids))
        print("FIRST 20 TOKENS:", input_ids[:20])
        print("DECODED TAIL:")
        print(tokenizer.decode(input_ids[-300:], skip_special_tokens=False))


if __name__ == "__main__":
    main()
