import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B"
DATA_ROOT = "/root/autodl-tmp/datasets/MATH/algebra/train"

MAX_LENGTH = 2048


def extract_boxed_answer(solution):
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
            matches.append(
                solution[pos + len(r"\boxed{"):i - 1].strip()
            )

        start = i

    return matches[-1] if matches else None


def main():
    print("=" * 80)
    print("Loading tokenizer")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_from_disk(DATA_ROOT)

    samples = []

    for i in range(4):
        question = ds[i]["problem"]
        solution = ds[i]["solution"]

        answer = extract_boxed_answer(solution)
        assert answer is not None

        prompt = f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
"""

        response = (
            solution.strip()
            + f"\n</think> <answer>{answer}</answer>"
        )

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

        # Explicitly construct prompt + response.
        input_ids = prompt_ids + response_ids

        if len(input_ids) > MAX_LENGTH:
            # Keep the response intact whenever possible.
            # If response itself exceeds MAX_LENGTH, retain its tail.
            if len(response_ids) >= MAX_LENGTH:
                input_ids = response_ids[-MAX_LENGTH:]
                response_start = 0
            else:
                keep_prompt = MAX_LENGTH - len(response_ids)
                input_ids = (
                    prompt_ids[-keep_prompt:]
                    + response_ids
                )
                response_start = keep_prompt
        else:
            response_start = len(prompt_ids)

        response_mask = (
            [0] * response_start
            + [1] * (len(input_ids) - response_start)
        )

        assert len(input_ids) == len(response_mask)
        assert sum(response_mask) > 0

        samples.append((input_ids, response_mask))

        print(
            f"sample {i}: "
            f"total={len(input_ids)}, "
            f"response_tokens={sum(response_mask)}, "
            f"prompt_tokens={len(input_ids)-sum(response_mask)}"
        )

    max_len = max(len(x[0]) for x in samples)

    input_ids = []
    attention_mask = []
    response_mask = []

    for ids, mask in samples:
        pad = max_len - len(ids)

        input_ids.append(
            ids + [tokenizer.pad_token_id] * pad
        )
        attention_mask.append(
            [1] * len(ids) + [0] * pad
        )
        response_mask.append(
            mask + [0] * pad
        )

    input_ids = torch.tensor(
        input_ids,
        dtype=torch.long,
        device="cuda",
    )
    attention_mask = torch.tensor(
        attention_mask,
        dtype=torch.long,
        device="cuda",
    )
    response_mask = torch.tensor(
        response_mask,
        dtype=torch.float32,
        device="cuda",
    )

    print()
    print("=" * 80)
    print("Loading model")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).cuda()

    model.train()

    total_params = sum(
        p.numel() for p in model.parameters()
    )
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("total parameters:", total_params)
    print("trainable parameters:", trainable_params)
    print(
        "trainable ratio:",
        trainable_params / total_params,
    )

    assert total_params == trainable_params

    print()
    print("=" * 80)
    print("Forward + response-only loss")
    print("=" * 80)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = response_mask[:, 1:].contiguous()

    vocab_size = shift_logits.size(-1)

    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)

    response_loss = (
        token_loss * shift_mask
    ).sum() / shift_mask.sum()

    print("loss:", response_loss.item())
    print("response tokens:", int(shift_mask.sum().item()))

    print()
    print("=" * 80)
    print("Backward")
    print("=" * 80)

    response_loss.backward()

    grad_count = 0
    grad_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            grad_count += 1
            grad_norm += p.grad.detach().float().norm().item() ** 2

    grad_norm = grad_norm ** 0.5

    print("parameters with gradients:", grad_count)
    print("gradient norm:", grad_norm)

    assert grad_count > 0
    assert grad_norm > 0

    print()
    print("SFT smoke test PASSED.")


if __name__ == "__main__":
    main()
