import os
import sys
import json
import csv
import re
import math
import argparse
import multiprocessing as mp

# ============================================================
# vLLM multiprocessing / CUDA environment
# ============================================================

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ============================================================
# Imports
# ============================================================

import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


# ============================================================
# Default paths
# ============================================================

DEFAULT_BASE_MODEL = (
    "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-SelfPlay/step-15"
)

DEFAULT_PEFT_MODEL = (
    "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-PEFT/final"
)

DEFAULT_PHY_A = (
    "/root/autodl-tmp/ABench/Physics/data/Phy_A_fixed_400.csv"
)

DEFAULT_PHY_B = (
    "/root/autodl-tmp/ABench/Physics/data/Phy_B_dynamic_100.csv"
)

DEFAULT_OUTPUT_DIR = (
    "/root/autodl-tmp/self_play/physics_eval"
)


# ============================================================
# LoRA implementation
#
# Must be identical to the training-time implementation
# ============================================================

class LoRALinear(torch.nn.Module):

    def __init__(
        self,
        base_linear,
        r,
        alpha,
        dropout,
    ):
        super().__init__()

        self.base = base_linear

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = torch.nn.Dropout(dropout)

        # Original linear layer is frozen.
        self.base.weight.requires_grad = False

        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # Training code used:
        #
        # A: [out_features, r]
        # B: [r, in_features]
        #
        # A initialized to zero
        # B initialized with Gaussian noise

        self.lora_A = torch.nn.Parameter(
            torch.zeros(
                self.base.out_features,
                r,
                dtype=self.base.weight.dtype,
                device=self.base.weight.device,
            )
        )

        self.lora_B = torch.nn.Parameter(
            torch.empty(
                r,
                self.base.in_features,
                dtype=self.base.weight.dtype,
                device=self.base.weight.device,
            )
        )

        torch.nn.init.normal_(
            self.lora_B,
            mean=0.0,
            std=0.02,
        )

    def forward(self, x):

        base_out = self.base(x)

        lora_out = torch.nn.functional.linear(
            self.dropout(x),
            self.lora_B,
        )

        lora_out = torch.nn.functional.linear(
            lora_out,
            self.lora_A,
        )

        return base_out + self.scaling * lora_out


# ============================================================
# Utility: replace target Linear layers with LoRA layers
# ============================================================

def replace_lora_modules(
    model,
    target_modules,
    r,
    alpha,
    dropout,
):
    replaced = []

    for name, module in list(model.named_modules()):

        if not isinstance(module, torch.nn.Linear):
            continue

        module_name = name.split(".")[-1]

        if module_name not in target_modules:
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]

        parent = model

        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)

        old_linear = getattr(parent, child_name)

        new_linear = LoRALinear(
            old_linear,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        setattr(parent, child_name, new_linear)

        replaced.append(name)

    return replaced


# ============================================================
# Load PEFT model
# ============================================================

def load_peft_model(model_dir):

    config_path = os.path.join(
        model_dir,
        "adapter_config.json",
    )

    adapter_path = os.path.join(
        model_dir,
        "adapter_model.pt",
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found:\n{config_path}"
        )

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"adapter_model.pt not found:\n{adapter_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_model = config["base_model"]

    r = int(config["r"])
    alpha = float(config["alpha"])
    dropout = float(config["dropout"])

    target_modules = config["target_modules"]

    print()
    print("=" * 80)
    print("Loading PEFT model")
    print("=" * 80)

    print("Adapter directory :", model_dir)
    print("Base model        :", base_model)
    print("LoRA rank         :", r)
    print("LoRA alpha        :", alpha)
    print("LoRA dropout      :", dropout)
    print("Target modules    :", target_modules)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    replaced = replace_lora_modules(
        model,
        target_modules=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )

    print()
    print(f"LoRA modules replaced: {len(replaced)}")

    adapter_state = torch.load(
        adapter_path,
        map_location="cpu",
    )

    missing, unexpected = model.load_state_dict(
        adapter_state,
        strict=False,
    )

    print(
        f"Missing keys    : {len(missing)}"
    )

    print(
        f"Unexpected keys : {len(unexpected)}"
    )

    if len(missing) > 0:
        print("First missing keys:")
        for key in missing[:10]:
            print("  ", key)

    if len(unexpected) > 0:
        print("First unexpected keys:")
        for key in unexpected[:10]:
            print("  ", key)

    model.eval()

    return model, tokenizer


# ============================================================
# Numerical extraction
#
# This intentionally follows ABench's philosophy:
# numerical answer inside the final boxed result.
# ============================================================

def extract_boxed_content(text):

    if not isinstance(text, str):
        return None

    start = text.rfind(r"\boxed{")

    if start == -1:
        return None

    content = text[
        start + len(r"\boxed{"):
    ]

    depth = 0

    for i, char in enumerate(content):

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == -1:
                return content[:i].strip()

    return None


def remove_latex_units(text):

    if text is None:
        return None

    # Remove common textual unit constructs.
    text = re.sub(
        r"\\text\s*\{[^{}]*\}",
        "",
        text,
    )

    text = re.sub(
        r"\\mathrm\s*\{[^{}]*\}",
        "",
        text,
    )

    # Remove simple alphabetic units after numbers.
    text = re.sub(
        r"(?<=\d)\s*[A-Za-z]+(?:/[A-Za-z]+)?\b",
        "",
        text,
    )

    return text.strip()


def normalize_numeric_text(text):

    if text is None:
        return None

    text = text.strip()

    # Remove commas used as thousands separators.
    text = text.replace(",", "")

    # LaTeX spaces
    text = text.replace(r"\,", "")
    text = text.replace(r"\;", "")
    text = text.replace(r"\!", "")

    # \times 10^{x}
    text = re.sub(
        r"\\times\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?",
        r"e\1",
        text,
    )

    # × 10^x
    text = re.sub(
        r"×\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?",
        r"e\1",
        text,
    )

    # 10^x
    text = re.sub(
        r"10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?",
        r"1e\1",
        text,
    )

    # Remove remaining LaTeX braces.
    text = text.replace("{", "")
    text = text.replace("}", "")

    return text.strip()


def extract_number(text):

    """
    Extract one numerical value from the boxed answer.

    Supports:
      30
      -3.14
      1.2e-5
      1.2 × 10^{-5}
      3.0 \\times 10^{8}
    """

    if text is None:
        return None

    text = remove_latex_units(text)
    text = normalize_numeric_text(text)

    # If the boxed result is an equation:
    #
    # x = 30
    # x \approx 30
    #
    if "=" in text:
        text = text.split("=")[-1].strip()

    text = text.replace(r"\approx", "")
    text = text.strip()

    # Scientific notation
    scientific = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        text,
    )

    if scientific is None:
        return None

    try:
        return float(scientific.group(0))
    except Exception:
        return None


# ============================================================
# ABench official-style scoring
# ============================================================

def compare_answer(
    model_response,
    standard_answer,
    threshold=0.01,
):
    """
    ABench official criterion:

        relative error <= 1%

    Official evaluator first extracts \\boxed{...}.
    """

    boxed = extract_boxed_content(model_response)

    if boxed is None:
        return False, None, None

    model_value = extract_number(boxed)

    if model_value is None:
        return False, None, None

    standard_value = extract_number(
        str(standard_answer)
    )

    if standard_value is None:

        # Standard answer may itself be a raw number.
        try:
            standard_value = float(
                str(standard_answer).strip()
            )
        except Exception:
            return False, model_value, None

    # Official implementation effectively uses:
    #
    # abs(answer - result) <=
    #     threshold * abs(answer)
    #
    # with threshold = 0.01.

    if standard_value == 0:

        correct = (
            abs(model_value - standard_value)
            <= threshold
        )

    else:

        correct = (
            abs(
                standard_value - model_value
            )
            <= threshold
            * abs(standard_value)
        )

    return (
        bool(correct),
        model_value,
        standard_value,
    )


# ============================================================
# Prompt
# ============================================================

def build_prompt(question):

    return f"""A conversation between User and Assistant. The User asks a physics problem, and the Assistant solves it. The Assistant first thinks through the reasoning process and then provides the final answer. The reasoning process is enclosed within <think> </think> and the final answer is enclosed within <answer> </answer> tags. The final numerical answer must be enclosed in \\boxed{{}}.

User: {question}

Assistant: <think>"""


# ============================================================
# Read benchmark
# ============================================================

def load_physics_dataset(
    path,
    benchmark,
):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    print()
    print("=" * 80)
    print(f"Loading {benchmark}")
    print("=" * 80)

    print("File:", path)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    required = [
        "mid",
        "standard_question",
        "standard_answer",
    ]

    for col in required:

        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )

    return df


# ============================================================
# Generate answers with vLLM
# ============================================================

def generate_answers(
    llm,
    prompts,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=1024,
):

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=None,
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    responses = []

    for output in outputs:

        if len(output.outputs) == 0:
            responses.append("")
        else:
            responses.append(
                output.outputs[0].text
            )

    return responses


# ============================================================
# Evaluate Phy_A
# ============================================================

def evaluate_phy_a(
    df,
    responses,
):

    records = []

    correct_count = 0
    format_count = 0

    for i, row in df.iterrows():

        response = responses[i]

        boxed = extract_boxed_content(
            response
        )

        has_format = boxed is not None

        if has_format:
            format_count += 1

        correct, model_value, gt_value = (
            compare_answer(
                response,
                row["standard_answer"],
            )
        )

        if correct:
            correct_count += 1

        records.append(
            {
                "mid": row["mid"],
                "question": row["standard_question"],
                "ground_truth": row["standard_answer"],
                "response": response,
                "boxed_answer": boxed,
                "model_value": model_value,
                "ground_truth_value": gt_value,
                "format_correct": has_format,
                "answer_correct": correct,
            }
        )

    total = len(df)

    return {
        "records": records,
        "total": total,
        "format_count": format_count,
        "correct_count": correct_count,
    }


# ============================================================
# Evaluate Phy_B
# ============================================================

def evaluate_phy_b(
    df,
    responses,
):

    records = []

    # Row-level statistics
    row_correct = 0
    row_format = 0

    for i, row in df.iterrows():

        response = responses[i]

        boxed = extract_boxed_content(
            response
        )

        has_format = boxed is not None

        if has_format:
            row_format += 1

        correct, model_value, gt_value = (
            compare_answer(
                response,
                row["standard_answer"],
            )
        )

        if correct:
            row_correct += 1

        records.append(
            {
                "mid": row["mid"],
                "subid": row["subid"],
                "question": row["standard_question"],
                "ground_truth": row["standard_answer"],
                "response": response,
                "boxed_answer": boxed,
                "model_value": model_value,
                "ground_truth_value": gt_value,
                "format_correct": has_format,
                "answer_correct": correct,
            }
        )

    result_df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Dynamic accuracy
    #
    # ABench:
    # all SubIDs belonging to one MID must be correct.
    # --------------------------------------------------------

    mid_results = []

    for mid, group in result_df.groupby(
        "mid",
        sort=False,
    ):

        all_correct = bool(
            group["answer_correct"].all()
        )

        mid_results.append(
            {
                "mid": mid,
                "num_subquestions": len(group),
                "all_correct": all_correct,
            }
        )

    mid_df = pd.DataFrame(mid_results)

    dynamic_correct = int(
        mid_df["all_correct"].sum()
    )

    dynamic_total = len(mid_df)

    return {
        "records": records,
        "row_total": len(df),
        "row_format_count": row_format,
        "row_correct_count": row_correct,
        "mid_total": dynamic_total,
        "mid_correct_count": dynamic_correct,
        "mid_results": mid_results,
    }


# ============================================================
# Save JSONL
# ============================================================

def save_jsonl(
    records,
    path,
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    with open(
        path,
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


# ============================================================
# Print summary
# ============================================================

def print_phy_a_summary(result):

    total = result["total"]

    format_acc = (
        result["format_count"] / total
        if total
        else 0.0
    )

    answer_acc = (
        result["correct_count"] / total
        if total
        else 0.0
    )

    print()
    print("=" * 80)
    print("Phy_A Evaluation summary")
    print("=" * 80)

    print(
        f"Evaluated examples : {total}"
    )

    print(
        f"Format accuracy    : {format_acc:.4f}"
    )

    print(
        f"Answer accuracy    : {answer_acc:.4f}"
    )

    print(
        f"Average reward     : {answer_acc:.4f}"
    )

    print("-" * 80)

    print(
        f"Correct            : "
        f"{result['correct_count']} / {total}"
    )

    print(
        f"Format valid       : "
        f"{result['format_count']} / {total}"
    )

    print("=" * 80)


def print_phy_b_summary(result):

    row_total = result["row_total"]

    row_format_acc = (
        result["row_format_count"]
        / row_total
        if row_total
        else 0.0
    )

    row_answer_acc = (
        result["row_correct_count"]
        / row_total
        if row_total
        else 0.0
    )

    dynamic_acc = (
        result["mid_correct_count"]
        / result["mid_total"]
        if result["mid_total"]
        else 0.0
    )

    print()
    print("=" * 80)
    print("Phy_B Evaluation summary")
    print("=" * 80)

    print(
        f"Evaluated rows     : {row_total}"
    )

    print(
        f"Evaluated MIDs     : "
        f"{result['mid_total']}"
    )

    print(
        f"Format accuracy    : "
        f"{row_format_acc:.4f}"
    )

    print(
        f"SubID accuracy     : "
        f"{row_answer_acc:.4f}"
    )

    print(
        f"Dynamic accuracy   : "
        f"{dynamic_acc:.4f}"
    )

    print(
        f"Average reward     : "
        f"{dynamic_acc:.4f}"
    )

    print("-" * 80)

    print(
        f"Correct MIDs       : "
        f"{result['mid_correct_count']} / "
        f"{result['mid_total']}"
    )

    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model path. Can be either "
            "SelfPlay-15 base model or "
            "PEFT final adapter."
        ),
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["A", "B", "both"],
        default="both",
    )

    parser.add_argument(
        "--phy_a",
        type=str,
        default=DEFAULT_PHY_A,
    )

    parser.add_argument(
        "--phy_b",
        type=str,
        default=DEFAULT_PHY_B,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--max_model_len",
        type=int,
        default=3072,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For debugging, evaluate only first N examples.",
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    print()
    print("=" * 80)
    print("ABench-Physics Evaluation")
    print("=" * 80)

    print("Model:", args.model)
    print("Benchmark:", args.benchmark)
    print("GPU memory utilization:",
          args.gpu_memory_utilization)
    print("Max model length:",
          args.max_model_len)
    print("Max new tokens:",
          args.max_new_tokens)
    print("Temperature:",
          args.temperature)
    print("Top-p:",
          args.top_p)

    # --------------------------------------------------------
    # Determine whether this is a PEFT adapter
    # --------------------------------------------------------

    is_peft = os.path.exists(
        os.path.join(
            args.model,
            "adapter_config.json",
        )
    )

    # --------------------------------------------------------
    # Base model
    # --------------------------------------------------------

    if is_peft:

        print()
        print("Detected manual LoRA adapter.")

        model, tokenizer = load_peft_model(
            args.model
        )

        # Save merged temporary model for vLLM.
        #
        # IMPORTANT:
        # vLLM does not directly understand this custom
        # LoRALinear implementation.
        #
        # Therefore we merge LoRA weights into the base
        # Linear layers before starting vLLM.
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("Merging LoRA weights for vLLM")
        print("=" * 80)

        model = merge_lora_weights(model)

        vllm_model_path = os.path.join(
            args.output_dir,
            "_merged_model",
        )

        os.makedirs(
            vllm_model_path,
            exist_ok=True,
        )

        model.save_pretrained(
            vllm_model_path,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(
            vllm_model_path
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_path_for_vllm = (
            vllm_model_path
        )

    else:

        model_path_for_vllm = args.model

    # --------------------------------------------------------
    # vLLM
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Loading vLLM")
    print("=" * 80)

    llm = LLM(
        model=model_path_for_vllm,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),
        max_model_len=args.max_model_len,
    )

    # --------------------------------------------------------
    # Run Phy_A
    # --------------------------------------------------------

    if args.benchmark in ["A", "both"]:

        df_a = load_physics_dataset(
            args.phy_a,
            "Phy_A",
        )

        if args.limit is not None:
            df_a = df_a.iloc[
                :args.limit
            ].reset_index(drop=True)

        prompts_a = [
            build_prompt(q)
            for q in df_a[
                "standard_question"
            ].tolist()
        ]

        print()
        print("=" * 80)
        print(
            f"Generating Phy_A answers "
            f"({len(prompts_a)} examples)"
        )
        print("=" * 80)

        responses_a = generate_answers(
            llm,
            prompts_a,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

        result_a = evaluate_phy_a(
            df_a,
            responses_a,
        )

        output_a = os.path.join(
            args.output_dir,
            "Phy_A_predictions.jsonl",
        )

        save_jsonl(
            result_a["records"],
            output_a,
        )

        print_phy_a_summary(
            result_a
        )

        print(
            f"Predictions saved to:\n"
            f"{output_a}"
        )

    # --------------------------------------------------------
    # Run Phy_B
    # --------------------------------------------------------

    if args.benchmark in ["B", "both"]:

        df_b = load_physics_dataset(
            args.phy_b,
            "Phy_B",
        )

        if "subid" not in df_b.columns:

            raise ValueError(
                "Phy_B must contain "
                "'subid' column."
            )

        if args.limit is not None:
            df_b = df_b.iloc[
                :args.limit
            ].reset_index(drop=True)

        prompts_b = [
            build_prompt(q)
            for q in df_b[
                "standard_question"
            ].tolist()
        ]

        print()
        print("=" * 80)
        print(
            f"Generating Phy_B answers "
            f"({len(prompts_b)} rows)"
        )
        print("=" * 80)

        responses_b = generate_answers(
            llm,
            prompts_b,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

        result_b = evaluate_phy_b(
            df_b,
            responses_b,
        )

        output_b = os.path.join(
            args.output_dir,
            "Phy_B_predictions.jsonl",
        )

        save_jsonl(
            result_b["records"],
            output_b,
        )

        print_phy_b_summary(
            result_b
        )

        print(
            f"Predictions saved to:\n"
            f"{output_b}"
        )

    print()
    print("=" * 80)
    print("Evaluation finished.")
    print("=" * 80)


# ============================================================
# Merge LoRA
# ============================================================

def merge_lora_weights(model):

    merged_count = 0

    for name, module in list(
        model.named_modules()
    ):

        if not isinstance(
            module,
            LoRALinear,
        ):
            continue

        base = module.base

        # LoRA delta:
        #
        # A @ B
        #
        delta = (
            module.lora_A
            @ module.lora_B
        )

        delta = (
            delta
            * module.scaling
        )

        with torch.no_grad():

            base.weight.data += (
                delta.to(
                    base.weight.dtype
                )
            )

        parent_name = ".".join(
            name.split(".")[:-1]
        )

        child_name = name.split(".")[-1]

        parent = model

        if parent_name:

            for part in parent_name.split("."):
                parent = getattr(
                    parent,
                    part,
                )

        setattr(
            parent,
            child_name,
            base,
        )

        merged_count += 1

    print(
        f"Merged LoRA modules: "
        f"{merged_count}"
    )

    return model


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    mp.set_start_method(
        "spawn",
        force=True,
    )

    main()