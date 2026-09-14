import os
import re
import json
import math
import argparse
import multiprocessing as mp
from pathlib import Path

# ============================================================
# vLLM multiprocessing
# ============================================================

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


# ============================================================
# Default paths
# ============================================================

DEFAULT_HELDOUT = (
    "/root/autodl-tmp/datasets/physics_peft/test"
)

DEFAULT_BASE_MODEL = (
    "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-SelfPlay/step-15"
)

DEFAULT_PEFT_MODEL = (
    "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-PEFT/final"
)

DEFAULT_OUTPUT_DIR = (
    "/root/autodl-tmp/self_play/physics_eval/heldout"
)


# ============================================================
# Manual LoRA
# ============================================================

class LoRALinear(nn.Module):
    """
    Manual LoRA implementation.

    Base:
        W

    LoRA:
        Delta W = scaling * A @ B

    A:
        [out_features, r]

    B:
        [r, in_features]
    """

    def __init__(
        self,
        base_linear,
        r,
        alpha,
        dropout=0.0,
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError(
                f"Expected nn.Linear, got {type(base_linear)}"
            )

        self.base = base_linear

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(dropout)

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(
            torch.zeros(
                out_features,
                self.r,
                dtype=base_linear.weight.dtype,
                device=base_linear.weight.device,
            )
        )

        self.lora_B = nn.Parameter(
            torch.empty(
                self.r,
                in_features,
                dtype=base_linear.weight.dtype,
                device=base_linear.weight.device,
            )
        )

        nn.init.normal_(
            self.lora_B,
            mean=0.0,
            std=0.02,
        )

        # Base model frozen.
        self.base.weight.requires_grad = False

        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)

        dropped = self.dropout(x)

        # x @ B^T
        lora_hidden = F.linear(
            dropped,
            self.lora_B,
        )

        # (x @ B^T) @ A^T
        lora_out = F.linear(
            lora_hidden,
            self.lora_A,
        )

        return base_out + self.scaling * lora_out


# ============================================================
# Utilities
# ============================================================

def get_parent_module(model, module_name):
    """
    For:
        model.layers.0.mlp.down_proj

    return:
        parent module
        "down_proj"
    """

    parts = module_name.split(".")

    parent = model

    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)

    return parent, parts[-1]


def replace_target_modules(
    model,
    target_modules,
    r,
    alpha,
    dropout,
):
    replaced = 0

    for name, module in list(model.named_modules()):

        if not isinstance(module, nn.Linear):
            continue

        short_name = name.split(".")[-1]

        if short_name not in target_modules:
            continue

        parent, child_name = get_parent_module(
            model,
            name,
        )

        wrapped = LoRALinear(
            module,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        setattr(
            parent,
            child_name,
            wrapped,
        )

        replaced += 1

    return replaced


def load_peft_model(model_dir):
    """
    Load manual LoRA adapter and reconstruct the PEFT model.
    """

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
            f"Missing: {config_path}"
        )

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Missing: {adapter_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)

    base_model = config["base_model"]
    r = int(config["r"])
    alpha = float(config["alpha"])
    dropout = float(config["dropout"])
    target_modules = config["target_modules"]

    print()
    print("=" * 70)
    print("Loading manual LoRA adapter")
    print("=" * 70)

    print("Adapter:", model_dir)
    print("Base   :", base_model)
    print("r      :", r)
    print("alpha  :", alpha)
    print("dropout:", dropout)
    print("target :", target_modules)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    replaced = replace_target_modules(
        model,
        target_modules=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
    )

    print("LoRA modules replaced:", replaced)

    state = torch.load(
        adapter_path,
        map_location="cpu",
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    print("Missing keys   :", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print("First missing keys:")
        for x in missing[:10]:
            print("  ", x)

    if unexpected:
        print("First unexpected keys:")
        for x in unexpected[:10]:
            print("  ", x)

    return model, tokenizer


# ============================================================
# Merge LoRA
# ============================================================

def merge_lora_weights(model):
    """
    Merge:
        W' = W + scaling * A @ B

    Then replace LoRALinear with ordinary nn.Linear.
    """

    merged = 0

    for name, module in list(model.named_modules()):

        if not isinstance(module, LoRALinear):
            continue

        with torch.no_grad():

            delta = (
                module.lora_A @
                module.lora_B
            ) * module.scaling

            module.base.weight.data.add_(
                delta.to(
                    module.base.weight.dtype
                )
            )

        parent, child_name = get_parent_module(
            model,
            name,
        )

        setattr(
            parent,
            child_name,
            module.base,
        )

        merged += 1

    print("Merged LoRA modules:", merged)

    return model


# ============================================================
# Prompt
# ============================================================

def build_prompt(question):
    """
    Physics prompt.

    We deliberately do NOT require boxed output here,
    because camel-ai/physics reference solutions do not
    consistently use \\boxed{}.
    """

    return f"""A conversation between User and Assistant. The User asks a physics problem, and the Assistant solves it. The Assistant first thinks through the reasoning process and then provides the final answer. The reasoning process is enclosed within <think> </think> and the final answer is enclosed within <answer> </answer> tags.

User: {question}

Assistant: <think>"""


# ============================================================
# Number parsing
# ============================================================

NUMBER_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    [+-]?
    (?:
        \d+(?:\.\d*)?
        |
        \.\d+
    )
    (?:
        [eE][+-]?\d+
    )?
    (?![A-Za-z0-9_])
    """,
    re.VERBOSE,
)


FRACTION_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9_])
    [+-]?
    \d+(?:\.\d+)?
    \s*/\s*
    [+-]?
    \d+(?:\.\d+)?
    (?![A-Za-z0-9_])
    """,
    re.VERBOSE,
)


def parse_number(text):
    """
    Convert a single numerical string to float.

    Supports:
        2000
        16.67
        1.4e-2
        1/137.03599920611
    """

    if text is None:
        return None

    text = text.strip()

    # Remove common TeX wrappers.
    text = text.replace(
        r"\,", ""
    )
    text = text.replace(
        ",",
        "",
    )

    # Remove TeX commands around numbers.
    text = re.sub(
        r"\\mathrm\s*\{([^}]*)\}",
        r"\1",
        text,
    )

    text = text.strip()

    # Fraction.
    m = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*/\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
    )

    if m:
        denominator = float(m.group(2))

        if denominator == 0:
            return None

        return float(m.group(1)) / denominator

    # Plain number.
    m = re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][+-]?\d+)?",
        text,
    )

    if m:
        try:
            return float(text)
        except Exception:
            return None

    return None


def normalize_math_text(text):
    if text is None:
        return ""

    text = text.strip()

    # Unicode minus.
    text = text.replace("−", "-")

    # Multiplication signs.
    text = text.replace("×", "*")
    text = text.replace("·", "*")

    # TeX spacing.
    text = re.sub(
        r"\\[,;:!]\s*",
        "",
        text,
    )

    return text


# ============================================================
# Extract reference answer
# ============================================================

CONCLUSION_MARKERS = [
    "therefore",
    "thus",
    "hence",
    "so",
    "the answer is",
    "the final answer is",
    "the result is",
    "is approximately",
    "is about",
    "is equal to",
    "equals",
    "equal to",
]


def split_sentences(text):
    """
    Lightweight sentence splitting.
    """

    if not text:
        return []

    text = text.replace(
        "\n",
        " ",
    )

    # Keep decimal points intact reasonably well.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        s.strip()
        for s in sentences
        if s.strip()
    ]


def extract_number_from_sentence(sentence):
    """
    Extract the likely answer number from one sentence.

    Priority:
    1. fraction
    2. scientific / decimal / integer numbers
    3. last number in the sentence
    """

    sentence = normalize_math_text(
        sentence
    )

    # First look for explicit fraction.
    fractions = FRACTION_PATTERN.findall(
        sentence
    )

    if fractions:
        # Last fraction in the sentence.
        value = parse_number(
            fractions[-1]
        )

        if value is not None:
            return value

    numbers = NUMBER_PATTERN.findall(
        sentence
    )

    if not numbers:
        return None

    # Last number is usually the final numerical result.
    for candidate in reversed(numbers):
        value = parse_number(candidate)

        if value is not None:
            return value

    return None


def reference_is_likely_numeric(question, solution):
    """
    Determine whether the problem has a numerical answer.

    This is deliberately conservative.

    We first inspect conclusion-like sentences.
    If none provides a numerical result, the problem is
    treated as open-ended.
    """

    sentences = split_sentences(solution)

    if not sentences:
        return None

    # --------------------------------------------------------
    # Pass 1:
    # conclusion-like sentences
    # --------------------------------------------------------

    conclusion_candidates = []

    for sentence in sentences:

        lower = sentence.lower()

        if any(
            marker in lower
            for marker in CONCLUSION_MARKERS
        ):
            value = extract_number_from_sentence(
                sentence
            )

            if value is not None:
                conclusion_candidates.append(
                    (sentence, value)
                )

    if conclusion_candidates:
        # Last conclusion sentence.
        return conclusion_candidates[-1][1]

    # --------------------------------------------------------
    # Pass 2:
    # Last few sentences.
    # --------------------------------------------------------

    tail = sentences[-3:]

    for sentence in reversed(tail):

        value = extract_number_from_sentence(
            sentence
        )

        if value is not None:
            return value

    # No numerical final answer.
    return None


# ============================================================
# Model answer extraction
# ============================================================

def extract_answer_section(response):
    """
    Extract content inside <answer>...</answer>.

    If tags are malformed, use the remaining text.
    """

    if not response:
        return ""

    # Normal case.
    m = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        return m.group(1).strip()

    # Sometimes model emits answer without closing tag.
    m = re.search(
        r"<answer>\s*(.*)",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if m:
        return m.group(1).strip()

    return response.strip()


def extract_boxed_content(text):
    """
    Extract the last \\boxed{...}.

    Supports nested braces reasonably well.
    """

    if not text:
        return None

    marker = r"\boxed"

    start = text.rfind(marker)

    if start < 0:
        return None

    brace_start = text.find(
        "{",
        start + len(marker),
    )

    if brace_start < 0:
        return None

    depth = 0

    for i in range(
        brace_start,
        len(text),
    ):

        ch = text[i]

        if ch == "{":
            depth += 1

        elif ch == "}":
            depth -= 1

            if depth == 0:
                return text[
                    brace_start + 1:i
                ].strip()

    return None


def extract_model_number(response):
    """
    Try to extract a numerical final answer.

    Priority:
        1. boxed answer
        2. answer section
        3. conclusion-like sentence
        4. last numerical expression
    """

    if not response:
        return None

    answer_section = extract_answer_section(
        response
    )

    # --------------------------------------------------------
    # 1. boxed
    # --------------------------------------------------------

    boxed = extract_boxed_content(
        answer_section
    )

    if boxed is not None:

        # Handle:
        # x = 2000
        # x \\approx 2000
        if "=" in boxed:
            boxed = boxed.split("=")[-1]

        boxed = re.sub(
            r"\\approx|≈|\\sim|~",
            "",
            boxed,
        ).strip()

        value = parse_number(
            boxed
        )

        if value is not None:
            return value

        # Sometimes boxed contains text + number.
        numbers = NUMBER_PATTERN.findall(
            boxed
        )

        if numbers:
            for candidate in reversed(numbers):
                value = parse_number(candidate)

                if value is not None:
                    return value

    # --------------------------------------------------------
    # 2. Explicit conclusion sentences
    # --------------------------------------------------------

    sentences = split_sentences(
        answer_section
    )

    for sentence in reversed(sentences):

        lower = sentence.lower()

        if any(
            marker in lower
            for marker in CONCLUSION_MARKERS
        ):
            value = extract_number_from_sentence(
                sentence
            )

            if value is not None:
                return value

    # --------------------------------------------------------
    # 3. Last few lines
    # --------------------------------------------------------

    lines = [
        x.strip()
        for x in answer_section.splitlines()
        if x.strip()
    ]

    for line in reversed(lines[-5:]):

        value = extract_number_from_sentence(
            line
        )

        if value is not None:
            return value

    return None


# ============================================================
# Numerical evaluation
# ============================================================

def numerical_match(
    prediction,
    reference,
    relative_tolerance=0.01,
):
    """
    ABench-style 1% relative tolerance.

    For reference == 0:
        use a small absolute tolerance.
    """

    if prediction is None:
        return False

    if reference is None:
        return False

    if not (
        math.isfinite(prediction)
        and math.isfinite(reference)
    ):
        return False

    if reference == 0:
        return abs(prediction) <= 1e-9

    relative_error = abs(
        prediction - reference
    ) / abs(reference)

    return relative_error <= relative_tolerance


def compute_relative_error(
    prediction,
    reference,
):
    if prediction is None or reference is None:
        return None

    if reference == 0:
        return abs(prediction)

    return abs(
        prediction - reference
    ) / abs(reference)


# ============================================================
# Dataset
# ============================================================

def load_heldout_dataset(path):
    ds = load_from_disk(path)

    required = [
        "message_1",
        "message_2",
    ]

    for column in required:
        if column not in ds.column_names:
            raise ValueError(
                f"Missing dataset column: {column}"
            )

    return ds


# ============================================================
# Evaluation
# ============================================================

def evaluate_reference_types(ds):
    """
    Classify all examples before generation.

    This lets us know how many examples have
    an automatically extractable numerical reference.
    """

    results = []

    numeric_count = 0
    open_count = 0

    for idx, item in enumerate(ds):

        question = item["message_1"]
        solution = item["message_2"]

        reference = reference_is_likely_numeric(
            question,
            solution,
        )

        if reference is None:
            category = "open-ended"
            open_count += 1
        else:
            category = "numeric"
            numeric_count += 1

        results.append(
            {
                "index": idx,
                "question": question,
                "reference_solution": solution,
                "reference_answer": reference,
                "category": category,
            }
        )

    return results, numeric_count, open_count


# ============================================================
# Main model evaluation
# ============================================================

def run_model(
    model_path,
    reference_records,
    output_path,
    gpu_memory_utilization=0.70,
    max_model_len=3072,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
):
    """
    Generate answers for the same held-out examples.
    """

    is_peft = os.path.exists(
        os.path.join(
            model_path,
            "adapter_config.json",
        )
    )

    print()
    print("=" * 70)
    print("MODEL")
    print("=" * 70)

    print("Path :", model_path)
    print("PEFT :", is_peft)

    tokenizer = None
    model_for_vllm = model_path

    # --------------------------------------------------------
    # PEFT
    # --------------------------------------------------------

    if is_peft:

        model, tokenizer = load_peft_model(
            model_path
        )

        print()
        print("Merging LoRA weights...")

        model = merge_lora_weights(
            model
        )

        merged_dir = os.path.join(
            os.path.dirname(output_path),
            "_merged_"
            + Path(model_path).name,
        )

        os.makedirs(
            merged_dir,
            exist_ok=True,
        )

        print(
            "Saving merged model:",
            merged_dir,
        )

        model.save_pretrained(
            merged_dir,
            safe_serialization=True,
        )

        tokenizer.save_pretrained(
            merged_dir
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model_for_vllm = merged_dir

    # --------------------------------------------------------
    # vLLM
    # --------------------------------------------------------

    print()
    print("Loading vLLM...")
    print("model =", model_for_vllm)

    llm = LLM(
        model=model_for_vllm,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=(
            gpu_memory_utilization
        ),
        max_model_len=max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    prompts = [
        build_prompt(
            record["question"]
        )
        for record in reference_records
    ]

    print()
    print("=" * 70)
    print("GENERATION")
    print("=" * 70)

    print(
        "Examples:",
        len(prompts),
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    numerical_total = 0
    numerical_correct = 0

    open_total = 0

    boxed_total = 0
    valid_answer_tag = 0

    output_records = []

    for record, output in zip(
        reference_records,
        outputs,
    ):

        response = ""

        if output.outputs:
            response = output.outputs[0].text

        reference = record[
            "reference_answer"
        ]

        category = record[
            "category"
        ]

        answer_section = extract_answer_section(
            response
        )

        boxed = extract_boxed_content(
            answer_section
        )

        prediction = extract_model_number(
            response
        )

        if boxed is not None:
            boxed_total += 1

        if re.search(
            r"<answer>",
            response,
            flags=re.IGNORECASE,
        ):
            valid_answer_tag += 1

        if category == "numeric":

            numerical_total += 1

            correct = numerical_match(
                prediction,
                reference,
            )

            if correct:
                numerical_correct += 1

        else:

            open_total += 1
            correct = None

        relative_error = (
            compute_relative_error(
                prediction,
                reference,
            )
            if reference is not None
            else None
        )

        item = {
            "index": record["index"],
            "category": category,
            "question": record["question"],
            "reference_solution": record[
                "reference_solution"
            ],
            "reference_answer": reference,
            "model_response": response,
            "answer_section": answer_section,
            "boxed_content": boxed,
            "predicted_number": prediction,
            "numerical_correct": correct,
            "relative_error": relative_error,
        }

        output_records.append(item)

    # --------------------------------------------------------
    # Save JSONL
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        for item in output_records:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    numerical_accuracy = (
        numerical_correct / numerical_total
        if numerical_total > 0
        else 0.0
    )

    boxed_rate = (
        boxed_total / len(output_records)
        if output_records
        else 0.0
    )

    answer_tag_rate = (
        valid_answer_tag / len(output_records)
        if output_records
        else 0.0
    )

    summary = {
        "model": model_path,
        "total": len(output_records),
        "numeric_total": numerical_total,
        "numeric_correct": numerical_correct,
        "numeric_accuracy": numerical_accuracy,
        "open_ended_total": open_total,
        "boxed_rate": boxed_rate,
        "answer_tag_rate": answer_tag_rate,
    }

    summary_path = (
        output_path
        + ".summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total examples       : {len(output_records)}"
    )

    print(
        f"Numeric questions    : {numerical_total}"
    )

    print(
        f"Numeric correct      : "
        f"{numerical_correct} / {numerical_total}"
    )

    print(
        f"Numeric accuracy     : "
        f"{numerical_accuracy:.4f}"
    )

    print(
        f"Open-ended questions : {open_total}"
    )

    print(
        f"Boxed output rate    : "
        f"{boxed_rate:.4f}"
    )

    print(
        f"<answer> tag rate    : "
        f"{answer_tag_rate:.4f}"
    )

    print()
    print(
        "Predictions saved to:",
        output_path,
    )

    print(
        "Summary saved to:",
        summary_path,
    )

    del llm

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model path. Can be SelfPlay-15 "
            "or PEFT final."
        ),
    )

    parser.add_argument(
        "--heldout",
        default=DEFAULT_HELDOUT,
        help="Fixed 2,000-example held-out dataset.",
    )

    parser.add_argument(
        "--output_dir",
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
        help=(
            "Only evaluate first N examples. "
            "Use 20 for smoke test."
        ),
    )

    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Output name prefix. "
            "Example: selfplay15 / peft"
        ),
    )

    return parser.parse_args()


# ============================================================
# Entry
# ============================================================

def main():

    args = parse_args()

    print("=" * 70)
    print("Physics Held-out Evaluation")
    print("=" * 70)

    print("Dataset:", args.heldout)

    # --------------------------------------------------------
    # Load fixed test split
    # --------------------------------------------------------

    ds = load_heldout_dataset(
        args.heldout
    )

    print(
        "Loaded examples:",
        len(ds),
    )

    # --------------------------------------------------------
    # Limit
    # --------------------------------------------------------

    if args.limit is not None:

        if args.limit <= 0:
            raise ValueError(
                "--limit must be > 0"
            )

        ds = ds.select(
            range(
                min(
                    args.limit,
                    len(ds),
                )
            )
        )

        print(
            "Limited examples:",
            len(ds),
        )

    # --------------------------------------------------------
    # Reference classification
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("REFERENCE ANALYSIS")
    print("=" * 70)

    records, numeric_count, open_count = (
        evaluate_reference_types(ds)
    )

    print(
        "Total      :",
        len(records),
    )

    print(
        "Numeric    :",
        numeric_count,
    )

    print(
        "Open-ended :",
        open_count,
    )

    print(
        "Numeric ratio:",
        f"{numeric_count / len(records):.4f}"
        if records
        else "0",
    )

    # --------------------------------------------------------
    # Output name
    # --------------------------------------------------------

    model_name = (
        args.name
        if args.name
        else Path(
            args.model.rstrip("/")
        ).name
    )

    output_path = os.path.join(
        args.output_dir,
        model_name
        + "_predictions.jsonl",
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    run_model(
        model_path=args.model,
        reference_records=records,
        output_path=output_path,
        gpu_memory_utilization=(
            args.gpu_memory_utilization
        ),
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":

    mp.set_start_method(
        "spawn",
        force=True,
    )

    main()