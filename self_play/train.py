"""
GRPO-200 model
    ↓
same model generates new math problems
    ↓
problem quality filtering
    ↓
same model independently solves each problem twice
    ↓
answer agreement verification
    ↓
verified self-play samples
    ↓
SFT update
    ↓
updated policy
    ↓
next self-play step

Memory strategy:
    - vLLM is only alive during generation / solving.
    - HF policy stays on CPU during vLLM rollout.
    - vLLM is destroyed before HF SFT.
    - HF model moves to GPU only during SFT.
    - HF model moves back to CPU after SFT.
    - Each updated policy is saved as step-N.
"""

import os
import gc
import json
import random
import re
import time
import math
import multiprocessing as mp

from pathlib import Path
from collections import Counter, defaultdict

# RTX 5090 / vLLM multiprocessing
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


# ============================================================
# 1. PATHS
# ============================================================

MODEL_PATH = (
    "/root/autodl-tmp/models/"
    "Qwen2.5-Math-1.5B-GRPO"
)

OUTPUT_ROOT = Path(
    "/root/autodl-tmp/models/"
    "Qwen2.5-Math-1.5B-SelfPlay-25"
)

DATASET_ROOT = Path(
    "/root/autodl-tmp/datasets/MATH"
)

LOG_ROOT = Path(
    "/root/autodl-tmp/self_play/data"
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


# ============================================================
# 2. SELF-PLAY PARAMETERS
# ============================================================

# 第一轮建议 5 steps。
N_STEPS = 20

# 每轮从 MATH seed 中取多少道题。
SEED_BATCH_SIZE = 32

# 每道 seed 让当前模型出多少道新题。
GEN_ROLLOUTS_PER_SEED = 4


# ============================================================
# 3. PROBLEM GENERATION
# ============================================================

GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 320

GENERATION_MAX_MODEL_LEN = 1536


# ============================================================
# 4. SOLVER
# ============================================================

SOLVE_TEMPERATURE = 0.7
SOLVE_TOP_P = 0.9
SOLVE_MAX_NEW_TOKENS = 1024

SOLVER_MAX_MODEL_LEN = 2560

NUM_SOLVER_ROLLOUTS = 2


# ============================================================
# 5. vLLM MEMORY
# ============================================================

VLLM_GPU_MEMORY_UTILIZATION = 0.82


# ============================================================
# 6. PROBLEM FILTER
# ============================================================

MIN_PROBLEM_TOKENS = 15
MAX_PROBLEM_TOKENS = 512

SIMILARITY_THRESHOLD = 0.93


# ============================================================
# 7. SFT
# ============================================================

SFT_LEARNING_RATE = 1e-5
SFT_EPOCHS = 2

SFT_BATCH_SIZE = 1
SFT_GRADIENT_ACCUMULATION_STEPS = 32

SFT_MAX_LENGTH = 1536

SFT_WEIGHT_DECAY = 0.0
SFT_MAX_GRAD_NORM = 1.0


# 至少有一定数量的 verified pair 才进行一次更新。
MIN_VERIFIED_FOR_SFT = 8


# replay buffer 只保存经过双解一致验证 + audit 的样本。
REPLAY_BUFFER_SIZE = 256


SEED = 42


# ============================================================
# 8. PROMPTS
# ============================================================

SOLVE_SYSTEM_PROMPT = r"""# Instruction
Your task is to solve the math problem below.

You should reason carefully and provide the final answer in the required format.

# Query:
```{instruction}```

# Answer:
"""


SOLVE_USER_PROMPT = r"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""


GENERATOR_PROMPT = r"""You are a mathematical problem generator.

Generate ONE new math problem based on the seed problem.

The new problem must:
1. test the same main mathematical skill;
2. change the numbers, variables, or context;
3. be self-contained;
4. have a unique, well-defined answer;
5. be solvable using the same type of mathematical reasoning.

Do NOT solve the problem.
Do NOT give the answer.
Do NOT explain anything.
Do NOT discuss the seed problem.

Seed problem:
{seed_problem}

New problem:
"""


def build_solve_prompt(question):

    return (
        SOLVE_SYSTEM_PROMPT.format(
            instruction=question
        )
        + "\n"
        + SOLVE_USER_PROMPT.format(
            question=question
        )
    )


# ============================================================
# 9. UTILITIES
# ============================================================

def write_jsonl(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


def clean_text(text):

    text = text or ""

    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = text.replace(
        "\x00",
        " "
    )

    return text.strip()


def normalize_ws(text):

    return re.sub(
        r"\s+",
        " ",
        clean_text(text)
    ).strip().lower()


def similarity_ratio(a, b):

    try:

        from rapidfuzz.fuzz import ratio

        return (
            ratio(
                normalize_ws(a),
                normalize_ws(b)
            )
            / 100.0
        )

    except Exception:

        import difflib

        return difflib.SequenceMatcher(
            None,
            normalize_ws(a),
            normalize_ws(b)
        ).ratio()


def token_len(tokenizer, text):

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


def strip_code_fences(text):

    text = clean_text(text)

    text = re.sub(
        r"^```(?:text|latex|markdown)?\s*",
        "",
        text,
        flags=re.I
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


# ============================================================
# 10. LOAD MATH DATA
# ============================================================

def load_all_train_data():

    samples = []

    for config in CONFIGS:

        dataset_path = (
            DATASET_ROOT
            / config
            / "train"
        )

        dataset = load_from_disk(
            str(dataset_path)
        )

        for idx, item in enumerate(dataset):

            samples.append(
                {
                    "id": f"{config}_{idx}",
                    "config": config,
                    "index": idx,

                    # MATH schema:
                    # problem / solution
                    "question": item["problem"],
                    "ground_truth": item["solution"],
                }
            )

    return samples


def make_seed_order(samples):

    rng = random.Random(SEED)

    by_config = defaultdict(list)

    for item in samples:

        by_config[
            item["config"]
        ].append(item)

    for config in CONFIGS:

        rng.shuffle(
            by_config[config]
        )

    ordered = []

    pos = 0

    while True:

        added = False

        for config in CONFIGS:

            if pos < len(
                by_config[config]
            ):

                ordered.append(
                    by_config[config][pos]
                )

                added = True

        if not added:
            break

        pos += 1

    rng.shuffle(ordered)

    return ordered


# ============================================================
# 11. GENERATOR PARSER
# ============================================================

GENERATION_PREFIXES = [
    r"^\s*problem\s*:\s*",
    r"^\s*new\s+problem\s*:\s*",
    r"^\s*problem\s+statement\s*:\s*",
]


META_PREFIX_PATTERNS = [
    r"^\s*the\s+new\s+problem\s+is\s*:?\s*",
    r"^\s*one\s+new\s+problem\s+is\s*:?\s*",
    r"^\s*a\s+new\s+problem\s+is\s*:?\s*",
    r"^\s*this\s+new\s+problem\s+is\s*:?\s*",
    r"^\s*the\s+original\s+problem\s*:?\s*",
    r"^\s*the\s+seed\s+problem\s*:?\s*",
    r"^\s*same\s+mathematical\s+skill\s*:?\s*",
    r"^\s*changed\s+(?:numbers|values|context)\s*:?\s*",
]


SOLUTION_TRANSITIONS = [

    r"\bto\s+solve\s+the\s+problem\b",

    r"\bto\s+determine\s+the\s+answer\b",

    r"\bto\s+find\s+the\s+answer\b",

    r"\bwe\s+need\s+to\s+solve\b",

    r"\bwe\s+need\s+to\s+calculate\b",

    r"\bwe\s+can\s+solve\b",

    r"\blet'?s\s+solve\b",

    r"\bthe\s+solution\s+is\b",

    r"\bthe\s+answer\s+is\b",

    r"\bfinal\s+answer\b",

    r"\btherefore\b",

    r"\bthus\b",

    r"\bsubstituting\b",

    r"\bsimplifying\b",

    r"\bnow\s+we\s+calculate\b",

    r"\bplugging\s+(?:this|these|the)\b",

    r"\bwe\s+get\b",

    r"\bhence\b",

    r"\bso\s+the\s+answer\s+is\b",
]


SOLUTION_RE = re.compile(
    "|".join(
        SOLUTION_TRANSITIONS
    ),
    flags=re.I
)


CODE_OR_PLACEHOLDER_PATTERNS = [

    r"\[new\s+problem\]",

    r"\[same\s+as\s+seed\s+problem\]",

    r"\[generate_problem\]",

    r"\[generate_answer\]",

    r"\bplaceholder\b",

    r"\bTODO\b",

    r"```",

    r"<think>",

    r"<answer>",
]


META_GENERATION_PATTERNS = [

    r"\bgenerate\s+(?:a|one)\s+new\s+problem\b",

    r"\bbased\s+on\s+the\s+seed\b",

    r"\bsame\s+mathematical\s+skill\b",

    r"\bchanged\s+(?:numbers|values|context)\b",

    r"\bwe\s+need\s+to\s+create\b",

    r"\bhere\s+is\s+(?:a|the)\s+new\s+problem\b",
]


def extract_last_boxed(text):

    matches = re.findall(
        r"\\boxed\s*\{"
        r"([^{}]*(?:\{[^{}]*\}[^{}]*)*)"
        r"\}",
        text
    )

    if matches:

        return matches[-1].strip()

    return None


def strip_generator_prefix(text):

    text = strip_code_fences(text)

    for pattern in GENERATION_PREFIXES:

        new_text = re.sub(
            pattern,
            "",
            text,
            count=1,
            flags=re.I
        )

        if new_text != text:

            return (
                new_text.strip(),
                "heading"
            )

    for pattern in META_PREFIX_PATTERNS:

        new_text = re.sub(
            pattern,
            "",
            text,
            count=1,
            flags=re.I
        )

        if new_text != text:

            return (
                new_text.strip(),
                "strip_meta_preamble"
            )

    return (
        text.strip(),
        "none"
    )


def cut_solution_contamination(text):

    text = clean_text(text)

    lines = text.splitlines()

    kept = []

    for line in lines:

        if (
            not kept
            and not line.strip()
        ):
            continue

        if SOLUTION_RE.search(line):

            if len(
                " ".join(kept)
            ) >= 60:

                return (
                    "\n".join(
                        kept
                    ).strip(),
                    "truncate_ordinary_solution"
                )

        kept.append(line)

    text2 = "\n".join(
        kept
    ).strip()

    for pattern in SOLUTION_TRANSITIONS:

        m = re.search(
            pattern,
            text2,
            flags=re.I
        )

        if (
            m
            and m.start() >= 80
        ):

            return (
                text2[:m.start()]
                .rstrip(" \n:;-"),
                "truncate_ordinary_solution"
            )

    return (
        text2,
        "none"
    )


def generated_problem_copies_seed(
    seed_problem,
    generated_problem
):

    seed_norm = normalize_ws(
        seed_problem
    )

    gen_norm = normalize_ws(
        generated_problem
    )

    if not seed_norm or not gen_norm:

        return False

    seed_prefix = seed_norm[:240]
    gen_prefix = gen_norm[:240]

    if similarity_ratio(
        seed_prefix,
        gen_prefix
    ) >= 0.97:

        return True

    if (
        len(seed_norm) >= 160
        and gen_norm.startswith(
            seed_norm[:160]
        )
    ):

        return True

    return False


def problem_contains_obvious_solution(text):

    patterns = [

        r"\bsolution\s*:",

        r"\banswer\s*:",

        r"\bfinal\s+answer\s*:",

        r"\bthe\s+answer\s+is\b",

        r"\bthe\s+solution\s+is\b",

        r"\bwe\s+therefore\b",

        r"\btherefore\s+the\s+answer\b",

        r"\\boxed\s*\{",
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            flags=re.I
        ):

            return True

    return False


def generated_problem_is_obviously_bad(text):

    norm = normalize_ws(text)

    if not norm:

        return "empty"

    if any(
        re.search(
            p,
            text,
            flags=re.I
        )
        for p in CODE_OR_PLACEHOLDER_PATTERNS
    ):

        return "code_or_placeholder"

    if any(
        re.search(
            p,
            text,
            flags=re.I
        )
        for p in META_GENERATION_PATTERNS
    ):

        return "meta_generation_prefix"

    if len(norm) < 50:

        return "too_short"

    math_signal = re.search(
        r"(?:"
        r"\d|=|\\frac|\\sqrt|\+|-|\*|/"
        r"|percent|probability"
        r"|triangle|circle|angle"
        r"|area|volume|integer"
        r"|equation|function|sequence"
        r"|polynomial|ratio|distance"
        r"|length|perimeter|radius"
        r"|factor|divisible|prime"
        r")",
        text,
        flags=re.I
    )

    question_signal = re.search(
        r"(?:"
        r"\?|find|determine|calculate"
        r"|how many|what is|what are"
        r"|compute|evaluate|solve"
        r"|probability"
        r")",
        text,
        flags=re.I
    )

    if (
        not math_signal
        or not question_signal
    ):

        return "not_problem_like"

    return None


def parse_generator_output(
    raw_output,
    seed_problem,
    tokenizer
):

    raw_output = clean_text(
        raw_output
    )

    parsed, action = (
        strip_generator_prefix(
            raw_output
        )
    )

    parsed, action2 = (
        cut_solution_contamination(
            parsed
        )
    )

    if (
        action == "none"
        and action2 != "none"
    ):

        action = action2

    elif (
        action == "heading"
        and action2 != "none"
    ):

        action = (
            f"{action}+{action2}"
        )

    parsed = strip_code_fences(
        parsed
    ).strip()

    reason = (
        generated_problem_is_obviously_bad(
            parsed
        )
    )

    if reason:

        return {
            "accepted": False,
            "problem": parsed,
            "cleaning_action": action,
            "reason": reason,
            "similarity": similarity_ratio(
                seed_problem,
                parsed
            ),
        }

    if generated_problem_copies_seed(
        seed_problem,
        parsed
    ):

        return {
            "accepted": False,
            "problem": parsed,
            "cleaning_action": action,
            "reason": "seed_copy",
            "similarity": similarity_ratio(
                seed_problem,
                parsed
            ),
        }

    if problem_contains_obvious_solution(
        parsed
    ):

        return {
            "accepted": False,
            "problem": parsed,
            "cleaning_action": action,
            "reason": "answer_leak_in_problem",
            "similarity": similarity_ratio(
                seed_problem,
                parsed
            ),
        }

    sim = similarity_ratio(
        seed_problem,
        parsed
    )

    if sim >= SIMILARITY_THRESHOLD:

        return {
            "accepted": False,
            "problem": parsed,
            "cleaning_action": action,
            "reason": "seed_copy",
            "similarity": sim,
        }

    n_tokens = token_len(
        tokenizer,
        parsed
    )

    if n_tokens < MIN_PROBLEM_TOKENS:

        return {
            "accepted": False,
            "problem": parsed,
            "cleaning_action": action,
            "reason": "too_short",
            "similarity": sim,
        }

    if n_tokens > MAX_PROBLEM_TOKENS:

        return {
            "accepted": False,
            "problem": parsed,
            "cleaning_action": action,
            "reason": "too_long",
            "similarity": sim,
        }

    return {
        "accepted": True,
        "problem": parsed,
        "cleaning_action": action,
        "reason": "ok",
        "similarity": sim,
    }


# ============================================================
# 12. SOLVER ANSWER EXTRACTION
# ============================================================

def normalize_answer_text(answer):

    if answer is None:

        return None

    answer = clean_text(
        answer
    ).strip()

    answer = answer.strip(
        " \t\n.,;"
    )

    if not answer:

        return None

    return answer


def extract_solver_answer(response):

    response = clean_text(
        response
    )

    # 1. Exact expected format.
    m = re.search(
        r"</think>\s*"
        r"<answer>\s*"
        r"(.*?)"
        r"\s*</answer>",
        response,
        flags=re.I | re.S
    )

    if m:

        ans = normalize_answer_text(
            m.group(1)
        )

        if ans:

            return (
                ans,
                "answer_block"
            )

    # 2. Truncated answer block.
    m = re.search(
        r"<answer>\s*(.*)$",
        response,
        flags=re.I | re.S
    )

    if m:

        ans = normalize_answer_text(
            m.group(1)
        )

        if ans:

            return (
                ans,
                "truncated_answer_block"
            )

    # 3. Last boxed answer.
    boxed = extract_last_boxed(
        response
    )

    if boxed:

        ans = normalize_answer_text(
            boxed
        )

        if ans:

            return (
                ans,
                "boxed"
            )

    # 4. Explicit answer line.
    matches = list(
        re.finditer(
            r"(?:final\s+answer|answer)"
            r"\s*:\s*(.+)",
            response,
            flags=re.I
        )
    )

    if matches:

        ans = normalize_answer_text(
            matches[-1].group(1)
        )

        if ans:

            return (
                ans,
                "answer_line"
            )

    return (
        None,
        "missing"
    )


def canonical_answer_for_agreement(
    answer
):

    answer = normalize_answer_text(
        answer
    )

    if not answer:

        return None

    norm = normalize_ws(
        answer
    )

    norm = norm.replace(
        "$",
        ""
    )

    norm = norm.replace(
        "−",
        "-"
    )

    norm = norm.replace(
        " ",
        ""
    )

    return norm


def answers_agree(a, b):

    a = normalize_answer_text(a)
    b = normalize_answer_text(b)

    if not a or not b:

        return False

    ca = canonical_answer_for_agreement(
        a
    )

    cb = canonical_answer_for_agreement(
        b
    )

    if ca == cb:

        return True

    # Try project grader if available.
    try:

        from grader import grade_answer_mathd

        if grade_answer_mathd(
            a,
            b
        ):

            return True

    except Exception:
        pass

    try:

        from drgrpo_grader import (
            grade_answer_mathd
        )

        if grade_answer_mathd(
            a,
            b
        ):

            return True

    except Exception:
        pass

    # SymPy fallback.
    try:

        import sympy as sp
        from sympy.parsing.latex import (
            parse_latex
        )

        aa = parse_latex(a)
        bb = parse_latex(b)

        if sp.simplify(
            aa - bb
        ) == 0:

            return True

    except Exception:
        pass

    return False


# ============================================================
# 13. VLLM
# ============================================================

def build_llm(
    model_path,
    max_model_len
):

    print(
        f"[vLLM] loading: {model_path}"
    )

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=(
            VLLM_GPU_MEMORY_UTILIZATION
        ),
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    return llm


def destroy_llm(llm):

    if llm is None:

        return

    try:

        del llm

    except Exception:
        pass

    gc.collect()

    try:

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception:
        pass

    time.sleep(2)


# ============================================================
# 14. GENERATE NEW PROBLEMS
# ============================================================

def generate_candidates(
    llm,
    tokenizer,
    seeds,
    step
):

    prompts = []
    meta = []

    for seed in seeds:

        for rollout_id in range(
            GEN_ROLLOUTS_PER_SEED
        ):

            prompts.append(
                GENERATOR_PROMPT.format(
                    seed_problem=seed["question"]
                )
            )

            meta.append(
                {
                    "seed_id": seed["id"],
                    "config": seed["config"],
                    "seed_question": seed["question"],
                    "rollout_id": rollout_id,
                }
            )

    sampling_params = SamplingParams(
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_NEW_TOKENS,
        min_tokens=8,
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    accepted = []
    diagnostics = []

    for i, out in enumerate(outputs):

        raw = (
            out.outputs[0].text
            if out.outputs
            else ""
        )

        info = parse_generator_output(
            raw,
            meta[i]["seed_question"],
            tokenizer
        )

        record = {
            "step": step,
            **meta[i],
            "raw_output": raw,
            "parsed_problem": info["problem"],
            "cleaning_action": info[
                "cleaning_action"
            ],
            "reason": info["reason"],
            "similarity": info["similarity"],
            "accepted": info["accepted"],
        }

        diagnostics.append(
            record
        )

        if info["accepted"]:

            accepted.append(
                {
                    "step": step,
                    "seed_id": meta[i]["seed_id"],
                    "config": meta[i]["config"],
                    "seed_question": meta[i][
                        "seed_question"
                    ],
                    "rollout_id": meta[i][
                        "rollout_id"
                    ],
                    "problem": info["problem"],
                    "similarity": info[
                        "similarity"
                    ],
                    "cleaning_action": info[
                        "cleaning_action"
                    ],
                    "raw_generator_output": raw,
                }
            )

    return (
        accepted,
        diagnostics
    )


# ============================================================
# 15. SOLVE TWICE
# ============================================================

def solve_verified_problems(
    llm,
    accepted_problems,
    step
):

    if not accepted_problems:

        return (
            [],
            [],
            {
                "verified": 0,
                "missing": 0,
                "disagreement": 0,
            }
        )

    prompts = []
    meta = []

    for problem_index, item in enumerate(
        accepted_problems
    ):

        prompt = build_solve_prompt(
            item["problem"]
        )

        for rollout_id in range(
            NUM_SOLVER_ROLLOUTS
        ):

            prompts.append(prompt)

            meta.append(
                {
                    "problem_index": problem_index,
                    "rollout_id": rollout_id,
                    "item": item,
                }
            )

    sampling_params = SamplingParams(
        temperature=SOLVE_TEMPERATURE,
        top_p=SOLVE_TOP_P,
        max_tokens=SOLVE_MAX_NEW_TOKENS,
        min_tokens=4,
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    grouped = defaultdict(list)

    for i, out in enumerate(outputs):

        response = (
            out.outputs[0].text
            if out.outputs
            else ""
        )

        answer, method = (
            extract_solver_answer(
                response
            )
        )

        grouped[
            meta[i]["problem_index"]
        ].append(
            {
                "rollout_id": meta[i][
                    "rollout_id"
                ],
                "response": response,
                "answer": answer,
                "extraction_method": method,
            }
        )

    verified = []
    solver_records = []

    missing = 0
    disagreement = 0

    for problem_index, item in enumerate(
        accepted_problems
    ):

        sols = grouped.get(
            problem_index,
            []
        )

        while len(sols) < 2:

            sols.append(
                {
                    "rollout_id": len(sols),
                    "response": "",
                    "answer": None,
                    "extraction_method": (
                        "missing_rollout"
                    ),
                }
            )

        a = sols[0]["answer"]
        b = sols[1]["answer"]

        if not a or not b:

            status = "missing_answer"
            missing += 1

        elif answers_agree(a, b):

            status = "verified"

            verified.append(
                {
                    "step": step,
                    "config": item["config"],
                    "seed_id": item["seed_id"],
                    "seed_question": item[
                        "seed_question"
                    ],
                    "problem": item["problem"],

                    "answer": a,

                    "solution": sols[0][
                        "response"
                    ],

                    "solution_1": sols[0][
                        "response"
                    ],

                    "solution_2": sols[1][
                        "response"
                    ],

                    "answer_1": a,
                    "answer_2": b,

                    "similarity": item[
                        "similarity"
                    ],

                    "generator_cleaning_action":
                        item[
                            "cleaning_action"
                        ],

                    "verification":
                        "two_solver_agreement",
                }
            )

        else:

            status = "disagreement"
            disagreement += 1

        solver_records.append(
            {
                "step": step,
                "config": item["config"],
                "seed_id": item["seed_id"],
                "problem": item["problem"],

                "solver_1": sols[0],
                "solver_2": sols[1],

                "status": status,

                "similarity": item[
                    "similarity"
                ],
            }
        )

    stats = {
        "verified": len(verified),
        "missing": missing,
        "disagreement": disagreement,
    }

    return (
        verified,
        solver_records,
        stats
    )


# ============================================================
# 16. V7.4 AUDIT
# ============================================================

def audit_verified_sample(
    item,
    tokenizer
):

    problem = clean_text(
        item.get(
            "problem",
            ""
        )
    )

    if not problem:

        return (
            False,
            "empty"
        )

    n_tokens = token_len(
        tokenizer,
        problem
    )

    if n_tokens < MIN_PROBLEM_TOKENS:

        return (
            False,
            "too_short"
        )

    if n_tokens > MAX_PROBLEM_TOKENS:

        return (
            False,
            "too_long"
        )

    if problem_contains_obvious_solution(
        problem
    ):

        return (
            False,
            "answer_leak"
        )

    # NOTE:
    # [asy] is legitimate MATH content and is
    # intentionally NOT rejected.
    if any(
        re.search(
            p,
            problem,
            flags=re.I
        )
        for p in CODE_OR_PLACEHOLDER_PATTERNS
    ):

        return (
            False,
            "code_or_placeholder"
        )

    if any(
        re.search(
            p,
            problem,
            flags=re.I
        )
        for p in META_GENERATION_PATTERNS
    ):

        return (
            False,
            "meta_generation"
        )

    if re.search(
        r"\b(?:"
        r"to solve"
        r"|solution"
        r"|reasoning"
        r"|let's calculate"
        r"|we calculate"
        r"|substituting"
        r"|simplifying"
        r")\b",
        problem,
        flags=re.I
    ):

        return (
            False,
            "solution_leak"
        )

    bad = generated_problem_is_obviously_bad(
        problem
    )

    if bad:

        return (
            False,
            bad
        )

    return (
        True,
        "ok"
    )


def audit_verified(
    verified,
    tokenizer
):

    clean = []
    rejected = []

    for item in verified:

        ok, reason = (
            audit_verified_sample(
                item,
                tokenizer
            )
        )

        audit_item = {
            **item,

            "audit_status":
                "HIGH_QUALITY"
                if ok
                else "REJECTED",

            "audit_reason":
                reason,
        }

        if ok:

            clean.append(
                audit_item
            )

        else:

            rejected.append(
                audit_item
            )

    return (
        clean,
        rejected
    )


# ============================================================
# 17. SFT DATASET
# ============================================================

class SelfPlaySFTDataset(
    Dataset
):

    def __init__(
        self,
        rows,
        tokenizer,
        max_length
    ):

        self.examples = []

        for row in rows:

            problem = row["problem"]
            response = row["solution"]

            prompt = build_solve_prompt(
                problem
            )

            prompt_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False
            )

            response_ids = tokenizer.encode(
                response,
                add_special_tokens=False
            )

            if not response_ids:

                continue

            total_len = (
                len(prompt_ids)
                + len(response_ids)
            )

            if total_len > max_length:

                available_response = (
                    max_length
                    - len(prompt_ids)
                )

                if available_response <= 8:

                    continue

                response_ids = (
                    response_ids[
                        :available_response
                    ]
                )

            input_ids = (
                prompt_ids
                + response_ids
            )

            labels = (
                [-100] * len(prompt_ids)
                + response_ids
            )

            if len(input_ids) > max_length:

                input_ids = (
                    input_ids[:max_length]
                )

                labels = (
                    labels[:max_length]
                )

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "labels": labels,
                }
            )

    def __len__(self):

        return len(
            self.examples
        )

    def __getitem__(self, idx):

        return self.examples[idx]


class SFTCollator:

    def __init__(
        self,
        tokenizer
    ):

        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id
            is not None
            else tokenizer.eos_token_id
        )

    def __call__(
        self,
        batch
    ):

        max_len = max(
            len(x["input_ids"])
            for x in batch
        )

        input_ids = []
        labels = []
        attention_mask = []

        for x in batch:

            n = len(
                x["input_ids"]
            )

            pad = (
                max_len - n
            )

            input_ids.append(
                x["input_ids"]
                + [self.pad_id] * pad
            )

            labels.append(
                x["labels"]
                + [-100] * pad
            )

            attention_mask.append(
                [1] * n
                + [0] * pad
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),

            "labels": torch.tensor(
                labels,
                dtype=torch.long
            ),

            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long
            ),
        }


# ============================================================
# 18. SFT
# ============================================================

def run_sft(
    policy,
    tokenizer,
    rows,
    step
):

    if len(rows) < MIN_VERIFIED_FOR_SFT:

        print(
            f"[SFT] only {len(rows)} examples; "
            f"minimum={MIN_VERIFIED_FOR_SFT}. "
            "Skip update."
        )

        return {
            "updated": False,
            "examples": len(rows),
            "steps": 0,
            "mean_loss": None,
        }

    dataset = SelfPlaySFTDataset(
        rows,
        tokenizer,
        SFT_MAX_LENGTH
    )

    if len(dataset) < MIN_VERIFIED_FOR_SFT:

        print(
            f"[SFT] tokenized examples="
            f"{len(dataset)}; skip."
        )

        return {
            "updated": False,
            "examples": len(dataset),
            "steps": 0,
            "mean_loss": None,
        }

    loader = DataLoader(
        dataset,
        batch_size=SFT_BATCH_SIZE,
        shuffle=True,
        collate_fn=SFTCollator(
            tokenizer
        ),
        num_workers=0,
        pin_memory=True,
    )

    policy.train()

    policy.config.use_cache = False

    if hasattr(
        policy,
        "gradient_checkpointing_enable"
    ):

        policy.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        policy.parameters(),

        lr=SFT_LEARNING_RATE,

        weight_decay=SFT_WEIGHT_DECAY,

        betas=(0.9, 0.95),
    )

    device = next(
        policy.parameters()
    ).device

    total_loss = 0.0
    loss_count = 0
    optimizer_steps = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        SFT_EPOCHS
    ):

        for batch_idx, batch in enumerate(
            loader
        ):

            batch = {
                k: v.to(
                    device,
                    non_blocking=True
                )
                for k, v in batch.items()
            }

            outputs = policy(
                input_ids=batch[
                    "input_ids"
                ],

                attention_mask=batch[
                    "attention_mask"
                ],

                labels=batch[
                    "labels"
                ],

                use_cache=False,
            )

            loss = outputs.loss

            scaled_loss = (
                loss
                / SFT_GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

            total_loss += float(
                loss.detach().cpu()
            )

            loss_count += 1

            should_step = (
                (
                    batch_idx + 1
                )
                % SFT_GRADIENT_ACCUMULATION_STEPS
                == 0
                or batch_idx + 1
                == len(loader)
            )

            if should_step:

                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    SFT_MAX_GRAD_NORM
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_steps += 1

        print(
            f"[SFT] "
            f"step={step} "
            f"epoch={epoch + 1}/{SFT_EPOCHS} "
            f"mean_loss="
            f"{total_loss / max(loss_count, 1):.6f}"
        )

    mean_loss = (
        total_loss
        / max(loss_count, 1)
    )

    del optimizer

    gc.collect()

    torch.cuda.empty_cache()

    return {
        "updated": True,
        "examples": len(dataset),
        "steps": optimizer_steps,
        "mean_loss": mean_loss,
    }


# ============================================================
# 19. MODEL SAVE / CPU OFFLOAD
# ============================================================

def move_policy_to_cpu(
    policy
):

    policy.eval()

    policy.to("cpu")

    gc.collect()

    try:

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception:
        pass


def save_policy(
    policy,
    tokenizer,
    path
):

    path.mkdir(
        parents=True,
        exist_ok=True
    )

    policy.save_pretrained(
        str(path),
        safe_serialization=True
    )

    tokenizer.save_pretrained(
        str(path)
    )


# ============================================================
# 20. GENERATION SUMMARY
# ============================================================

def summarize_generation(
    diagnostics
):

    reasons = Counter()
    actions = Counter()

    accepted = 0

    for x in diagnostics:

        if x["accepted"]:

            accepted += 1

        reasons[
            x["reason"]
        ] += 1

        actions[
            x["cleaning_action"]
        ] += 1

    sims = [
        x["similarity"]
        for x in diagnostics
        if x["accepted"]
    ]

    return {
        "total_candidates":
            len(diagnostics),

        "accepted_candidates":
            accepted,

        "acceptance_rate":
            (
                accepted
                / len(diagnostics)
                if diagnostics
                else 0.0
            ),

        "reasons":
            dict(reasons),

        "cleaning_actions":
            dict(actions),

        "accepted_similarity_mean":
            (
                sum(sims)
                / len(sims)
                if sims
                else None
            ),

        "accepted_similarity_min":
            (
                min(sims)
                if sims
                else None
            ),

        "accepted_similarity_max":
            (
                max(sims)
                if sims
                else None
            ),
    }


def summarize_configs(rows):

    counter = Counter()

    for row in rows:

        counter[
            row["config"]
        ] += 1

    return dict(counter)


# ============================================================
# 21. MAIN
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    LOG_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 80)
    print("SELF-PLAY V8")
    print("=" * 80)

    print(
        f"Base model         : {MODEL_PATH}"
    )

    print(
        f"Output root        : {OUTPUT_ROOT}"
    )

    print(
        f"Steps              : {N_STEPS}"
    )

    print(
        f"Seed batch         : {SEED_BATCH_SIZE}"
    )

    print(
        f"Generator rollouts : "
        f"{GEN_ROLLOUTS_PER_SEED}"
    )

    print(
        f"Solver rollouts    : "
        f"{NUM_SOLVER_ROLLOUTS}"
    )

    print(
        f"Replay buffer      : "
        f"{REPLAY_BUFFER_SIZE}"
    )

    print("=" * 80)

    random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=True
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # --------------------------------------------------------
    # MATH seeds
    # --------------------------------------------------------

    print(
        "[DATA] loading MATH..."
    )

    all_samples = (
        load_all_train_data()
    )

    seed_order = (
        make_seed_order(
            all_samples
        )
    )

    print(
        f"[DATA] total seeds: "
        f"{len(seed_order)}"
    )

    # --------------------------------------------------------
    # HF policy lives on CPU.
    # --------------------------------------------------------

    print(
        "[HF] loading policy on CPU..."
    )

    policy = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    )

    policy.to("cpu")
    policy.eval()

    current_model_path = MODEL_PATH

    seed_cursor = 0

    replay_buffer = []

    all_step_summaries = []

    # ========================================================
    # SELF-PLAY LOOP
    # ========================================================

    for step in range(
        1,
        N_STEPS + 1
    ):

        step_start = time.time()

        print("\n")
        print("=" * 80)
        print(
            f"SELF-PLAY STEP "
            f"{step}/{N_STEPS}"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # A. Select seeds
        # ----------------------------------------------------

        if (
            seed_cursor
            + SEED_BATCH_SIZE
            <= len(seed_order)
        ):

            seeds = seed_order[
                seed_cursor:
                seed_cursor
                + SEED_BATCH_SIZE
            ]

            seed_cursor += (
                SEED_BATCH_SIZE
            )

        else:

            remaining = seed_order[
                seed_cursor:
            ]

            rng = random.Random(
                SEED + step
            )

            rng.shuffle(
                seed_order
            )

            need = (
                SEED_BATCH_SIZE
                - len(remaining)
            )

            seeds = (
                remaining
                + seed_order[:need]
            )

            seed_cursor = need

        print(
            "[SEEDS]",
            Counter(
                x["config"]
                for x in seeds
            )
        )

        # ----------------------------------------------------
        # B. vLLM generation
        # ----------------------------------------------------

        llm = None

        try:

            llm = build_llm(
                current_model_path,
                GENERATION_MAX_MODEL_LEN
            )

            (
                accepted,
                generation_records
            ) = generate_candidates(
                llm,
                tokenizer,
                seeds,
                step
            )

            gen_summary = (
                summarize_generation(
                    generation_records
                )
            )

            print(
                "\n[GENERATION]"
            )

            print(
                "Total candidates :",
                gen_summary[
                    "total_candidates"
                ]
            )

            print(
                "Accepted         :",
                gen_summary[
                    "accepted_candidates"
                ]
            )

            print(
                "Acceptance rate  :",
                f"{gen_summary['acceptance_rate']:.4f}"
            )

            print(
                "Reasons:",
                json.dumps(
                    gen_summary[
                        "reasons"
                    ],
                    ensure_ascii=False
                )
            )

            # ------------------------------------------------
            # Destroy generator vLLM.
            # Rebuild with longer solver context.
            # ------------------------------------------------

            destroy_llm(
                llm
            )

            llm = None

            llm = build_llm(
                current_model_path,
                SOLVER_MAX_MODEL_LEN
            )

            # ------------------------------------------------
            # C. Solve twice
            # ------------------------------------------------

            (
                verified_raw,
                solver_records,
                solver_stats
            ) = solve_verified_problems(
                llm,
                accepted,
                step
            )

            print(
                "\n[SOLVER VERIFICATION]"
            )

            print(
                "Accepted problems :",
                len(accepted)
            )

            print(
                "Verified          :",
                solver_stats[
                    "verified"
                ]
            )

            print(
                "Missing answer    :",
                solver_stats[
                    "missing"
                ]
            )

            print(
                "Disagreement      :",
                solver_stats[
                    "disagreement"
                ]
            )

            agreement_rate = (
                solver_stats["verified"]
                / len(accepted)
                if accepted
                else 0.0
            )

            print(
                "Agreement rate    :",
                f"{agreement_rate:.4f}"
            )

            # ------------------------------------------------
            # D. Final V7.4-style audit
            # ------------------------------------------------

            (
                clean_verified,
                audit_rejected
            ) = audit_verified(
                verified_raw,
                tokenizer
            )

            print(
                "\n[AUDIT]"
            )

            print(
                "Verified raw      :",
                len(verified_raw)
            )

            print(
                "High quality      :",
                len(clean_verified)
            )

            print(
                "Rejected          :",
                len(audit_rejected)
            )

            print(
                "Verified configs  :",
                summarize_configs(
                    clean_verified
                )
            )

            # ------------------------------------------------
            # Save logs
            # ------------------------------------------------

            step_dir = (
                LOG_ROOT
                / "v8_steps"
            )

            step_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            write_jsonl(
                step_dir
                / f"step_{step:03d}_generation.jsonl",
                generation_records
            )

            write_jsonl(
                step_dir
                / f"step_{step:03d}_solver.jsonl",
                solver_records
            )

            write_jsonl(
                step_dir
                / f"step_{step:03d}_verified_raw.jsonl",
                verified_raw
            )

            write_jsonl(
                step_dir
                / f"step_{step:03d}_verified_clean.jsonl",
                clean_verified
            )

            write_jsonl(
                step_dir
                / f"step_{step:03d}_verified_rejected.jsonl",
                audit_rejected
            )

        finally:

            # ------------------------------------------------
            # IMPORTANT:
            # vLLM MUST be destroyed before HF SFT.
            # ------------------------------------------------

            destroy_llm(
                llm
            )

            llm = None

        # ----------------------------------------------------
        # E. Replay buffer
        # ----------------------------------------------------

        replay_buffer.extend(
            clean_verified
        )

        if (
            len(replay_buffer)
            > REPLAY_BUFFER_SIZE
        ):

            replay_buffer = (
                replay_buffer[
                    -REPLAY_BUFFER_SIZE:
                ]
            )

        train_rows = list(
            replay_buffer
        )

        print(
            "\n[SFT DATA]"
        )

        print(
            "Current verified :",
            len(clean_verified)
        )

        print(
            "Replay buffer    :",
            len(replay_buffer)
        )

        print(
            "SFT examples     :",
            len(train_rows)
        )

        # ----------------------------------------------------
        # F. HF SFT
        # ----------------------------------------------------

        sft_stats = {
            "updated": False,
            "examples": len(train_rows),
            "steps": 0,
            "mean_loss": None,
        }

        if (
            len(train_rows)
            >= MIN_VERIFIED_FOR_SFT
        ):

            print(
                "[HF] CPU -> GPU"
            )

            policy.to("cuda")

            sft_stats = run_sft(
                policy,
                tokenizer,
                train_rows,
                step
            )

            print(
                f"[SFT] "
                f"updated={sft_stats['updated']} "
                f"examples={sft_stats['examples']} "
                f"optimizer_steps="
                f"{sft_stats['steps']} "
                f"mean_loss="
                f"{sft_stats['mean_loss']}"
            )

            # ------------------------------------------------
            # GPU -> CPU
            # ------------------------------------------------

            move_policy_to_cpu(
                policy
            )

            # ------------------------------------------------
            # Save updated policy
            # ------------------------------------------------

            step_path = (
                OUTPUT_ROOT
                / f"step-{step}"
            )

            print(
                "[SAVE]",
                step_path
            )

            save_policy(
                policy,
                tokenizer,
                step_path
            )

            current_model_path = (
                str(step_path)
            )

        else:

            print(
                "[SFT] insufficient "
                "verified samples; "
                "policy unchanged."
            )

        # ----------------------------------------------------
        # G. Summary
        # ----------------------------------------------------

        elapsed = (
            time.time()
            - step_start
        )

        summary = {

            "step": step,

            "seed_batch_size":
                len(seeds),

            "generation":
                gen_summary,

            "solver":
                solver_stats,

            "agreement_rate":
                agreement_rate,

            "audit_clean":
                len(clean_verified),

            "audit_rejected":
                len(audit_rejected),

            "sft":
                sft_stats,

            "replay_buffer_size":
                len(replay_buffer),

            "model_path_after_step":
                current_model_path,

            "step_seconds":
                elapsed,
        }

        all_step_summaries.append(
            summary
        )

        print(
            "\n[STEP SUMMARY]"
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2
            )
        )

        write_jsonl(
            LOG_ROOT
            / "self_play_v8_step_summaries.jsonl",
            all_step_summaries
        )

        gc.collect()

        try:

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        except Exception:
            pass

    # ========================================================
    # FINAL SAVE
    # ========================================================

    final_path = (
        OUTPUT_ROOT
        / "final"
    )

    print("\n")
    print("=" * 80)
    print("FINAL SAVE")
    print("=" * 80)

    move_policy_to_cpu(
        policy
    )

    save_policy(
        policy,
        tokenizer,
        final_path
    )

    write_jsonl(
        OUTPUT_ROOT
        / "step_summaries.jsonl",
        all_step_summaries
    )

    print(
        "Final model:",
        final_path
    )

    print(
        "SELF-PLAY V8 COMPLETE"
    )


if __name__ == "__main__":

    mp.set_start_method(
        "spawn",
        force=True
    )

    main()