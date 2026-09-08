#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Self-Play V7.4 Diagnostic
=======================

目标：
1. 仍然严格保持 Self-Play 主线：
      GRPO-200 自己出题
          ↓
      GRPO-200 自己解题
          ↓
      两次独立 rollout 答案一致
          ↓
      verified sample
          ↓
      SFT

2. 本版本主要用于诊断“模型到底会不会出题”，因此：
   - 每个 seed 生成 4 个 candidate；
   - generator 不要求输出 Answer；
   - 大幅放宽问题解析/过滤；
   - 保存 EVERY candidate 的原始输出和分类原因；
   - 不因为格式问题轻易丢弃 candidate；
   - 只有明显的 seed 原题复制、明显 solution contamination、代码/空输出等才过滤；
   - solver 仍然使用严格的 answer extraction；
   - 最终只对通过基础问题检查的 candidate 做双 rollout 验证。

注意：
这是诊断版，不要直接把输出数据当成最终大规模 Self-Play 数据。
第一轮只跑 100 seeds。
"""

import os
import sys
import json
import math
import random
import re
import multiprocessing as mp
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------
# CUDA / vLLM multiprocessing
# ---------------------------------------------------------------------

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-Math-1.5B-GRPO"
DATASET_ROOT = Path("/root/autodl-tmp/datasets/MATH")

OUTPUT_ROOT = Path("/root/autodl-tmp/self_play/data")

ALL_CANDIDATES_PATH = OUTPUT_ROOT / "self_play_candidates_v73.jsonl"
ACCEPTED_PROBLEMS_PATH = OUTPUT_ROOT / "self_play_problems_v73.jsonl"
SOLUTIONS_PATH = OUTPUT_ROOT / "self_play_solutions_v73.jsonl"
VERIFIED_PATH = OUTPUT_ROOT / "self_play_verified_v73.jsonl"
REJECTED_PATH = OUTPUT_ROOT / "self_play_rejected_v73.jsonl"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

NUM_SEEDS = 100
GEN_ROLLOUTS_PER_SEED = 4

# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------

GEN_TEMPERATURE = 0.9
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 320
GENERATION_MAX_MODEL_LEN = 1536

# Only obvious failures are filtered.
MIN_PROBLEM_CHARS = 20
MAX_PROBLEM_CHARS = 4000

# ---------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------

SOLVE_TEMPERATURE = 0.7
SOLVE_TOP_P = 0.9
SOLVE_MAX_NEW_TOKENS = 1024
SOLVER_MAX_MODEL_LEN = 2560
NUM_SOLVER_ROLLOUTS = 2

VLLM_GPU_MEMORY_UTILIZATION = 0.82

# ---------------------------------------------------------------------
# Diagnostic mode
# ---------------------------------------------------------------------

# V7.3 deliberately stops after generation + double-solver verification.
# No HF SFT model is loaded, so vLLM cannot conflict with HF CUDA memory.

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

GENERATOR_PROMPT = r"""You are a mathematical problem generator.

Create ONE NEW math problem based on the seed problem.

Requirements:
- Test the same main mathematical skill as the seed.
- Change the numbers, variables, or context.
- The new problem must be self-contained.
- The new problem must have one definite answer.
- The new problem must be independently solvable.

Important:
- Output ONLY the new problem statement.
- Do NOT solve it.
- Do NOT give its answer.
- Do NOT explain your reasoning.
- Do NOT discuss the seed problem.
- Do NOT copy the seed problem verbatim.

Seed problem:
{seed_problem}

New problem:
"""

# Reuse the project's actual solve prompt format.
SOLVE_USER_PROMPT = r"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>"""

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(text):
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    return text


def normalize_ws(text):
    return re.sub(r"\s+", " ", clean_text(text)).strip().lower()


def similarity_ratio(a, b):
    try:
        from rapidfuzz.fuzz import ratio
        return ratio(normalize_ws(a), normalize_ws(b)) / 100.0
    except Exception:
        import difflib
        return difflib.SequenceMatcher(
            None,
            normalize_ws(a),
            normalize_ws(b),
        ).ratio()


def extract_last_boxed(text):
    """
    Robust enough for diagnostics.
    Returns the content of the last \\boxed{...}.
    """
    if not text:
        return None

    positions = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not positions:
        return None

    start = positions[-1]
    brace_start = text.find("{", start)

    depth = 0
    for i in range(brace_start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i].strip()

    return None


def normalize_answer_text(answer):
    if answer is None:
        return None

    answer = clean_text(answer)
    answer = answer.strip()

    if not answer:
        return None

    # Remove accidental trailing punctuation.
    answer = answer.strip(" \t\r\n.,;")

    if not answer:
        return None

    return answer


# ---------------------------------------------------------------------
# Generator output parsing
# ---------------------------------------------------------------------

GENERATION_BOUNDARIES = [
    r"\bto solve the problem\b",
    r"\bto determine the answer\b",
    r"\bto determine the value\b",
    r"\bto find the answer\b",
    r"\bto find the value\b",
    r"\bwe need to solve\b",
    r"\bwe need to calculate\b",
    r"\bwe can solve\b",
    r"\blet'?s solve\b",
    r"\bthe solution is\b",
    r"\bthe answer is\b",
    r"\bfinal answer\s*:",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bsubstituting\b",
    r"\bsimplifying\b",
    r"\bnow we calculate\b",
    r"\bplugging (?:this|these|the)\b",
    r"\bwe get\b",
    r"\bhence\b",
    r"\bso the answer is\b",
]

META_GENERATION_PATTERNS = [
    r"\bthe new problem is\b",
    r"\bone new problem\b",
    r"\ba new problem\b",
    r"\bthis new problem is\b",
    r"\bthe original problem\b",
    r"\bthe seed problem\b",
    r"\bsame mathematical skill\b",
    r"\bsame skill\b",
    r"\bchanged numbers\b",
    r"\bchanged values\b",
    r"\bchanged context\b",
    r"\bderived from the seed\b",
    r"\bbased on the seed\b",
    r"\bvariant of the seed\b",
    r"\bvariant of the original\b",
]

ANSWER_LEAK_PATTERNS = [
    r"\banswer\s*:",
    r"\bfinal answer\s*:",
    r"\bthe answer is\b",
    r"\bthe final answer is\b",
    r"\\boxed\s*\{",
]

CODE_PATTERNS = [
    r"```",
    r"\bimport\s+\w+",
    r"\bdef\s+\w+\s*\(",
    r"\bprint\s*\(",
    r"\bpython\b",
]

PLACEHOLDER_PATTERNS = [
    r"\[generate_problem\]",
    r"\[generate_answer\]",
    r"\[new problem\]",
    r"\[same as seed problem\]",
    r"\bplaceholder\b",
]


def _first_match(text, patterns):
    hits=[]
    for pat in patterns:
        m=re.search(pat, text, flags=re.IGNORECASE|re.DOTALL)
        if m: hits.append(m)
    return min(hits, key=lambda m:m.start()) if hits else None


def strip_generator_prefix(text):
    text=clean_text(text)
    m=re.match(r"^(?:new\s+)?problem(?:\s+statement)?\s*:\s*", text, flags=re.I)
    if m: text=text[m.end():].strip()
    return text


def _strip_meta_preamble(text):
    patterns=[
        r"(?is)\bthe new problem is\s*:?\s*",
        r"(?is)\bthis new problem is\s*:?\s*",
        r"(?is)\bone new problem is\s*:?\s*",
        r"(?is)\ba new problem is\s*:?\s*",
    ]
    m=_first_match(text,patterns)
    if not m: return text.strip(),False
    return text[m.end():].strip(),True


def _extract_heading_payload(raw):
    patterns=[
        r"(?is)\bproblem\s+statement\s*:\s*",
        r"(?is)\bnew\s+problem\s*:\s*",
        r"(?is)\bproblem\s*:\s*",
    ]
    m=_first_match(raw,patterns)
    if not m: return raw.strip(),False
    return raw[m.end():].strip(),True


def _cut_solution_contamination(text):
    m=_first_match(text,GENERATION_BOUNDARIES)
    if not m: return text.strip(),"none",None
    return text[:m.start()].strip(" \n\r\t:;-"),"truncate_ordinary_solution",m.start()


def _looks_like_problem(text):
    n=normalize_ws(text)
    if len(n)<MIN_PROBLEM_CHARS: return False
    low=n.lower()
    signals=[r"\bfind\b",r"\bcalculate\b",r"\bdetermine\b",r"\bsolve\b",r"\bwhat\s+is\b",r"\bwhat\s+are\b",r"\bhow\s+many\b",r"\bcompute\b",r"\bevaluate\b",r"\bprobability\b",r"\bvalue\s+of\b",r"\barea\b",r"\bperimeter\b",r"\bangle\b",r"\bslope\b",r"\bequation\b"]
    return "?" in n or any(re.search(x,low) for x in signals)


def generated_problem_copies_seed(seed, generated):
    seed_n=normalize_ws(seed).lower(); gen_n=normalize_ws(generated).lower()
    if not seed_n or not gen_n: return False
    full=similarity_ratio(seed_n,gen_n)
    prefix=similarity_ratio(seed_n[:240],gen_n[:240])
    exact=len(gen_n)>=120 and gen_n.startswith(seed_n[:min(160,len(seed_n))])
    return exact or full>=0.97 or prefix>=0.985


def generated_problem_is_obviously_bad(problem):
    p=normalize_ws(problem); low=p.lower()
    if len(p)<MIN_PROBLEM_CHARS: return True,"too_short"
    if len(p)>MAX_PROBLEM_CHARS: return True,"too_long"
    if any(re.search(x,low) for x in CODE_PATTERNS): return True,"code_like"
    if any(re.search(x,low) for x in PLACEHOLDER_PATTERNS): return True,"meta_or_placeholder"
    return False,None


def _suffix_is_ordinary_solution(suffix):
    if not suffix: return True,"no_suffix"
    low=suffix.lower()
    if any(re.search(x,low) for x in META_GENERATION_PATTERNS): return False,"meta_generation_in_suffix"
    if any(re.search(x,low) for x in ANSWER_LEAK_PATTERNS): return False,"answer_leak_in_suffix"
    return True,"ordinary_solution_suffix"


def problem_contains_obvious_solution(text):
    low=normalize_ws(text).lower()
    return any(re.search(x,low) for x in ANSWER_LEAK_PATTERNS)


def parse_generator_output(seed_problem, raw_output):
    raw=clean_text(raw_output)
    if not raw: return None,"rejected","empty_output",0.0,"none"
    raw=re.sub(r"^\s*(?:assistant|output)\s*:\s*","",raw,flags=re.I)

    if any(re.search(x,raw,flags=re.I) for x in CODE_PATTERNS+PLACEHOLDER_PATTERNS):
        return None,"rejected","code_or_placeholder",similarity_ratio(seed_problem,raw),"none"

    payload,had_heading=_extract_heading_payload(raw)
    action="heading" if had_heading else "none"

    # If the model talked about the generation task before stating the problem,
    # remove that preamble only when a known "new problem is" transition exists.
    if not had_heading:
        payload,did=_strip_meta_preamble(payload)
        if did: action="strip_meta_preamble"

    # If meta text still occurs before the first plausible problem signal,
    # reject rather than accepting generation commentary as a problem.
    meta=_first_match(payload,META_GENERATION_PATTERNS)
    problem_markers=[]
    for pat in [r"\bwhat\s+is\b",r"\bcalculate\b",r"\bfind\b",r"\bdetermine\b",r"\bcompute\b",r"\bevaluate\b",r"\bhow\s+many\b",r"\bprobability\b"]:
        m=re.search(pat,payload,flags=re.I)
        if m: problem_markers.append(m.start())
    if meta and problem_markers and meta.start()<min(problem_markers):
        return None,"rejected","meta_generation_prefix",similarity_ratio(seed_problem,payload),action

    problem,suffix,pos=_cut_solution_contamination(payload)
    if pos is not None:
        action="truncate_ordinary_solution"
        ok,sreason=_suffix_is_ordinary_solution(suffix)
        if not ok:
            return None,"rejected",sreason,similarity_ratio(seed_problem,problem),action

    problem=normalize_ws(problem).strip("`").strip()
    if not problem: return None,"rejected","empty_after_cleaning",0.0,action
    if re.search(r"\banswer\s*:",problem,flags=re.I) or re.search(r"\\boxed\s*\{",problem):
        return None,"rejected","answer_leak_in_problem",similarity_ratio(seed_problem,problem),action
    bad,reason=generated_problem_is_obviously_bad(problem)
    if bad: return None,"rejected",reason,similarity_ratio(seed_problem,problem),action
    if not _looks_like_problem(problem): return None,"rejected","not_problem_like",similarity_ratio(seed_problem,problem),action

    sim=similarity_ratio(seed_problem,problem)
    if generated_problem_copies_seed(seed_problem,problem):
        return None,"rejected","seed_copy",sim,action
    return problem,"accepted","ok",sim,action

# ---------------------------------------------------------------------
# Solver answer extraction
# ---------------------------------------------------------------------

def extract_solver_answer(response):
    """
    Priority:
      1. complete <answer>...</answer>
      2. truncated <answer>...
      3. last \\boxed{...}
      4. Answer:/Final Answer:
    """
    if not response:
        return None

    response = clean_text(response)

    candidates = []

    # Complete answer tag.
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for x in matches:
        candidates.append(x)

    # Truncated answer tag.
    matches = re.findall(
        r"<answer>\s*(.*)$",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for x in matches:
        candidates.append(x)

    # Last boxed answer.
    boxed = extract_last_boxed(response)
    if boxed:
        candidates.append(boxed)

    # Explicit answer line.
    matches = re.findall(
        r"(?:final\s+answer|answer)\s*:\s*(.+)",
        response,
        flags=re.IGNORECASE,
    )
    for x in matches:
        candidates.append(x)

    for candidate in candidates:
        candidate = normalize_answer_text(candidate)
        if candidate:
            # Strip accidental closing tags.
            candidate = re.sub(
                r"</answer>.*$",
                "",
                candidate,
                flags=re.IGNORECASE,
            ).strip()

            if candidate:
                return candidate

    return None


# ---------------------------------------------------------------------
# MATH data
# ---------------------------------------------------------------------

def load_all_train_data():
    samples = []

    for config in CONFIGS:
        dataset_path = DATASET_ROOT / config / "train"

        if not dataset_path.exists():
            print(f"[WARN] missing dataset: {dataset_path}")
            continue

        from datasets import load_from_disk

        dataset = load_from_disk(str(dataset_path))
        print(f"{config:30s}: {len(dataset)} train examples")

        for idx, item in enumerate(dataset):
            problem = item["problem"]
            solution = item["solution"]

            samples.append({
                "id": f"{config}_{idx}",
                "config": config,
                "index": idx,
                "question": problem,
                "ground_truth": solution,
            })

    return samples


def select_seeds(samples):
    if len(samples) < NUM_SEEDS:
        raise RuntimeError(
            f"Only {len(samples)} seeds available, "
            f"but NUM_SEEDS={NUM_SEEDS}"
        )

    return random.sample(samples, NUM_SEEDS)


# ---------------------------------------------------------------------
# vLLM
# ---------------------------------------------------------------------

def build_llm(max_model_len):
    from vllm import LLM

    print("=" * 70)
    print("Loading vLLM")
    print("=" * 70)

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=max_model_len,
    )

    return llm


def generate_candidates(llm, seeds, tokenizer):
    from vllm import SamplingParams

    prompts = []
    metadata = []

    for seed_idx, seed in enumerate(seeds):
        prompt = GENERATOR_PROMPT.format(
            seed_problem=seed["question"]
        )

        token_ids = tokenizer.encode(prompt)

        if len(token_ids) + GEN_MAX_NEW_TOKENS >= GENERATION_MAX_MODEL_LEN:
            print(
                f"[FILTER PROMPT] seed={seed['id']} "
                f"prompt_tokens={len(token_ids)}"
            )
            continue

        prompts.append(prompt)
        metadata.append({
            "seed_idx": seed_idx,
            "seed": seed,
        })

    sampling_params = SamplingParams(
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        n=GEN_ROLLOUTS_PER_SEED,
        max_tokens=GEN_MAX_NEW_TOKENS,
    )

    print()
    print("=" * 70)
    print("SELF-PLAY V7.3 DIAGNOSTIC: PROBLEM GENERATION")
    print("=" * 70)
    print(f"Seeds selected        : {len(seeds)}")
    print(f"Generation rollouts   : {GEN_ROLLOUTS_PER_SEED}")
    print(f"Candidates expected    : {len(prompts) * GEN_ROLLOUTS_PER_SEED}")

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    records = []

    for meta, output in zip(metadata, outputs):
        seed = meta["seed"]

        for rollout_idx, completion in enumerate(output.outputs):
            raw = completion.text

            problem, status, reason, sim, cleaning_action = parse_generator_output(
                seed["question"],
                raw,
            )

            record = {
                "seed_id": seed["id"],
                "config": seed["config"],
                "seed_index": seed["index"],
                "generation_rollout": rollout_idx,
                "seed_problem": seed["question"],
                "raw_output": raw,
                "parsed_problem": problem,
                "status": status,
                "reason": reason,
                "similarity": sim,
                "cleaning_action": cleaning_action,
            }

            records.append(record)

    return records


# ---------------------------------------------------------------------
# Solver verification
# ---------------------------------------------------------------------

def solve_verified_problems(llm, accepted_records):
    from vllm import SamplingParams

    if not accepted_records:
        return [], []

    solve_prompts = [
        SOLVE_USER_PROMPT.format(
            question=r["parsed_problem"]
        )
        for r in accepted_records
    ]

    sampling_params = SamplingParams(
        temperature=SOLVE_TEMPERATURE,
        top_p=SOLVE_TOP_P,
        n=NUM_SOLVER_ROLLOUTS,
        max_tokens=SOLVE_MAX_NEW_TOKENS,
        stop=None,
    )

    print()
    print("=" * 70)
    print("SELF-PLAY V7.3 DIAGNOSTIC: SOLVER VERIFICATION")
    print("=" * 70)
    print(f"Accepted generated problems : {len(solve_prompts)}")
    print(f"Solver rollouts/problem     : {NUM_SOLVER_ROLLOUTS}")

    outputs = llm.generate(
        solve_prompts,
        sampling_params,
        use_tqdm=True,
    )

    solution_records = []
    verified_records = []

    for problem_record, output in zip(accepted_records, outputs):
        answers = []
        raw_responses = []

        for completion in output.outputs:
            response = completion.text
            answer = extract_solver_answer(response)

            raw_responses.append(response)
            answers.append(answer)

        base = {
            "seed_id": problem_record["seed_id"],
            "config": problem_record["config"],
            "seed_index": problem_record["seed_index"],
            "generation_rollout": problem_record["generation_rollout"],
            "problem": problem_record["parsed_problem"],
            "generation_raw_output": problem_record["raw_output"],
            "solver_responses": raw_responses,
            "solver_answers": answers,
        }

        solution_records.append(base)

        if any(a is None for a in answers):
            base["verification"] = "missing_answer"
            base["verified"] = False
            continue

        normalized = [
            normalize_ws(a)
            for a in answers
        ]

        if len(set(normalized)) == 1:
            base["verification"] = "agreement"
            base["verified"] = True

            verified_records.append({
                "seed_id": problem_record["seed_id"],
                "config": problem_record["config"],
                "seed_index": problem_record["seed_index"],
                "problem": problem_record["parsed_problem"],
                "solution": raw_responses[0],
                "answer": answers[0],
                "solver_answers": answers,
                "generation_raw_output": problem_record["raw_output"],
            })
        else:
            base["verification"] = "disagreement"
            base["verified"] = False

    return solution_records, verified_records


# ---------------------------------------------------------------------
# Diagnostics

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def print_generation_diagnostics(all_records):
    total = len(all_records)

    status_counter = Counter()
    reason_counter = Counter()
    action_counter = Counter()

    accepted = []

    for r in all_records:
        status_counter[r["status"]] += 1
        reason_counter[r["reason"]] += 1
        action_counter[r.get("cleaning_action", "none")] += 1

        if r["status"] == "accepted":
            accepted.append(r)

    print()
    print("=" * 70)
    print("GENERATOR DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(f"Total candidates       : {total}")
    print(f"Accepted candidates    : {len(accepted)}")
    print(
        f"Acceptance rate        : "
        f"{len(accepted) / max(1, total):.4f}"
    )

    print()
    print("Status:")
    for k, v in status_counter.most_common():
        print(f"  {k:35s} {v}")

    print()
    print("Reasons:")
    for k, v in reason_counter.most_common():
        print(f"  {k:35s} {v}")

    print()
    print("Cleaning actions:")
    for k, v in action_counter.most_common():
        print(f"  {k:35s} {v}")

    if accepted:
        sims = [
            r["similarity"]
            for r in accepted
            if r["similarity"] is not None
        ]

        if sims:
            print()
            print(
                f"Accepted similarity mean/min/max: "
                f"{sum(sims)/len(sims):.4f} / "
                f"{min(sims):.4f} / "
                f"{max(sims):.4f}"
            )

        print()
        print("-" * 70)
        print("FIRST 10 ACCEPTED CANDIDATES")
        print("-" * 70)

        for i, r in enumerate(accepted[:10]):
            print()
            print(f"[{i + 1}] seed={r['seed_id']} rollout={r['generation_rollout']}")
            print(f"similarity={r['similarity']:.4f}")
            print("PROBLEM:")
            print(r["parsed_problem"][:1500])
            print("-" * 70)


def print_solver_diagnostics(solution_records, verified_records):
    missing = 0
    disagreement = 0
    agreement = 0

    for r in solution_records:
        if r["verification"] == "missing_answer":
            missing += 1
        elif r["verification"] == "disagreement":
            disagreement += 1
        elif r["verification"] == "agreement":
            agreement += 1

    print()
    print("=" * 70)
    print("SELF-PLAY VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"Generated candidates   : {len(solution_records)}")
    print(f"Verified problems      : {len(verified_records)}")
    print(f"Solver agreement       : {agreement}")
    print(f"Missing answer         : {missing}")
    print(f"Solver disagreement    : {disagreement}")
    print(f"Verified output        : {VERIFIED_PATH}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    random.seed(42)

    print("=" * 70)
    print("SELF-PLAY V7.3 DIAGNOSTIC")
    print("=" * 70)
    print(f"Model      : {MODEL_PATH}")
    print(f"Seeds      : {NUM_SEEDS}")
    print(f"Rollouts   : {GEN_ROLLOUTS_PER_SEED}")
    print(f"Total cand.: {NUM_SEEDS * GEN_ROLLOUTS_PER_SEED}")

    samples = load_all_train_data()
    seeds = select_seeds(samples)

    # vLLM must have enough context for generation.
    llm = build_llm(GENERATION_MAX_MODEL_LEN)
    tokenizer = llm.get_tokenizer()

    # ---------------------------------------------------------------
    # 1. Generate 4 candidate problems per seed.
    # ---------------------------------------------------------------

    all_records = generate_candidates(
        llm,
        seeds,
        tokenizer,
    )

    write_jsonl(
        ALL_CANDIDATES_PATH,
        all_records,
    )

    print_generation_diagnostics(all_records)

    # Accepted generated problems only.
    accepted_records = [
        r for r in all_records
        if r["status"] == "accepted"
    ]

    write_jsonl(
        ACCEPTED_PROBLEMS_PATH,
        accepted_records,
    )

    rejected_records = [
        r for r in all_records
        if r["status"] != "accepted"
    ]

    write_jsonl(
        REJECTED_PATH,
        rejected_records,
    )

    print()
    print(f"All candidates saved : {ALL_CANDIDATES_PATH}")
    print(f"Accepted problems    : {ACCEPTED_PROBLEMS_PATH}")
    print(f"Rejected candidates  : {REJECTED_PATH}")

    # ---------------------------------------------------------------
    # 2. Self-play solver verification.
    # ---------------------------------------------------------------

    solution_records, verified_records = solve_verified_problems(
        llm,
        accepted_records,
    )

    write_jsonl(
        SOLUTIONS_PATH,
        solution_records,
    )

    write_jsonl(
        VERIFIED_PATH,
        verified_records,
    )

    print_solver_diagnostics(
        solution_records,
        verified_records,
    )

    print(f"Solutions output      : {SOLUTIONS_PATH}")

    # ---------------------------------------------------------------
    # 3. Diagnostic only: no SFT in V7.3.
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("SELF-PLAY V7.3 DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print(f"Generation candidates : {len(all_records)}")
    print(f"Accepted problems     : {len(accepted_records)}")
    print(f"Verified samples      : {len(verified_records)}")
    print("SFT                   : SKIPPED (diagnostic-only)")

    print()
    print("IMPORTANT:")
    print("This run is diagnostic. Inspect self_play_candidates_v73.jsonl")
    print("before deciding whether to scale to 500/1000 seeds.")



# ============================================================
# V7.4 VERIFIED DATASET QUALITY AUDIT
# ============================================================

AUDIT_INPUT = OUTPUT_ROOT / "self_play_verified_v73.jsonl"
AUDIT_OUTPUT = OUTPUT_ROOT / "self_play_audit_v74.jsonl"
AUDIT_CLEAN_OUTPUT = OUTPUT_ROOT / "self_play_verified_clean_v74.jsonl"
AUDIT_REJECTED_OUTPUT = OUTPUT_ROOT / "self_play_verified_rejected_v74.jsonl"

AUDIT_MIN_PROBLEM_WORDS = 8
AUDIT_HIGH_SIMILARITY = 0.93

AUDIT_META_PATTERNS = [
    r"\bthe original problem\b",
    r"\bthe seed problem\b",
    r"\bbased on the seed\b",
    r"\bderived from the seed\b",
    r"\bsame mathematical skill\b",
    r"\bsame skill\b",
    r"\bchanged numbers\b",
    r"\bchanged values\b",
    r"\bchanged context\b",
    r"\bthe new problem is\b",
    r"\bnew problem statement\b",
    r"\bto generate a new problem\b",
    r"\bwe need to generate\b",
]

AUDIT_ANSWER_PATTERNS = [
    r"\banswer\s*:",
    r"\bfinal answer\s*:",
    r"\bthe answer is\b",
    r"\bthe final answer is\b",
    r"\\boxed\s*\{",
]

AUDIT_SOLUTION_PATTERNS = [
    r"\bto solve the problem\b",
    r"\bto determine the answer\b",
    r"\bto find the answer\b",
    r"\bwe need to solve\b",
    r"\bwe need to calculate\b",
    r"\bwe can solve\b",
    r"\blet'?s solve\b",
    r"\btherefore\b",
    r"\bthus\b",
    r"\bsubstituting\b",
    r"\bsimplifying\b",
    r"\bnow we calculate\b",
    r"\bplugging (?:this|these|the)\b",
    r"\bhence\b",
    r"\bso the answer is\b",
]

AUDIT_CODE_PATTERNS = [
    r"```",
    r"\bimport\s+\w+",
    r"\bdef\s+\w+\s*\(",
    r"\bprint\s*\(",
    r"\breturn\s+",
]

def _audit_norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _audit_matches(text, patterns):
    low = _audit_norm(text).lower()
    hits = []
    for pat in patterns:
        m = re.search(pat, low, flags=re.IGNORECASE | re.DOTALL)
        if m:
            hits.append((m.start(), m.group(0), pat))
    return sorted(hits, key=lambda x: x[0])

def _audit_extract_problem(record):
    # Prefer the exact field that V7.3's solver should have used.
    for key in ("problem", "parsed_problem", "audit_problem", "question"):
        value = record.get(key)
        if value:
            return _audit_norm(value)
    return ""

def _audit_extract_seed(record):
    for key in ("seed_problem", "source_problem", "seed"):
        value = record.get(key)
        if value:
            return _audit_norm(value)
    return ""

def _audit_solver_answers(record):
    """
    V7.3 schemas can differ slightly. We inspect common answer fields,
    but absence of answer fields is NOT by itself a rejection because
    V7.3's verification stage already established agreement.
    """
    possible_pairs = [
        ("answer_1", "answer_2"),
        ("answer1", "answer2"),
        ("solver_answer_1", "solver_answer_2"),
        ("solver_answer1", "solver_answer2"),
        ("solution_1", "solution_2"),
    ]
    for a, b in possible_pairs:
        if record.get(a) is not None or record.get(b) is not None:
            return _audit_norm(record.get(a)), _audit_norm(record.get(b))
    return "", ""

def audit_record(record):
    result = dict(record)

    problem = _audit_extract_problem(record)
    seed = _audit_extract_seed(record)

    result["audit_problem"] = problem
    result["audit_status"] = "REJECTED"
    result["audit_reason"] = "unknown"
    result["audit_match"] = ""
    result["audit_similarity"] = 0.0

    if not problem:
        result["audit_reason"] = "empty_problem"
        return result

    if len(problem.split()) < AUDIT_MIN_PROBLEM_WORDS:
        result["audit_reason"] = "too_short"
        return result

    # Explicit answer leakage is always disqualifying.
    hits = _audit_matches(problem, AUDIT_ANSWER_PATTERNS)
    if hits:
        result["audit_reason"] = "answer_leak"
        result["audit_match"] = hits[0][1]
        return result

    # Generation/seed discussion is not a mathematical problem statement.
    hits = _audit_matches(problem, AUDIT_META_PATTERNS)
    if hits:
        result["audit_reason"] = "meta_generation"
        result["audit_match"] = hits[0][1]
        return result

    # If the verified artifact still contains reasoning, reject it instead
    # of silently changing a verified sample. V7.4 is an audit, not a parser.
    hits = _audit_matches(problem, AUDIT_SOLUTION_PATTERNS)
    if hits:
        result["audit_reason"] = "solution_leak"
        result["audit_match"] = hits[0][1]
        return result

    # Do NOT reject [asy]: it is legitimate MATH problem content.
    code_hits = _audit_matches(problem, AUDIT_CODE_PATTERNS)
    if code_hits:
        result["audit_reason"] = "code_artifact"
        result["audit_match"] = code_hits[0][1]
        return result

    # Seed-copy check on the actual problem.
    if seed:
        sim = similarity_ratio(seed, problem)
        result["audit_similarity"] = sim
        if sim >= AUDIT_HIGH_SIMILARITY:
            result["audit_reason"] = "seed_copy"
            return result

    # Optional sanity check for the verification record.
    a1, a2 = _audit_solver_answers(record)
    if (a1 and not a2) or (a2 and not a1):
        result["audit_reason"] = "malformed_verification"
        return result

    result["audit_status"] = "HIGH_QUALITY"
    result["audit_reason"] = "ok"
    return result

def run_v74_audit(input_path=AUDIT_INPUT):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing V7.3 verified dataset: {input_path}\n"
            "Run V7.3 first."
        )

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    audited = [audit_record(r) for r in records]
    clean = [r for r in audited if r["audit_status"] == "HIGH_QUALITY"]
    rejected = [r for r in audited if r["audit_status"] != "HIGH_QUALITY"]

    write_jsonl(AUDIT_OUTPUT, audited)
    write_jsonl(AUDIT_CLEAN_OUTPUT, clean)
    write_jsonl(AUDIT_REJECTED_OUTPUT, rejected)

    from collections import Counter
    reasons = Counter(r["audit_reason"] for r in audited)
    configs = Counter(r.get("config", "unknown") for r in clean)

    print("\n" + "=" * 70)
    print("SELF-PLAY V7.4 VERIFIED DATASET AUDIT")
    print("=" * 70)
    print(f"Input verified samples : {len(records)}")
    print(f"HIGH_QUALITY           : {len(clean)}")
    print(f"REJECTED               : {len(rejected)}")
    print(
        f"Clean rate             : "
        f"{len(clean) / len(records):.4f}"
        if records else
        "Clean rate             : 0.0000"
    )

    print("\nAudit reasons:")
    for k, v in reasons.most_common():
        print(f"  {k:<32} {v}")

    print("\nHIGH_QUALITY by config:")
    for k, v in configs.most_common():
        print(f"  {k:<32} {v}")

    print("\nOutputs:")
    print(f"  Full audit            : {AUDIT_OUTPUT}")
    print(f"  Clean verified        : {AUDIT_CLEAN_OUTPUT}")
    print(f"  Rejected              : {AUDIT_REJECTED_OUTPUT}")
    print("=" * 70)

    return audited, clean, rejected



if __name__ == "__main__":
    run_v74_audit()
