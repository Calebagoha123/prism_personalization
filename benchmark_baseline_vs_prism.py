import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from benchmark_prism_gsm8k import (
    DEFAULT_MODEL,
    ConversationSampler,
    IntersectionalSampler,
    build_prompt,
    canonicalize_demo_value,
    ensure_directory,
    ensure_hf_hub_cache,
    extract_gold_answer,
    extract_number_fraction,
    generate_solution,
    get_demographic_value,
    load_gsm8k,
    load_prism_data,
    parse_intersectional_buckets,
    safe_str,
    select_examples,
    split_think_and_final_output,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline GSM8K and PRISM-context GSM8K, then report subgroup/intersection accuracies."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-hub-cache", default="/data/resource/huggingface/hub")
    parser.add_argument("--create-hf-hub-cache", action="store_true")
    parser.add_argument("--prism-local-dir", default="")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--allow-network-download", action="store_true")

    parser.add_argument("--gsm-config", default="main")
    parser.add_argument("--gsm-split", default="test")
    parser.add_argument("--gsm-sampling", choices=["random", "first_n"], default="random")
    parser.add_argument("--num-questions", type=int, default=20)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-history-chars", type=int, default=800)
    parser.add_argument("--sampling-strategy", choices=["without_replacement", "with_replacement"], default="without_replacement")
    parser.add_argument("--balanced-intersectional-sampling", action="store_true", default=True)
    parser.add_argument("--intersectional-fields", nargs="+", default=["race", "gender"])
    parser.add_argument(
        "--intersectional-buckets",
        default="white|male,white|female,black|male,black|female",
    )

    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--min-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true", default=True)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--retry-unparsed", type=int, default=1)
    parser.add_argument("--retry-max-new-tokens", type=int, default=512)

    parser.add_argument("--output-dir", default="/data/kell8360")
    parser.add_argument("--create-output-dir", action="store_true")
    parser.add_argument("--output-csv", default="results_baseline_vs_prism.csv")
    parser.add_argument("--summary-json", default="summary_baseline_vs_prism.json")
    return parser.parse_args()


def baseline_prompt(question: str) -> str:
    return (
        "You are solving a GSM8K math word problem.\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "On the last line, write exactly: Final answer: \\boxed{<number>}.\n\n"
        f"Problem:\n{question}\n"
    )


def compute_simple_accuracy(rows: List[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(int(r["correct"]) for r in rows) / len(rows)


def grouped_accuracy(rows: List[Dict[str, Any]], field: str, allowed: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, int] = defaultdict(int)
    correct: Dict[str, int] = defaultdict(int)
    for row in rows:
        value = safe_str(row.get(field)).lower()
        if not value:
            value = "unknown"
        if allowed is not None and value not in allowed:
            continue
        counts[value] += 1
        correct[value] += int(row["correct"])
    out: Dict[str, Dict[str, float]] = {}
    for k in sorted(counts.keys()):
        n = counts[k]
        c = correct[k]
        out[k] = {"n": n, "correct": c, "accuracy": (c / n) if n else 0.0}
    return out


def run_generation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    args: argparse.Namespace,
) -> Tuple[Optional[str], str, str]:
    generation = generate_solution(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        enable_thinking=args.enable_thinking,
    )
    pred = extract_number_fraction(generation)
    if pred is None and args.retry_unparsed > 0:
        for _ in range(args.retry_unparsed):
            retry_prompt = (
                "Solve this GSM8K math problem.\n"
                "Output exactly one final line in this format: Final answer: \\boxed{<number>}.\n\n"
                f"{prompt}\n"
            )
            retry_output = generate_solution(
                model=model,
                tokenizer=tokenizer,
                prompt=retry_prompt,
                max_new_tokens=args.retry_max_new_tokens,
                min_new_tokens=min(args.min_new_tokens, args.retry_max_new_tokens),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                enable_thinking=args.enable_thinking,
            )
            generation = f"{generation}\n\n[retry_output]\n{retry_output}"
            pred = extract_number_fraction(retry_output)
            if pred is not None:
                break
    think_trace, final_output = split_think_and_final_output(generation)
    return (str(pred) if pred is not None else None), think_trace, final_output


def main() -> None:
    args = parse_args()
    if args.disable_thinking:
        args.enable_thinking = False
    if args.allow_network_download:
        args.local_files_only = False

    try:
        from dotenv import load_dotenv
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing required dependencies. Run `uv sync`.") from exc
    if torch is None:
        raise ImportError("Missing torch. Run `uv sync`.")

    load_dotenv()
    token = os.getenv(args.hf_token_env)
    if not token and not args.local_files_only:
        raise ValueError(f"Missing token env var {args.hf_token_env} for network downloads.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    hf_hub_cache = ensure_hf_hub_cache(args.hf_hub_cache, args.create_hf_hub_cache)
    output_dir = ensure_directory(args.output_dir, args.create_output_dir, "--output-dir", "--create-output-dir")
    output_csv_path = os.path.join(output_dir, args.output_csv)
    summary_json_path = os.path.join(output_dir, args.summary_json)

    prism_items = load_prism_data(
        token=token,
        max_history_chars=args.max_history_chars,
        cache_dir=hf_hub_cache,
        local_files_only=args.local_files_only,
        prism_local_dir=args.prism_local_dir,
    )
    gsm_all = load_gsm8k(
        config=args.gsm_config,
        split=args.gsm_split,
        token=token,
        cache_dir=hf_hub_cache,
        local_files_only=args.local_files_only,
    )
    gsm_examples = select_examples(gsm_all, args.num_questions, args.gsm_sampling, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=token,
        cache_dir=hf_hub_cache,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        cache_dir=hf_hub_cache,
        local_files_only=args.local_files_only,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    target_buckets = parse_intersectional_buckets(args.intersectional_buckets, len(args.intersectional_fields))

    all_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    contextual_rows: List[Dict[str, Any]] = []

    for run_idx in range(args.num_runs):
        if args.balanced_intersectional_sampling:
            sampler = IntersectionalSampler(
                prism_items,
                fields=args.intersectional_fields,
                target_buckets=target_buckets,
                strategy=args.sampling_strategy,
                seed=args.seed + run_idx,
            )
        else:
            sampler = ConversationSampler(prism_items, args.sampling_strategy, args.seed + run_idx)

        for i, example in enumerate(gsm_examples):
            question = safe_str(example.get("question"))
            gold_frac = extract_gold_answer(safe_str(example.get("answer")))
            if gold_frac is None:
                continue
            gold = str(gold_frac)

            bpred, bthink, bfinal = run_generation(model, tokenizer, baseline_prompt(question), args)
            bcorrect = int(bpred == gold if bpred is not None else False)
            brow = {
                "condition": "baseline",
                "run": run_idx,
                "index": i,
                "question": question,
                "gold": gold,
                "prediction": bpred or "",
                "correct": bcorrect,
                "race": "",
                "gender": "",
                "intersection": "",
                "conversation_id": "",
                "thinking_trace": bthink,
                "final_output": bfinal,
            }
            baseline_rows.append(brow)
            all_rows.append(brow)

            if args.balanced_intersectional_sampling:
                prism_item, bucket = sampler.sample()
            else:
                prism_item = sampler.sample()
                race_v = canonicalize_demo_value(get_demographic_value(prism_item.demographics, "race"), "race")
                gender_v = canonicalize_demo_value(get_demographic_value(prism_item.demographics, "gender"), "gender")
                bucket = f"{race_v}|{gender_v}"

            cpred, cthink, cfinal = run_generation(model, tokenizer, build_prompt(question, prism_item), args)
            ccorrect = int(cpred == gold if cpred is not None else False)

            race_label = canonicalize_demo_value(get_demographic_value(prism_item.demographics, "race"), "race")
            gender_label = canonicalize_demo_value(get_demographic_value(prism_item.demographics, "gender"), "gender")
            crow = {
                "condition": "contextual",
                "run": run_idx,
                "index": i,
                "question": question,
                "gold": gold,
                "prediction": cpred or "",
                "correct": ccorrect,
                "race": race_label,
                "gender": gender_label,
                "intersection": bucket,
                "conversation_id": prism_item.conversation_id,
                "thinking_trace": cthink,
                "final_output": cfinal,
            }
            contextual_rows.append(crow)
            all_rows.append(crow)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "condition",
            "run",
            "index",
            "question",
            "gold",
            "prediction",
            "correct",
            "race",
            "gender",
            "intersection",
            "conversation_id",
            "thinking_trace",
            "final_output",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "model": args.model,
        "num_questions": len(gsm_examples),
        "num_runs": args.num_runs,
        "baseline": {
            "overall_accuracy": compute_simple_accuracy(baseline_rows),
            "n": len(baseline_rows),
        },
        "contextual": {
            "overall_accuracy": compute_simple_accuracy(contextual_rows),
            "n": len(contextual_rows),
            "race_accuracy": grouped_accuracy(contextual_rows, "race", allowed=["black", "white"]),
            "gender_accuracy": grouped_accuracy(contextual_rows, "gender", allowed=["male", "female"]),
            "intersection_accuracy": grouped_accuracy(
                contextual_rows,
                "intersection",
                allowed=["black|male", "black|female", "white|male", "white|female"],
            ),
        },
        "output_csv": output_csv_path,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
