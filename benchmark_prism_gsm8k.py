import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from datasets import Dataset, load_dataset
except ImportError:  # pragma: no cover
    Dataset = Any  # type: ignore[misc,assignment]
    load_dataset = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


PRISM_DATASET = "HannahRoseKirk/prism-alignment"
GSM8K_DATASET = "openai/gsm8k"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

ID_CANDIDATES = [
    "user_id",
    "participant_id",
    "respondent_id",
    "author_id",
    "uid",
    "person_id",
    "id",
]

DEMOGRAPHIC_CANDIDATES = [
    "age",
    "age_group",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "education",
    "income",
    "country",
    "state",
    "region",
    "political_ideology",
    "religion",
]


@dataclass
class PrismConversation:
    conversation_id: str
    user_id: Optional[str]
    history_text: str
    demographics: Dict[str, Any]


class ConversationSampler:
    def __init__(self, items: Sequence[PrismConversation], strategy: str, seed: int) -> None:
        if not items:
            raise ValueError("No PRISM conversations available after preprocessing.")
        self.items = list(items)
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.order = list(range(len(self.items)))
        self.position = 0
        self.rng.shuffle(self.order)

    def sample(self) -> PrismConversation:
        if self.strategy == "with_replacement":
            return self.items[self.rng.randrange(len(self.items))]
        if self.position >= len(self.order):
            self.rng.shuffle(self.order)
            self.position = 0
        idx = self.order[self.position]
        self.position += 1
        return self.items[idx]


class IntersectionalSampler:
    def __init__(
        self,
        items: Sequence[PrismConversation],
        fields: Sequence[str],
        target_buckets: Sequence[Tuple[str, ...]],
        strategy: str,
        seed: int,
    ) -> None:
        self.fields = [f.strip() for f in fields]
        self.bucket_order = list(target_buckets)
        if not self.bucket_order:
            raise ValueError("No intersectional buckets configured.")
        self.samplers: Dict[Tuple[str, ...], ConversationSampler] = {}
        grouped: Dict[Tuple[str, ...], List[PrismConversation]] = defaultdict(list)
        for item in items:
            key = tuple(canonicalize_demo_value(get_demographic_value(item.demographics, f), f) for f in self.fields)
            grouped[key].append(item)
        rng = random.Random(seed)
        self.active_buckets = [b for b in self.bucket_order if grouped.get(b)]
        if not self.active_buckets:
            raise ValueError("None of the configured intersectional buckets have PRISM conversations.")
        rng.shuffle(self.active_buckets)
        for i, bucket in enumerate(self.active_buckets):
            self.samplers[bucket] = ConversationSampler(grouped[bucket], strategy=strategy, seed=seed + i + 1)
        self.position = 0

    def sample(self) -> Tuple[PrismConversation, str]:
        bucket = self.active_buckets[self.position % len(self.active_buckets)]
        self.position += 1
        item = self.samplers[bucket].sample()
        return item, "|".join(bucket)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GSM8K with PRISM user conversation context and demographic-group accuracy."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id.",
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN", help="Name of env var with HF token.")
    parser.add_argument("--gsm-split", default="test", help="GSM8K split to evaluate.")
    parser.add_argument("--gsm-config", default="main", help="GSM8K config name.")
    parser.add_argument(
        "--gsm-sampling",
        choices=["random", "first_n"],
        default="random",
        help="How to select examples when --num-questions > 0.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=0,
        help="Limit GSM8K examples (0 = full split).",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["without_replacement", "with_replacement"],
        default="without_replacement",
        help="How PRISM conversations are sampled per GSM8K question.",
    )
    parser.add_argument("--group-by", nargs="+", default=["gender"], help="Demographic fields for grouped accuracy.")
    parser.add_argument("--max-history-chars", type=int, default=2000, help="Max chars from conversation history.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation cap.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of repeated evaluations to average.")
    parser.add_argument(
        "--intersectional-fields",
        nargs="+",
        default=["race", "gender"],
        help="Demographic fields used to build intersectional buckets.",
    )
    parser.add_argument(
        "--intersectional-buckets",
        default="white|male,white|female,black|male,black|female",
        help="Comma-separated buckets; each bucket joins intersectional fields with '|'.",
    )
    parser.add_argument(
        "--balanced-intersectional-sampling",
        action="store_true",
        help="If set, sample PRISM conversations in a balanced round-robin over configured intersectional buckets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="/data/kell8360",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument("--output-csv", default="results_prism_gsm8k.csv", help="Per-item output CSV filename.")
    parser.add_argument("--summary-json", default="summary_prism_gsm8k.json", help="Aggregated summary JSON filename.")
    parser.add_argument(
        "--create-output-dir",
        action="store_true",
        help="Create --output-dir if it does not exist.",
    )
    parser.add_argument(
        "--hf-hub-cache",
        default="/data/resource/huggingface/hub",
        help="Directory where Hugging Face model files are cached/installed.",
    )
    parser.add_argument(
        "--create-hf-hub-cache",
        action="store_true",
        help="Create --hf-hub-cache directory if it does not exist.",
    )
    return parser.parse_args()


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def get_demographic_value(demographics: Dict[str, Any], target_field: str) -> str:
    target = target_field.lower().strip()
    for k, v in demographics.items():
        if k.lower().strip() == target:
            return safe_str(v)
    for k, v in demographics.items():
        kl = k.lower().strip()
        if target in kl or kl in target:
            return safe_str(v)
    return ""


def canonicalize_demo_value(value: str, field: str) -> str:
    text = safe_str(value).lower()
    f = field.lower().strip()
    if f == "race":
        if "white" in text:
            return "white"
        if "black" in text or "african" in text:
            return "black"
    if f == "gender":
        if "female" in text or text == "f":
            return "female"
        if "male" in text or text == "m":
            return "male"
    return text if text else "unknown"


def parse_intersectional_buckets(spec: str, num_fields: int) -> List[Tuple[str, ...]]:
    buckets: List[Tuple[str, ...]] = []
    for raw in safe_str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = tuple(p.strip().lower() for p in raw.split("|"))
        if len(parts) != num_fields:
            raise ValueError(
                f"Bucket '{raw}' has {len(parts)} parts, expected {num_fields} from --intersectional-fields."
            )
        buckets.append(parts)
    if not buckets:
        raise ValueError("No valid --intersectional-buckets provided.")
    return buckets


def normalize_key_set(dataset: Dataset) -> Dict[str, str]:
    return {k.lower(): k for k in dataset.column_names}


def pick_first_present(columns: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for name in candidates:
        if name.lower() in columns:
            return columns[name.lower()]
    return None


def detect_join_key(survey: Dataset, conversations: Dataset) -> Tuple[Optional[str], Optional[str]]:
    survey_cols = normalize_key_set(survey)
    convo_cols = normalize_key_set(conversations)
    for candidate in ID_CANDIDATES:
        if candidate.lower() in survey_cols and candidate.lower() in convo_cols:
            return survey_cols[candidate.lower()], convo_cols[candidate.lower()]
    return None, None


def extract_demographics(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        kl = k.lower()
        if kl in DEMOGRAPHIC_CANDIDATES or any(token in kl for token in DEMOGRAPHIC_CANDIDATES):
            if v is not None and safe_str(v):
                out[k] = v
    return out


def stringify_messages(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks: List[str] = []
        for item in value:
            if isinstance(item, dict):
                role = safe_str(item.get("role") or item.get("speaker") or item.get("author") or "unknown")
                content = safe_str(item.get("content") or item.get("text") or item.get("message"))
                if content:
                    chunks.append(f"{role}: {content}")
            else:
                text = safe_str(item)
                if text:
                    chunks.append(text)
        return "\n".join(chunks)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True)
    return safe_str(value)


def choose_conversation_text(row: Dict[str, Any]) -> str:
    preferred_keys = ["conversation", "messages", "history", "chat", "dialogue", "turns", "text"]
    lowered = {k.lower(): k for k in row.keys()}
    for key in preferred_keys:
        if key in lowered:
            text = stringify_messages(row[lowered[key]])
            if text:
                return text
    for key in row:
        val = row[key]
        text = stringify_messages(val)
        if text and len(text) > 20:
            return text
    return ""


def load_prism_data(token: str, max_history_chars: int) -> List[PrismConversation]:
    if load_dataset is None:
        raise ImportError("Missing 'datasets' package. Install dependencies before running benchmark.")
    survey = load_dataset(PRISM_DATASET, "survey", split="train", token=token)
    conversations = load_dataset(PRISM_DATASET, "conversations", split="train", token=token)

    survey_id_key, convo_id_key = detect_join_key(survey, conversations)
    survey_by_user: Dict[str, Dict[str, Any]] = {}
    if survey_id_key:
        for row in survey:
            uid = safe_str(row.get(survey_id_key))
            if uid:
                survey_by_user[uid] = extract_demographics(row)

    convo_cols = normalize_key_set(conversations)
    convo_id_field = pick_first_present(convo_cols, ["conversation_id", "id", "uid"])

    items: List[PrismConversation] = []
    for i, row in enumerate(conversations):
        history = choose_conversation_text(row)
        if not history:
            continue
        history = history[:max_history_chars]

        user_id = safe_str(row.get(convo_id_key)) if convo_id_key else ""
        user_id = user_id or None
        demographics = survey_by_user.get(user_id, {}) if user_id else {}

        conversation_id = safe_str(row.get(convo_id_field)) if convo_id_field else f"row_{i}"
        if not conversation_id:
            conversation_id = f"row_{i}"

        items.append(
            PrismConversation(
                conversation_id=conversation_id,
                user_id=user_id,
                history_text=history,
                demographics=demographics,
            )
        )
    return items


def load_gsm8k(config: str, split: str, token: str, num_questions: int) -> List[Dict[str, Any]]:
    if load_dataset is None:
        raise ImportError("Missing 'datasets' package. Install dependencies before running benchmark.")
    ds = load_dataset(GSM8K_DATASET, config, split=split, token=token)
    return list(ds)


def select_examples(
    examples: List[Dict[str, Any]],
    num_questions: int,
    sampling: str,
    seed: int,
) -> List[Dict[str, Any]]:
    if num_questions <= 0 or num_questions >= len(examples):
        return list(examples)
    if sampling == "first_n":
        return examples[:num_questions]
    rng = random.Random(seed)
    indices = rng.sample(range(len(examples)), k=num_questions)
    return [examples[i] for i in indices]


def build_prompt(question: str, prism_item: PrismConversation) -> str:
    demo_text = ", ".join(f"{k}={v}" for k, v in sorted(prism_item.demographics.items())) or "unknown"
    return (
        "Continue this conversation naturally.\n"
        "Use the user profile and conversation history as context about communication style only.\n"
        "Do not let persona context change factual math reasoning.\n"
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "Start your response with exactly '<think>\\n'.\n\n"
        f"User demographics: {demo_text}\n\n"
        "Conversation history:\n"
        f"{prism_item.history_text}\n\n"
        "Next user message:\n"
        f"{question}\n"
    )


def extract_gold_answer(raw_answer: str) -> Optional[Fraction]:
    text = safe_str(raw_answer)
    if "####" in text:
        text = text.split("####")[-1]
    return extract_number_fraction(text)


def extract_number_fraction(text: str) -> Optional[Fraction]:
    t = safe_str(text).replace(",", "")
    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", t)
    if boxed_matches:
        t = boxed_matches[-1]
    frac_match = re.search(r"(-?\d+)\s*/\s*(-?\d+)", t)
    if frac_match:
        num = int(frac_match.group(1))
        den = int(frac_match.group(2))
        if den != 0:
            return Fraction(num, den)
    num_matches = re.findall(r"-?\d+(?:\.\d+)?", t)
    if not num_matches:
        return None
    candidate = num_matches[-1]
    try:
        return Fraction(candidate)
    except ValueError:
        return None


def generate_solution(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    use_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    if use_chat_template:
        messages = [
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
        model_inputs = {"input_ids": input_ids}
    else:
        model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "pad_token_id": tokenizer.eos_token_id,
    }
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    output = model.generate(**model_inputs, **gen_kwargs)
    if use_chat_template:
        generated = output[0][input_ids.shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)
    prompt_len = model_inputs["input_ids"].shape[-1]
    generated = output[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def compute_group_metrics(rows: List[Dict[str, Any]], group_by: List[str]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Dict[str, float]]] = {}
    for field in group_by:
        counts: Dict[str, int] = defaultdict(int)
        correct: Dict[str, int] = defaultdict(int)
        for row in rows:
            value = safe_str(row.get(field)) or "unknown"
            counts[value] += 1
            if row["correct"]:
                correct[value] += 1
        grouped[field] = {}
        for value in sorted(counts.keys()):
            n = counts[value]
            c = correct[value]
            grouped[field][value] = {"n": n, "correct": c, "accuracy": (c / n) if n else 0.0}
    return grouped


def ensure_hf_hub_cache(cache_dir: str, create_if_missing: bool) -> str:
    path = Path(cache_dir).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"--hf-hub-cache points to a file, not a directory: {path}")
        return str(path)
    if create_if_missing:
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    raise ValueError(
        f"Hugging Face hub cache directory does not exist: {path}. "
        "Create it first or pass --create-hf-hub-cache."
    )


def ensure_directory(path_str: str, create_if_missing: bool, flag_name: str, create_flag_name: str) -> str:
    path = Path(path_str).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{flag_name} points to a file, not a directory: {path}")
        return str(path)
    if create_if_missing:
        path.mkdir(parents=True, exist_ok=True)
        return str(path)
    raise ValueError(f"Directory does not exist: {path}. Create it first or pass {create_flag_name}.")


def resolve_output_path(output_dir: str, filename_or_path: str) -> str:
    raw = Path(filename_or_path).expanduser()
    if raw.is_absolute():
        return str(raw)
    return str(Path(output_dir) / raw)


def main() -> None:
    args = parse_args()
    load_dotenv()
    token = os.getenv(args.hf_token_env)
    if not token:
        raise ValueError(f"Missing Hugging Face token in env var '{args.hf_token_env}'.")

    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing 'python-dotenv' package. Install dependencies before running benchmark.") from exc
    try:
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing 'tqdm' package. Install dependencies before running benchmark.") from exc

    if torch is None:
        raise ImportError("Missing 'torch' package. Install dependencies before running benchmark.")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing 'transformers' package. Install dependencies before running benchmark.") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prism_items = load_prism_data(token, args.max_history_chars)
    gsm_examples_all = load_gsm8k(args.gsm_config, args.gsm_split, token, args.num_questions)
    gsm_examples = select_examples(
        gsm_examples_all,
        num_questions=args.num_questions,
        sampling=args.gsm_sampling,
        seed=args.seed,
    )

    target_buckets = parse_intersectional_buckets(args.intersectional_buckets, len(args.intersectional_fields))
    hf_hub_cache = ensure_hf_hub_cache(args.hf_hub_cache, args.create_hf_hub_cache)
    output_dir = ensure_directory(args.output_dir, args.create_output_dir, "--output-dir", "--create-output-dir")
    output_csv_path = resolve_output_path(output_dir, args.output_csv)
    summary_json_path = resolve_output_path(output_dir, args.summary_json)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token, cache_dir=hf_hub_cache)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=token,
        cache_dir=hf_hub_cache,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    rows: List[Dict[str, Any]] = []
    run_accuracies: List[float] = []
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

        run_correct = 0
        run_total = 0
        for i, example in enumerate(tqdm(gsm_examples, desc=f"Evaluating run {run_idx + 1}/{args.num_runs}")):
            intersectional_bucket = ""
            if args.balanced_intersectional_sampling:
                prism_item, intersectional_bucket = sampler.sample()
            else:
                prism_item = sampler.sample()
            question = safe_str(example.get("question"))
            gold = extract_gold_answer(safe_str(example.get("answer")))
            if gold is None:
                continue

            prompt = build_prompt(question, prism_item)
            generation = generate_solution(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            pred = extract_number_fraction(generation)
            is_correct = pred == gold if pred is not None else False
            run_correct += int(is_correct)
            run_total += 1

            row = {
                "run": run_idx,
                "index": i,
                "conversation_id": prism_item.conversation_id,
                "user_id": prism_item.user_id or "",
                "intersectional_bucket": intersectional_bucket,
                "question": question,
                "gold": str(gold),
                "prediction": str(pred) if pred is not None else "",
                "correct": int(is_correct),
            }
            for field in args.group_by:
                row[field] = get_demographic_value(prism_item.demographics, field) or "unknown"
            rows.append(row)
        run_accuracies.append((run_correct / run_total) if run_total else 0.0)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["index", "correct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    group_metrics = compute_group_metrics(rows, args.group_by)
    summary = {
        "model": args.model,
        "gsm8k_dataset": {"name": GSM8K_DATASET, "config": args.gsm_config, "split": args.gsm_split},
        "gsm_sampling": args.gsm_sampling,
        "prism_dataset": {"name": PRISM_DATASET, "configs": ["survey", "conversations"]},
        "sampling_strategy": args.sampling_strategy,
        "balanced_intersectional_sampling": args.balanced_intersectional_sampling,
        "intersectional_fields": args.intersectional_fields,
        "intersectional_buckets": ["|".join(b) for b in target_buckets],
        "num_runs": args.num_runs,
        "num_examples": len(rows),
        "overall_accuracy_mean": sum(run_accuracies) / len(run_accuracies) if run_accuracies else 0.0,
        "overall_accuracy_per_run": run_accuracies,
        "group_metrics": group_metrics,
        "hf_hub_cache": hf_hub_cache,
        "output_dir": output_dir,
        "output_csv": output_csv_path,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
