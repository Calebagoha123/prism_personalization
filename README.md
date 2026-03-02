# prism_personalization

Run GSM8K with PRISM conversation context and report accuracy by demographic group.

## Setup

```bash
uv sync
```

Create `.env` only if you want network fallback downloads:

```bash
HF_TOKEN=your_huggingface_token
```

## Run

```bash
uv run benchmark_prism_gsm8k.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --hf-hub-cache /data/resource/huggingface/hub \
  --prism-local-dir /Users/calebagoha/Downloads/prism \
  --output-dir /data/kell8360 \
  --create-output-dir \
  --num-questions 100 \
  --gsm-sampling random \
  --num-runs 3 \
  --max-new-tokens 32768 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --sampling-strategy without_replacement \
  --group-by gender race age_group \
  --balanced-intersectional-sampling \
  --intersectional-fields race gender \
  --intersectional-buckets "white|male,white|female,black|male,black|female"
```

Model files and datasets are loaded from `/data/resource/huggingface/hub` by default.
The script runs in local-cache-only mode by default (`--local-files-only`).
If you want download fallback from Hub, pass `--allow-network-download` and provide `HF_TOKEN`.
If you already exported PRISM as JSONL files, pass `--prism-local-dir` and the script will load `conversations.jsonl` and `survey.jsonl` directly.
Benchmark outputs are written under `/data/kell8360` by default (or `--output-dir`).
When `--num-questions` is set, GSM8K examples are randomly sampled by seed by default (`--gsm-sampling random`) to reduce ordering bias.

## Unit tests

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

## Baseline vs Contextual Script

Run both conditions on the same sampled GSM8K set:

```bash
uv run benchmark_baseline_vs_prism.py \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --hf-hub-cache /data/resource/huggingface/hub \
  --prism-local-dir /data/kell8360/prism \
  --output-dir /data/kell8360 \
  --create-output-dir \
  --num-questions 100 \
  --num-runs 1 \
  --gsm-sampling random \
  --balanced-intersectional-sampling \
  --intersectional-fields race gender \
  --intersectional-buckets "white|male,white|female,black|male,black|female"
```

This writes:
- baseline overall accuracy (question only)
- contextual overall accuracy
- contextual race accuracy (`black`, `white`)
- contextual gender accuracy (`male`, `female`)
- contextual intersection accuracy (`black|male`, `black|female`, `white|male`, `white|female`)

Generate plots from the CSV:

```bash
uv run plot_baseline_vs_prism.py \
  --csv /data/kell8360/results_baseline_vs_prism.csv \
  --output-dir /data/kell8360/plots_baseline_vs_prism
```

Outputs:
- `overall_comparison.png`
- `race_gender_comparison.png`
- `intersectional_comparison.png`

Outputs:
- `results_prism_gsm8k.csv`: one row per GSM8K question, including sampled PRISM conversation and correctness.
- `summary_prism_gsm8k.json`: overall and per-group accuracy.

## Sampling strategy recommendation

Use `without_replacement` as default (implemented as epoch-wise shuffle: each conversation is used once before reuse). This reduces repeated-context bias versus pure with-replacement sampling while still allowing runs larger than the PRISM pool.

For intersectional sampling, enable `--balanced-intersectional-sampling` so the script rotates across available configured buckets (for example white male, white female, black male, black female).
