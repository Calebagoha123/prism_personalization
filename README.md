# prism_personalization

Run GSM8K with PRISM conversation context and report accuracy by demographic group.

## Setup

```bash
uv sync
```

Create `.env`:

```bash
HF_TOKEN=your_huggingface_token
```

## Run

```bash
uv run benchmark_prism_gsm8k.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --hf-hub-cache /data/resource/huggingface/hub \
  --create-hf-hub-cache \
  --output-dir /data/kell8360 \
  --create-output-dir \
  --num-questions 100 \
  --gsm-sampling random \
  --num-runs 3 \
  --temperature 0.6 \
  --sampling-strategy without_replacement \
  --group-by gender race age_group \
  --balanced-intersectional-sampling \
  --intersectional-fields race gender \
  --intersectional-buckets "white|male,white|female,black|male,black|female"
```

Model files are cached/installed under `/data/resource/huggingface/hub`. The script checks this directory first and can create it when `--create-hf-hub-cache` is provided.
Benchmark outputs are written under `/data/kell8360` by default (or `--output-dir`).
When `--num-questions` is set, GSM8K examples are randomly sampled by seed by default (`--gsm-sampling random`) to reduce ordering bias.

## Unit tests

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```

Outputs:
- `results_prism_gsm8k.csv`: one row per GSM8K question, including sampled PRISM conversation and correctness.
- `summary_prism_gsm8k.json`: overall and per-group accuracy.

## Sampling strategy recommendation

Use `without_replacement` as default (implemented as epoch-wise shuffle: each conversation is used once before reuse). This reduces repeated-context bias versus pure with-replacement sampling while still allowing runs larger than the PRISM pool.

For intersectional sampling, enable `--balanced-intersectional-sampling` so the script rotates across available configured buckets (for example white male, white female, black male, black female).
