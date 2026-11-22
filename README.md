# Speculative Decoding for LLM Inference

A from-scratch PyTorch implementation of speculative decoding (draft-and-verify) for Hugging Face causal LMs, with a benchmark harness that sweeps draft length `k` across code, factual QA, and creative workloads.

## Method

A small draft model proposes `k` tokens autoregressively. The target model scores all `k` positions in one forward pass. Tokens are accepted left to right:

- **Greedy (temperature 0):** accept draft token `x_i` iff it equals the target argmax. On the first mismatch, emit the target argmax and stop the round. Output is identical to greedy decoding with the target alone.
- **Sampling (temperature > 0):** accept `x_i` with probability `min(1, p(x_i) / q(x_i))`, where `p` is the target distribution and `q` the draft distribution. On rejection, sample from `norm(max(0, p - q))`. The output distribution equals the target's.

If all `k` drafts are accepted, a bonus token is sampled from the target's distribution at position `k+1`, so every round commits at least one token.

KV caches for both models are cropped back to the committed length after each round (`DynamicCache.crop`), so rejected draft tokens are discarded without recomputation.

## Layout

- `spec_decode.py`: `generate_baseline`, `generate_speculative`, and `Stats` (tokens/s, acceptance rate, mean accepted per round).
- `bench.py`: runs baseline and speculative for each `k` and workload, and writes `results.json`.
- `tests/test_equivalence.py`: correctness tests on tiny random GPT-2 models (no downloads).

## Setup

```bash
pip install torch transformers pytest
pytest -q tests/
```

## Benchmark

Use a same-tokenizer pair, for example:

```bash
python bench.py \
  --target Qwen/Qwen2.5-1.5B-Instruct \
  --draft Qwen/Qwen2.5-0.5B-Instruct \
  --ks 2 4 6 8 --max-new 256 --temperature 0 --out results.json
```

The script prints, per workload and `k`: speedup over baseline, draft-token acceptance rate, and the fraction of outputs identical to baseline (should be 1.0 in greedy mode, modulo fp16 numerical noise).

## Results

Fill from `results.json` after running on your hardware:

| Workload | Best k | Speedup | Acceptance rate |
|---|---|---|---|
| Code | | | |
| Factual QA | | | |
| Creative | | | |

Hardware, dtype, and model pair: ______

## Notes and limitations

- Baseline and speculative runs both use a KV cache, so the speedup isolates the draft-and-verify effect.
- Batch size is 1. Batched speculative decoding is more involved (ragged acceptance lengths) and not implemented here.
- Speedup depends on the target/draft compute ratio and on how well the draft matches the target on your workload; expect creative and open-ended text to accept less often than code.
