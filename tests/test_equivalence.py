"""Checks correctness with tiny random models (no downloads needed).

Run:  pytest -q tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from spec_decode import generate_baseline, generate_speculative

VOCAB = 64
EOS = 0


def make(n_layer, n_embd, seed):
    torch.manual_seed(seed)
    cfg = GPT2Config(vocab_size=VOCAB, n_positions=256, n_embd=n_embd, n_layer=n_layer, n_head=2)
    return GPT2LMHeadModel(cfg).eval()


def test_greedy_speculative_matches_baseline():
    target = make(2, 32, seed=1)
    draft = make(1, 16, seed=2)  # different weights => many rejections, good coverage
    for prompt in ([5, 9, 13], [1], [7, 7, 7, 7, 3, 2]):
        for k in (1, 3, 5):
            base, _ = generate_baseline(target, prompt, 20, EOS, temperature=0.0)
            spec, st = generate_speculative(target, draft, prompt, 20, k, EOS, temperature=0.0)
            assert spec == base, f"prompt={prompt} k={k}"
            assert st.new_tokens == len(spec) <= 20


def test_self_draft_accepts_everything_greedy():
    target = make(2, 32, seed=3)
    spec, st = generate_speculative(target, target, [4, 8], 24, 4, EOS, temperature=0.0)
    base, _ = generate_baseline(target, [4, 8], 24, EOS, temperature=0.0)
    assert spec == base
    assert st.acceptance_rate == 1.0


def test_eos_stops_generation():
    target = make(2, 32, seed=4)
    base, _ = generate_baseline(target, [3, 3], 50, EOS, temperature=0.0)
    spec, _ = generate_speculative(target, make(1, 16, seed=5), [3, 3], 50, 4, EOS, temperature=0.0)
    assert spec == base
    if EOS in base:
        assert spec[-1] == EOS and spec.index(EOS) == len(spec) - 1


def test_sampling_mode_runs_and_respects_budget():
    torch.manual_seed(0)
    target = make(2, 32, seed=6)
    draft = make(1, 16, seed=7)
    spec, st = generate_speculative(target, draft, [1, 2, 3], 30, 4, EOS, temperature=1.0)
    assert 1 <= len(spec) <= 30
    assert 0.0 <= st.acceptance_rate <= 1.0
