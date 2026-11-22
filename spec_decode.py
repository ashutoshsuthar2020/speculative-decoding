"""Speculative decoding (draft-and-verify) for Hugging Face causal LMs.

Invariant used throughout: a KV cache covers tokens[:len(tokens) - 1].
The last committed token is always the next model input.

Greedy mode (temperature=0): accept draft token i iff it equals the target argmax.
The output is identical to greedy decoding with the target alone.

Sampling mode (temperature>0): standard rejection sampling (Leviathan et al. 2023,
Chen et al. 2023). Accept x ~ q with prob min(1, p(x)/q(x)); on reject, sample from
norm(max(0, p - q)). The output distribution equals the target's distribution.
"""

import time
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache


@dataclass
class Stats:
    new_tokens: int = 0
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    elapsed: float = 0.0

    @property
    def tokens_per_s(self) -> float:
        return self.new_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def mean_accepted_per_round(self) -> float:
        return self.accepted / self.rounds if self.rounds else 0.0


@torch.inference_mode()
def _forward(model, ids: list[int], cache: DynamicCache) -> torch.Tensor:
    """Run model on ids, extend cache, return float32 logits [len(ids), V]."""
    x = torch.tensor([ids], device=model.device)
    out = model(input_ids=x, past_key_values=cache, use_cache=True)
    return out.logits[0].float()


def _sync(model, cache: DynamicCache, tokens: list[int], target_len: int) -> None:
    """Make the cache cover exactly tokens[:target_len]."""
    cur = cache.get_seq_length()
    if cur > target_len:
        cache.crop(target_len - cur)  # transformers requires a negative count to remove
    elif cur < target_len:
        _forward(model, tokens[cur:target_len], cache)


def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature == 0:
        return int(torch.argmax(logits))
    p = F.softmax(logits / temperature, dim=-1)
    return int(torch.multinomial(p, 1))


@torch.inference_mode()
def generate_baseline(target, prompt_ids, max_new, eos_id, temperature=0.0):
    tokens = list(prompt_ids)
    tc = DynamicCache()
    st = Stats()
    t0 = time.perf_counter()
    _sync(target, tc, tokens, len(tokens) - 1)
    while st.new_tokens < max_new:
        logits = _forward(target, [tokens[-1]], tc)[-1]
        nxt = _sample(logits, temperature)
        tokens.append(nxt)
        st.new_tokens += 1
        if nxt == eos_id:
            break
    st.elapsed = time.perf_counter() - t0
    return tokens[len(prompt_ids):], st


@torch.inference_mode()
def generate_speculative(target, draft, prompt_ids, max_new, k, eos_id, temperature=0.0):
    assert k >= 1
    tokens = list(prompt_ids)
    prompt_len = len(tokens)
    tc, dc = DynamicCache(), DynamicCache()
    _sync(target, tc, tokens, len(tokens) - 1)
    _sync(draft, dc, tokens, len(tokens) - 1)
    st = Stats()
    t0 = time.perf_counter()

    while st.new_tokens < max_new:
        # 1) Draft k tokens autoregressively.
        drafted, q_dists = [], []
        cur = tokens[-1]
        for _ in range(k):
            dl = _forward(draft, [cur], dc)[-1]
            if temperature == 0:
                q = None
                nxt = int(torch.argmax(dl))
            else:
                q = F.softmax(dl / temperature, dim=-1)
                nxt = int(torch.multinomial(q, 1))
            drafted.append(nxt)
            q_dists.append(q)
            cur = nxt

        # 2) Verify all k drafts in one target forward pass.
        #    tl[i] is the target distribution for the token at position len(tokens)+i.
        tl = _forward(target, [tokens[-1]] + drafted, tc)
        st.rounds += 1
        st.drafted += k

        # 3) Accept/reject.
        accepted: list[int] = []
        extra = None
        for i in range(k):
            if temperature == 0:
                t_arg = int(torch.argmax(tl[i]))
                if t_arg == drafted[i]:
                    accepted.append(drafted[i])
                    continue
                extra = t_arg
                break
            p = F.softmax(tl[i] / temperature, dim=-1)
            q = q_dists[i]
            x = drafted[i]
            if random.random() < min(1.0, float(p[x] / q[x])):
                accepted.append(x)
                continue
            residual = (p - q).clamp(min=0)
            s = residual.sum()
            residual = residual / s if s > 0 else p
            extra = int(torch.multinomial(residual, 1))
            break
        else:
            # All k accepted: take a bonus token from the target's last distribution.
            extra = _sample(tl[k], temperature)

        st.accepted += len(accepted)

        # 4) Commit tokens, stopping at EOS or the token budget.
        for t in accepted + [extra]:
            tokens.append(t)
            st.new_tokens += 1
            if t == eos_id or st.new_tokens >= max_new:
                break
        if tokens[-1] == eos_id or st.new_tokens >= max_new:
            break

        # 5) Re-sync caches to the new committed length minus one.
        _sync(target, tc, tokens, len(tokens) - 1)
        _sync(draft, dc, tokens, len(tokens) - 1)

    st.elapsed = time.perf_counter() - t0
    return tokens[prompt_len:], st
