"""Benchmark speculative decoding vs. baseline across draft lengths and workloads.

Example:
  python bench.py --target Qwen/Qwen2.5-1.5B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct \
      --ks 2 4 6 8 --max-new 256 --temperature 0 --out results.json
"""

import argparse
import json
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from spec_decode import generate_baseline, generate_speculative

WORKLOADS = {
    "code": [
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
        "Write a Python class for a thread-safe LRU cache with get and put methods.",
        "Write a SQL query that returns the top 5 customers by total order value per year.",
        "Write a Python function that merges two sorted lists in linear time.",
        "Write a bash script that renames all .jpeg files in a folder to .jpg.",
        "Implement binary search on a sorted list in Python with a docstring.",
    ],
    "factual_qa": [
        "What causes the seasons on Earth? Answer in one paragraph.",
        "Explain the difference between a virus and a bacterium.",
        "Who wrote 'The Origin of Species' and when was it published?",
        "What is the boiling point of water at sea level in Fahrenheit and Kelvin?",
        "Briefly explain how a transistor works.",
        "What is the capital of Australia and why was it chosen?",
    ],
    "creative": [
        "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        "Write a haiku sequence about a city in the rain.",
        "Describe an imaginary planet with three moons in vivid detail.",
        "Write the opening paragraph of a mystery novel set on a train.",
        "Write a limerick about a programmer who never sleeps.",
        "Write a dialogue between the sun and the moon arguing about who is brighter.",
    ],
}


def encode(tokenizer, text):
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer.encode(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device
    dtype = {"auto": torch.float16 if device in ("cuda", "mps") else torch.float32,
             "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.target)
    target = AutoModelForCausalLM.from_pretrained(args.target, torch_dtype=dtype).to(device).eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft, torch_dtype=dtype).to(device).eval()
    assert tok.vocab_size == AutoTokenizer.from_pretrained(args.draft).vocab_size, "draft/target vocab mismatch"
    eos = tok.eos_token_id

    # Warmup so kernel compilation / allocation is not billed to the first run.
    w = encode(tok, WORKLOADS["code"][0])
    generate_baseline(target, w, 16, eos, args.temperature)
    generate_speculative(target, draft, w, 16, 4, eos, args.temperature)

    results = {"config": vars(args) | {"device": device, "dtype": str(dtype)}, "workloads": {}}
    for wl, prompts in WORKLOADS.items():
        enc = [encode(tok, p) for p in prompts]
        base_tps, base_out = [], []
        for ids in enc:
            out, st = generate_baseline(target, ids, args.max_new, eos, args.temperature)
            base_tps.append(st.tokens_per_s)
            base_out.append(out)
        base_mean = statistics.mean(base_tps)
        wl_res = {"baseline_tokens_per_s": base_mean, "by_k": {}}

        for k in args.ks:
            tps, drafted, accepted, rounds, same = [], 0, 0, 0, 0
            for ids, bo in zip(enc, base_out):
                out, st = generate_speculative(target, draft, ids, args.max_new, k, eos, args.temperature)
                tps.append(st.tokens_per_s)
                drafted += st.drafted
                accepted += st.accepted
                rounds += st.rounds
                same += int(out == bo)
            mean_tps = statistics.mean(tps)
            wl_res["by_k"][k] = {
                "tokens_per_s": mean_tps,
                "speedup": mean_tps / base_mean,
                "acceptance_rate": accepted / drafted if drafted else 0.0,
                "mean_accepted_per_round": accepted / rounds if rounds else 0.0,
                "identical_to_baseline_frac": same / len(enc),
            }
            print(f"{wl:12s} k={k:<2d} speedup={mean_tps / base_mean:5.2f}x "
                  f"accept={wl_res['by_k'][k]['acceptance_rate']:.1%} "
                  f"identical={same}/{len(enc)}")
        results["workloads"][wl] = wl_res

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
