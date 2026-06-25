"""Extract mid-layer logit-lens decodability of MMLU answers for base vs RL/instruct models.

For each model we run the same raw multiple-choice prompt, capture the residual
stream at the final input token (the position that predicts the answer letter)
for a set of evenly spaced layers, apply the logit lens (final norm + unembed)
at every layer restricted to the four option-letter tokens, and record the
model's actual final-layer option prediction. Saves per model to .npz.

Tests the elicitation hypothesis: is the correct answer already linearly present
in the BASE model's mid-layer states even when its final output is wrong?

Python 3.10 (system python3 on the GPU host). No match, no X|Y annotations.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# (label, hf_id, is_base)
MODELS = [
    ("qwen2.5-7b-base", "Qwen/Qwen2.5-7B", True),
    ("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct", False),
    ("llama-3.1-8b-base", "meta-llama/Llama-3.1-8B", True),
    ("llama-3.1-8b-instruct", "meta-llama/Llama-3.1-8B-Instruct", False),
]
LETTERS = ["A", "B", "C", "D"]
PROMPT = ("The following is a multiple choice question. Answer with the letter of "
          "the correct option.\n\nQuestion: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:")


def load_mmlu(n, seed):
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n]
    rows = []
    for i in idx:
        r = ds[int(i)]
        if len(r["choices"]) != 4 or not (0 <= r["answer"] <= 3):
            continue
        rows.append({"q": r["question"], "ch": r["choices"], "ans": int(r["answer"]),
                     "subj": r.get("subject", "")})
    return rows


def option_token_ids(tok):
    # token id for " A", " B", ... (leading space, the continuation after "Answer:")
    ids = []
    for L in LETTERS:
        t = tok.encode(" " + L, add_special_tokens=False)
        ids.append(t[-1])
    return ids


def pick_layers(n_layers):
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return sorted(set(max(1, min(n_layers, int(round(fr * n_layers)))) for fr in fracs))


@torch.no_grad()
def run_model(label, hf_id, rows, out_dir, seed, batch_size=8):
    print("[load]", hf_id, flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so last token is at position -1 for all rows
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map="cuda", output_hidden_states=True)
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = pick_layers(n_layers)
    opt_ids = option_token_ids(tok)
    # final norm + unembed for logit lens; project onto ONLY the 4 option rows
    # of the unembedding (avoids materializing the full-vocab logits per layer,
    # which OOMs on large-vocab models like Qwen2.5 with ~152k vocab).
    norm = model.model.norm
    W_opt = model.lm_head.weight[opt_ids].detach()  # [4, d]
    gold = np.array([r["ans"] for r in rows], dtype=np.int64)

    hid = {li: [] for li in layers}        # stored hidden states for the probe
    lens_logits = {li: [] for li in layers}  # 4-way option logits via logit lens
    final_pred = []                         # model final-layer option argmax
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        prompts = [PROMPT.format(q=r["q"], a=r["ch"][0], b=r["ch"][1], c=r["ch"][2], d=r["ch"][3]) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        out = model(**enc)
        hs = out.hidden_states  # tuple n_layers+1, each [B, T, d]
        last = -1  # left padding => last real token at -1
        for li in layers:
            h = hs[li][:, last, :]  # [B, d]
            hid[li].append(h.float().cpu().numpy())
            opt = (norm(h) @ W_opt.T).float().cpu().numpy()  # [B,4] option logits
            lens_logits[li].append(opt)
        # final-layer prediction via the same option projection on the last hidden state
        fl = (norm(hs[-1][:, last, :]) @ W_opt.T).float().cpu().numpy()
        final_pred.append(fl.argmax(axis=1))
        if start % (batch_size * 20) == 0:
            print("  [%s] %d/%d" % (label, start, len(rows)), flush=True)

    final_pred = np.concatenate(final_pred)
    np.savez_compressed(
        out_dir / ("lens_%s.npz" % label),
        layers=np.array(layers), gold=gold, final_pred=final_pred,
        subj=np.array([r["subj"] for r in rows], dtype=object),
        eval_order=np.arange(len(rows), dtype=np.int64),
        **{("hid_L%02d" % li): np.concatenate(hid[li], axis=0) for li in layers},
        **{("lens_L%02d" % li): np.concatenate(lens_logits[li], axis=0) for li in layers},
    )
    acc = float((final_pred == gold).mean())
    print("  [saved] %s  final-output acc=%.3f  n=%d  layers=%s" % (label, acc, len(rows), layers), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/real/acts")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--only", default="")  # comma list of labels to run
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_mmlu(args.n, args.seed)
    print("[mmlu] loaded", len(rows), "items", flush=True)
    (out_dir / "extract_meta.json").write_text(json.dumps(
        {"prompt": PROMPT, "n": len(rows), "seed": args.seed,
         "models": [m[0] for m in MODELS], "letters": LETTERS}, indent=2))
    only = set(s for s in args.only.split(",") if s)
    for label, hf_id, is_base in MODELS:
        if only and label not in only:
            continue
        run_model(label, hf_id, rows, out_dir, args.seed, args.batch_size)
    print("[done] extraction complete", flush=True)


if __name__ == "__main__":
    main()
