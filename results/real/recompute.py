"""recompute.py — reproduce analysis_summary.txt from per-example data alone.

Pure standard library. Reads eval_points.jsonl (per-example {section, eval_order,
pred, gold}) from the current directory, recomputes every accuracy + bootstrap CI,
and prints analysis_summary.txt byte-for-byte.

Gate: `cd results/real && python3 recompute.py | diff - analysis_summary.txt` empty.
"""
import json
import random
from collections import defaultdict

BOOT_N = 2000
BOOT_SEED = 42
CHANCE = 0.25

SECTIONS = [
    ("Qwen2.5-7B base: final-output MMLU accuracy (raw prompt, zero-shot)", "acc", "qwen_base_final"),
    ("Qwen2.5-7B base: best mid-layer logit-lens accuracy", "acc", "qwen_base_lens"),
    ("Qwen2.5-7B base: best mid-layer linear-probe accuracy (5-fold OOF)", "acc", "qwen_base_probe"),
    ("Qwen2.5-7B-Instruct: final-output MMLU accuracy (raw prompt)", "acc", "qwen_instruct_final"),
    ("Qwen2.5-7B base on RL-unlocked subset (instruct right, base output wrong): mid-layer logit-lens accuracy", "acc", "qwen_unlocked_lens"),
    ("Qwen2.5-7B base on RL-unlocked subset: mid-layer linear-probe accuracy", "acc", "qwen_unlocked_probe"),
    ("Llama-3.1-8B base: final-output MMLU accuracy (raw prompt, zero-shot)", "acc", "llama_base_final"),
    ("Llama-3.1-8B base: best mid-layer logit-lens accuracy", "acc", "llama_base_lens"),
    ("Llama-3.1-8B base: best mid-layer linear-probe accuracy (5-fold OOF)", "acc", "llama_base_probe"),
    ("Llama-3.1-8B-Instruct: final-output MMLU accuracy (raw prompt)", "acc", "llama_instruct_final"),
    ("Llama-3.1-8B base on RL-unlocked subset (instruct right, base output wrong): mid-layer logit-lens accuracy", "acc", "llama_unlocked_lens"),
    ("Llama-3.1-8B base on RL-unlocked subset: mid-layer linear-probe accuracy", "acc", "llama_unlocked_probe"),
]


def percentile(s, q):
    if not s:
        return float("nan")
    pos = q / 100.0 * (len(s) - 1)
    lo = int(pos); frac = pos - lo
    if lo + 1 < len(s):
        return s[lo] * (1 - frac) + s[lo + 1] * frac
    return s[lo]


def acc_ci(preds, golds):
    n = len(preds)
    point = sum(1 for p, g in zip(preds, golds) if p == g) / n if n else float("nan")
    rng = random.Random(BOOT_SEED)
    boots = []
    for _ in range(BOOT_N):
        c = 0
        for _ in range(n):
            i = rng.randrange(n)
            if preds[i] == golds[i]:
                c += 1
        boots.append(c / n)
    boots.sort()
    return point, percentile(boots, 2.5), percentile(boots, 97.5), n


def main():
    by = defaultdict(list)
    with open("eval_points.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by[r["section"]].append(r)
    for k in by:
        by[k].sort(key=lambda r: r["eval_order"])
    lines = []
    lines.append("# RL-unlocked answers and base-model mid-layer decodability (MMLU)")
    lines.append("")
    lines.append("Logit lens: final norm + unembedding applied to mid-layer residual, argmax over A/B/C/D.")
    lines.append("Linear probe: multinomial logistic regression on mid-layer residual, 5-fold OOF, 4-way.")
    lines.append("Same raw multiple-choice prompt for base and instruct. Chance = 0.25.")
    lines.append("Bootstrap: 2000 resamples, percentile 95% CI, seed 42.")
    lines.append("")
    for title, kind, key in SECTIONS:
        rows = by.get(key, [])
        preds = [r["pred"] for r in rows]
        golds = [r["gold"] for r in rows]
        p, lo, hi, n = acc_ci(preds, golds)
        lines.append("## %s" % title)
        lines.append("  accuracy = %.4f  (95%% CI %.4f-%.4f, n=%d)" % (p, lo, hi, n))
        lines.append("")
    print("\n".join(lines).rstrip("\n"))


if __name__ == "__main__":
    main()
