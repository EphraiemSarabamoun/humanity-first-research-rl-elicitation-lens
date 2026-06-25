"""Write REPRO_CONTRACT sidecars, per-figure CSVs, sources.json for rl-elicitation-lens."""
import csv
import json
from pathlib import Path

OUT = Path("results/real")
GIT = "ae0598b0916f5f9a0f08a3cc877c8e60872dbaa1"
rows = list(csv.DictReader(open(OUT / "curve.csv")))


def sidecar(name, columns, analysis_cmd):
    body = """# %s — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py
Command: python3 src/analyze.py --acts_dir results/real/acts --out_dir results/real
Git-Commit: %s
Seeds: 42 (MMLU sampling in extract_lens.py; 5-fold StratifiedKFold for the probe; per-fold PCA(256); 2000-resample percentile bootstrap for CIs)
Source-Data: results/real/acts/lens_<model>.npz (last-token residual-stream activations at selected layers + logit-lens option logits + final-output prediction, Qwen2.5-7B base/instruct and Llama-3.1-8B base/instruct, RTX 5090, 2026-06-24, torch 2.12 cu130; built from cais/mmlu test split, 2000 items, via src/extract_lens.py with a fixed raw multiple-choice prompt)
Analysis-Command: %s
Columns:
%s
""" % (name, GIT, analysis_cmd, columns)
    (OUT / (name + ".md")).write_text(body)


sidecar("curve.csv",
        "  model (HF model label: qwen2.5-7b/llama-3.1-8b, base or instruct);\n"
        "  layer (residual-stream block index into hidden_states);\n"
        "  lens_acc (logit-lens MMLU accuracy at this layer: final-norm + unembedding applied to the last-token residual, argmax over the four option tokens A/B/C/D, unitless 0-1);\n"
        "  probe_acc (5-fold out-of-fold multinomial logistic-regression accuracy on PCA-256 of the last-token residual, 4-way, 0-1);\n"
        "  final_acc (the model's own final-output MMLU accuracy under the raw prompt, constant per model, 0-1);\n"
        "  chance (0.25); n (number of MMLU items evaluated)",
        "this file is the figure data of record; the cited headline numbers are reproduced by `cd results/real && python3 recompute.py | diff - analysis_summary.txt` (empty) from eval_points.jsonl")

sidecar("eval_points.jsonl",
        "  section (which summary arm this row backs, e.g. qwen_base_probe, llama_unlocked_lens);\n"
        "  eval_order (integer position within the arm, for seeded-bootstrap replay);\n"
        "  pred (predicted option index 0-3); gold (correct option index 0-3)",
        "cd results/real && python3 recompute.py  (rebuilds every accuracy + 95% bootstrap CI from these rows)")

# per-figure CSV + md (stem-named)
def w(stem, rs, cols, desc):
    with open(OUT / (stem + ".csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for r in rs:
            wr.writerow({c: r[c] for c in cols})
    (OUT / (stem + ".md")).write_text(
        "# %s.csv / %s.png\n\n%s\n\nSource: curve.csv (slice). Generated-By: src/analyze.py + src/meta.py. Git-Commit: %s\n"
        % (stem, stem, desc, GIT))

base = ["model", "layer", "lens_acc", "probe_acc", "final_acc", "chance", "n"]
qwen = [r for r in rows if r["model"].startswith("qwen")]
w("figure_main", qwen, base, "Qwen2.5-7B base vs instruct: logit-lens and linear-probe MMLU accuracy across layers, with each model's final-output accuracy and chance.")
w("figure_layers_models", rows, base, "Linear-probe MMLU accuracy across layers for all four models (base and instruct).")
w("figure_lens_vs_probe", [r for r in rows if r["model"] == "qwen2.5-7b-base"], base, "Qwen2.5-7B base: logit-lens (own basis) vs trained linear probe (any basis) across layers.")
w("figure_gap", rows, base, "Base final-output vs base best mid-layer probe vs instruct final-output accuracy per model (the elicitation gap).")
# figure_unlocked is backed by the unlocked eval_points sections; emit a small derived csv
unl = []
import collections
acc = collections.defaultdict(lambda: [0, 0])
for line in open(OUT / "eval_points.jsonl"):
    r = json.loads(line)
    if "unlocked" in r["section"]:
        acc[r["section"]][1] += 1
        if r["pred"] == r["gold"]:
            acc[r["section"]][0] += 1
with open(OUT / "figure_unlocked.csv", "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["section", "correct", "n", "accuracy", "chance"])
    for s, (c, n) in sorted(acc.items()):
        wr.writerow([s, c, n, "%.4f" % (c / n if n else 0), 0.25])
(OUT / "figure_unlocked.md").write_text(
    "# figure_unlocked.csv / figure_unlocked.png\n\nBase-model mid-layer logit-lens and linear-probe accuracy on the RL-unlocked subset (instruct right, base output wrong), per model, vs chance 0.25.\n\nSource: eval_points.jsonl (unlocked sections). Generated-By: src/analyze.py + src/meta.py. Git-Commit: %s\n" % GIT)

(OUT / "sources.json").write_text(json.dumps(
    {"metrics": {"*": {"csv": "curve.csv"}}, "per_example": ["eval_points.jsonl"]}, indent=2))
print("wrote sidecars + per-figure csv/md + sources.json")
