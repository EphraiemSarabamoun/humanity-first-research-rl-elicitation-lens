"""analyze_lens.py — logit-lens / linear-probe decodability of MMLU answers
from base vs RL/instruct mid-layer hidden states.

For each model and layer: logit-lens accuracy (argmax of unembed-projected
option logits), 5-fold OOF linear-probe accuracy, and the model's final-output
accuracy. For each base<->instruct pair: the RL-unlocked subset (instruct
right, base final output wrong) and the base model's mid-layer decodability on it.

Writes results/real/{curve.csv, eval_points.jsonl} + sidecars, sources.json,
analysis_summary.txt (via recompute.py), figures. Python 3.10 safe.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PROBE_PCA = 256  # reduce residual dim before the linear probe (fit per-fold, no leakage)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAIRS = [("qwen2.5-7b", "qwen2.5-7b-base", "qwen2.5-7b-instruct"),
         ("llama-3.1-8b", "llama-3.1-8b-base", "llama-3.1-8b-instruct")]
SEED = 42
CHANCE = 0.25


def load(acts_dir, label):
    d = np.load(acts_dir / ("lens_%s.npz" % label), allow_pickle=True)
    layers = [int(x) for x in d["layers"]]
    return d, layers


def probe_oof(X, y, seed=SEED):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    k = min(PROBE_PCA, X.shape[1], X.shape[0] - 1)
    clf = make_pipeline(StandardScaler(),
                        PCA(n_components=k, random_state=seed),
                        LogisticRegression(max_iter=2000, C=1.0))
    return cross_val_predict(clf, X, y, cv=skf, n_jobs=5)  # predicted class per item


def analyze_model(d, layers):
    gold = d["gold"].astype(int)
    final_pred = d["final_pred"].astype(int)
    per_layer = {}
    for li in layers:
        lens = d["lens_L%02d" % li]            # [n,4]
        lens_pred = lens.argmax(axis=1)
        H = d["hid_L%02d" % li].astype(np.float64)
        oof = probe_oof(H, gold)
        per_layer[li] = {
            "lens_pred": lens_pred,
            "probe_pred": oof,
            "lens_acc": float((lens_pred == gold).mean()),
            "probe_acc": float((oof == gold).mean()),
        }
    final_acc = float((final_pred == gold).mean())
    return {"gold": gold, "final_pred": final_pred, "final_acc": final_acc,
            "per_layer": per_layer, "layers": layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_dir", default="results/real/acts")
    ap.add_argument("--out_dir", default="results/real")
    args = ap.parse_args()
    acts_dir = Path(args.acts_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    M = {}
    for _pair, base, inst in PAIRS:
        for label in (base, inst):
            fn = acts_dir / ("lens_%s.npz" % label)
            if not fn.exists():
                print("[skip] missing", label, flush=True); continue
            d, layers = load(acts_dir, label)
            M[label] = analyze_model(d, layers)
            print("[done]", label, "final_acc=%.3f" % M[label]["final_acc"], flush=True)

    # headline mid-layer per model = argmax base/inst probe_acc over non-final layers
    def headline_mid(res):
        ls = res["layers"][:-1] if len(res["layers"]) > 1 else res["layers"]
        return max(ls, key=lambda li: res["per_layer"][li]["probe_acc"])

    rows = []
    for label, res in M.items():
        for li in res["layers"]:
            pl = res["per_layer"][li]
            rows.append({"model": label, "layer": li, "lens_acc": pl["lens_acc"],
                         "probe_acc": pl["probe_acc"], "final_acc": res["final_acc"],
                         "chance": CHANCE, "n": len(res["gold"])})
    import csv as _csv
    cols = ["model", "layer", "lens_acc", "probe_acc", "final_acc", "chance", "n"]
    with open(out / "curve.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: ("%.6f" % r[c] if isinstance(r[c], float) else r[c]) for c in cols})

    # cited eval_points
    ep = []
    def add(section, pred, gold, idx=None):
        if idx is None:
            idx = range(len(gold))
        for k, i in enumerate(idx):
            ep.append({"section": section, "eval_order": k,
                       "pred": int(pred[i]), "gold": int(gold[i])})

    headline = {}
    unlocked_info = {}
    for pair, base, inst in PAIRS:
        if base not in M or inst not in M:
            continue
        rb, ri = M[base], M[inst]
        hb = headline_mid(rb)
        headline[base] = hb
        gold = rb["gold"]
        # unlocked subset: instruct final right AND base final wrong
        unlocked = np.where((ri["final_pred"] == gold) & (rb["final_pred"] != gold))[0]
        unlocked_info[pair] = {"n": int(len(unlocked)), "mid_layer": hb,
                               "instruct_acc": ri["final_acc"], "base_final_acc": rb["final_acc"]}
        pfx = "qwen" if pair.startswith("qwen") else "llama"
        add("%s_base_final" % pfx, rb["final_pred"], gold)
        add("%s_base_lens" % pfx, rb["per_layer"][hb]["lens_pred"], gold)
        add("%s_base_probe" % pfx, rb["per_layer"][hb]["probe_pred"], gold)
        add("%s_instruct_final" % pfx, ri["final_pred"], gold)
        add("%s_unlocked_lens" % pfx, rb["per_layer"][hb]["lens_pred"], gold, unlocked)
        add("%s_unlocked_probe" % pfx, rb["per_layer"][hb]["probe_pred"], gold, unlocked)

    with open(out / "eval_points.jsonl", "w") as f:
        for r in ep:
            f.write(json.dumps(r) + "\n")

    # summary via recompute (reproducible by construction)
    with open(out / "analysis_summary.txt", "w") as f:
        subprocess.run(["python3", "recompute.py"], cwd=str(out), stdout=f, check=True)

    (out / "headline_layers.json").write_text(json.dumps(
        {"headline_mid": headline, "unlocked": unlocked_info}, indent=2))
    make_figures(M, headline, unlocked_info, out)
    print("[analyze] complete. headline mid-layers:", headline, flush=True)
    print("unlocked:", unlocked_info, flush=True)


def make_figures(M, headline, unlocked_info, out):
    qb, qi = "qwen2.5-7b-base", "qwen2.5-7b-instruct"
    # Fig 1: Qwen base vs instruct, probe + lens across layers, with output-acc lines
    if qb in M and qi in M:
        fig, ax = plt.subplots(figsize=(7.6, 4.7))
        rb, ri = M[qb], M[qi]
        L = rb["layers"]
        ax.plot(L, [rb["per_layer"][li]["probe_acc"] for li in L], "o-", color="#1b9e77", label="base linear-probe")
        ax.plot(L, [rb["per_layer"][li]["lens_acc"] for li in L], "s--", color="#1b9e77", alpha=0.6, label="base logit-lens")
        ax.plot(L, [ri["per_layer"][li]["probe_acc"] for li in ri["layers"]], "o-", color="#d95f02", label="instruct linear-probe")
        ax.axhline(rb["final_acc"], color="#1b9e77", ls=":", lw=1.4, label="base final-output acc=%.2f" % rb["final_acc"])
        ax.axhline(ri["final_acc"], color="#d95f02", ls=":", lw=1.4, label="instruct final-output acc=%.2f" % ri["final_acc"])
        ax.axhline(CHANCE, color="k", ls=":", lw=0.8, label="chance")
        ax.set_xlabel("residual-stream layer"); ax.set_ylabel("MMLU accuracy")
        ax.set_ylim(0, 1.02); ax.set_title("Qwen2.5-7B: the MMLU answer is decodable from base mid-layers\nwell above what the base model outputs")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(out / "figure_main.png", dpi=150); plt.close(fig)

    # Fig 2: unlocked-subset base decodability bars
    fig, ax = plt.subplots(figsize=(7, 4.4))
    pairs = [p for p in ("qwen2.5-7b", "llama-3.1-8b") if p in unlocked_info]
    x = np.arange(len(pairs)); wd = 0.35
    lens_v, probe_v, ns = [], [], []
    for pair in pairs:
        pfx = "qwen" if pair.startswith("qwen") else "llama"
        base = pair + "-base"; hb = headline[base]; res = M[base]
        info = unlocked_info[pair]; un = info["n"]
        gold = res["gold"]
        ri = M[pair + "-instruct"]
        idx = np.where((ri["final_pred"] == gold) & (res["final_pred"] != gold))[0]
        lens_v.append(float((res["per_layer"][hb]["lens_pred"][idx] == gold[idx]).mean()))
        probe_v.append(float((res["per_layer"][hb]["probe_pred"][idx] == gold[idx]).mean()))
        ns.append(un)
    ax.bar(x - wd/2, lens_v, wd, color="#7570b3", label="logit-lens")
    ax.bar(x + wd/2, probe_v, wd, color="#1b9e77", label="linear-probe")
    ax.axhline(CHANCE, color="k", ls=":", lw=1, label="chance (0.25)")
    ax.set_xticks(x); ax.set_xticklabels(["%s\n(n_unlocked=%d)" % (p, n) for p, n in zip(pairs, ns)])
    ax.set_ylabel("base mid-layer accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("RL-unlocked items (instruct right, base output wrong):\nthe answer is already in the base model's mid layers")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "figure_unlocked.png", dpi=150); plt.close(fig)

    # Fig 3: probe_acc across layers, all models
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {"qwen2.5-7b-base": "#1b9e77", "qwen2.5-7b-instruct": "#d95f02",
              "llama-3.1-8b-base": "#66c2a5", "llama-3.1-8b-instruct": "#fc8d62"}
    for label, res in M.items():
        L = res["layers"]
        ax.plot(L, [res["per_layer"][li]["probe_acc"] for li in L], "o-",
                color=colors.get(label, None), label=label)
    ax.axhline(CHANCE, color="k", ls=":", lw=0.8)
    ax.set_xlabel("residual-stream layer"); ax.set_ylabel("linear-probe MMLU accuracy (5-fold OOF)")
    ax.set_ylim(0, 1.02); ax.set_title("Mid-layer linear decodability of the MMLU answer, base vs instruct")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "figure_layers_models.png", dpi=150); plt.close(fig)

    # Fig 4: logit-lens vs probe across layers, Qwen base
    if qb in M:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        rb = M[qb]; L = rb["layers"]
        ax.plot(L, [rb["per_layer"][li]["lens_acc"] for li in L], "s-", color="#7570b3", label="logit-lens (own basis)")
        ax.plot(L, [rb["per_layer"][li]["probe_acc"] for li in L], "o-", color="#1b9e77", label="linear-probe (any basis)")
        ax.axhline(CHANCE, color="k", ls=":", lw=0.8)
        ax.set_xlabel("residual-stream layer"); ax.set_ylabel("MMLU accuracy")
        ax.set_ylim(0, 1.02); ax.set_title("Qwen2.5-7B base: logit-lens vs trained linear probe")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "figure_lens_vs_probe.png", dpi=150); plt.close(fig)

    # Fig 5: elicitation gap bars per model
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pairs = [p for p in ("qwen2.5-7b", "llama-3.1-8b") if (p + "-base") in M]
    x = np.arange(len(pairs)); wd = 0.27
    bf = [M[p + "-base"]["final_acc"] for p in pairs]
    bp = [M[p + "-base"]["per_layer"][headline[p + "-base"]]["probe_acc"] for p in pairs]
    inf = [M[p + "-instruct"]["final_acc"] for p in pairs]
    ax.bar(x - wd, bf, wd, color="#bdbdbd", label="base final-output")
    ax.bar(x, bp, wd, color="#1b9e77", label="base best mid-layer probe")
    ax.bar(x + wd, inf, wd, color="#d95f02", label="instruct final-output")
    ax.axhline(CHANCE, color="k", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(pairs); ax.set_ylabel("MMLU accuracy"); ax.set_ylim(0, 1.0)
    ax.set_title("The elicitation gap: base output vs base internal vs instruct output")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "figure_gap.png", dpi=150); plt.close(fig)
    print("[figures] wrote 5 figures", flush=True)


if __name__ == "__main__":
    main()
