# curve.csv — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py
Command: python3 src/analyze.py --acts_dir results/real/acts --out_dir results/real
Git-Commit: ae0598b0916f5f9a0f08a3cc877c8e60872dbaa1
Seeds: 42 (MMLU sampling in extract_lens.py; 5-fold StratifiedKFold for the probe; per-fold PCA(256); 2000-resample percentile bootstrap for CIs)
Source-Data: results/real/acts/lens_<model>.npz (last-token residual-stream activations at selected layers + logit-lens option logits + final-output prediction, Qwen2.5-7B base/instruct and Llama-3.1-8B base/instruct, RTX 5090, 2026-06-24, torch 2.12 cu130; built from cais/mmlu test split, 2000 items, via src/extract_lens.py with a fixed raw multiple-choice prompt)
Analysis-Command: this file is the figure data of record; the cited headline numbers are reproduced by `cd results/real && python3 recompute.py | diff - analysis_summary.txt` (empty) from eval_points.jsonl
Columns:
  model (HF model label: qwen2.5-7b/llama-3.1-8b, base or instruct);
  layer (residual-stream block index into hidden_states);
  lens_acc (logit-lens MMLU accuracy at this layer: final-norm + unembedding applied to the last-token residual, argmax over the four option tokens A/B/C/D, unitless 0-1);
  probe_acc (5-fold out-of-fold multinomial logistic-regression accuracy on PCA-256 of the last-token residual, 4-way, 0-1);
  final_acc (the model's own final-output MMLU accuracy under the raw prompt, constant per model, 0-1);
  chance (0.25); n (number of MMLU items evaluated)
