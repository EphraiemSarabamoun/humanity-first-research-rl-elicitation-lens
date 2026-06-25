# eval_points.jsonl — sidecar (REPRO_CONTRACT)

Generated-By: src/analyze.py
Command: python3 src/analyze.py --acts_dir results/real/acts --out_dir results/real
Git-Commit: ae0598b0916f5f9a0f08a3cc877c8e60872dbaa1
Seeds: 42 (MMLU sampling in extract_lens.py; 5-fold StratifiedKFold for the probe; per-fold PCA(256); 2000-resample percentile bootstrap for CIs)
Source-Data: results/real/acts/lens_<model>.npz (last-token residual-stream activations at selected layers + logit-lens option logits + final-output prediction, Qwen2.5-7B base/instruct and Llama-3.1-8B base/instruct, RTX 5090, 2026-06-24, torch 2.12 cu130; built from cais/mmlu test split, 2000 items, via src/extract_lens.py with a fixed raw multiple-choice prompt)
Analysis-Command: cd results/real && python3 recompute.py  (rebuilds every accuracy + 95% bootstrap CI from these rows)
Columns:
  section (which summary arm this row backs, e.g. qwen_base_probe, llama_unlocked_lens);
  eval_order (integer position within the arm, for seeded-bootstrap replay);
  pred (predicted option index 0-3); gold (correct option index 0-3)
