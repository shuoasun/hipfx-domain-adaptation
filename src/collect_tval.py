#!/usr/bin/env python3
"""Collect the per-trial outputs of run_tval.py into one CSV per selection rule, in the format read by ci_participant.py.
  python collect_tval.py --job female_primary [--suffix _fu2_tval] --out_dir ../results/ci
Writes <job><suffix>_{proxy,tval,oracle}.csv: the trial chosen by the label-free proxy score, by the target-validation AUC
(main analysis) or by the test AUC (upper bound only; never used for reporting)."""
import os, glob, json, argparse, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--job", required=True); ap.add_argument("--suffix", default="_fu2_tval")
ap.add_argument("--results_dir", default=os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")))
ap.add_argument("--out_dir", required=True); a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
units = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.results_dir, a.job + a.suffix, "units", "*.json")))]
for rule, key in (("proxy", "proxy"), ("tval", "tval_auc"), ("oracle", "test_auc")):
    rows = []
    for u in units:
        b = max(u["trials"], key=lambda t: t[key])
        rows.append(dict(rep=u["rep"], seed=u["seed"], Variant=u["Variant"], AUC=b["test_auc"], ACC=float("nan"), SEN=float("nan"), SPE=float("nan"),
                         PREC=float("nan"), F1=float("nan"), best_hp_json=json.dumps(b["hp"]), tgt_y_true_json=u["tgt_y_true_json"], tgt_scores_json=json.dumps(b["test_scores"])))
    out = os.path.join(a.out_dir, f"{a.job}{a.suffix}_{rule}.csv"); pd.DataFrame(rows).to_csv(out, index=False); print("wrote", out, len(rows))
