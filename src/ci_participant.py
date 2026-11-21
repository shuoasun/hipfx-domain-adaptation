#!/usr/bin/env python3
"""AUC confidence intervals with a participant-level bootstrap, from saved test-set predictions (no retraining).
The 6 repetitions use overlapping random 20% test sets drawn from one target cohort, so participants (not test-set rows)
are resampled: one stratified resample of the whole target cohort (cases and controls separately) is drawn per bootstrap
replicate and applied to every repetition, seed and variant; AUC is averaged over seeds and repetitions.
Also writes mean/SD of run-level metrics, calibration and selected hyperparameters.
  python ci_participant.py --res <results csv> [--extra <conv csv>] --uk <UK csv> [--complete] --out <prefix>
"""
import argparse, json, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--res", required=True); ap.add_argument("--extra", default=None); ap.add_argument("--uk", required=True)
ap.add_argument("--complete", action="store_true"); ap.add_argument("--out", required=True)
ap.add_argument("--B", type=int, default=2000); ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args()
BMD = ["HA_THD", "HA_LSD", "HA_FND"]
uk = pd.read_csv(a.uk); uk.columns = [c.strip().upper() for c in uk.columns]
if a.complete: uk = uk[uk[BMD].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)].reset_index(drop=True)
yt = pd.to_numeric(uk["HA_HIPFX"], errors="coerce").to_numpy().astype(int); N = len(yt)
def test_idx(rep):
    rng = np.random.default_rng(12000 + rep); idx = np.arange(N); rng.shuffle(idx); return idx[:int(round(N*0.2))]
df = pd.read_csv(a.res)
if a.extra:
    ex = pd.read_csv(a.extra); ex = ex[ex.Variant != "Baseline"]; df = pd.concat([df, ex], ignore_index=True)
order = ["Baseline", "+MMD", "+CORAL", "+GRL", "+MMD+CORAL", "+MMD+GRL", "+CORAL+GRL", "+MMD+CORAL+GRL"]
VARS = [v for v in order if v in set(df.Variant)]
df["y"] = df.tgt_y_true_json.map(lambda s: np.array(json.loads(s), dtype=int))
df["p"] = df.tgt_scores_json.map(lambda s: np.array(json.loads(s), dtype=float))
reps = sorted(df.rep.unique()); seeds = sorted(df.seed.unique())
TE = {r: test_idx(r) for r in reps}
for r in reps:
    for yy in df[df.rep == r].y: assert np.array_equal(yy, yt[TE[r]]), f"test set of rep {r} not reproduced"
union = np.unique(np.concatenate(list(TE.values())))
print(f"N={N} cases={yt.sum()} | test sets reproduced | union of test sets: {len(union)} participants, {yt[union].sum()} cases; cases per test set: {[int(yt[TE[r]].sum()) for r in reps]}")
def prep(y, p):
    u, g = np.unique(p, return_inverse=True); return g, len(u)
def wauc(y, g, G, w):
    wn = np.bincount(g, weights=w*(y == 0), minlength=G); below = np.cumsum(wn) - wn
    pos = y == 1; Wp = w[pos].sum(); Wn = wn.sum()
    if Wp == 0 or Wn == 0: return np.nan
    return float((w[pos]*(below[g[pos]] + 0.5*wn[g[pos]])).sum()/(Wp*Wn))
P = {}
for r in reps:
    for v in VARS:
        for s in seeds:
            row = df[(df.rep == r) & (df.Variant == v) & (df.seed == s)]; assert len(row) == 1, (r, v, s, len(row))
            p = row.p.iloc[0]; P[(r, v, s)] = prep(yt[TE[r]], p)
def stat(counts):
    out = np.full((len(reps), len(VARS)), np.nan)
    for i, r in enumerate(reps):
        w = counts[TE[r]].astype(float); y = yt[TE[r]]
        for j, v in enumerate(VARS):
            out[i, j] = np.mean([wauc(y, *P[(r, v, s)], w) for s in seeds])
    return np.nanmean(out, axis=0)
obs = stat(np.ones(N))
rng = np.random.default_rng(a.seed); pos = np.where(yt == 1)[0]; neg = np.where(yt == 0)[0]
boot = np.zeros((a.B, len(VARS)))
for b in range(a.B):
    c = np.zeros(N); c[pos] = rng.multinomial(len(pos), np.ones(len(pos))/len(pos)); c[neg] = rng.multinomial(len(neg), np.ones(len(neg))/len(neg))
    boot[b] = stat(c)
rows = []
m = df.groupby("Variant")[["AUC", "SEN", "SPE", "ACC", "PREC", "F1"]].agg(["mean", "std"])
def logit_fit(x, y, it=50):
    X = np.column_stack([np.ones_like(x), x]); b = np.zeros(2)
    for _ in range(it):
        z = np.clip(X@b, -30, 30); mu = 1/(1+np.exp(-z)); W = mu*(1-mu)+1e-12
        step = np.linalg.solve(X.T@(X*W[:, None])+1e-8*np.eye(2), X.T@(y-mu)); b = b+step
        if np.abs(step).max() < 1e-8: break
    return b
for j, v in enumerate(VARS):
    d = boot[:, j] - boot[:, 0]; g = df[df.Variant == v]; br = []; sl = []; ic = []; mp = []; pr = []
    for _, r in g.iterrows():
        y, p = r.y, np.clip(r.p, 1e-6, 1-1e-6); b0, b1 = logit_fit(np.log(p/(1-p)), y.astype(float))
        br.append(np.mean((p-y)**2)); sl.append(b1); ic.append(b0); mp.append(p.mean()); pr.append(y.mean())
    row = dict(Variant=v, n_runs=len(g), AUC=obs[j], AUC_lo=np.percentile(boot[:, j], 2.5), AUC_hi=np.percentile(boot[:, j], 97.5),
               dAUC=obs[j]-obs[0], dAUC_lo=np.percentile(d, 2.5) if j else np.nan, dAUC_hi=np.percentile(d, 97.5) if j else np.nan,
               boot_SE=boot[:, j].std(ddof=1))
    for k in ["AUC", "SEN", "SPE", "ACC", "PREC", "F1"]: row[k+"_mean"] = m.loc[v, (k, "mean")]; row[k+"_sd"] = m.loc[v, (k, "std")]
    row.update(Brier=np.mean(br), cal_slope=np.mean(sl), cal_intercept=np.mean(ic), mean_pred=np.mean(mp), obs_rate=np.mean(pr))
    if "best_hp_json" in g:
        hp = pd.DataFrame([json.loads(s) for s in g.best_hp_json])
        for k in ["lam_mmd", "lam_coral", "grl_lambda"]: row[k+"_mean"] = hp[k].mean(); row[k+"_sd"] = hp[k].std(); row[k+"_med"] = hp[k].median()
        row["hidden_mode"] = int(hp.hidden.mode()[0]); row["depth_mode"] = int(hp.depth.mode()[0])
    rows.append(row)
out = pd.DataFrame(rows); out.to_csv(a.out + "_ci.csv", index=False)
pd.set_option("display.width", 250)
print(out[["Variant", "AUC", "AUC_lo", "AUC_hi", "dAUC", "dAUC_lo", "dAUC_hi", "boot_SE", "AUC_mean", "AUC_sd", "SEN_mean", "SEN_sd"]].round(3).to_string(index=False))
