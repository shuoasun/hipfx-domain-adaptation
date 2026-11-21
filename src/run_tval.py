#!/usr/bin/env python3
"""Target-validation re-run. Same splits, seeds, random-search stream and trials as run_chunk.py (so the proxy-selected
model is reproduced exactly), but every trial's predictions are saved, and a labelled TARGET VALIDATION set is carved
from the target training pool (stratified, 25%, disjoint from the test set). Its labels are used only to choose among
trials; the test set is never used for selection. Three selection rules are then compared from the same models:
  proxy  (label-free, as in the paper), tval (target-validation labels, legitimate), oracle (test labels; upper bound only)
  python run_tval.py --worker 0 --nworkers 4 --budget 1500 [--jobs ...]     |  --status  |  --merge
"""
import os, sys, json, time, argparse
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(HERE), "data")); RES = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(HERE), "results")); sys.path.insert(0, HERE); sys.path.insert(0, HERE)
SUF = os.environ.get("JOB_SUFFIX", "_fu2") + "_tval" + os.environ.get("TVAL_TAG", "")
import importlib.util
spec = importlib.util.spec_from_file_location("rc_cfg", os.path.join(HERE, "run_chunk.py"))
src = open(os.path.join(HERE, "run_chunk.py")).read(); cfg = src[:src.index("ap = argparse.ArgumentParser()")]
G = {"__file__": os.path.join(HERE, "run_chunk.py")}; exec(cfg, G)
JOBS, N_REP, SEEDS, N_TRIALS = G["JOBS"], G["N_REP"], G["SEEDS"], G["N_TRIALS"]
VAL_FRAC = float(os.environ.get("TVAL_FRAC", "0.25"))
N_TRIALS = int(os.environ.get("N_TRIALS_OVERRIDE", N_TRIALS))
ap = argparse.ArgumentParser()
ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--budget", type=float, default=1500); ap.add_argument("--status", action="store_true")
ap.add_argument("--merge", action="store_true"); ap.add_argument("--jobs", nargs="*", default=["female_primary", "male_primary", "female_C1_std_all", "male_C1_std_all"])
a = ap.parse_args(); NV = 8
def upath(job, rep, seed, vi): return os.path.join(RES, job+SUF, "units", f"rep{rep}_seed{seed}_v{vi}.json")
ALL = [(job, rep, seed, vi) for rep in range(N_REP) for seed in SEEDS for vi in range(NV) for job in a.jobs]
if a.status:
    for job in a.jobs: print(job, f"{sum(os.path.exists(upath(*u)) for u in ALL if u[0]==job)}/{N_REP*len(SEEDS)*NV}")
    sys.exit()
if a.merge:
    import pandas as pd
    for job in a.jobs:
        rows = [json.load(open(upath(*u))) for u in ALL if u[0]==job and os.path.exists(upath(*u))]
        print(job, len(rows))
        if len(rows) == N_REP*len(SEEDS)*NV: pd.DataFrame(rows).to_csv(os.path.join(RES, job+SUF, "trials.csv"), index=False)
    sys.exit()
mine = [u for u in ALL if not os.path.exists(upath(*u))][a.worker::a.nworkers]
if not mine: print("nothing pending"); sys.exit()
import numpy as np, torch, io, contextlib
torch.set_num_threads(1)
import da_bmd_sensitivity_std as M
from sklearn.metrics import roc_auc_score
_data, _split = {}, {}
def get_data(job):
    if job not in _data:
        j = JOBS[job]; os.makedirs(os.path.join(RES, job+SUF, "units"), exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            _data[job] = M.prepare_source_target(j["sof"], j["uk"], "HA_HIPFX", j["drop"], j["complete"], j["std"])
    return _data[job]
def get_split(job, rep):
    if (job, rep) not in _split:
        d = get_data(job)
        u_idx, te_idx = M.build_full_random_split(d.yt, rep_seed=12000+rep, test_frac=0.2)
        rng = np.random.default_rng(55000 + rep); yu = d.yt[u_idx]
        val = np.concatenate([rng.choice(u_idx[yu == c], int(round((yu == c).sum()*VAL_FRAC)), replace=False) for c in (0, 1)])
        Xs_tr_raw, Xs_val_raw, ys_tr, ys_val = M.split_source_train_val(d.Xs, d.ys, seed=7000+rep, val_frac=0.2)
        pp = M.fit_preprocess(Xs_tr_raw); A_ = lambda X: M.apply_preprocess(pp, X)
        _split[(job, rep)] = (A_(Xs_tr_raw), ys_tr, A_(Xs_val_raw), ys_val, A_(d.Xt[u_idx]), A_(d.Xt[te_idx]), d.yt[te_idx].astype(int),
                              A_(d.Xt[val]), d.yt[val].astype(int))
    return _split[(job, rep)]
est = 75.0; done = 0
for (job, rep, seed, vi) in mine:
    if os.path.exists(upath(job, rep, seed, vi)): continue
    if time.time() - T0 + est*(0.55 if vi == 0 else 1.0)*1.15 > a.budget: continue
    t1 = time.time()
    Xs_tr, ys_tr, Xs_val, ys_val, Xt_u, Xt_te, yt_te, Xt_v, yt_v = get_split(job, rep)
    rng = np.random.default_rng(100000 + rep*100 + seed); hps = []
    for vname, flags in M.VARIANTS:
        trials = N_TRIALS if vname != "Baseline" else max(4, N_TRIALS//2)
        hps.append([M.sample_hp_mid(rng, flags) for _ in range(trials)])
    vname, flags = M.VARIANTS[vi]; tr_rows = []
    for tr, hp in enumerate(hps[vi]):
        diag, feat, clf = M.train_one_trial(flags, hp, Xs_tr, ys_tr, Xs_val, ys_val, Xt_u, "cpu", seed=seed + 37*tr + 1000*rep)
        pv = M.predict_probs(feat, clf, Xt_v, device="cpu"); pt = M.predict_probs(feat, clf, Xt_te, device="cpu")
        tr_rows.append(dict(trial=tr, proxy=float(diag["proxy"]), src_val_auc=float(diag["src_val_auc"]),
                            tval_auc=float(roc_auc_score(yt_v, pv)), test_auc=float(roc_auc_score(yt_te, pt)),
                            hp=hp, test_scores=[round(float(x), 6) for x in pt]))
    row = dict(rep=rep, seed=seed, Variant=vname, n_tval=int(len(yt_v)), cases_tval=int(yt_v.sum()), cases_te=int(yt_te.sum()),
               tgt_y_true_json=json.dumps(yt_te.tolist()), trials=tr_rows)
    tmp = upath(job, rep, seed, vi) + f".tmp{a.worker}"; json.dump(row, open(tmp, "w")); os.replace(tmp, upath(job, rep, seed, vi))
    dt = time.time() - t1; done += 1
    est = max(est*0.5 + dt*0.5, dt) if vi > 0 else max(est, dt/0.55*0.8)
    b = lambda k: max(tr_rows, key=lambda r: r[k])["test_auc"]
    print(f"w{a.worker} {job} rep{rep} seed{seed} {vname} proxy={b('proxy'):.3f} tval={b('tval_auc'):.3f} oracle={b('test_auc'):.3f} {dt:.0f}s", flush=True)
print(f"w{a.worker} done={done} elapsed={time.time()-T0:.0f}s")
