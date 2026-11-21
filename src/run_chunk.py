#!/usr/bin/env python3
"""Analysis definitions (JOBS) and resumable runs with label-free proxy model selection. Uses the functions of
da_bmd_sensitivity_std.py (same splits, seeds, random search and proxy selection) with per-(rep, seed, variant)
checkpointing, so that runs can be split into chunks and resumed.
  python run_chunk.py --worker 0 --nworkers 4 --budget 165     # run pending units for up to ~budget seconds
  python run_chunk.py --status                                 # progress
  python run_chunk.py --merge                                  # write <job>/results_best_with_scores.csv when a job is complete
"""
import os, sys, json, time, argparse
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(HERE), "data")); RES = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(HERE), "results"))
sys.path.insert(0, HERE)
BMD = ["HA_THD", "HA_LSD", "HA_FND"]
SUF = os.environ.get("JOB_SUFFIX", "")  # e.g. JOB_SUFFIX=_fu2 keeps results of the follow-up-restricted rerun separate
SOF = os.path.join(ROOT, "SOF_for_DA_enriched_smokefix.csv"); UKF = os.path.join(ROOT, os.environ.get("UKF_NAME","UKfemale_for_DA_aligned.csv"))
MROS = os.path.join(ROOT, "MrOS_for_DA_rawBMD.csv"); UKM = os.path.join(ROOT, os.environ.get("UKM_NAME","UKmale_for_DA_aligned.csv"))
JOBS = {
    "female_primary":        dict(sof=SOF, uk=UKF, drop=[], complete=[], std=[]),
    "male_primary":          dict(sof=MROS, uk=UKM, drop=[], complete=[], std=[]),
    "male_A_noBMD":          dict(sof=MROS, uk=UKM, drop=BMD, complete=[], std=[]),
    "male_B_completeBMD":    dict(sof=MROS, uk=UKM, drop=[], complete=BMD, std=[]),
    "female_A_noBMD":        dict(sof=SOF, uk=UKF, drop=BMD, complete=[], std=[]),
    "female_B_completeBMD":  dict(sof=SOF, uk=UKF, drop=[], complete=BMD, std=[]),
    "female_C1_std_all":     dict(sof=SOF, uk=UKF, drop=[], complete=[], std=BMD),
    "female_C2_std_complete":dict(sof=SOF, uk=UKF, drop=[], complete=BMD, std=BMD),
    # male C re-run from the original-scale MrOS BMD file (the earlier male C runs started from MrOS_for_DA_aligned.csv)
    "male_C1_std_all":       dict(sof=MROS, uk=UKM, drop=[], complete=[], std=BMD),
    "male_C2_std_complete":  dict(sof=MROS, uk=UKM, drop=[], complete=BMD, std=BMD),
    # exploratory female feature sets (selection on the target validation set only; see run_tval.py)
    "female_C1_noGrip":      dict(sof=SOF, uk=UKF, drop=["HA_GRIPAVG"], complete=[], std=BMD),
    "female_C1_core":        dict(sof=SOF, uk=UKF, drop=["HA_GRIPAVG","WALK_ORD","SMOKE_ORD","HA_HEIGHT","HA_WEIGHT"], complete=[], std=BMD),
}
N_REP, SEEDS, N_TRIALS = 6, [0, 1, 2], 10
ap = argparse.ArgumentParser()
ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--budget", type=float, default=165); ap.add_argument("--status", action="store_true")
ap.add_argument("--merge", action="store_true"); ap.add_argument("--jobs", nargs="*", default=list(JOBS))
a = ap.parse_args()
NV = 8
def upath(job, rep, seed, vi): return os.path.join(RES, job+SUF, "units", f"rep{rep}_seed{seed}_v{vi}.json")
ALL = [(job, rep, seed, vi) for rep in range(N_REP) for seed in SEEDS for vi in range(NV) for job in a.jobs]
pending = [u for u in ALL if not os.path.exists(upath(*u))]
if a.status:
    for job in a.jobs:
        n = sum(os.path.exists(upath(job, r, s, v)) for r in range(N_REP) for s in SEEDS for v in range(NV))
        print(job, f"{n}/{N_REP*len(SEEDS)*NV}")
    sys.exit()
if a.merge:
    import pandas as pd
    for job in a.jobs:
        rows = []
        for r in range(N_REP):
            for s in SEEDS:
                for v in range(NV):
                    if os.path.exists(upath(job, r, s, v)): rows.append(json.load(open(upath(job, r, s, v))))
        if len(rows) == N_REP*len(SEEDS)*NV:
            pd.DataFrame(rows).to_csv(os.path.join(RES, job+SUF, "results_best_with_scores.csv"), index=False); print("merged", job)
        else: print("incomplete", job, len(rows))
    sys.exit()
mine = pending[a.worker::a.nworkers]
if not mine: print("nothing pending"); sys.exit()
import numpy as np, torch
torch.set_num_threads(1)
import da_bmd_sensitivity_std as M
_data, _split = {}, {}
def get_data(job):
    if job not in _data:
        j = JOBS[job]; os.makedirs(os.path.join(RES, job+SUF, "units"), exist_ok=True)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            d = M.prepare_source_target(j["sof"], j["uk"], "HA_HIPFX", j["drop"], j["complete"], j["std"])
        _data[job] = d
        info = os.path.join(RES, job+SUF, "run_info.json")
        if not os.path.exists(info):
            json.dump(dict(feat_names=d.feat_names, n_target=int(len(d.yt)), n_cases=int(d.yt.sum()), job=j,
                           n_rep=N_REP, seeds=SEEDS, n_trials=N_TRIALS, torch=torch.__version__), open(info, "w"), indent=1)
    return _data[job]
def get_split(job, rep):
    if (job, rep) not in _split:
        d = get_data(job)
        u_idx, te_idx = M.build_full_random_split(d.yt, rep_seed=12000+rep, test_frac=0.2)
        Xs_tr_raw, Xs_val_raw, ys_tr, ys_val = M.split_source_train_val(d.Xs, d.ys, seed=7000+rep, val_frac=0.2)
        pp = M.fit_preprocess(Xs_tr_raw)
        _split[(job, rep)] = (M.apply_preprocess(pp, Xs_tr_raw), ys_tr, M.apply_preprocess(pp, Xs_val_raw), ys_val,
                              M.apply_preprocess(pp, d.Xt[u_idx]), M.apply_preprocess(pp, d.Xt[te_idx]), d.yt[te_idx].astype(int))
    return _split[(job, rep)]
est = 75.0; done = 0
for (job, rep, seed, vi) in mine:
    if os.path.exists(upath(job, rep, seed, vi)): continue
    need = est * (0.55 if vi == 0 else 1.0) * 1.15
    if time.time() - T0 + need > a.budget: continue
    t1 = time.time()
    Xs_tr, ys_tr, Xs_val, ys_val, Xt_u, Xt_te, yt_te = get_split(job, rep)
    rng = np.random.default_rng(100000 + rep*100 + seed)
    hps = []
    for vname, flags in M.VARIANTS:                       # replay the random search stream exactly as in the original loop
        trials = N_TRIALS if vname != "Baseline" else max(4, N_TRIALS//2)
        hps.append([M.sample_hp_mid(rng, flags) for _ in range(trials)])
    vname, flags = M.VARIANTS[vi]
    best = None
    for tr, hp in enumerate(hps[vi]):
        diag, feat, clf = M.train_one_trial(flags, hp, Xs_tr, ys_tr, Xs_val, ys_val, Xt_u, "cpu", seed=seed + 37*tr + 1000*rep)
        if best is None or diag["proxy"] > best[0]: best = (diag["proxy"], hp, feat, clf, diag)
    p_te = M.predict_probs(best[2], best[3], Xt_te, device="cpu")
    met = M.metrics_from_scores(yt_te, p_te, thr=0.5)
    row = {"rep": rep, "seed": seed, "Variant": vname, **{k: met[k] for k in ["AUC", "ACC", "SEN", "SPE", "PREC", "F1"]},
           "proxy": float(best[0]), "src_val_auc": best[4]["src_val_auc"], "dom_acc": best[4]["dom_acc"], "tgt_entropy": best[4]["tgt_entropy"],
           "cases_te": int((yt_te == 1).sum()), "best_hp_json": json.dumps(best[1]),
           "tgt_y_true_json": json.dumps(yt_te.tolist()), "tgt_scores_json": json.dumps(p_te.tolist())}
    tmp = upath(job, rep, seed, vi) + f".tmp{a.worker}"
    json.dump(row, open(tmp, "w")); os.replace(tmp, upath(job, rep, seed, vi))
    dt = time.time() - t1; done += 1
    if vi > 0: est = max(est*0.5 + dt*0.5, dt)
    else: est = max(est, dt/0.55*0.8)
    print(f"w{a.worker} {job} rep{rep} seed{seed} {vname} AUC={met['AUC']:.3f} {dt:.0f}s", flush=True)
print(f"w{a.worker} done={done} elapsed={time.time()-T0:.0f}s")
