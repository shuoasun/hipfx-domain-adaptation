#!/usr/bin/env python3
"""Step 2: target fine-tuning (supervised transfer). For each (rep, seed, variant) the trial chosen with the target
validation set in run_tval.py (10 trials) is retrained exactly (same seed), then fine-tuned on the labelled target
training pool (target training data minus the validation set). Fine-tuning learning rate and number of epochs are
chosen on the target validation set; the test set is used once.
  python run_ft.py --worker 0 --nworkers 4 --budget 1500 [--jobs ...]  |  --status  |  --merge
"""
import os, sys, json, time, argparse
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(HERE), "data")); RES = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(HERE), "results")); sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "run_chunk.py")).read(); G = {"__file__": os.path.join(HERE, "run_chunk.py")}
exec(src[:src.index("ap = argparse.ArgumentParser()")], G)
JOBS, N_REP, SEEDS = G["JOBS"], G["N_REP"], G["SEEDS"]
TV = os.environ.get("JOB_SUFFIX", "_fu2") + "_tval"; SUF = os.environ.get("JOB_SUFFIX", "_fu2") + "_ft"; VAL_FRAC = 0.25
ap = argparse.ArgumentParser()
ap.add_argument("--worker", type=int, default=0); ap.add_argument("--nworkers", type=int, default=1)
ap.add_argument("--budget", type=float, default=1500); ap.add_argument("--status", action="store_true")
ap.add_argument("--merge", action="store_true"); ap.add_argument("--jobs", nargs="*", default=["female_primary", "female_C1_std_all", "male_primary", "male_C1_std_all"])
a = ap.parse_args(); NV = 8
def upath(job, rep, seed, vi): return os.path.join(RES, job+SUF, "units", f"rep{rep}_seed{seed}_v{vi}.json")
ALL = [(job, rep, seed, vi) for job in a.jobs for rep in range(N_REP) for seed in SEEDS for vi in range(NV)]
if a.status:
    for job in a.jobs: print(job, f"{sum(os.path.exists(upath(*u)) for u in ALL if u[0]==job)}/{N_REP*len(SEEDS)*NV}")
    sys.exit()
if a.merge:
    import pandas as pd
    for job in a.jobs:
        rows = [json.load(open(upath(*u))) for u in ALL if u[0]==job and os.path.exists(upath(*u))]
        print(job, len(rows))
        if len(rows) == N_REP*len(SEEDS)*NV: pd.DataFrame(rows).to_json(os.path.join(RES, job+SUF, "ft.json"))
    sys.exit()
mine = [u for u in ALL if not os.path.exists(upath(*u))][a.worker::a.nworkers]
if not mine: print("nothing pending"); sys.exit()
import numpy as np, torch, io, contextlib, copy
torch.set_num_threads(1)
import da_bmd_sensitivity_std as M
from sklearn.metrics import roc_auc_score
_data, _split = {}, {}
def get_split(job, rep):
    if (job, rep) not in _split:
        if job not in _data:
            j = JOBS[job]; os.makedirs(os.path.join(RES, job+SUF, "units"), exist_ok=True)
            with contextlib.redirect_stdout(io.StringIO()):
                _data[job] = M.prepare_source_target(j["sof"], j["uk"], "HA_HIPFX", j["drop"], j["complete"], j["std"])
        d = _data[job]
        u_idx, te_idx = M.build_full_random_split(d.yt, rep_seed=12000+rep, test_frac=0.2)
        rng = np.random.default_rng(55000 + rep); yu = d.yt[u_idx]
        val = np.concatenate([rng.choice(u_idx[yu == c], int(round((yu == c).sum()*VAL_FRAC)), replace=False) for c in (0, 1)])
        ft = np.setdiff1d(u_idx, val)
        Xs_tr_raw, Xs_val_raw, ys_tr, ys_val = M.split_source_train_val(d.Xs, d.ys, seed=7000+rep, val_frac=0.2)
        pp = M.fit_preprocess(Xs_tr_raw); A_ = lambda X: M.apply_preprocess(pp, X)
        _split[(job, rep)] = dict(Xs_tr=A_(Xs_tr_raw), ys_tr=ys_tr, Xs_val=A_(Xs_val_raw), ys_val=ys_val, Xt_u=A_(d.Xt[u_idx]),
            Xte=A_(d.Xt[te_idx]), yte=d.yt[te_idx].astype(int), Xv=A_(d.Xt[val]), yv=d.yt[val].astype(int), Xf=A_(d.Xt[ft]), yf=d.yt[ft].astype(int))
    return _split[(job, rep)]
def finetune(feat, clf, S, seed):
    best = (roc_auc_score(S["yv"], M.predict_probs(feat, clf, S["Xv"], "cpu")), 0.0, 0, copy.deepcopy(feat.state_dict()), copy.deepcopy(clf.state_dict()))
    f0, c0 = copy.deepcopy(feat.state_dict()), copy.deepcopy(clf.state_dict())
    X = torch.from_numpy(S["Xf"].astype(np.float32)); y = torch.from_numpy(S["yf"].astype(np.float32))
    pw = torch.tensor([(S["yf"] == 0).sum() / max(1, (S["yf"] == 1).sum())], dtype=torch.float32)
    for lr in (1e-4, 3e-4, 1e-3):
        feat.load_state_dict(f0); clf.load_state_dict(c0); M.seed_all(seed)
        opt = torch.optim.Adam(list(feat.parameters()) + list(clf.parameters()), lr=lr, weight_decay=1e-4)
        for ep in range(1, 31):
            feat.train(); clf.train(); perm = torch.randperm(len(y))
            for i in range(0, len(y), 256):
                b = perm[i:i+256]; opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(feat(X[b])).squeeze(-1), y[b], pos_weight=pw); loss.backward(); opt.step()
            if ep % 2 == 0:
                v = roc_auc_score(S["yv"], M.predict_probs(feat, clf, S["Xv"], "cpu"))
                if v > best[0]: best = (v, lr, ep, copy.deepcopy(feat.state_dict()), copy.deepcopy(clf.state_dict()))
    feat.load_state_dict(best[3]); clf.load_state_dict(best[4]); return best[:3]
for (job, rep, seed, vi) in mine:
    if os.path.exists(upath(job, rep, seed, vi)): continue
    if time.time() - T0 > a.budget - 120: break
    t1 = time.time(); S = get_split(job, rep)
    tvp = os.path.join(RES, job+TV, "units", f"rep{rep}_seed{seed}_v{vi}.json")
    if not os.path.exists(tvp): print("skip (run_tval.py unit missing):", tvp); continue
    tv = json.load(open(tvp))
    tr = max(tv["trials"], key=lambda t: t["tval_auc"]); vname, flags = M.VARIANTS[vi]
    diag, feat, clf = M.train_one_trial(flags, tr["hp"], S["Xs_tr"], S["ys_tr"], S["Xs_val"], S["ys_val"], S["Xt_u"], "cpu", seed=seed + 37*tr["trial"] + 1000*rep)
    p0 = M.predict_probs(feat, clf, S["Xte"], "cpu"); auc0 = float(roc_auc_score(S["yte"], p0))
    v, lr, ep = finetune(feat, clf, S, seed + 1000*rep + 7*vi)
    p1 = M.predict_probs(feat, clf, S["Xte"], "cpu")
    row = dict(rep=rep, seed=seed, Variant=vname, trial=tr["trial"], auc_noft=auc0, auc_noft_saved=tr["test_auc"], ft_lr=lr, ft_epochs=ep, ft_tval_auc=v,
               AUC=float(roc_auc_score(S["yte"], p1)), tgt_y_true_json=json.dumps(S["yte"].tolist()), tgt_scores_json=json.dumps([round(float(x), 6) for x in p1]),
               best_hp_json=json.dumps(tr["hp"]), cases_ft=int(S["yf"].sum()))
    tmp = upath(job, rep, seed, vi) + f".tmp{a.worker}"; json.dump(row, open(tmp, "w")); os.replace(tmp, upath(job, rep, seed, vi))
    print(f"w{a.worker} {job} rep{rep} seed{seed} {vname} noFT={auc0:.3f}(saved {tr['test_auc']:.3f}) FT={row['AUC']:.3f} lr={lr} ep={ep} {time.time()-t1:.0f}s", flush=True)
print(f"w{a.worker} elapsed={time.time()-T0:.0f}s")
