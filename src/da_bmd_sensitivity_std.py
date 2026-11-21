#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core components of the domain-adaptation pipeline: data preparation (shared predictors, optional dropped predictors,
within-cohort BMD standardization), splits, preprocessing, the MLP feature extractor, classifier and domain
discriminator, MMD / CORAL / gradient-reversal losses, the random-search space and the label-free proxy score.
The analyses are driven by run_chunk.py (label-free selection) and run_tval.py (target-validation selection);
this file can also be run on its own (see --help).
"""

import os, time, json, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------- utils ----------
def seed_all(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def metrics_from_scores(y_true, y_score, thr: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
    return {"AUC": safe_auc(y_true, y_score), "ACC": acc, "SEN": sens, "SPE": spec, "PREC": prec, "F1": f1}

def entropy_from_probs(p: np.ndarray, eps: float = 1e-8) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(np.mean(h))

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


# ---------- data ----------
@dataclass
class PreparedData:
    Xs: np.ndarray
    ys: np.ndarray
    Xt: np.ndarray
    yt: np.ndarray
    feat_names: List[str]

def prepare_source_target(sof_csv: str, uk_csv: str, target_col: str, drop_feats=(), uk_complete_cols=(), std_cols=()) -> PreparedData:
    target_col = target_col.strip().upper()
    sof = load_csv(sof_csv)
    uk  = load_csv(uk_csv)

    drop_feats = {c.strip().upper() for c in drop_feats}
    uk_complete_cols = [c.strip().upper() for c in uk_complete_cols]
    if uk_complete_cols:
        n0 = len(uk); m = uk[uk_complete_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        uk = uk[m].reset_index(drop=True)
        print(f"[INFO] UKB restricted to rows with non-missing {uk_complete_cols}: {len(uk)}/{n0} rows, "
              f"{int(pd.to_numeric(uk[target_col], errors='coerce').sum())} cases")
    # within-cohort z-score of selected columns (features only, no labels; uses each cohort's own observed values)
    for c in [x.strip().upper() for x in std_cols]:
        for df in (sof, uk):
            if c in df.columns:
                v = pd.to_numeric(df[c], errors="coerce")
                df[c] = (v - v.mean()) / v.std()
        print(f"[INFO] within-cohort standardized {c}")
    feats = sorted(list((set(sof.columns) & set(uk.columns)) - {target_col} - drop_feats))
    print("[INFO] features used:", feats)
    Xs = sof[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    Xt = uk[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    # drop all-NaN
    keep_idx, drop = [], []
    for j, name in enumerate(feats):
        if np.all(np.isnan(Xs[:, j])) or np.all(np.isnan(Xt[:, j])):
            drop.append(name)
        else:
            keep_idx.append(j)
    if drop:
        print("[WARN] Dropping all-NaN features:", drop)
    Xs = Xs[:, keep_idx]; Xt = Xt[:, keep_idx]
    feats = [feats[j] for j in keep_idx]

    ys = pd.to_numeric(sof[target_col], errors="coerce").to_numpy()
    yt = pd.to_numeric(uk[target_col], errors="coerce").to_numpy()

    m = np.isfinite(ys)
    if (~m).any():
        print("[WARN] Dropping SOF rows with NaN labels:", int((~m).sum()))
        Xs = Xs[m]; ys = ys[m]

    return PreparedData(Xs=Xs, ys=ys.astype(np.int64), Xt=Xt, yt=yt.astype(np.int64), feat_names=feats)


@dataclass
class PreprocessFit:
    imputer: SimpleImputer
    scaler: StandardScaler

def fit_preprocess(X_train: np.ndarray) -> PreprocessFit:
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_train)
    sc = StandardScaler()
    sc.fit(X_imp)
    return PreprocessFit(imputer=imp, scaler=sc)

def apply_preprocess(pp: PreprocessFit, X: np.ndarray) -> np.ndarray:
    return pp.scaler.transform(pp.imputer.transform(X)).astype(np.float32)


# ---------- splits ----------
def split_source_train_val(Xs, ys, seed, val_frac):
    return train_test_split(Xs, ys, test_size=val_frac, random_state=seed, stratify=ys)

def build_full_random_split(y: np.ndarray, rep_seed: int, test_frac: float):
    rng = np.random.default_rng(rep_seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    n_test = int(round(len(idx) * test_frac))
    test = idx[:n_test]
    train = idx[n_test:]
    return train, test


# ---------- models ----------
class MLPFeat(nn.Module):
    def __init__(self, d_in: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers=[]; d=d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ClassHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.fc = nn.Linear(d_in, 1)
    def forward(self, z): return self.fc(z).squeeze(-1)

class GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd): return GradReverseFn.apply(x, lambd)

class DomainDisc(nn.Module):
    def __init__(self, d_in: int, hidden: int, depth: int, dropout: float):
        super().__init__()
        layers=[]; d=d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z).squeeze(-1)


# ---------- losses ----------
def mmd_rbf(x: torch.Tensor, y: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    xx = torch.cdist(x, x) ** 2
    yy = torch.cdist(y, y) ** 2
    xy = torch.cdist(x, y) ** 2
    if gamma is None:
        with torch.no_grad():
            vals = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()], dim=0)
            med = torch.median(vals)
            gamma = 1.0 / (med + 1e-8)
    return torch.exp(-gamma*xx).mean() + torch.exp(-gamma*yy).mean() - 2.0*torch.exp(-gamma*xy).mean()

def coral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cx = (x.T @ x) / (x.shape[0] - 1 + 1e-8)
    cy = (y.T @ y) / (y.shape[0] - 1 + 1e-8)
    return ((cx - cy) ** 2).mean()

@torch.no_grad()
def predict_probs(feat, clf, X, device, bs=4096):
    feat.eval(); clf.eval()
    x = torch.from_numpy(X.astype(np.float32)).to(device)
    out=[]
    for i in range(0, x.shape[0], bs):
        z = feat(x[i:i+bs])
        p = torch.sigmoid(clf(z))
        out.append(p.detach().cpu().numpy())
    return np.concatenate(out, axis=0)

@torch.no_grad()
def domain_acc_on_features(feat, Xs, Xt, device):
    feat.eval()
    z_s = feat(torch.from_numpy(Xs.astype(np.float32)).to(device)).detach().cpu().numpy()
    z_t = feat(torch.from_numpy(Xt.astype(np.float32)).to(device)).detach().cpu().numpy()
    Z = np.vstack([z_s, z_t])
    y = np.concatenate([np.zeros(len(z_s)), np.ones(len(z_t))]).astype(int)
    rng = np.random.default_rng(123)
    idx = np.arange(len(y)); rng.shuffle(idx)
    tr = idx[:int(0.7*len(idx))]; te = idx[int(0.7*len(idx)):]
    dom = LogisticRegression(max_iter=800, solver="liblinear")
    dom.fit(Z[tr], y[tr])
    return float(dom.score(Z[te], y[te]))

def make_loaders(Xs, ys, Xt_u, bs, device):
    xs = torch.from_numpy(Xs).to(device)
    ys_t = torch.from_numpy(ys.astype(np.float32)).to(device)
    xt = torch.from_numpy(Xt_u).to(device)
    dl_s = DataLoader(TensorDataset(xs, ys_t), batch_size=bs, shuffle=True, drop_last=True)
    dl_t = DataLoader(TensorDataset(xt), batch_size=bs, shuffle=True, drop_last=True)
    return dl_s, dl_t


# ---------- variants (8 total) ----------
VARIANTS = [
    ("Baseline", dict(mmd=False, coral=False, grl=False)),
    ("+MMD", dict(mmd=True, coral=False, grl=False)),
    ("+CORAL", dict(mmd=False, coral=True, grl=False)),
    ("+GRL", dict(mmd=False, coral=False, grl=True)),
    ("+MMD+CORAL", dict(mmd=True, coral=True, grl=False)),
    ("+MMD+GRL", dict(mmd=True, coral=False, grl=True)),
    ("+CORAL+GRL", dict(mmd=False, coral=True, grl=True)),
    ("+MMD+CORAL+GRL", dict(mmd=True, coral=True, grl=True)),
]


def sample_hp_mid(rng, flags):
    hidden = int(rng.choice([64,128,256]))
    depth  = int(rng.choice([1,2,3]))
    dropout = float(rng.choice([0.0,0.1,0.2]))
    lr = float(10 ** rng.uniform(-3.5, -2.7))
    wd = float(10 ** rng.uniform(-6, -4.0))
    bs = int(rng.choice([128,256,512]))
    epochs = int(rng.choice([40,60,80]))

    lam_mmd = float(10 ** rng.uniform(-2.0, 0.3)) if flags["mmd"] else 0.0
    lam_coral = float(10 ** rng.uniform(-2.0, 0.3)) if flags["coral"] else 0.0
    grl_lambda = float(10 ** rng.uniform(-1.0, 0.3)) if flags["grl"] else 0.0

    return dict(hidden=hidden, depth=depth, dropout=dropout, lr=lr, wd=wd, bs=bs, epochs=epochs,
                lam_mmd=lam_mmd, lam_coral=lam_coral, grl_lambda=grl_lambda,
                alpha_dom=0.2, alpha_da=0.03, alpha_ent=0.1)


def train_one_trial(flags, hp, Xs_tr, ys_tr, Xs_val, ys_val, Xt_u, device, seed):
    seed_all(seed)
    d_in = Xs_tr.shape[1]

    feat = MLPFeat(d_in, hp["hidden"], hp["depth"], hp["dropout"]).to(device)
    clf  = ClassHead(hp["hidden"]).to(device)
    disc = DomainDisc(hp["hidden"], hp["hidden"], 2, hp["dropout"]).to(device) if flags["grl"] else None

    params = list(feat.parameters()) + list(clf.parameters())
    if disc is not None:
        params += list(disc.parameters())
    opt = torch.optim.Adam(params, lr=hp["lr"], weight_decay=hp["wd"])

    n_pos = float((ys_tr==1).sum()); n_neg = float((ys_tr==0).sum())
    pos_weight = torch.tensor([n_neg/max(1.0,n_pos)], device=device, dtype=torch.float32)

    dl_s, dl_t = make_loaders(Xs_tr, ys_tr, Xt_u, hp["bs"], device)
    for _ in range(hp["epochs"]):
        steps = min(len(dl_s), len(dl_t))
        it_s, it_t = iter(dl_s), iter(dl_t)
        for _ in range(steps):
            xs, ys_b = next(it_s); (xt,) = next(it_t)
            z_s = feat(xs); z_t = feat(xt)

            logits = clf(z_s)
            loss_cls = F.binary_cross_entropy_with_logits(logits, ys_b, pos_weight=pos_weight)

            loss_da = torch.tensor(0.0, device=device)
            if flags["mmd"]:
                loss_da = loss_da + hp["lam_mmd"] * mmd_rbf(z_s, z_t)
            if flags["coral"]:
                loss_da = loss_da + hp["lam_coral"] * coral_loss(z_s, z_t)

            loss_dom = torch.tensor(0.0, device=device)
            if flags["grl"]:
                dom_s = torch.zeros(z_s.shape[0], device=device)
                dom_t = torch.ones(z_t.shape[0], device=device)
                z_cat = torch.cat([z_s, z_t], dim=0)
                dom_y = torch.cat([dom_s, dom_t], dim=0)
                dom_logits = disc(grad_reverse(z_cat, hp["grl_lambda"]))
                loss_dom = F.binary_cross_entropy_with_logits(dom_logits, dom_y)

            loss = loss_cls + loss_da + loss_dom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    p_val = predict_probs(feat, clf, Xs_val, device=device)
    src_val_auc = safe_auc(ys_val, p_val)
    p_tu = predict_probs(feat, clf, Xt_u, device=device)
    tgt_ent = entropy_from_probs(p_tu)
    dom_acc = domain_acc_on_features(feat, Xs_val, Xt_u[:min(len(Xt_u), len(Xs_val))], device=device)

    proxy = (src_val_auc
             - hp["alpha_dom"]*dom_acc
             - hp["alpha_da"]*float(loss_da.detach().cpu())
             - hp["alpha_ent"]*tgt_ent)
    diag = dict(proxy=float(proxy), src_val_auc=float(src_val_auc), dom_acc=float(dom_acc), tgt_entropy=float(tgt_ent))
    return diag, feat, clf


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sof_csv", type=str, default=os.path.join(HERE, "SOF_for_DA_enriched.csv"))
    ap.add_argument("--uk_csv",  type=str, default=os.path.join(HERE, "UKfemale_for_DA_aligned.csv"))
    ap.add_argument("--target_col", type=str, default="HA_HIPFX")
    ap.add_argument("--drop_feats", nargs="*", default=[], help="predictors to exclude, e.g. HA_THD HA_LSD HA_FND for the no-BMD analysis")
    ap.add_argument("--uk_complete_cols", nargs="*", default=[], help="keep only UKB rows with these columns observed, e.g. HA_THD HA_LSD HA_FND")
    ap.add_argument("--std_cols", nargs="*", default=[], help="z-score these columns within each cohort, e.g. HA_THD HA_LSD HA_FND")
    ap.add_argument("--out_dir", type=str, default=os.path.join(HERE, "out_sens_female"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # balanced but not too slow
    ap.add_argument("--n_rep", type=int, default=6)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--n_trials", type=int, default=10)
    ap.add_argument("--val_frac_src", type=float, default=0.2)
    ap.add_argument("--full_test_frac", type=float, default=0.2)
    ap.add_argument("--eval_thr", type=float, default=0.5)

    args, _ = ap.parse_known_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = prepare_source_target(args.sof_csv, args.uk_csv, args.target_col, args.drop_feats, args.uk_complete_cols, args.std_cols)
    json.dump(dict(feat_names=data.feat_names, n_target=int(len(data.yt)), n_cases=int(data.yt.sum()), args=vars(args)), open(os.path.join(args.out_dir, 'run_info.json'), 'w'), indent=1)
    print("[INFO] device:", args.device, "| nfeat:", len(data.feat_names), "| variants:", len(VARIANTS))

    rows_best=[]
    t0=time.time()

    for rep in range(args.n_rep):
        tgt_u_idx, tgt_te_idx = build_full_random_split(data.yt, rep_seed=12000+rep, test_frac=args.full_test_frac)
        Xt_u_raw  = data.Xt[tgt_u_idx]
        Xt_te_raw = data.Xt[tgt_te_idx]
        yt_te     = data.yt[tgt_te_idx].astype(int)

        Xs_tr_raw, Xs_val_raw, ys_tr, ys_val = split_source_train_val(data.Xs, data.ys, seed=7000+rep, val_frac=args.val_frac_src)
        pp = fit_preprocess(Xs_tr_raw)
        Xs_tr  = apply_preprocess(pp, Xs_tr_raw)
        Xs_val = apply_preprocess(pp, Xs_val_raw)
        Xt_u   = apply_preprocess(pp, Xt_u_raw)
        Xt_te  = apply_preprocess(pp, Xt_te_raw)

        for seed in args.seeds:
            rng = np.random.default_rng(100000 + rep*100 + seed)

            for vname, flags in VARIANTS:
                trials = args.n_trials if vname!="Baseline" else max(4, args.n_trials//2)

                best=None; best_hp=None; best_feat=None; best_clf=None; best_diag=None
                for tr in range(trials):
                    hp = sample_hp_mid(rng, flags)
                    diag, feat, clf = train_one_trial(flags, hp, Xs_tr, ys_tr, Xs_val, ys_val, Xt_u,
                                                     args.device, seed=seed + 37*tr + 1000*rep)
                    if (best is None) or (diag["proxy"] > best):
                        best = diag["proxy"]; best_hp=hp; best_feat=feat; best_clf=clf; best_diag=diag

                p_te = predict_probs(best_feat, best_clf, Xt_te, device=args.device)
                met  = metrics_from_scores(yt_te, p_te, thr=args.eval_thr)

                rows_best.append({
                    "rep": rep, "seed": seed, "Variant": vname,
                    "AUC": met["AUC"], "ACC": met["ACC"], "SEN": met["SEN"], "SPE": met["SPE"], "PREC": met["PREC"], "F1": met["F1"],
                    "proxy": float(best),
                    "src_val_auc": best_diag["src_val_auc"], "dom_acc": best_diag["dom_acc"], "tgt_entropy": best_diag["tgt_entropy"],
                    "cases_te": int((yt_te==1).sum()),
                    "best_hp_json": json.dumps(best_hp),
                    "tgt_y_true_json": json.dumps(yt_te.tolist()),
                    "tgt_scores_json": json.dumps(p_te.tolist()),
                })

                print(f"[rep {rep+1}/{args.n_rep}] seed={seed} {vname:>14s} AUC={met['AUC']:.4f} SEN={met['SEN']:.4f} cases_te={int((yt_te==1).sum())}")

    df_best = pd.DataFrame(rows_best)
    out_best = os.path.join(args.out_dir, "results_best_with_scores.csv")
    df_best.to_csv(out_best, index=False)
    print("\n[DONE] wrote:", out_best)
    print(f"[TIME] total_sec={time.time()-t0:.1f}")

if __name__ == "__main__":
    main()
