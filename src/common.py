"""Shared loading, indexing and full-ranking evaluation."""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
WINDOWS = list(range(1, 8))
KS = (10, 50)


class Split:
    """Training graph + held-out applications for one window and one stage.

    stage='val'  -> train graph = fit,   held out = val
    stage='test' -> train graph = train, held out = test
    """

    def __init__(self, window, stage):
        base = "fit" if stage == "val" else "train"
        self.window, self.stage = window, stage
        tr = pd.read_parquet(DATA / f"w{window}_{base}.parquet")
        ho = pd.read_parquet(DATA / f"w{window}_{stage}.parquet")
        self.users = np.sort(tr.UserID.unique())
        self.jobs = np.sort(tr.JobID.unique())
        self.uidx = pd.Series(np.arange(len(self.users)), index=self.users)
        self.jidx = pd.Series(np.arange(len(self.jobs)), index=self.jobs)
        tr = tr.assign(u=self.uidx[tr.UserID].values, j=self.jidx[tr.JobID].values)
        self.train = tr
        self.n_users, self.n_jobs = len(self.users), len(self.jobs)
        self.X = sp.csr_matrix(
            (np.ones(len(tr), dtype=np.float32), (tr.u.values, tr.j.values)),
            shape=(self.n_users, self.n_jobs))
        warm = ho[ho.UserID.isin(self.uidx.index) & ho.JobID.isin(self.jidx.index)]
        warm = warm.assign(u=self.uidx[warm.UserID].values, j=self.jidx[warm.JobID].values)
        self.heldout = warm
        self.eval_users = np.sort(warm.u.unique())
        self.truth = warm.groupby("u").j.apply(np.array).to_dict()
        # cold users: held-out apps from users with no training apps, to warm jobs
        cold = ho[~ho.UserID.isin(self.uidx.index) & ho.JobID.isin(self.jidx.index)]
        self.cold_heldout = cold.assign(j=self.jidx[cold.JobID].values)
        self.t_end = tr.ts.max()


def topk_from_scores(scores, k):
    part = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    return part[rows, order]


def metrics_for_user(top, truth):
    hits = np.isin(top, truth)
    out = {}
    for k in KS:
        out[f"recall@{k}"] = hits[:k].sum() / min(len(truth), k)
    out["hit@10"] = float(hits[:10].any())
    disc = 1.0 / np.log2(np.arange(2, 12))
    dcg = (hits[:10] * disc).sum()
    idcg = disc[: min(len(truth), 10)].sum()
    out["ndcg@10"] = dcg / idcg
    return out


def evaluate(split, score_fn, batch=1024, users=None, truth=None, exclude_train=True):
    """score_fn(user_idx_array) -> dense (B, n_jobs) scores. Full ranking over all
    warm jobs in the window; jobs the user already applied to are excluded."""
    users = split.eval_users if users is None else users
    truth = split.truth if truth is None else truth
    kmax = max(KS)
    rows = []
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        sc = np.asarray(score_fn(ub), dtype=np.float32)
        if exclude_train:
            seen = split.X[ub]
            sc[seen.nonzero()] = -np.inf
        top = topk_from_scores(sc, kmax)
        for i, u in enumerate(ub):
            m = metrics_for_user(top[i], truth[u])
            m["u"] = int(u)
            rows.append(m)
    return pd.DataFrame(rows)


def popularity_scores(split, half_life_days=2.0):
    """Recency-weighted application counts from the training graph."""
    tr = split.train
    age = (split.t_end - tr.ts).dt.total_seconds() / 86400
    w = 0.5 ** (age / half_life_days)
    return np.bincount(tr.j.values, weights=w.values, minlength=split.n_jobs).astype(np.float32)
