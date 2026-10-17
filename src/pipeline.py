"""Two-stage job recommender, end to end, on real CareerBuilder applications.

Per window:
  val stage  : retrievers trained on `fit`, candidates scored against `val` labels
               -> trains the LambdaRank reranker
  test stage : retrievers retrained on `train`, reranker applied, evaluated on `test`
Cold-start users (no prior applications) are evaluated with the inductive two-tower.
"""
import json
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from baselines import itemknn
from common import (DATA, RESULTS, Split, evaluate, metrics_for_user, popularity_scores,
                    topk_from_scores)
from models import (UserFeaturizer, train_lightgcn, train_two_tower, two_tower_job_emb,
                    two_tower_user_emb)

warnings.filterwarnings("ignore")
N_CAND = 100
LGCN_CFG = dict(epochs=15, lr=5e-3, batch=2048, l2=1e-4, layers=3)  # v2: smaller batch (validated better on window 3)
FEATS_NO_GNN = ["s_knn", "r_knn", "pop", "job_deg", "u_deg", "early_career", "exp", "years_since_grad"]
PROMO_LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
FEATS = ["s_lgcn", "s_knn", "r_lgcn", "r_knn", "pop", "job_deg", "u_deg",
         "early_career", "exp", "years_since_grad"]


def stage1(s):
    t = time.time()
    U, J = train_lightgcn(s, **LGCN_CFG)
    S = itemknn(s)
    pop = popularity_scores(s)
    print(f"   stage1 trained {time.time() - t:.0f}s", flush=True)
    return U, J, S, pop


def candidates(s, U, J, S, pop, uf, use_gnn=True):
    job_deg = np.asarray(s.X.sum(0)).ravel()
    u_deg = np.asarray(s.X.sum(1)).ravel()
    ufi = uf.set_index("UserID")
    rows = []
    for b in range(0, len(s.eval_users), 512):
        users = s.eval_users[b:b + 512]
        seen = s.X[users]
        lg = U[users] @ J.T if use_gnn else np.zeros((len(users), s.n_jobs), np.float32)
        kn = (s.X[users] @ S).toarray() + pop / pop.max() * 1e-3
        for m in (lg, kn):
            m[seen.nonzero()] = -np.inf
        if use_gnn:
            tl, tk = topk_from_scores(lg, N_CAND), topk_from_scores(kn, N_CAND)
        else:  # ablation: KNN-only candidates, same budget (~160)
            tk = topk_from_scores(kn, 160); tl = tk[:, :0]
        for i, u in enumerate(users):
            cand = np.unique(np.concatenate([tl[i], tk[i]]))
            rl = pd.Series(np.arange(tl.shape[1]), index=tl[i]).reindex(cand).fillna(N_CAND).values
            rk = pd.Series(np.arange(tk.shape[1]), index=tk[i]).reindex(cand).fillna(tk.shape[1]).values
            f = ufi.loc[s.users[u]]
            rows.append(pd.DataFrame({
                "u": u, "j": cand, "s_lgcn": lg[i, cand], "s_knn": kn[i, cand], "r_lgcn": rl, "r_knn": rk,
                "pop": pop[cand], "job_deg": job_deg[cand], "u_deg": u_deg[u],
                "early_career": f.early_career, "exp": f.exp, "years_since_grad": f.years_since_grad,
                "label": np.isin(cand, s.truth[u]).astype(int)}))
        del lg, kn
    return pd.concat(rows, ignore_index=True), None, None


def promotion_tradeoff(s, ct):
    """Employer-side promotion proxy (no price data exists). 'Under-served' jobs = bottom half
    of jobs by prior applicants, the jobs an employer would pay to promote. Score per user =
    rank-percentile of reranker score + lambda * under_served. Measures what share of top-10
    slots and of real applications go to under-served jobs, and the relevance cost."""
    job_deg = np.asarray(s.X.sum(0)).ravel()
    under = (job_deg <= np.median(job_deg)).astype(float)
    c = ct[["u", "j", "s_rerank", "label"]].copy()
    c["pct"] = c.groupby("u").s_rerank.rank(pct=True)
    c["under"] = under[c.j.values]
    n_truth = pd.Series({u: len(t) for u, t in s.truth.items()})
    rows = []
    for lam in PROMO_LAMBDAS:
        c["sc"] = c.pct + lam * c.under
        top = c.sort_values(["u", "sc"], ascending=[True, False]).groupby("u").head(10)
        hits = top.groupby("u").label.sum()
        rows.append({"lambda": lam,
                     "recall@10": float((hits / np.minimum(n_truth.reindex(hits.index), 10)).mean()),
                     "under_served_slot_share": float(top.under.mean()),
                     "applications_to_under_served": int(top[top.label == 1].under.sum()),
                     "applications_total": int(top.label.sum())})
    return rows


def eval_from_ranked(s, cand, score_col):
    rows = []
    for u, g in cand.groupby("u", sort=False):
        top = g.sort_values(score_col, ascending=False).j.values[:50]
        if len(top) < 50:
            top = np.concatenate([top, -np.ones(50 - len(top), int)])
        m = metrics_for_user(top, s.truth[u]); m["u"] = int(u); rows.append(m)
    return pd.DataFrame(rows)


def run_window(w, uf_all):
    uf = uf_all[uf_all.WindowID == w]
    out = {}
    # --- val stage: build reranker training data
    sv = Split(w, "val")
    print(f"w{w} val: users={sv.n_users} jobs={sv.n_jobs} edges={len(sv.train)}", flush=True)
    s1v = stage1(sv)
    rankers = {}
    for name, use_gnn, feats in (("two_stage_rerank", True, FEATS), ("rerank_no_gnn", False, FEATS_NO_GNN)):
        cv, _, _ = candidates(sv, *s1v, uf, use_gnn=use_gnn)
        cv = cv[cv.groupby("u").label.transform("max") == 1]  # queries with at least one positive
        r = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                           num_leaves=31, min_child_samples=50, verbose=-1)
        r.fit(cv[feats], cv.label, group=cv.groupby("u", sort=False).size().values)
        rankers[name] = r
    ranker = rankers["two_stage_rerank"]
    # --- test stage
    st = Split(w, "test")
    print(f"w{w} test: users={st.n_users} jobs={st.n_jobs} eval_users={len(st.eval_users)}", flush=True)
    U, J, S, pop = stage1(st)
    ct, lg, kn = candidates(st, U, J, S, pop, uf)
    ct["s_rerank"] = ranker.predict(ct[FEATS])
    res = {
        "popularity": evaluate(st, batch=512, score_fn=lambda ub: np.tile(pop, (len(ub), 1))),
        "itemknn": evaluate(st, batch=512, score_fn=lambda ub: (st.X[ub] @ S).toarray() + pop / pop.max() * 1e-3),
        "lightgcn": evaluate(st, batch=512, score_fn=lambda ub: U[ub] @ J.T),
        "two_stage_rerank": eval_from_ranked(st, ct, "s_rerank"),
    }
    cn, _, _ = candidates(st, U, J, S, pop, uf, use_gnn=False)
    cn["s_rerank"] = rankers["rerank_no_gnn"].predict(cn[FEATS_NO_GNN])
    res["rerank_no_gnn"] = eval_from_ranked(st, cn, "s_rerank")
    out["promotion_tradeoff"] = promotion_tradeoff(st, ct)
    recall_cand = (ct.groupby("u").label.sum() / pd.Series({u: min(len(t), 10**9) for u, t in st.truth.items()})).mean()
    out["candidate_recall"] = float(recall_cand)
    out["candidates_per_user"] = float(ct.groupby("u").size().mean())
    out["feature_importance"] = dict(zip(FEATS, ranker.booster_.feature_importance("gain").round(1).tolist()))
    # notification policy on reranker scores: send only the user's top-1 job if its score clears a threshold
    top1 = ct.sort_values("s_rerank", ascending=False).groupby("u").head(1)
    out["notif_top1"] = top1[["u", "s_rerank", "label"]].assign(window=w)
    # --- cold-start users with the inductive two-tower (trained on the test-stage graph)
    out["cold_users"] = None  # cold-start two-tower measured in v1 (results_v1); not rerun
    early = uf.set_index("UserID").early_career
    for k, r in res.items():
        r["model"], r["window"] = k, w
        r["early_career"] = early.reindex(st.users[r.u.values]).values
    out["per_user"] = pd.concat(res.values())
    print(f"w{w} done:", out["per_user"].groupby("model")[["recall@10", "recall@50", "ndcg@10"]].mean().round(4).to_dict("index"),
          "| cold:", out["cold_users"], flush=True)
    return out


def main(windows):
    uf = pd.read_parquet(DATA / "user_features.parquet")
    per_user, notif, meta = [], [], {}
    for w in windows:
        o = run_window(w, uf)
        per_user.append(o.pop("per_user")); notif.append(o.pop("notif_top1")); meta[w] = o
        pd.concat(per_user).to_parquet(RESULTS / "per_user_test.parquet")
        pd.concat(notif).to_parquet(RESULTS / "notif_top1.parquet")
        json.dump(meta, open(RESULTS / "window_meta.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [3, 4, 5])
