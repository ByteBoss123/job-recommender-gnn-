"""Non-neural baselines: recency popularity and co-application ItemKNN."""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import RESULTS, WINDOWS, Split, evaluate, popularity_scores


def itemknn(split, shrink=10.0):
    X = split.X.astype(np.float32)
    co = (X.T @ X).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    n = np.asarray(X.sum(0)).ravel()
    co = co.tocoo()
    vals = co.data / (np.sqrt(n[co.row] * n[co.col]) + shrink)
    S = sp.csr_matrix((vals.astype(np.float32), (co.row, co.col)), shape=co.shape)
    return S


def run(stage):
    out = {}
    per_user = []
    for w in WINDOWS:
        s = Split(w, stage)
        pop = popularity_scores(s)
        S = itemknn(s)
        res_pop = evaluate(s, lambda ub: np.tile(pop, (len(ub), 1)))
        # ItemKNN, backfilled with a tiny popularity term so users with no co-applied
        # neighbours still get a ranking
        pop_n = pop / (pop.max() + 1e-9) * 1e-3
        res_knn = evaluate(s, lambda ub: (s.X[ub] @ S).toarray() + pop_n)
        for name, r in (("popularity", res_pop), ("itemknn", res_knn)):
            per_user.append(r.assign(model=name, window=w))
        print(w, "pop", res_pop.mean().round(4).to_dict(), "| knn", res_knn.mean().round(4).to_dict(), flush=True)
    pu = pd.concat(per_user)
    pu.to_parquet(RESULTS / f"per_user_baselines_{stage}.parquet")
    summ = pu.drop(columns=["u", "window"]).groupby("model").mean().round(4)
    print(summ)
    return summ


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "test")
