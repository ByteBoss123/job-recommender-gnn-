import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import DATA, Split, metrics_for_user, topk_from_scores  # noqa: E402


def test_metrics_perfect_and_miss():
    m = metrics_for_user(np.array([5, 1, 2] + list(range(10, 57))), np.array([5]))
    assert m["recall@10"] == 1.0 and m["ndcg@10"] == 1.0 and m["hit@10"] == 1.0
    m = metrics_for_user(np.arange(100, 150), np.array([5]))
    assert m["recall@50"] == 0.0 and m["ndcg@10"] == 0.0


def test_topk_order():
    s = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert topk_from_scores(s, 3).tolist() == [[1, 3, 2]]


def test_split_is_temporal_and_disjoint():
    for w in (3, 4, 5):
        tr = pd.read_parquet(DATA / f"w{w}_train.parquet")
        te = pd.read_parquet(DATA / f"w{w}_test.parquet")
        assert tr.ts.max() < te.ts.min()
        assert not tr.merge(te, on=["UserID", "JobID"]).shape[0]  # no leaked (user, job) pairs


def test_eval_excludes_only_unseen_pairs():
    s = Split(3, "test")
    # every held-out (user, job) is absent from the training graph
    assert s.X[s.heldout.u.values, s.heldout.j.values].sum() == 0
    # cold users truly have no training applications
    assert not s.cold_heldout.UserID.isin(s.users).any()
