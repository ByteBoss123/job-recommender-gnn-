"""Aggregate test results with user-level bootstrap CIs."""
import json

import numpy as np
import pandas as pd

from common import RESULTS

rng = np.random.default_rng(0)


def boot_diff(a, b, n=2000):
    """Paired bootstrap over users of mean(a - b)."""
    d = (a - b).values
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


pu = pd.read_parquet(RESULTS / "per_user_test.parquet")
meta = json.load(open(RESULTS / "window_meta.json"))
cols = ["recall@10", "recall@50", "hit@10", "ndcg@10"]
summary = pu.groupby("model")[cols].mean().round(4)
early = pu[pu.early_career == 1].groupby("model")[cols].mean().round(4)
wide = {c: pu.pivot_table(index=["window", "u"], columns="model", values=c) for c in ["recall@10", "ndcg@10", "recall@50"]}
comparisons = {}
for c, t in wide.items():
    for a, b in (("lightgcn", "itemknn"), ("two_stage_rerank", "itemknn"), ("two_stage_rerank", "lightgcn")):
        comparisons[f"{c}: {a} - {b}"] = [round(x, 4) for x in boot_diff(t[a], t[b])]
# notification policy: top-1 job per user, send only if reranker score >= threshold
nt = pd.read_parquet(RESULTS / "notif_top1.parquet")
notif = []
for q in (0.0, 0.25, 0.5, 0.75, 0.9):
    thr = nt.s_rerank.quantile(q)
    sent = nt[nt.s_rerank >= thr]
    notif.append({"withhold_bottom": q, "share_users_notified": round(len(sent) / len(nt), 3),
                  "precision_per_send": round(sent.label.mean(), 4), "hits": int(sent.label.sum())})
cold = {w: m["cold_users"] for w, m in meta.items()}
cold_tt = np.average([c["two_tower_profile"]["recall@50"] for c in cold.values()], weights=[c["n_users"] for c in cold.values()])
cold_pop = np.average([c["popularity"]["recall@50"] for c in cold.values()], weights=[c["n_users"] for c in cold.values()])
out = {
    "n_eval_users": int(pu[pu.model == "itemknn"].shape[0]),
    "n_early_career_users": int(pu[(pu.model == "itemknn") & (pu.early_career == 1)].shape[0]),
    "summary": summary.to_dict("index"),
    "early_career": early.to_dict("index"),
    "paired_bootstrap_mean_ci95": comparisons,
    "candidate_recall": {w: round(m["candidate_recall"], 4) for w, m in meta.items()},
    "candidates_per_user": {w: round(m["candidates_per_user"], 1) for w, m in meta.items()},
    "reranker_gain_importance": {w: m["feature_importance"] for w, m in meta.items()},
    "notification_policy": notif,
    "cold_users": cold,
    "cold_users_weighted_recall@50": {"two_tower_profile": round(cold_tt, 4), "popularity": round(cold_pop, 4)},
}
json.dump(out, open(RESULTS / "summary.json", "w"), indent=1)
print(summary, "\nEarly career\n", early)
print(json.dumps({k: out[k] for k in ("n_eval_users", "n_early_career_users", "paired_bootstrap_mean_ci95", "candidate_recall", "notification_policy", "cold_users_weighted_recall@50")}, indent=1))
