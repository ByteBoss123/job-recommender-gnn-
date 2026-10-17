"""Build per-window temporal splits from the real CareerBuilder job-application data.

Each CareerBuilder window is an independent marketplace snapshot: no user or job
appears in more than one window. Within each window we split applications by time:

  fit   : ApplicationDate <  p70     (stage-1 training for the reranker's training labels)
  val   : p70 <= date     <  p80     (labels to train the LambdaRank reranker)
  train : ApplicationDate <  p80     (stage-1 training for the final test)
  test  : ApplicationDate >= p80     (held-out future applications)

A held-out application is scoreable only if its user and its job both appear in the
corresponding training graph. Applications to jobs with no prior applicants are
counted and reported as cold-start, not silently dropped.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

EARLY_CAREER_GRAD_YEAR = 2010  # graduated within ~2 years of the 2012 data period


def load():
    apps = pd.read_csv(RAW / "apps.tsv", sep="\t")
    apps["ts"] = pd.to_datetime(apps.ApplicationDate, format="mixed")
    users = pd.read_csv(RAW / "users.tsv", sep="\t", dtype={"ZipCode": str})
    hist = pd.read_csv(RAW / "user_history.tsv", sep="\t")
    return apps, users, hist


def tokens(s):
    return re.findall(r"[a-z]{3,}", str(s).lower())


def user_features(users, hist):
    u = users.copy()
    grad = pd.to_datetime(u.GraduationDate, errors="coerce")
    u["grad_year"] = grad.dt.year
    u["years_since_grad"] = (2012 - u.grad_year).clip(-5, 60)
    u["early_career"] = (u.grad_year >= EARLY_CAREER_GRAD_YEAR).astype(int)
    u["employed"] = (u.CurrentlyEmployed == "Yes").astype(int)
    u["managed"] = (u.ManagedOthers == "Yes").astype(int)
    u["exp"] = u.TotalYearsExperience.fillna(-1).clip(-1, 60)
    u["degree"] = u.DegreeType.fillna("Unknown")
    u["state"] = u.State.fillna("UNK")
    # text profile: prior job titles + major
    titles = hist.groupby("UserID").JobTitle.apply(lambda s: " ".join(map(str, s)))
    u["text"] = u.UserID.map(titles).fillna("") + " " + u.Major.fillna("")
    return u[["UserID", "WindowID", "Split", "degree", "state", "years_since_grad",
              "early_career", "employed", "managed", "exp", "WorkHistoryCount",
              "ManagedHowMany", "text"]]


def main():
    apps, users, hist = load()
    uf = user_features(users, hist)
    uf.to_parquet(OUT / "user_features.parquet")
    report = {}
    for w, g in apps.groupby("WindowID"):
        p70, p80 = g.ts.quantile([0.7, 0.8])
        parts = {
            "fit": g[g.ts < p70],
            "val": g[(g.ts >= p70) & (g.ts < p80)],
            "train": g[g.ts < p80],
            "test": g[g.ts >= p80],
        }
        for k, df in parts.items():
            df[["UserID", "JobID", "ts"]].to_parquet(OUT / f"w{w}_{k}.parquet")
        stats = {"p70": str(p70), "p80": str(p80), "n_apps": int(len(g))}
        for hold, base in (("val", "fit"), ("test", "train")):
            h, b = parts[hold], parts[base]
            wu, wj = h.UserID.isin(b.UserID), h.JobID.isin(b.JobID)
            stats[hold] = {
                "apps": int(len(h)),
                "warm_user_warm_job": int((wu & wj).sum()),
                "warm_user_cold_job": int((wu & ~wj).sum()),
                "cold_user": int((~wu).sum()),
                "eval_users": int(h[wu & wj].UserID.nunique()),
            }
        report[int(w)] = stats
        print(w, json.dumps(stats))
    json.dump(report, open(OUT / "split_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
