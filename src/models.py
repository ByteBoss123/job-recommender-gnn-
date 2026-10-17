"""Stage-1 retrievers: LightGCN (graph neural network) and a feature two-tower.

LightGCN (He et al., SIGIR 2020): user and job ID embeddings propagated over the
normalised user-job application graph for L layers, layer outputs averaged, BPR loss.

Two-tower: the user tower is inductive. It encodes profile features (degree, state,
experience, years since graduation, prior job titles + major as a bag of words) plus
a mean-pool of the job embeddings the user has already applied to. A brand-new user
with no applications is encoded from profile features alone, so the same model can
rank jobs for cold-start users, which LightGCN cannot. The job tower is an ID embedding.
"""
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
DIM = 64


# ----------------------------------------------------------------------------- LightGCN
def norm_adj(split):
    nu, nj = split.n_users, split.n_jobs
    X = split.X.tocoo()
    rows = np.concatenate([X.row, X.col + nu])
    cols = np.concatenate([X.col + nu, X.row])
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(nu + nj, nu + nj))
    d = np.asarray(A.sum(1)).ravel()
    dinv = np.where(d > 0, d ** -0.5, 0).astype(np.float32)
    A = sp.diags(dinv) @ A @ sp.diags(dinv)
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(A.data), A.shape).coalesce()


class LightGCN(nn.Module):
    def __init__(self, nu, nj, adj, layers=3, dim=DIM):
        super().__init__()
        self.nu, self.nj, self.adj, self.layers = nu, nj, adj, layers
        self.emb = nn.Embedding(nu + nj, dim)
        nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self):
        h = self.emb.weight
        out = [h]
        for _ in range(self.layers):
            h = torch.sparse.mm(self.adj, h)
            out.append(h)
        z = torch.stack(out).mean(0)
        return z[: self.nu], z[self.nu:]


def bpr_batches(split, batch, rng):
    u, j = split.train.u.values, split.train.j.values
    perm = rng.permutation(len(u))
    for s in range(0, len(u), batch):
        b = perm[s:s + batch]
        neg = rng.integers(0, split.n_jobs, len(b))
        yield u[b], j[b], neg


def train_lightgcn(split, epochs=30, lr=5e-3, l2=1e-4, layers=3, batch=8192, seed=0, log=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = LightGCN(split.n_users, split.n_jobs, norm_adj(split), layers)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        for u, jp, jn in bpr_batches(split, batch, rng):
            U, J = m.propagate()
            u_, p_, n_ = (torch.as_tensor(x) for x in (u, jp, jn))
            eu, ep_, en = U[u_], J[p_], J[n_]
            loss = -F.logsigmoid((eu * ep_).sum(1) - (eu * en).sum(1)).mean()
            e0 = m.emb.weight
            reg = (e0[u_].pow(2).sum() + e0[p_ + m.nu].pow(2).sum() + e0[n_ + m.nu].pow(2).sum()) / len(u)
            loss = loss + l2 * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(u)
        if log:
            print(f"  lightgcn ep{ep} loss={tot / len(split.train):.4f} {time.time() - t0:.1f}s", flush=True)
    m.eval()
    with torch.no_grad():
        U, J = m.propagate()
    return U.numpy(), J.numpy()


# ----------------------------------------------------------------------------- Two-tower
class UserFeaturizer:
    """Fit on profile text of all users (no labels involved)."""

    def __init__(self, uf, vocab_size=20000):
        cnt = Counter()
        for t in uf.text.values:
            cnt.update(set(re.findall(r"[a-z]{3,}", t.lower())))
        self.vocab = {w: i + 1 for i, (w, _) in enumerate(cnt.most_common(vocab_size))}
        self.degrees = {d: i for i, d in enumerate(sorted(uf.degree.unique()))}
        self.states = {s: i for i, s in enumerate(sorted(uf.state.unique()))}
        self.uf = uf.set_index("UserID")

    def encode(self, user_ids, max_tok=64):
        f = self.uf.loc[user_ids]
        toks = np.zeros((len(f), max_tok), dtype=np.int64)
        for i, t in enumerate(f.text.values):
            ids = [self.vocab[w] for w in dict.fromkeys(re.findall(r"[a-z]{3,}", t.lower())) if w in self.vocab][:max_tok]
            toks[i, : len(ids)] = ids
        num = np.stack([
            np.log1p(f.exp.clip(0).values), (f.exp < 0).astype(float).values,
            np.nan_to_num(f.years_since_grad.values / 10.0), f.years_since_grad.isna().astype(float).values,
            f.employed.values, f.managed.values, np.log1p(f.WorkHistoryCount.values),
            np.log1p(f.ManagedHowMany.clip(0).values)], 1).astype(np.float32)
        return {
            "tok": torch.tensor(toks),
            "deg": torch.tensor(f.degree.map(self.degrees).values),
            "state": torch.tensor(f.state.map(self.states).values),
            "num": torch.tensor(num),
        }


class TwoTower(nn.Module):
    def __init__(self, nj, n_vocab, n_deg, n_state, dim=DIM):
        super().__init__()
        self.job = nn.Embedding(nj, dim)
        nn.init.normal_(self.job.weight, std=0.05)
        self.tok = nn.EmbeddingBag(n_vocab + 1, dim, mode="mean", padding_idx=0)
        self.deg = nn.Embedding(n_deg, 8)
        self.state = nn.Embedding(n_state, 16)
        self.mlp = nn.Sequential(nn.Linear(dim + 8 + 16 + 8 + dim, 128), nn.ReLU(), nn.Linear(128, dim))

    def user(self, feats, hist):
        """hist: sparse (B, nj) row-normalised matrix of the user's past applications."""
        h = torch.sparse.mm(hist, self.job.weight)
        x = torch.cat([self.tok(feats["tok"]), self.deg(feats["deg"]), self.state(feats["state"]),
                       feats["num"], h], 1)
        return F.normalize(self.mlp(x), dim=1)

    def jobs(self):
        return F.normalize(self.job.weight, dim=1)


def hist_tensor(X_rows):
    X = X_rows.tocoo()
    d = np.asarray(X_rows.sum(1)).ravel()
    vals = X.data / np.maximum(d[X.row], 1)
    idx = torch.tensor(np.vstack([X.row, X.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), X.shape).coalesce()


def train_two_tower(split, fz, epochs=8, lr=2e-3, batch=2048, n_neg=512, temp=0.1, drop_hist=0.5,
                    seed=0, log=False):
    """Sampled softmax. Each positive (u, j) is scored against shared uniform negatives.
    The target job is removed from the user's history input (leave-one-out) so the model
    cannot copy it, and with prob drop_hist the history is dropped entirely so the
    profile-only path is trained for cold-start users."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = TwoTower(split.n_jobs, len(fz.vocab), len(fz.degrees), len(fz.states))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    feats_all = fz.encode(split.users)
    X = split.X.tolil(copy=True)
    Xc = split.X.tocsr()
    tr_u, tr_j = split.train.u.values, split.train.j.values
    for ep in range(epochs):
        t0, tot = time.time(), 0.0
        perm = rng.permutation(len(tr_u))
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            u, j = tr_u[b], tr_j[b]
            H = Xc[u].tolil()
            for r in range(len(u)):           # leave-one-out
                H[r, j[r]] = 0
            H = H.tocsr()
            keep = (rng.random(len(u)) >= drop_hist).astype(np.float32)
            H = sp.diags(keep) @ H
            feats = {k: v[torch.as_tensor(u)] for k, v in feats_all.items()}
            eu = m.user(feats, hist_tensor(H))
            J = m.jobs()
            neg = torch.as_tensor(rng.integers(0, split.n_jobs, n_neg))
            pos = (eu * J[torch.as_tensor(j)]).sum(1, keepdim=True)
            negs = eu @ J[neg].T
            logits = torch.cat([pos, negs], 1) / temp
            loss = F.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(u)
        if log:
            print(f"  twotower ep{ep} loss={tot / len(tr_u):.4f} {time.time() - t0:.1f}s", flush=True)
    m.eval()
    return m, feats_all


def two_tower_user_emb(m, fz, user_ids, X_rows):
    with torch.no_grad():
        return m.user(fz.encode(user_ids), hist_tensor(X_rows)).numpy()


def two_tower_job_emb(m):
    with torch.no_grad():
        return m.jobs().numpy()
