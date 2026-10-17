# Job Recommendation with a Graph Neural Network and Two-Stage Ranking

Recommends jobs to job seekers from real CareerBuilder application data (Kaggle "Job
Recommendation Challenge", 2012). It answers one question: given what a person has applied to so far, and their
profile, which open jobs will they apply to next?

## Data
- `apps.tsv`: 1,603,111 real applications, 321,235 users, 365,668 jobs, 7 independent
  time windows (no user or job appears in more than one window).
- `users.tsv` (degree, major, graduation date, experience, state) and `user_history.tsv`
  (prior job titles).
- Source: GitHub mirror `ruchikahazariwal/Job_Recommendation_System` (the original Kaggle
  competition files). SHA-256 of each file is in `data_raw/SHA256SUMS`.
- `jobs.tsv` (job title/description) is not in the mirror, so the models have **no job
  content**. Jobs are represented only by who applied to them.

## Evaluation (windows 3, 4, 5)
Within each window, applications are split by time: the model trains on the first 80%
and is scored on the last 20% (future applications). Each user's held-out jobs are ranked
against **every** job in the window (~44K-47K jobs), not a sampled subset. Jobs a user
already applied to are excluded. 23,151 users with history are evaluated, plus 12,095
brand-new users with no history.

## Models
1. **Popularity**: recency-weighted application counts.
2. **ItemKNN**: "people who applied to X also applied to Y" (cosine co-application).
3. **LightGCN** (graph neural network): user and job embeddings propagated 3 layers over
   the user-job application graph, BPR loss, PyTorch.
4. **Two-stage**: LightGCN top-100 + ItemKNN top-100 as candidates (~160 per user), then a
   LightGBM **LambdaRank** reranker on retriever scores/ranks, job demand and user profile
   features. The reranker is trained on a separate earlier period (70-80%) so it never sees
   test labels.
5. **Profile two-tower** (cold start): the user tower encodes degree, state, experience,
   years since graduation and prior job titles + major; it also pools past applications
   when they exist. Scores new users who have zero applications, which LightGCN cannot.

## Results (held-out future applications, 23,151 users)
| Model | Recall@10 | Recall@50 | nDCG@10 |
|---|---|---|---|
| Popularity | 0.006 | 0.022 | 0.004 |
| LightGCN (GNN) alone | 0.112 | 0.287 | 0.075 |
| ItemKNN | 0.140 | 0.310 | 0.095 |
| **Two-stage (GNN + KNN → LambdaRank)** | **0.167** | **0.370** | **0.110** |

- Two-stage vs ItemKNN, paired bootstrap over users (95% CI): Recall@10 +0.028
  [+0.025, +0.031], a 19.9% relative lift; nDCG@10 +0.016 [+0.014, +0.017].
- Same direction in all 3 windows (Recall@10 0.176 / 0.163 / 0.164 vs ItemKNN 0.142 / 0.137 / 0.140).
- **LightGCN alone lost to ItemKNN** (Recall@10 −0.027 [−0.031, −0.024]). Its value is as a
  candidate generator and reranker signal, shown by the ablation below.
- Early-career users (graduated 2010 or later, 4,697 users): two-stage Recall@10 0.165 vs
  ItemKNN 0.133.

### GNN ablation (window 3, 7,309 users, `results_v2/`)
Same reranker, same candidate budget (~160/user), with vs without the LightGCN candidates and
features (LightGCN retrained with batch 2048):

| | Recall@10 | Recall@50 | nDCG@10 |
|---|---|---|---|
| Reranker without GNN (ItemKNN candidates only) | 0.160 | 0.335 | 0.104 |
| **Reranker with GNN** | **0.173** | **0.375** | **0.113** |

Adding the GNN: Recall@10 +0.013 [95% CI +0.007, +0.018] (+8.0% relative), Recall@50 +0.040
[+0.034, +0.046] (+11.8%), nDCG@10 +0.009 [+0.006, +0.012]. One window only.

### Employer-side promotion tradeoff (window 3, `results_v2/`)
No price or revenue data exists in this dataset, so this is a proxy for promoted-job
products, not a revenue result. "Under-served" jobs are the bottom half of jobs by prior
applicants (the jobs an employer would pay to promote). Ranking score = per-user rank
percentile of the reranker score + λ × under-served.

| λ | Recall@10 | Top-10 slots to under-served jobs | Real applications to under-served jobs |
|---|---|---|---|
| 0 | 0.1732 | 4.2% | 81 |
| 0.05 | 0.1736 | 7.5% | 139 |
| 0.10 | 0.1712 | 11.6% | 196 |
| 0.20 | 0.1605 | 20.1% | 272 |
| 0.50 | 0.1317 | 42.0% | 435 |

At λ=0.10, applications delivered to under-served jobs rose 2.4x (81 → 196) for a 1.2%
relative Recall@10 cost; past λ=0.2 the relevance cost grows quickly.

**New users with no applications (12,095):** profile two-tower Recall@10 0.042 vs
popularity 0.005; Recall@50 0.141 vs 0.020.

**Notification send policy:** sending each user only their top-ranked job, precision per
notification is 7.0% if everyone is notified, rising to 11.2% if only the top 10% most
confident users are notified. That 10% captures 16% of the hits (259 of 1,625), so
precision is bought by sending far fewer notifications.

## Limitations
- 2012 data from a general job board, not a student network.
- No job content: 11.3% of held-out applications go to jobs with no prior applicants and
  cannot be scored. They are reported, not hidden.
- 3 of 7 windows evaluated, one seed, limited tuning.
- No revenue or pricing data: the promotion analysis is an exposure/application proxy, not revenue.

## Run
Download `apps.tsv`, `users.tsv` and `user_history.tsv` from the mirror into `data_raw/`
and check them against `data_raw/SHA256SUMS`.
```
pip install -r requirements.txt
python src/prep.py            # splits + user features
python src/pipeline.py 3 4 5  # trains and evaluates (~13 min/window on 2 CPUs)
python src/report.py          # bootstrap CIs -> results/summary.json
# results_v2/: GNN ablation + promotion tradeoff (pipeline.py 3); src/pipeline_v1.py produced results/
pytest -q tests               # 4 tests
```
