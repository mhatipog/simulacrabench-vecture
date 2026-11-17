# SimulacraBench 2026 — team Vecture

Cross-fitted, gate-factorised stacking of small survey experts. This repository reproduces the team's single test-phase submission (Codabench submission 1053704, 15 Nov 2026) and every offline number in the technical report.

| | |
|---|---|
| Submitted archive | `SimulacraBench_F5_KumoStackRobust.zip`, SHA-256 `07e4ba37ac77bc2d78b8695ba0a984cc21df74ba61a7de02395cf29b59c76f75` |
| Test phase (leaderboard, 2 decimals) | skill 0.49 · UNICEF 0.34 · World Bank 0.24 · UNHCR 0.88 |
| Development (same archive, 13 Nov) | skill 0.4878 · UNICEF 0.3499 · World Bank 0.2486 · UNHCR 0.8650 (noised and bucketed by the platform) |
| Licence | MIT (this code). Kumo Tabular weights: OpenMDW 1.1, downloaded by the platform from `nvidia/Kumo-Tabular`. |

## Layout

| Path | What |
|---|---|
| `submission/` | The eleven files of the uploaded archive, unchanged (`SUBMISSION_SHA256.txt` lists their hashes). |
| `tools/build_submission_zip.py` | Rebuilds the uploaded archive byte for byte (Python 3.13; other zlib builds may compress differently, the file contents still match). |
| `ablation/main.py` | `submission/main.py` with the stacking stage wrapped so it can be re-run under ablation flags. With no flag it reproduces the submitted predictions exactly. |
| `eval/` | Dataset builders, evaluation, ablation sweep, paired bootstrap, table generation. |
| `results/` | Official score files, per-dataset ablation logs, summaries and the tables used in the papers. |
| `paper/report/` | Four-page competition report (organizers' template, NeurIPS 2026 style). |
| `paper/tmlr/` | Extended manuscript (TMLR style). |

## Reproduce

```bash
# 0. environment (Python 3.13)
pip install -r requirements-eval.txt
git clone https://github.com/SituatedEvals/public starter-kit    # organizers' scorer, MIT
export SB_OFFICIAL=$PWD/starter-kit

# 1. the submitted archive, byte for byte
python tools/build_submission_zip.py

# 2. evaluation data (GSS 2022 Stata file from NORC, public)
curl -LO https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/2022_stata.zip && unzip 2022_stata.zip
bash eval/make_datasets.sh starter-kit 2022_stata/GSS2022.dta data

# 3. run the submission on one dataset (no GPU: Kumo off)
SBENCH_KUMO=off python eval/run_eval.py submission data/sandbox/unicef 1

# 4. ablation sweep and tables
bash eval/ablate_all.sh data results/ablation real/gss0 real/gss1 real/gss2 sim/unhcr_A sim/unhcr_B sim/unhcr_C sim/twin_C sim/twin_D sandbox/unicef sandbox/world_bank sandbox/unhcr
python eval/make_tables.py results/ablation results/tables

# 5. papers
cd paper/report && tectonic report.tex
cd ../tmlr && tectonic main.tex
```

The hosted run used one H100, 8 CPUs, 16 GB, no network. `predict()` times every expert on the host and picks the largest plan that fits the phase limit (900 s Development, 3,600 s Test), so a CPU-only machine may choose fewer experts or folds. `SBENCH_BUDGET=100000` forces the full plan; the ablations use it so that every variant shares one plan.

## What is and is not verified

- The ablation build's `full` variant gives per-cell log probabilities identical to the submitted code (maximum absolute difference 0 on the GSS check set).
- Refitting all experts with two other seeds (`SBA_SEED=1`, `2`, ablation build only) moves the GSS-1 full-stack loss by at most 0.11 milli-nats per cell and flips no non-null ablation effect (`results/tables/seed_check_gss0.json`).
- `results/ablation_logs/` holds the RESULT lines of every run; `results/tables/` the summaries (`ablation_summary.json`, `ablation_full.csv` with intervals) and the LaTeX tables. Per-cell log probabilities (`*.npz`, 66 MB) are attached to the GitHub release `v1.0-report`.
- All offline numbers use the organizers' `sample_rows`, `floored`, `marginal_reference` and `score`. They are proxies (a public survey, synthetic generators and the organizer sandbox), not leaderboard scores.
- Kumo was never run on a GPU locally. On the hosted machine the UNICEF and World Bank Kumo blend is mandatory (a failure would have failed the submission); whether the optional UNHCR Kumo pass covered any question is not observable from the returned files.
- No competition microdata, hidden labels or leaderboard probing are used anywhere. The GSS file is used for offline validation only.
