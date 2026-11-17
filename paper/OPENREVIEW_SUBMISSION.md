# OpenReview submission — SimulacraBench technical report (team Vecture)

Venue: <https://openreview.net/group?id=NeurIPS.cc/2026/Competition/SimulacraBench>
Form closes **20 Nov 2026 10:59 UTC** (11:59 Brussels), expiry 11:29 UTC. Read from the OpenReview API on 17 Nov.

Only the author can submit: it needs your OpenReview profile (`~Your_Name1`) and the three attestations are yours to give.

| Field | Value to enter |
|---|---|
| Title | SimulacraBench Technical Report: Cross-fitted, Gate-factorised Stacking of Small Survey Experts |
| Authors | Your OpenReview profile; institution Vecture BV, Belgium. Belgium is not one of the 14 data-collection countries, so the Roots of the Data Prize does not apply. |
| Team ID | `Vecture` (the "Your team name" column on the Codabench test leaderboard; participant `mhatipog`, submission 1053704) |
| Keywords | survey completion, stacking, cross-fitting, calibration, skip logic, log score, tabular foundation models |
| TL;DR | Cross-fitted per-question stacking of small survey experts, with the log score factorised into a skip term and an answer term for gated questions; 0.49 test skill. |
| Abstract | Copy the abstract from `report/report.tex` (plain text; the form takes Markdown/TeX). |
| PDF | `report/report.pdf` |
| Code URL | `https://github.com/mhatipog/simulacrabench-vecture` (currently **private**; must be made public before submitting, the form asks for a public repository) |
| Code licence | MIT |
| Base models | `nvidia/Kumo-Tabular` (small and large classifier checkpoints, Hugging Face main revision as pre-fetched by the platform; c5893d4 at time of writing) via structured-data-models 0.2.1 |
| External data | General Social Survey 2022 cross-section (NORC at the University of Chicago, https://gss.norc.org, public Stata release, 4,149 respondents, free for research use). Used offline only, to build three validation studies of 70 questions each for model selection and ablations. Not used by, or shipped with, the submission. |
| Data-use attestation | Your decision. Nothing in this work re-identified respondents, linked records or redistributed microdata. |
| E-mail sharing / data release | Your decision. |

## Before you submit

1. Fill `\authoremail` in `report/meta.tex`, rebuild the PDF (`tectonic report.tex`).
2. Make the code repository public (`gh repo edit mhatipog/simulacrabench-vecture --visibility public --accept-visibility-change-consequences`) and set `\codecommit` in `meta.tex` to the release commit.
3. Optional: download the **Test-phase** scoring output of submission 1053704 from Codabench and record the unrounded values. The files in Downloads (`scoring_result (10).zip`, `(11).zip`) are byte-identical to the 13 Nov Development F5 result, so they are not the test output.
