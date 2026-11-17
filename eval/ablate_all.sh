#!/bin/bash
# Ablation sweep: fit experts once per dataset, re-stack under 15 variants.
# usage: eval/ablate_all.sh <data_root> <out_dir> <dataset_rel>...   (two datasets in parallel)
# env:   SB_OFFICIAL (starter kit), PY (python 3.13 with requirements-eval.txt)
D=$1; O=$2; shift 2; mkdir -p "$O"
R=$(cd "$(dirname "$0")/.." && pwd); export R D O PY=${PY:-python}
# ablation/ holds the modified main.py; every other file is the submitted one
A=$(mktemp -d); cp "$R"/submission/* "$A"/; cp "$R"/ablation/main.py "$A"/main.py; export A
export SBENCH_KUMO=off SBENCH_BUDGET=100000
export SBA_VARIANTS="full;drop_lgb;drop_cat;drop_net;drop_eb;drop_lin;drop_nn;drop_fact;nofactstack;noiso;notemp;nopq;fit950;avg;best1"
printf '%s\n' "$@" | xargs -P 2 -I{} bash -c 'n=$(echo {} | tr / _); $PY -X utf8 "$R"/eval/run_ablate.py "$A" "$D"/{} 1 "$O"/$n > "$O"/$n.log 2>&1; echo done {}'
