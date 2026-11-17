#!/bin/bash
# Rebuild every evaluation set used in the report (exact commands and seeds).
# usage: eval/make_datasets.sh <starter_kit_dir> <GSS2022.dta> <out_root>
# Starter kit: https://github.com/SituatedEvals/public (commit aa03186f used).
# GSS 2022: https://gss.norc.org/content/dam/gss/get-the-data/documents/stata/2022_stata.zip
set -e
KIT=$(cd "$1" && pwd); GSS=$2; OUT=$3; E=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python}
mkdir -p "$OUT"/real "$OUT"/sim "$OUT"/sandbox
# Real survey: three studies, disjoint sets of 70 questions, IID respondent split
$PY "$E"/gss_build.py "$GSS" "$OUT"/real/gss 3 70
# Gate-mechanism simulators on the UNHCR schema: <seed> <n_train> <n_dev> <max_targets> <structural share> <gate mode>
for cfg in "A 1 0.5 noisy" "B 2 0.0 noisy" "C 3 0.5 strict"; do set -- $cfg
  $PY "$E"/realsim.py "$KIT"/data/unhcr.json "$OUT"/sim/unhcr_$1 $2 2600 600 45 $3 $4; done
# Panel-twin simulators: random roles (twin_C) and household-grouped roles (twin_D)
$PY "$E"/twinsim_conc.py    "$KIT"/data/unhcr.json "$OUT"/sim/twin_C 11 4000 600 50 .2 noisy
$PY "$E"/twinsim_grouped.py "$KIT"/data/unhcr.json "$OUT"/sim/twin_D 11 4000 600 50 .2 noisy
# Organizer sandboxes at declared sizes
for s in unicef world_bank unhcr; do (cd "$KIT" && $PY make_sandbox.py --schema data/$s.json --out "$OUT"/sandbox/$s); done
