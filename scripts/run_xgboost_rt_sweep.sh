#!/usr/bin/env bash

# Runs random-tail XGBoost sweep from within the baselines/ directory.

# ---- Config ----
BENCHMARK_NAME="bcb"
MODEL_NAME="olmo"
LAYERS=(64)

# Tail fractions to sweep over
TAIL_FRACTIONS=(0.001 0.05 0.25 0.5 0.75 0.95 1.0)

# ---- Execution ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="${REPO_ROOT}/baselines"

cd "${BASELINES_DIR}"

counter=1
total=${#TAIL_FRACTIONS[@]}

for tail_fraction in "${TAIL_FRACTIONS[@]}"; do
  # Convert to percentage without decimal for naming, e.g. 0.25 -> 25
  tail_pct=$(python - "$tail_fraction" <<'PY'
import sys
v = float(sys.argv[1])
print(int(round(v * 100)))
PY
  )

  model_path="outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/rt_${tail_pct}_model.json"

  echo "[${counter}/${total}] Sweeping: tail=${tail_fraction} -> ${model_path}" 
  python -u xgboost_random_tail_baseline.py \
    --model-path "${model_path}" \
    --data-root "/data/feats_${BENCHMARK_NAME}_${MODEL_NAME}31instruct_code_segment" \
    --layer "${LAYERS[@]}" \
    --tail-fraction "${tail_fraction}" \
    --filter-empty \
    --grid-search-learning-rate

  counter=$((counter + 1))
done

echo "Done."
