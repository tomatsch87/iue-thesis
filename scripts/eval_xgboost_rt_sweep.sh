#!/usr/bin/env bash

# Evaluate random-tail XGBoost sweep on the test split and append results as a Markdown table.

# ---- Config ----
BENCHMARK_NAME="bcb"
MODEL_NAME="gptoss"
LAYER=24
FILTER_EMPTY=true

# Tail fractions to evaluate (0.0 < value <= 1.0)
TAIL_FRACTIONS=(0.001 0.05 0.25 0.5 0.75 0.95 1.0)

# Data root for hidden-state features.
DATA_ROOT="/data/feats_${BENCHMARK_NAME}_${MODEL_NAME}_code_segment"

# Where to append the markdown table.
TARGET_MD="eval/${BENCHMARK_NAME}_rt_eval.md"

# Optional: keep per-model raw metric outputs.
WRITE_LOGS=false
LOG_DIR="xgboost_logs/${BENCHMARK_NAME}/${MODEL_NAME}/random_tail_test"

# ---- Derived config ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="${REPO_ROOT}/baselines"

cd "${BASELINES_DIR}"

mkdir -p "$(dirname "${TARGET_MD}")"
if [[ "${WRITE_LOGS}" == "true" ]]; then
  mkdir -p "${LOG_DIR}"
fi

_filter_flag=()
if [[ "${FILTER_EMPTY}" == "true" ]]; then
  _filter_flag=(--filter-empty)
fi

append_table() {
  {
    echo
    echo "### ${MODEL_NAME}"
    echo
    echo "Benchmark: ${BENCHMARK_NAME}  "
    echo "Layer: ${LAYER}  "
    echo
    echo "| Tail Fraction | roc_auc | bss | f1 | accuracy |"
    echo "|---------------|--------:|----:|---:|---------:|"
  } >> "${TARGET_MD}"

  for tail_fraction in "${TAIL_FRACTIONS[@]}"; do
    tail_pct=$(python - "$tail_fraction" <<'PY'
import sys
v = float(sys.argv[1])
print(int(round(v * 100)))
PY
    )

    model_path="outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/rt_${tail_pct}_model.json"

    if [[ ! -f "${model_path}" ]]; then
      {
        printf "| %s | %s | %s | %s | %s |\n" "${tail_fraction}" "N/A" "N/A" "N/A" "missing model"
      } >> "${TARGET_MD}"
      continue
    fi

    local log_path out roc_auc bss f1 accuracy
    log_path="${LOG_DIR}/rt_${tail_pct}.txt"

    if [[ "${WRITE_LOGS}" == "true" ]]; then
      out="$(
        python -u test_xgboost_random_tail.py \
          --model-path "${model_path}" \
          --data-root "${DATA_ROOT}" \
          --layer "${LAYER}" \
          --tail-fraction "${tail_fraction}" \
          "${_filter_flag[@]}" \
        |& tee "${log_path}"
      )"
    else
      out="$(
        python -u test_xgboost_random_tail.py \
          --model-path "${model_path}" \
          --data-root "${DATA_ROOT}" \
          --layer "${LAYER}" \
          --tail-fraction "${tail_fraction}" \
          "${_filter_flag[@]}"
      )"
    fi

    roc_auc="$(awk -F': ' '/^roc_auc:/ {print $2; exit}' <<< "${out}" || true)"
    bss="$(awk -F': ' '/^bss:/ {print $2; exit}' <<< "${out}" || true)"
    f1="$(awk -F': ' '/^f1:/ {print $2; exit}' <<< "${out}" || true)"
    accuracy="$(awk -F': ' '/^accuracy:/ {print $2; exit}' <<< "${out}" || true)"

    roc_auc="${roc_auc:-N/A}"
    bss="${bss:-N/A}"
    f1="${f1:-N/A}"
    accuracy="${accuracy:-N/A}"

    {
      printf "| %s | %s | %s | %s | %s |\n" "${tail_fraction}" "${roc_auc}" "${bss}" "${f1}" "${accuracy}"
    } >> "${TARGET_MD}"
  done
}

append_table

echo "Appended results to: ${BASELINES_DIR}/${TARGET_MD}"
if [[ "${WRITE_LOGS}" == "true" ]]; then
  echo "Raw outputs in: ${BASELINES_DIR}/${LOG_DIR}"
fi
