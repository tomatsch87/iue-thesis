#!/usr/bin/env bash

# Evaluate 6 trained XGBoost baselines on the test split and append results as a Markdown table.

# ---- Config ----
BENCHMARK_NAME="bcb_domain_general"
MODEL_NAME="qwen3"
LAYERS=(48)
FILTER_EMPTY=true

# Where to append the markdown table (path is relative to baselines/).
TARGET_MD="eval/bcb_domain_general.md"

# Optional: keep per-baseline raw metric outputs.
WRITE_LOGS=false
LOG_DIR="xgboost_logs/${BENCHMARK_NAME}/${MODEL_NAME}/test"

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

# Format: label|model_path|token_positions...
BASELINES=(
  "first token|outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/first_token_model.json|0"
  "last token|outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/last_token_model.json|-1"
  "first+last token|outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/first_last_tokens_model.json|0|-1"
  "first code token|outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/first_code_token_model.json|-3"
  "last code token|outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/last_code_token_model.json|-2"
  "first+last code token|outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/code_tokens_model.json|-3|-2"
)

append_table() {
  {
    echo
    echo "### ${MODEL_NAME}"
    echo
    echo "Benchmark: ${BENCHMARK_NAME}  "
    echo "Layers: ${LAYERS[*]}  "
    echo
    echo "| Baseline Model | roc_auc | bss | f1 | accuracy |"
    echo "|----------------|--------:|----:|---:|---------:|"
  } >> "${TARGET_MD}"

  for spec in "${BASELINES[@]}"; do
    IFS='|' read -r label model_path token1 token2 <<< "${spec}"

    token_args=("${token1}")
    if [[ -n "${token2:-}" ]]; then
      token_args+=("${token2}")
    fi

    if [[ ! -f "${model_path}" ]]; then
      # Model is missing: still append a row for traceability.
      {
        printf "| %s | %s | %s | %s | %s |\n" "${label}" "N/A" "N/A" "N/A" "missing model"
      } >> "${TARGET_MD}"
      continue
    fi

    local log_path out roc_auc bss f1 accuracy
    log_path="${LOG_DIR}/$(echo "${label}" | tr ' +' '__').txt"

    if [[ "${WRITE_LOGS}" == "true" ]]; then
      out="$(
        python -u test_xgboost.py \
          --model-path "${model_path}" \
          --layers "${LAYERS[@]}" \
          --token-positions "${token_args[@]}" \
          "${_filter_flag[@]}" \
        |& tee "${log_path}"
      )"
    else
      out="$(
        python -u test_xgboost.py \
          --model-path "${model_path}" \
          --layers "${LAYERS[@]}" \
          --token-positions "${token_args[@]}" \
          "${_filter_flag[@]}"
      )"
    fi

    roc_auc="$(awk -F': ' '/^roc_auc:/ {print $2; exit}' <<< "${out}" || true)"
    bss="$(awk -F': ' '/^bss:/ {print $2; exit}' <<< "${out}" || true)"
    f1="$(awk -F': ' '/^f1:/ {print $2; exit}' <<< "${out}" || true)"
    accuracy="$(awk -F': ' '/^accuracy:/ {print $2; exit}' <<< "${out}" || true)"

    # Fall back to N/A if metrics were not printed for some reason.
    roc_auc="${roc_auc:-N/A}"
    bss="${bss:-N/A}"
    f1="${f1:-N/A}"
    accuracy="${accuracy:-N/A}"

    {
      printf "| %s | %s | %s | %s | %s |\n" "${label}" "${roc_auc}" "${bss}" "${f1}" "${accuracy}"
    } >> "${TARGET_MD}"
  done
}

append_table

echo "Appended results to: ${BASELINES_DIR}/${TARGET_MD}"
if [[ "${WRITE_LOGS}" == "true" ]]; then
  echo "Raw outputs in: ${BASELINES_DIR}/${LOG_DIR}"
fi
