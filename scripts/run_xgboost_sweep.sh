#!/usr/bin/env bash

# Runs the XGBoost sweep from within the baselines/ directory.

# ---- Config ----
BENCHMARK_NAME="bcb_fusion"
MODEL_NAME="qwen3"
LAYERS=(48)

# Token-position sets for each sweep.
TOKEN_POS_FIRST=(0)
TOKEN_POS_LAST=(-1)
TOKEN_POS_FIRST_LAST=(0 -1)
TOKEN_POS_FIRST_CODE=(-3)
TOKEN_POS_LAST_CODE=(-2)
TOKEN_POS_CODE=(-3 -2)

# Output model paths.
MODEL_PATH_FIRST="outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/first_token_model.json"
MODEL_PATH_LAST="outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/last_token_model.json"
MODEL_PATH_FIRST_LAST="outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/first_last_tokens_model.json"
MODEL_PATH_FIRST_CODE="outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/first_code_token_model.json"
MODEL_PATH_LAST_CODE="outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/last_code_token_model.json"
MODEL_PATH_CODE="outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}/code_tokens_model.json"

# ---- Execution ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINES_DIR="${REPO_ROOT}/baselines"

# Per-command log files (absolute paths so the script can be launched from anywhere).
LOG_DIR="xgboost_logs/${BENCHMARK_NAME}/${MODEL_NAME}"
LOG_FIRST="${LOG_DIR}/sweep_first_token.txt"
LOG_LAST="${LOG_DIR}/sweep_last_token.txt"
LOG_FIRST_LAST="${LOG_DIR}/sweep_first_last_tokens.txt"
LOG_FIRST_CODE="${LOG_DIR}/sweep_first_code_token.txt"
LOG_LAST_CODE="${LOG_DIR}/sweep_last_code_token.txt"
LOG_CODE="${LOG_DIR}/sweep_code_tokens.txt"

cd "${BASELINES_DIR}"

mkdir -p "outputs/xgboost/${BENCHMARK_NAME}/${MODEL_NAME}"
mkdir -p "${LOG_DIR}"

echo "[1/6] Sweeping: first token -> ${MODEL_PATH_FIRST} (log: ${LOG_FIRST})" |& tee "${LOG_FIRST}"
python -u xgboost_baseline.py \
  --model-path "${MODEL_PATH_FIRST}" \
  --layers "${LAYERS[@]}" \
  --token-positions "${TOKEN_POS_FIRST[@]}" \
  --filter-empty \
  --grid-search-learning-rate |& tee -a "${LOG_FIRST}"

echo "[2/6] Sweeping: last token -> ${MODEL_PATH_LAST} (log: ${LOG_LAST})" |& tee "${LOG_LAST}"
python -u xgboost_baseline.py \
  --model-path "${MODEL_PATH_LAST}" \
  --layers "${LAYERS[@]}" \
  --token-positions "${TOKEN_POS_LAST[@]}" \
  --filter-empty \
  --grid-search-learning-rate |& tee -a "${LOG_LAST}"

echo "[3/6] Sweeping: first+last token -> ${MODEL_PATH_FIRST_LAST} (log: ${LOG_FIRST_LAST})" |& tee "${LOG_FIRST_LAST}"
python -u xgboost_baseline.py \
  --model-path "${MODEL_PATH_FIRST_LAST}" \
  --layers "${LAYERS[@]}" \
  --token-positions "${TOKEN_POS_FIRST_LAST[@]}" \
  --filter-empty \
  --grid-search-learning-rate |& tee -a "${LOG_FIRST_LAST}"

echo "[4/6] Sweeping: first code token -> ${MODEL_PATH_FIRST_CODE} (log: ${LOG_FIRST_CODE})" |& tee "${LOG_FIRST_CODE}"
python -u xgboost_baseline.py \
  --model-path "${MODEL_PATH_FIRST_CODE}" \
  --layers "${LAYERS[@]}" \
  --token-positions "${TOKEN_POS_FIRST_CODE[@]}" \
  --filter-empty \
  --grid-search-learning-rate |& tee -a "${LOG_FIRST_CODE}"

echo "[5/6] Sweeping: last code token -> ${MODEL_PATH_LAST_CODE} (log: ${LOG_LAST_CODE})" |& tee "${LOG_LAST_CODE}"
python -u xgboost_baseline.py \
  --model-path "${MODEL_PATH_LAST_CODE}" \
  --layers "${LAYERS[@]}" \
  --token-positions "${TOKEN_POS_LAST_CODE[@]}" \
  --filter-empty \
  --grid-search-learning-rate |& tee -a "${LOG_LAST_CODE}"

echo "[6/6] Sweeping: code tokens -> ${MODEL_PATH_CODE} (log: ${LOG_CODE})" |& tee "${LOG_CODE}"
python -u xgboost_baseline.py \
  --model-path "${MODEL_PATH_CODE}" \
  --layers "${LAYERS[@]}" \
  --token-positions "${TOKEN_POS_CODE[@]}" \
  --filter-empty \
  --grid-search-learning-rate |& tee -a "${LOG_CODE}"

echo "Done. Logs written to:"
echo "  ${LOG_FIRST}"
echo "  ${LOG_LAST}"
echo "  ${LOG_FIRST_LAST}"
echo "  ${LOG_FIRST_CODE}"
echo "  ${LOG_LAST_CODE}"
echo "  ${LOG_CODE}"