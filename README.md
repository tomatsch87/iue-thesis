# Introspective Uncertainty Estimation for LLM-Based Code Generation

Master-Thesis Repository -- Author: [Thomas Klassert](mailto:tomatsch87@protonmail.com)

## Abstract

Large Language Models (LLMs) are increasingly used for code generation but can produce fluent yet functionally incorrect outputs, which limits trust in their usage for practical software engineering workflows. This thesis investigates whether Introspective Uncertainty Estimation (IUE), based on internal hidden-state representations of LLMs, can reliably indicate correctness at the response and line levels for code generation tasks. The objective is to determine the extent to which hidden states encode information about functional code correctness and how this can be leveraged for practical risk assessment and fault localization. Methodologically, this thesis combines response-level evaluation on LiveCodeBench (LCB) and BigCodeBench (BCB) with an augmentation pipeline that derives token- and line-level labels from incorrect programs. In this setup, it compares static and dynamic response-level features, evaluates generalization across tasks, programming domains, and token positions, and studies line-level fault localization.

The results show that hidden states contain a strong response-level correctness signal. Static single-token probes perform best, reaching 0.90 AUROC and 0.96 F1 on LCB in the best settings, generally surpassing the thresholds of previously reported static probe baselines for IUE. More elaborate dynamic token-selection and sequence-modeling strategies yield no consistent gains. While generalization across tasks, domains, and token positions is feasible, setting-dependent degradation largely remains for real-world software projects. At a fine granularity, line-level prediction in mixed-program settings is substantially harder than response-level estimation. However, in a conditional localization setup with known-incorrect programs, Top-K point-of-failure ranking remains effective, achieving a Top-3 hit rate of 81% in the best setting. Overall, the findings suggest that hidden states are a robust and informative resource for estimating functional code correctness and localizing faults, supporting a two-stage workflow that combines response-level risk screening with targeted line-level prioritization, and motivate further research on IUE for LLM-based code generation.

## Paper

The paper can be found [here]().

## Datasets

The datasets can be found [here]().

## Citation

```

```

## General Usage Requirements

This repository contains data-generation, feature-extraction, feature-visualization, and IUE training pipelines for code generation benchmarks (LiveCodeBench and BigCodeBench).

### 1) System and Runtime

- Python 3.10+
- CUDA-capable GPU is strongly recommended for feature extraction and dataset generation

### 2) Data Requirements

Most training/evaluation scripts assume pre-extracted hidden-state `.pt` files organized by split:

```text
<data_root>/train/*.pt
<data_root>/validation/*.pt
<data_root>/test/*.pt
```

Default roots used by modules:

- `baselines/` defaults to `<IUE_DEFAULT_DATA_DIR>/feats_lcb_qwen3`
- `dynamic_token_select/` and `line_level/` default to `<IUE_DEFAULT_DATA_DIR>/feats_lcb_qwen3_code_segment`

Available Datasets:

- Published: the text datasets including generations and correctness labels used in my experiments (generated via `gen_dataset/`) are available see [Datasets](#datasets).
- Not published: the hidden-state feature datasets (`*.pt` files) used for model training are not distributed due to their size.

If the feature folders above do not exist, generate hidden-state features locally from a text dataset (either your own or the published text datasets see [Datasets](#datasets)) using `feat_extract/prefill_pipeline.py`.

### 3) Experiment Tracking / Optional Dependencies

- Training scripts in `baselines/`, `dynamic_token_select/`, and `line_level/` may use Weights & Biases (`wandb`) for runs and sweeps.
- Generation utilities under `gen_dataset/` rely on Hugging Face transformers/datasets/tokenizers and vLLM models.

### 4) Environment Variables

The project reads several environment variables from `global_variables.py`:

- `IUE_DEFAULT_DATA_DIR`: Base directory for dataset/features. Defaults to `../data` relative to the repository parent.
- `IUE_PROJECT`: Default W&B project name.
- `IUE_DEVICE`: Optional explicit device override (for example: `cpu`, `cuda`).
- `CUDA_VISIBLE_DEVICES`: Standard CUDA device visibility control.

## Installation

### 1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains the full research stack (PyTorch, vLLM, transformers, xgboost, etc.).

### 3) Sanity check

```bash
python -m baselines.baseline --help
python -m dynamic_token_select.train --help
python -m line_level.train_lstm --help
python -m token_level.xgboost_baseline --help
```

## High-Level Repository Structure

- `baselines/`: Baseline iue models (MLPs, XGBoost baselines, evaluation scripts, and markdown result artifacts).
- `dynamic_token_select/`: Sequence-aware token selection models (LSTM, ABMIL, fixed-token MLP), training/evaluation, and XGBoost variants.
- `token_level/`: Token-level XGBoost baselines and evaluation utilities.
- `line_level/`: Line-level iue models and evaluation scripts (LSTM, XGBoost, and pof (point-of-failure) evaluation).
- `feat_extract/`: Hidden-state extraction pipeline and domain mapping utilities.
- `gen_dataset/`: Dataset generation, filtering, augmentation, and evaluation scripts for LiveCodeBench and BigCodeBench.
- `feat_viz/`: Feature visualization scripts (t-SNE).
- `scripts/`: Shell automation for XGBoost sweeps and batch evaluation.
- `global_variables.py`, `global_utils.py`: Shared project defaults and common utilities.

## General End-to-End Workflow (Module-Level)

1. Generate benchmark samples (`gen_dataset/`) or skip if using the provided datasets.
2. Extract hidden-state features (`feat_extract/`).
3. Train uncertainty estimation models (`baselines/`, `dynamic_token_select/`, `token_level/`, `line_level/`).
4. Run evaluation scripts and collect metrics.

## Module Sections

All commands below assume you run them from the repository root with the virtual environment activated.

### gen_dataset

Overview of relevant files and subdirectories:

- `gen_dataset/livecodebench/build_dataset.py`: Generate LiveCodeBench responses with vLLM and attach response-level correctness.
- `gen_dataset/livecodebench/gen_split.py`: Create train/val/test splits from generation JSONL.
- `gen_dataset/livecodebench/augment_token_labels.py`: Augment token-level labels using minimal-edit fix generations.
- `gen_dataset/livecodebench/filter_augmented_jsonl.py`: Filter augmented candidates and add per-line error maps.
- `gen_dataset/livecodebench/gen_split_filtered.py`: Grouped stratified splitting for filtered augmented data.
- `gen_dataset/livecodebench/inspect_dataset.py`, `gen_dataset/livecodebench/inspect_token_labels.py`: Dataset analysis.
- `gen_dataset/bigcodebench/build_dataset.py`: Generate BigCodeBench responses with vLLM.
- `gen_dataset/bigcodebench/eval_jsonl_docker.py`: Re-evaluate responses and write `is_correct` labels.
- `gen_dataset/bigcodebench/to_raw_samples.py`: Flatten generation JSONL to sanitizer/evaluator sample format.
- `gen_dataset/bigcodebench/update_scores.py`: Merge external scored results back into original generation JSONL.

Typical workflow:

1. Generate model responses for the target benchmark.
2. Evaluate response-level correctness labels.
3. For LiveCodeBench token/line experiments, augment token/line labels and filter rows.
4. Split JSONL into train/validation/test files for downstream training tasks.

Example commands:

LiveCodeBench generation and preparation:

```bash
python -m gen_dataset.livecodebench.build_dataset \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --output-path data/livecodebench_qwen3.jsonl \
  --n 1 \
  --tensor-parallel-size 4

python -m gen_dataset.livecodebench.augment_token_labels \
  --input-path data/livecodebench_qwen3.jsonl \
  --output-path data/livecodebench_qwen3_augmented.jsonl \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --n 4 \
  --tensor-parallel-size 4

python -m gen_dataset.livecodebench.filter_augmented_jsonl \
  --input-path data/livecodebench_qwen3_augmented.jsonl \
  --output-path data/livecodebench_qwen3_augmented_filtered.jsonl \
  --add-line-error-map

python -m gen_dataset.livecodebench.gen_split_filtered \
  --input data/livecodebench_qwen3_augmented_filtered.jsonl \
  --output-dir data/livecodebench_filtered_splits
```

### feat_extract

Overview of relevant files:

- `feat_extract/prefill_pipeline.py`: Main hidden-state feature extraction pipeline based on JSONL model response datasets.
- `feat_extract/bcb_domain_mapper.py`: Reassemble BigCodeBench features into domain-based splits.
- `feat_extract/bcb_domains.json`: Library-to-domain mapping used by domain mapper.

Typical workflow:

1. Provide input text dataset JSONL files (train/validation/test).
2. Run prefill extraction pipeline to create feature `*.pt` files grouped by split.
3. Optionally create domain-holdout feature splits for cross-domain generalization evaluation.

Example commands:

```bash
python -m feat_extract.prefill_pipeline \
  --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --input-jsonl-map \
    train=data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_train.jsonl \
    validation=data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_val.jsonl \
    test=data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_test.jsonl \
  --output-dir /data/feats_lcb_qwen3 \
  --splits train validation test \
  --hidden-state-layers 48 \
  --hidden-state-token-positions first_code_token last_code_token

python -m feat_extract.bcb_domain_mapper \
  --data-dir /data/feats_bcb_qwen3 \
  --target-domain Network \
  --output-dir /data/feats_bcb_qwen3_ood_network
```

### feat_viz

Overview of relevant files:

- `feat_viz/tsne.py`: t-SNE visualization for a single layer/token setting.
- `feat_viz/multi_token_tsne.py`: t-SNE visualization over multiple token positions.

Typical workflow:

1. Select extracted features (`data-dir`, split, layer/token settings).
2. Run t-SNE script.

Example commands:

```bash
python -m feat_viz.tsne \
  --data-dir /data/feats_lcb_qwen3 \
  --split train \
  --layer 48 \
  --token-position 0 \
  --output figures/tsne_layer48_token0.png

python -m feat_viz.multi_token_tsne \
  --data-dir /data/feats_lcb_qwen3 \
  --split train \
  --layers 48 \
  --token-positions 0 -1 \
  --output figures/tsne_multi_layer_token.png
```

### baselines

Overview of relevant files and subdirectories:

- `baselines/baseline.py`: Train MLP baselines (`openia`, `snyder`) with optional W&B sweeps.
- `baselines/openia.py`, `baselines/snyder.py`: Model definitions.
- `baselines/dataset.py`: Feature dataset loaders used by MLP and XGBoost baselines.
- `baselines/xgboost_baseline.py`: Train XGBoost baseline with optional grid search.
- `baselines/test_openia_baseline.py`: Evaluate OpenIA checkpoint on test split.
- `baselines/test_xgboost.py`: Evaluate XGBoost checkpoint on test split.
- `baselines/xgboost_rt_baseline.py`, `baselines/test_xgboost_rt.py`: Random-tail XGBoost variants used for cross-token-position generalization experiments.
- `baselines/evaluation/compute_ood_metrics.py`: Aggregate OOD metrics from markdown eval tables.
- `baselines/evaluation/*.md`: Stored evaluation outputs and benchmark summaries.

Typical workflow:

1. Train an MLP or XGBoost baseline.
2. Evaluate the trained model checkpoint on the test split.

Example commands:

MLP baseline (OpenIA):

```bash
python -m baselines.baseline \
  --baseline openia \
  --data-root /data/feats_lcb_qwen3 \
  --layers 48 \
  --token-positions -2 \
  --output-dir outputs/openia

python -m baselines.test_openia_baseline \
  --checkpoint-path outputs/openia/best_model.pt \
  --test-split test
```

XGBoost baseline:

```bash
python -m baselines.xgboost_baseline \
  --data-root /data/feats_lcb_qwen3 \
  --layers 48 \
  --token-positions -1 \
  --model-path outputs/xgboost/model.json \
  --grid-search-learning-rate

python -m baselines.test_xgboost \
  --model-path outputs/xgboost/model.json \
  --data-root /data/feats_lcb_qwen3 \
  --layers 48 \
  --token-positions -1
```

### dynamic_token_select

Overview of relevant files:

- `dynamic_token_select/train.py`: Train script for sequence models (`dynamic_token_lstm`, `abmil`) and MLP baseline (`fixed_token_mlp`).
- `dynamic_token_select/lstm.py`, `dynamic_token_select/abmil.py`, `dynamic_token_select/mlp.py`: Model definitions and sweep configs.
- `dynamic_token_select/dataset.py`: Sequence dataset loader with dynamic token filtering options.
- `dynamic_token_select/xgboost_baseline.py`: Train XGBoost baseline on top-k entropy filtered sequence features.
- `dynamic_token_select/test_lstm.py`: Evaluation script for LSTM sequence models.

Typical workflow:

1. Train a LSTM sequence model with a chosen token filter.
2. Optionally launch W&B sweeps for the selected architecture.
3. Evaluate the trained model checkpoint on the test split.

Example commands:

```bash
python -m dynamic_token_select.train \
  --model dynamic_token_lstm \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --layers 48 \
  --token-filter top_k-token-entropy \
  --token-filter-config '{"k":5}' \
  --output-dir outputs/dynamic_token_lstm

python -m dynamic_token_select.test_lstm \
  --model-path outputs/dynamic_token_lstm/best_model.pt \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --layers 48 \
  --token-filter top_k-token-entropy \
  --token-filter-config '{"k":5}'
```

### token_level

Overview of relevant files:

- `token_level/dataset.py`: Maps filtered/augmented token labels to existing hidden-state feature vectors.
- `token_level/xgboost_baseline.py`: Train token-level XGBoost baseline.
- `token_level/test_xgboost.py`: Evaluate token-level model and report line-level metrics.

Typical workflow:

1. Prepare filtered/augmented JSONL dataset splits with token labels.
2. Train a token-level XGBoost model.
3. Evaluate the trained model checkpoint on the test split.

Example commands:

```bash
python -m token_level.xgboost_baseline \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --train-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_train.jsonl \
  --valid-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_val.jsonl \
  --layers 48 \
  --model-path outputs/token_level/xgboost/model.json

python -m token_level.test_xgboost \
  --model-path outputs/token_level/xgboost/model.json \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --test-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_test.jsonl \
  --layers 48
```

### line_level

Overview of relevant files:

- `line_level/sequence_dataset.py`: Builds per-line token sequences for sequence models.
- `line_level/dataset.py`: Builds per-line feature vectors for XGBoost baselines.
- `line_level/train_lstm.py`, `line_level/test_lstm.py`: Train/evaluate line-sequence LSTM models.
- `line_level/xgboost_baseline.py`, `line_level/test_xgboost.py`: Train/evaluate line-level XGBoost baselines.
- `line_level/pof_test_line_xgboost.py`: Evaluate pof (point-of-failure) localization, Hit@K, using line-level model scores.
- `line_level/pof_test_token_xgboost.py`: Evaluate pof localization, Hit@K, using token-level model scores aggregated to lines.

Typical workflow:

1. Train either line-sequence LSTM or line-level XGBoost on filtered/augmented JSONL dataset splits with line labels.
2. Evaluate the trained model checkpoint on the test split.
3. Optionally compute Hit@K localization metrics (pof-style evaluation).

Example commands:

Line-sequence LSTM path:

```bash
python -m line_level.train_lstm \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --train-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_train.jsonl \
  --valid-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_val.jsonl \
  --layers 48 \
  --output-dir outputs/line_level/lstm

python -m line_level.test_lstm \
  --model-path outputs/line_level/lstm/best_line_lstm.pt \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --test-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_test.jsonl \
  --layers 48
```

Line-level XGBoost path:

```bash
python -m line_level.xgboost_baseline \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --train-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_train.jsonl \
  --valid-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_val.jsonl \
  --layers 48 \
  --line-representation last_token \
  --model-path outputs/line_level/xgboost/model.json

python -m line_level.test_xgboost \
  --model-path outputs/line_level/xgboost/model.json \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --test-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_test.jsonl \
  --layers 48 \
  --line-representation last_token
```

Optional pof localization evaluation:

```bash
python -m line_level.pof_test_line_xgboost \
  --model-path outputs/line_level/xgboost/model.json \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --test-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_test.jsonl \
  --layers 48

python -m line_level.pof_test_token_xgboost \
  --model-path outputs/token_level/xgboost/model.json \
  --data-root /data/feats_lcb_qwen3_code_segment \
  --test-jsonl data/livecodebench_filtered_splits/livecodebench_qwen3_augmented_filtered_test.jsonl \
  --layers 48
```
