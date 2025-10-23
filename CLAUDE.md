# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PSY Agents NO-AUG** is a baseline implementation for clinical text analysis (Criteria and Evidence extraction from DSM-5 diagnostic posts) **without data augmentation**. This is a control repository for comparing against augmentation-enhanced models.

**Critical Principle:** This repository enforces **STRICT field separation** to prevent data leakage:
- **Criteria Task**: Uses ONLY `status` field from annotations
- **Evidence Task**: Uses ONLY `cases` field from annotations
- Any code violating this separation will fail with `AssertionError`

**Development Environment:** This project uses VS Code Dev Containers (`.devcontainer/`) for consistent development and training environments with CUDA support.

## Essential Commands

### Development Environment
```bash
# Open in VS Code Dev Container (recommended)
# 1. Install "Dev Containers" extension in VS Code
# 2. Open folder in VS Code
# 3. Click "Reopen in Container" when prompted
# Container includes: CUDA 12.1, Python 3.10, Poetry, PyTorch

# Or develop locally:
make setup              # Full setup: dependencies + pre-commit + tests
make install            # Install dependencies only
poetry install          # Direct poetry install
```

### Data Preparation
```bash
# Generate ground truth from HuggingFace (recommended)
make groundtruth
# Or: python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5

# Generate from local CSV
make groundtruth-local
```

### Training
```bash
# Train with defaults (criteria, roberta_base)
make train

# Train specific task/model
make train TASK=criteria MODEL=roberta_base

# Or use CLI directly with Hydra overrides
python -m psy_agents_noaug.cli train \
    task=criteria \
    model=roberta_base \
    training.num_epochs=20 \
    training.batch_size=32

# Train Criteria architecture standalone
python scripts/train_criteria.py

# Evaluate Criteria model
python scripts/eval_criteria.py checkpoint=outputs/checkpoints/best_checkpoint.pt
```

### Hyperparameter Optimization (HPO)
```bash
# Multi-stage HPO workflow
make hpo-s0             # Stage 0: Sanity (2 trials)
make hpo-s1             # Stage 1: Coarse (20 trials)
make hpo-s2             # Stage 2: Fine (50 trials)
make refit              # Stage 3: Refit on train+val

# Or run all stages
make full-hpo

# Use CLI for custom HPO
python -m psy_agents_noaug.cli hpo \
    hpo=stage1_coarse \
    task=criteria \
    model=roberta_base
```

### Testing and Quality
```bash
make test               # Run all tests
make test-cov           # With coverage report
make test-groundtruth   # Test strict validation rules only

make lint               # Run ruff + black --check
make format             # Format with ruff + black

make pre-commit-run     # Run all pre-commit hooks

# Single test file
poetry run pytest tests/test_groundtruth.py -v
```

### MLflow and Evaluation
```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Evaluate model
make eval CHECKPOINT=outputs/checkpoints/best_checkpoint.pt

# Export metrics to CSV
make export
```

## Architecture Overview

### Four Supported Architectures

Located in `src/psy_agents_noaug/architectures/` and `src/Project/`:

1. **Criteria** (`criteria/`): Post ↔ criterion pair encoder + binary classifier
   - Binary classification (status: present/absent)
   - Model: Transformer encoder → pooled output → classification head
   - Dataset: `CriteriaDataset` (uses ONLY `status` field)

2. **Evidence** (`evidence/`): Post ↔ criterion span extractor
   - Span prediction (start/end positions)
   - Model: Transformer encoder → span prediction head
   - Dataset: `EvidenceDataset` (uses ONLY `cases` field)

3. **Share** (`share/`): Shared encoder with dual heads
   - Single encoder for both tasks
   - Separate classification and span heads

4. **Joint** (`joint/`): Dual encoders with fusion
   - Separate encoders for criteria and evidence
   - Fusion layer before evidence head

### Data Pipeline Architecture

**STRICT Validation Flow:**
```
Raw Data → Field Map Validation → Groundtruth Generation
                                           ↓
                     +--------------------+--------------------+
                     ↓                                         ↓
            Criteria Groundtruth                     Evidence Groundtruth
            (status field ONLY)                      (cases field ONLY)
                     ↓                                         ↓
              CriteriaDataset                          EvidenceDataset
```

**Key Files:**
- `configs/data/field_map.yaml`: Defines field mappings and validation rules
- `src/psy_agents_noaug/data/groundtruth.py`: Groundtruth generation with assertions
- `src/psy_agents_noaug/data/loaders.py`: Data loading with strict validation
- `src/psy_agents_noaug/data/splits.py`: Train/val/test splitting

**Validation Guarantees:**
- `_assert_field_usage()` function fails if wrong field is accessed
- Status values normalized to binary (0/1)
- Cases parsed from JSON/list format and validated
- Tests in `tests/test_groundtruth.py` verify separation

### Training Infrastructure

**Reproducibility (NEW - 2025):**
- Enhanced seed management in `src/psy_agents_noaug/utils/reproducibility.py`
- Full determinism with `torch.use_deterministic_algorithms()`
- Hardware-optimized DataLoader settings (num_workers, pin_memory, persistent_workers)
- Mixed precision support (Float16/BFloat16 with automatic GPU detection)

**Training Scripts:**
- `scripts/train_criteria.py`: Standalone Criteria training (PRODUCTION-READY)
- `scripts/eval_criteria.py`: Standalone Criteria evaluation (PRODUCTION-READY)
- `scripts/train_best.py`: HPO integration router (routes to architecture-specific scripts)
- `scripts/run_hpo_stage.py`: HPO runner (objective function needs implementation)

**Training Configs:**
- `configs/training/default.yaml`: Standard settings with hardware optimizations
- `configs/training/optimized.yaml`: Comprehensive annotated config for max performance

**Core Training Loop:**
- `src/psy_agents_noaug/training/train_loop.py`: `Trainer` class with:
  - Mixed precision (AMP)
  - Gradient accumulation and clipping
  - Early stopping on validation metrics
  - MLflow logging
  - Checkpoint management

**Key Optimizations (2025 Best Practices):**
```yaml
# Mixed Precision
amp:
  enabled: true
  dtype: "float16"  # Use "bfloat16" for Ampere+ GPUs

# DataLoader (2-5x faster)
num_workers: 8          # Start with 2× CPU cores per GPU
pin_memory: true        # Always true for GPU training
persistent_workers: true
prefetch_factor: 2

# Reproducibility vs Speed
deterministic: true     # Full reproducibility (slower)
cudnn_benchmark: false  # Deterministic algorithms
```

### Configuration System (Hydra)

**Composition:**
```yaml
# configs/config.yaml
defaults:
  - data: hf_redsm5        # Data source
  - model: roberta_base     # Model architecture
  - training: default       # Training config
  - task: criteria          # Task definition
  - hpo: stage0_sanity      # HPO config
```

**Override Examples:**
```bash
# Single override
python -m psy_agents_noaug.cli train task=evidence

# Nested override
python -m psy_agents_noaug.cli train training.batch_size=32 training.optimizer.lr=3e-5

# Multiple models (multirun)
python -m psy_agents_noaug.cli train -m model=bert_base,roberta_base,deberta_v3_base
```

**Config Groups:**
- `data/`: Data sources (hf_redsm5, local_csv) + field_map.yaml
- `model/`: Model architectures (bert_base, roberta_base, deberta_v3_base)
- `training/`: Training hyperparameters
- `task/`: Task definitions (criteria, evidence)
- `hpo/`: HPO stages (stage0_sanity, stage1_coarse, stage2_fine, stage3_refit)

### CLI Architecture

**Entry Point:** `src/psy_agents_noaug/cli.py`

**Commands:**
1. `make_groundtruth`: Generate ground truth with strict validation
2. `train`: Train model with specified config
3. `hpo`: Run HPO stage (calls Optuna)
4. `refit`: Retrain best model on train+val
5. `evaluate_best`: Evaluate checkpoint on test set
6. `export_metrics`: Export MLflow metrics to CSV

**All commands use Hydra for configuration management.**

## Critical Data Rules

### Field Separation (ENFORCED)

```python
# In src/psy_agents_noaug/data/groundtruth.py
def _assert_field_usage(field_name: str, expected_field: str, operation: str):
    """Raises AssertionError if wrong field is used."""
    assert field_name == expected_field, (
        f"STRICT VALIDATION: {operation} must use '{expected_field}' field, "
        f"but '{field_name}' was provided"
    )

# Criteria uses ONLY status
_assert_field_usage(field_name, "status", "Criteria groundtruth generation")

# Evidence uses ONLY cases
_assert_field_usage(field_name, "cases", "Evidence groundtruth generation")
```

**Never mix these fields. Tests will fail if violated.**

### Field Mapping Format

```yaml
# configs/data/field_map.yaml
annotations:
  columns:
    status:
      required: true
      used_for: ["criteria"]
      type: "string"
      normalization:
        positive_values: ["positive", "present", "true", "1", 1, true]
        negative_values: ["negative", "absent", "false", "0", 0, false]

    cases:
      required: true
      used_for: ["evidence"]
      type: "json"
      structure:
        - text: "string"
        - start_char: "int"
        - end_char: "int"
```

## Key Testing Files

- `tests/test_groundtruth.py`: **Validates STRICT field separation** (highest priority)
- `tests/test_loaders.py`: Data loading validation
- `tests/test_training_smoke.py`: Training pipeline smoke tests
- `tests/test_hpo_config.py`: HPO config validation
- `tests/test_integration.py`: End-to-end workflow tests

## MLflow Tracking

**Backend:** SQLite (`mlflow.db`)
**Artifacts:** File system (`mlruns/`)

```python
# Tracking URI resolution
tracking_uri = "sqlite:///mlflow.db"
artifact_location = "./mlruns"

# Logged metrics
- Training: loss, accuracy (per step)
- Validation: loss, accuracy, F1 (macro/micro), precision, recall (per epoch)
- System: learning rate, epoch time
- Hyperparameters: all training config
```

**View UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Project Structure Highlights

**Note:** See `OPTIMIZATION_SUMMARY.md` for recent cleanup details (removed 3 redundant docs, cleaned caches, unified Docker config).

```
NoAug_Criteria_Evidence/
├── .devcontainer/             # Dev Container (SINGLE source for Docker)
│   ├── devcontainer.json      # VS Code settings + GPU config
│   ├── Dockerfile             # CUDA 12.1 + Python 3.10 + Poetry
│   └── docker-compose.yml     # Service composition
│
├── configs/                    # Hydra configs (YAML)
│   ├── config.yaml            # Main composition
│   ├── data/field_map.yaml    # STRICT field mappings
│   ├── model/                 # Model configs
│   ├── training/              # Training configs
│   │   ├── default.yaml       # Standard settings
│   │   └── optimized.yaml     # Max performance settings
│   ├── task/                  # Task definitions
│   └── hpo/                   # HPO stages
│
├── src/psy_agents_noaug/
│   ├── architectures/         # Four architectures
│   │   ├── criteria/          # Binary classification
│   │   ├── evidence/          # Span extraction
│   │   ├── share/             # Shared encoder
│   │   └── joint/             # Dual encoders
│   ├── data/
│   │   ├── groundtruth.py     # Groundtruth with strict validation
│   │   ├── loaders.py         # Data loading
│   │   └── splits.py          # Train/val/test splits
│   ├── training/
│   │   ├── train_loop.py      # Trainer class (AMP, early stopping)
│   │   └── evaluate.py        # Evaluator class
│   ├── utils/
│   │   ├── reproducibility.py # Enhanced seed + hardware utils
│   │   └── mlflow_utils.py    # MLflow helpers
│   └── cli.py                 # Unified CLI
│
├── src/Project/               # Architecture implementations (800KB)
│   │                          # Used by: train_criteria.py, eval_criteria.py
│   ├── Criteria/              # Simpler, standalone implementation
│   ├── Evidence/              # Binary/multi-class classification
│   ├── Joint/                 # Multi-task model
│   └── Share/                 # Shared encoder
│
├── scripts/
│   ├── train_criteria.py      # ✅ Production-ready Criteria training
│   ├── eval_criteria.py       # ✅ Production-ready Criteria evaluation
│   ├── train_best.py          # HPO integration router
│   ├── run_hpo_stage.py       # HPO runner
│   └── make_groundtruth.py    # Ground truth generation
│
├── tests/
│   ├── test_groundtruth.py    # ⚠️ CRITICAL: Tests field separation
│   └── ...
│
├── docs/
│   ├── TRAINING_GUIDE.md      # ✅ NEW: Comprehensive training guide
│   ├── TRAINING_SETUP_COMPLETE.md  # ✅ NEW: Setup summary
│   ├── DATA_PIPELINE_IMPLEMENTATION.md
│   ├── CLI_AND_MAKEFILE_GUIDE.md
│   └── ...
│
├── Makefile                   # Convenient command shortcuts
└── pyproject.toml            # Poetry dependencies
```

## Architecture Implementation Notes

**Duplicate Implementations Exist:**

Two architecture implementations coexist:
1. **`src/Project/`** (800KB) - Simpler, used by standalone training scripts
2. **`src/psy_agents_noaug/architectures/`** (1.9MB) - Extended features, self-contained

**Current Usage:**
- `src/Project/` powers `train_criteria.py` and `eval_criteria.py` (NEW, production-ready)
- `src/psy_agents_noaug/architectures/` has train/eval engines but not currently used by CLI
- Models are identical, datasets differ (psy_agents_noaug has criterion resolution)

**Why Both Exist:**
- src/Project: Simpler for standalone scripts
- psy_agents_noaug/architectures: More features, may be used for future CLI integration
- No conflicts (separate namespaces)

**Future Consolidation:**
See `OPTIMIZATION_SUMMARY.md` for consolidation plan (2-4 hours estimated).

## Common Pitfalls

1. **Field Mixing**: Never use `status` for evidence or `cases` for criteria. Tests will fail.

2. **Config Paths**: Hydra configs are relative to `configs/`, not full paths:
   ```bash
   # ✓ Correct
   python -m psy_agents_noaug.cli train task=criteria

   # ✗ Wrong
   python -m psy_agents_noaug.cli train task=configs/task/criteria.yaml
   ```

3. **Poetry Environment**: Always use `poetry run` or activate poetry shell:
   ```bash
   poetry run python -m psy_agents_noaug.cli train task=criteria
   # Or
   poetry shell
   python -m psy_agents_noaug.cli train task=criteria
   ```

4. **Reproducibility Trade-off**: `deterministic=true` ensures exact reproducibility but is 20% slower. Set to `false` for production/inference.

5. **Checkpoint Paths**: Use absolute paths or paths relative to project root:
   ```bash
   # ✓ Correct
   make eval CHECKPOINT=outputs/checkpoints/best_checkpoint.pt

   # ✗ May fail if run from wrong directory
   make eval CHECKPOINT=best_checkpoint.pt
   ```

6. **HPO Requirements**: Stage 3 (refit) requires best config from stage 2:
   ```bash
   make hpo-s2  # Must complete first
   make refit   # Uses outputs/hpo_stage2/best_config.yaml
   ```

## Development Workflow

1. **Make Changes**: Edit code in `src/` or `configs/`
2. **Format**: `make format`
3. **Lint**: `make lint`
4. **Test**: `make test`
5. **Pre-commit**: `make pre-commit-run`
6. **Commit**: Standard git workflow

## NO Augmentation

This is a **baseline repository** without data augmentation:
- ✅ Standard transformer models (BERT, RoBERTa, DeBERTa)
- ✅ Training and evaluation
- ✅ Hyperparameter optimization
- ❌ NO augmentation techniques
- ❌ NO augmentation code (nlpaug listed but unused)

## Quick Reference

```bash
# Complete workflow from scratch
make setup                      # 1. Setup
make groundtruth                # 2. Generate data
make hpo-s0                     # 3. Sanity check
make full-hpo                   # 4. Full HPO (stages 1-3)
make eval                       # 5. Evaluate
make export                     # 6. Export results

# Development cycle
make format && make lint && make test

# View documentation
cat docs/TRAINING_GUIDE.md      # Comprehensive training guide
cat docs/CLI_AND_MAKEFILE_GUIDE.md  # CLI reference
cat docs/QUICK_START.md         # Quick start guide
```

## Recent Updates (2025)

**NEW Training Infrastructure:**
- Enhanced reproducibility with full determinism support
- Hardware-optimized DataLoader settings (2-5x faster)
- Mixed precision (AMP) with Float16/BFloat16 auto-detection
- Production-ready Criteria training/evaluation scripts
- Comprehensive training documentation

**Project Optimization (Latest):**
- Removed 3 redundant documentation files (-985 lines)
- Unified Docker config (`.devcontainer/` only)
- Cleaned all cache files
- Verified no unused imports (ruff check)
- Documented architecture implementation status

See `OPTIMIZATION_SUMMARY.md` for full cleanup details.
See `docs/TRAINING_GUIDE.md` and `docs/TRAINING_SETUP_COMPLETE.md` for training details.
