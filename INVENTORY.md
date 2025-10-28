# Repository Inventory - PSY Agents NO-AUG to Augmentation Pipeline

**Last Updated**: October 26, 2025
**Purpose**: Comprehensive technical map for production readiness audit and transformation from NO-AUG baseline to augmentation-enabled system

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Python Packages](#python-packages)
3. [Entry Points & CLI](#entry-points--cli)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [HPO System](#hpo-system)
6. [Augmentation Pipeline](#augmentation-pipeline)
7. [Test Coverage](#test-coverage)
8. [Configuration System](#configuration-system)
9. [Critical Findings](#critical-findings)
10. [Naming Audit](#naming-audit---noaug-references)

---

## 1. Directory Structure

### Root Organization

```
DataAug_Criteria_Evidence/
├── src/                          # Source code (two parallel implementations)
│   ├── psy_agents_noaug/         # ACTIVE: CLI + augmentation + architectures
│   │   └── (see detailed breakdown below)
│   └── Project/                  # DUPLICATE: Standalone scripts use this
│       └── (see detailed breakdown below)
├── configs/                      # Hydra YAML configuration (12 subdirectories)
├── scripts/                      # Entry point scripts (17 Python scripts)
├── tests/                        # Test suite (24 test files + benchmarks)
├── docs/                         # Documentation (comprehensive guides)
├── Makefile                      # CLI shortcuts (60+ targets)
├── pyproject.toml               # Poetry configuration
└── [Root level docs]            # Strategy/status documents
```

### Source Directory: `src/psy_agents_noaug/` (ACTIVE - 5,356 LOC)

**Core Components:**

```
psy_agents_noaug/
├── __init__.py
│
├── augmentation/                 (4 files, ~500 LOC - CURRENTLY UNUSED)
│   ├── __init__.py
│   ├── pipeline.py              (156 LOC) - Orchestration + seeding
│   ├── registry.py              (300+ LOC) - nlpaug + textattack unification
│   └── tfidf_cache.py           (150+ LOC) - TF-IDF cache management
│
├── architectures/               (4 architectures × 4 components each)
│   ├── __init__.py
│   ├── utils/                   (2 files: heads.py, outputs.py, checkpoint.py)
│   │   └── dsm_criteria.py      - DSM-5 criterion definitions
│   │
│   ├── criteria/                (Binary classification)
│   │   ├── models/model.py      - Transformer + classification head
│   │   ├── data/dataset.py      - CriteriaDataset loader
│   │   ├── engine/
│   │   │   ├── train_engine.py  - Training orchestration
│   │   │   └── eval_engine.py   - Evaluation orchestration
│   │   └── utils/               (5 files: seed, checkpoint, logging, MLflow, optuna)
│   │
│   ├── evidence/                (Span extraction)
│   │   ├── models/model.py      - Span prediction head
│   │   ├── data/dataset.py      - EvidenceDataset loader
│   │   ├── engine/
│   │   │   ├── train_engine.py
│   │   │   └── eval_engine.py
│   │   └── utils/               (5 files: same pattern as criteria)
│   │
│   ├── share/                   (Shared encoder, dual heads)
│   │   └── (same structure as evidence)
│   │
│   └── joint/                   (Dual encoders, fusion)
│       └── (same structure as evidence)
│
├── data/                        (Core data pipeline - STRICT VALIDATION)
│   ├── __init__.py             (36 LOC)
│   ├── groundtruth.py          (500+ LOC) ⭐ CRITICAL - field separation
│   ├── loaders.py              (376 LOC) - ReDSM5DataLoader
│   ├── datasets.py             (300+ LOC) - ClassificationDataset + tokenization
│   ├── splits.py               (180 LOC) - Train/val/test splitting
│   ├── classification_loader.py (200+ LOC) - Legacy loader wrapper
│   └── augmentation_utils.py   (150+ LOC) - Augmentation integration
│
├── hpo/                         (Hyperparameter optimization)
│   ├── __init__.py
│   └── optuna_runner.py        (352 LOC) - OptunaRunner + search space builder
│
├── models/                      (Transformer heads)
│   ├── __init__.py
│   ├── encoders.py             (263 LOC) - Base encoder wrapper
│   ├── criteria_head.py         (132 LOC) - Binary classification head
│   └── evidence_head.py         (132 LOC) - Span prediction head
│
├── training/                    (Training infrastructure)
│   ├── __init__.py
│   ├── train_loop.py           (556 LOC) ⭐ CORE - Trainer class
│   ├── evaluate.py             (451 LOC) - Evaluator class
│   └── setup.py                (118 LOC) - Setup utilities
│
├── utils/                       (Infrastructure utilities)
│   ├── __init__.py
│   ├── reproducibility.py      (198 LOC) - Seeds + device utils
│   ├── logging.py              (73 LOC) - Structured logging
│   ├── logging_config.py       (115 LOC) - Logging setup
│   ├── mlflow_utils.py         (346 LOC) - MLflow tracking
│   └── type_aliases.py         (23 LOC) - Type definitions
│
└── cli.py                       (250+ LOC) ⭐ ENTRY POINT
    └── Typer CLI with train/tune/hpo subcommands
```

### Source Directory: `src/Project/` (DUPLICATE - 800+ LOC, used by standalone scripts)

**Structure mirrors psy_agents_noaug/architectures:**

```
Project/
├── utils/
│   ├── __init__.py             (23 LOC)
│   └── checkpoint.py           (248 LOC)
│
├── Criteria/                    (PRODUCTION-READY - used by train_criteria.py)
│   ├── models/model.py         - Standalone implementation
│   ├── data/dataset.py         - CriteriaDataset
│   ├── engine/
│   │   ├── train_engine.py
│   │   └── eval_engine.py
│   └── utils/                  (log, checkpoint, seed, optuna)
│
├── Evidence/
│   ├── models/model.py
│   ├── data/dataset.py
│   ├── engine/
│   └── utils/
│
├── Share/
│   └── (same structure)
│
└── Joint/
    └── (same structure)
```

**Critical Note**: `src/Project/` is 904KB of duplicate code. See [Critical Findings](#critical-findings) for consolidation plan.

---

## 2. Python Packages

### Main Package: `psy_agents_noaug`

| Module | Files | LOC | Purpose |
|--------|-------|-----|---------|
| `augmentation` | 4 | ~500 | Text augmentation (nlpaug + textattack unification) |
| `architectures` | 70+ | ~4K | 4 task architectures (criteria, evidence, share, joint) |
| `data` | 6 | ~1.5K | STRICT data validation + groundtruth generation |
| `hpo` | 2 | ~350 | Optuna-based hyperparameter optimization |
| `models` | 4 | ~500 | Transformer heads (encoder, classification, span) |
| `training` | 4 | ~1.1K | Training loop + evaluation infrastructure |
| `utils` | 7 | ~800 | Logging, MLflow, reproducibility, seeds |
| `cli` | 1 | ~250 | CLI entry point (Typer) |

### Key Dependencies (pyproject.toml)

**Core ML:**
- torch >= 2.6.0
- transformers >= 4.44
- optuna >= 4.5.0 (Optuna 4.5.0 with NSGAIISampler)

**Data & Processing:**
- datasets >= 2.20
- pandas >= 2.0
- scikit-learn >= 1.4
- nltk >= 3.8

**Augmentation (CURRENTLY UNUSED):**
- nlpaug >= 1.1.11 (28+ methods)
- textattack >= 0.3.10 (6+ recipes)

**Infrastructure:**
- hydra-core >= 1.3.0 (Configuration)
- mlflow >= 3.1.0 (Experiment tracking)
- typer >= 0.12 (CLI)
- pydantic >= 2.8 (Validation)

**Monitoring:**
- psutil >= 7.1.1
- pynvml >= 13.0.1

---

## 3. Entry Points & CLI

### 3.1 CLI Command Entry Point

**File**: `src/psy_agents_noaug/cli.py`
**Framework**: Typer
**Package Entry**: `psy-agents` (defined in pyproject.toml)

**Commands**:
1. `train` - Training job launcher (stubbed, ready for integration)
2. `tune` - HPO via Optuna (delegates to scripts/tune_max.py)
3. `show_best` - Print top-K HPO results

**Current Status**: Thin CLI (keeps imports isolated), ready to wire to training backend.

### 3.2 Makefile Targets (60+ targets, Makefile ~900 LOC)

**Training Commands:**
```
make train                 # Default: criteria + roberta_base
make train TASK=<t> MODEL=<m>  # Custom task/model combo
make train-evidence        # Evidence task
```

**HPO Commands:**
```
# Multi-stage progression
make hpo-s0 HPO_TASK=criteria       # Stage 0: 8 trials (sanity check)
make hpo-s1 HPO_TASK=criteria       # Stage 1: 20 trials (coarse)
make hpo-s2 HPO_TASK=criteria       # Stage 2: 50 trials (fine)
make refit HPO_TASK=criteria        # Stage 3: retrain on train+val

make full-hpo HPO_TASK=criteria     # All stages for one architecture
make full-hpo-all                   # All stages for all 4 architectures

# Maximal HPO
make tune-criteria-max              # 800 trials
make tune-evidence-max              # 1200 trials
make tune-share-max                 # 600 trials
make tune-joint-max                 # 600 trials
make maximal-hpo-all                # All 4 sequentially

# Super-max HPO
make tune-criteria-supermax         # 5000 trials, 100 epochs
make tune-evidence-supermax         # 8000 trials, 100 epochs
make tune-all-supermax              # ~19K trials total
```

**Ground Truth:**
```
make groundtruth                    # HuggingFace (default)
make groundtruth-local              # Local CSV
```

**Evaluation:**
```
make eval CHECKPOINT=<path>
make export                         # Export metrics to CSV
```

**Development:**
```
make format                         # Ruff + black
make lint                          # Ruff + black --check
make test                          # Pytest
make test-cov                      # With coverage
make test-groundtruth              # Field validation only
```

### 3.3 Script Entry Points (17 Python scripts in `scripts/`)

**Production-Ready**:
1. `train_criteria.py` (416 LOC) ✓ - Standalone Criteria training
   - Imports: Project.Criteria, transformers, torch
   - No CLI framework

2. `eval_criteria.py` (306 LOC) ✓ - Standalone evaluation
   - Evaluates checkpoint on test set
   - Checkpoint path via command line

**HPO Runners**:
3. `tune_max.py` (600+ LOC) ✓ - Maximal Optuna search
   - Conditional spaces (models, optimizers, heads, losses)
   - Multi-fidelity pruning (Hyperband + Percentile)
   - MLflow logging
   - Supports NSGAIISampler for multi-objective

4. `run_hpo_stage.py` (300+ LOC) ✓ - Multi-stage HPO runner
   - Hydra integration
   - Stages 0-3 with progressive refinement
   - Search space from config

5. `run_all_hpo.py` (200+ LOC) ✓ - Sequential HPO for all architectures
   - Runs tune_max.py for each: criteria, evidence, share, joint
   - Parallel worker support

6. `train_best.py` (200+ LOC) - HPO integration router
   - Routes to architecture-specific scripts
   - Uses best config from HPO stage

**Utilities**:
7. `make_groundtruth.py` - Ground truth generation
8. `export_metrics.py` - MLflow metrics export
9. `validate_installation.py` - Installation verification
10. `bench_dataloader.py` - DataLoader performance benchmarking
11. `profile_augmentation.py` - Augmentation profiling
12. `verify_determinism.py` - Reproducibility verification
13. `gpu_utilization.py` - GPU monitoring
14. `generate_sbom.py` - Bill of materials
15. `generate_licenses.py` - License report
16. `audit_security.py` - Security audit
17. `run_two_stage_hpo.py` - Legacy two-stage runner

---

## 4. Data Flow Pipeline

### 4.1 Data Ingestion

**Sources:**
- HuggingFace: `irlab-udc/redsm5` (preferred)
- Local CSV: `data/raw/redsm5/{posts,annotations}.csv`

**Loaders:**
```
src/psy_agents_noaug/data/loaders.py
├── ReDSM5DataLoader
│   ├── load_posts() -> DataFrame (DSM-5 diagnostic posts)
│   ├── load_annotations() -> DataFrame (status + cases fields)
│   └── load_dsm_criteria() -> DataFrame (criterion definitions)
```

### 4.2 STRICT Field Validation

**Critical File**: `src/psy_agents_noaug/data/groundtruth.py` (500+ LOC)

**Field Separation (ENFORCED):**
```python
# Criteria task
_assert_field_usage(field_name, "status", "Criteria labels")
# Returns: post_id, criterion_id, status → binary label (0/1)

# Evidence task
_assert_field_usage(field_name, "cases", "Evidence labels")
# Returns: post_id, criterion_id, cases → span list
```

**Functions:**
1. `load_field_map()` - Load field_map.yaml
2. `normalize_status_value()` - status → binary (0/1)
3. `parse_cases_field()` - cases → list of {start_char, end_char, sentence_id, text}
4. `create_criteria_groundtruth()` - Generate criteria ground truth
5. `create_evidence_groundtruth()` - Generate evidence ground truth
6. `_assert_field_usage()` - ENFORCE field separation (fails if violated)

**Configuration**: `configs/data/field_map.yaml`
```yaml
annotations:
  status: "status"        # ONLY for criteria
  cases: "cases"          # ONLY for evidence
status_values:
  positive: [positive, present, true, 1, True]
  negative: [negative, absent, false, 0, False]
validation:
  strict_mode: true
  allow_cross_contamination: false  # MUST remain false
  fail_on_invalid_criterion_id: true
  fail_on_missing_post_id: true
```

### 4.3 Dataset Construction & Tokenization

**Files:**
- `src/psy_agents_noaug/data/datasets.py` - Core dataset classes
- `src/psy_agents_noaug/data/splits.py` - Train/val/test splitting
- `src/psy_agents_noaug/data/augmentation_utils.py` - Augmentation integration

**Dataset Classes:**

1. **ClassificationDataset** (for criteria)
   - Eager tokenization (pre-tokenize) or lazy (defer to collate_fn)
   - Optional on-the-fly augmentation
   - Input: DataFrame + tokenizer + labels

2. **SpanDataset** (for evidence)
   - Span token classification format
   - Supports BIO/BIOES tagging
   - Token-level labels

3. **DatasetSplits** (container)
   - Holds train, val, test datasets
   - Splitter supports stratified/random

**Tokenization Modes:**
- **Eager**: Pre-tokenize at dataset creation (fewer CPU cycles)
- **Lazy**: Defer to collate_fn (required for on-the-fly augmentation)

### 4.4 Augmentation Hooks (CURRENTLY UNUSED)

**Pipeline Integration Points:**

1. **Dataset Level** (`datasets.py`):
   ```python
   # If augmenter is passed and lazy_encode=True:
   # - Store raw texts only
   # - In collate_fn: augment texts → tokenize batch
   ```

2. **Collate Function** (`classification_loader.py`):
   ```python
   # Apply augmentation before tokenization
   # Supports: nlpaug + textattack methods
   ```

3. **Configuration** (`configs/augmentation/default.yaml`):
   ```yaml
   lib: "none"              # none|nlpaug|textattack|both
   methods: ["all"]         # Method names or "all"
   p_apply: 0.15           # Probability of applying augmentation
   ops_per_sample: 1       # Augmentations per sample
   max_replace_ratio: 0.3  # Max percentage of text to replace
   ```

**Current State**: Augmentation code exists but is NEVER CALLED in production paths.

### 4.5 Data Flow Diagram

```
Raw Data Sources
       ↓
┌─────────────────────────────┐
│  ReDSM5DataLoader           │
│  (HF or local CSV)          │
└──────────┬──────────────────┘
           ↓
     Field Validation
     (field_map.yaml)
           ↓
   ┌───────┴────────┐
   ↓                ↓
Criteria GG      Evidence GG
(status field)  (cases field)
   ↓                ↓
   ├─────────────────┤
   ↓                ↓
CriteriaDataset  EvidenceDataset
   ├─[Lazy/Eager]──┤
   │    Tokenize   │
   │  (if lazy)    │
   │               │
   │[Augmentation] │ ← CURRENTLY UNUSED
   │  (if enabled) │
   │               │
   └──────┬────────┘
          ↓
    DataLoader
    (batch iteration)
          ↓
    Model Training
```

---

## 5. HPO System

### 5.1 HPO Architecture

**Three Modes:**

#### Mode 1: Multi-Stage HPO (Progressive Refinement)
- **Stage 0**: Sanity check (8 trials)
- **Stage 1**: Coarse search (20 trials)
- **Stage 2**: Fine search (50 trials)
- **Stage 3**: Refit on train+val (uses best config from stage 2)

**Runner**: `scripts/run_hpo_stage.py`
**Config**: `configs/hpo/stage{0,1,2,3}_*.yaml`

#### Mode 2: Maximal HPO (Single Large Run)
- Criteria: 800 trials
- Evidence: 1200 trials
- Share: 600 trials
- Joint: 600 trials

**Runner**: `scripts/tune_max.py`
**Config**: Environment variables + command-line args

#### Mode 3: Super-Max HPO (Ultra-Long Run)
- Criteria: 5000 trials, 100 epochs
- Evidence: 8000 trials, 100 epochs
- Share: 3000 trials, 100 epochs
- Joint: 3000 trials, 100 epochs

**Total**: ~19,000 trials

### 5.2 Search Space

**Core Parameters** (suggest_common):
- Tokenizer:
  - max_length: 128-1024 (step 32)
  - doc_stride: 32-256 (step 16)
  - use_fast: [True, False]
- Batch size: [8, 12, 16, 24, 32, 48, 64]
- Gradient accumulation: [1, 2, 3, 4, 6, 8]

**Optimizer**:
- Name: [adamw, adamw_8bit, adafactor, lion]
- LR: 5e-6 to 3e-4 (log scale)
- Weight decay: 1e-6 to 2e-1 (log scale)
- Beta1, Beta2, Eps (for Adam variants)

**Scheduler**:
- Name: [linear, cosine, cosine_restart, polynomial, one_cycle]
- Warmup ratio: 0.0-0.2

**Model**:
- Architecture: [bert-base, bert-large, roberta-base, roberta-large, deberta-v3-base, deberta-v3-large, electra-base, electra-large, xlm-roberta-base]

**Head-Specific Spaces**:
- **Classification** (Criteria): [ce, ce_label_smooth, focal]
- **Span** (Evidence): [qa_ce, qa_ce_ls, qa_focal]
- Pooling: [cls, mean, max, attn]
- Activation: [gelu, relu, silu]

**Evidence-Only**:
- Null span policy: [none, threshold, ratio, calibrated]
- Reranker: [sum, product, softmax]

### 5.3 HPO Components

**File**: `src/psy_agents_noaug/hpo/optuna_runner.py`

**Classes:**
1. `OptunaRunner` - Manages study creation, trial execution
2. Search space builders:
   - `create_search_space_from_config()` - Hydra-based
   - `suggest_common()` - Common hyperparameters
   - `suggest_criteria_head()` - Classification-specific
   - `suggest_evidence_head()` - Span-specific

**Pruning Strategies:**
- HyperbandPruner (multi-fidelity)
- PatientPruner (patience-based early stopping)

**Samplers:**
- TPESampler (default)
- NSGAIISampler (multi-objective, Optuna 4.5.0+)

**Storage:**
- SQLite: `_optuna/noaug.db` (default)
- In-memory: `study_name` only
- PostgreSQL: Custom URI support

### 5.4 HPO Execution Flow

```
make hpo-s0 HPO_TASK=criteria
       ↓
Makefile calls:
python scripts/run_hpo_stage.py hpo=stage0_sanity task=criteria
       ↓
Hydra loads config composition:
  - config.yaml (defaults)
  - overrides: hpo=stage0_sanity, task=criteria
       ↓
run_hpo_stage.py:
  1. Set seed for reproducibility
  2. Configure MLflow
  3. Create search space from cfg.hpo
  4. Create OptunaRunner
  5. Define objective(trial, params)
  6. Load data (criterion dataset)
  7. Train + eval for n_trials
  8. Report metric to Optuna
  9. Save best config to outputs/hpo_stageX/best_config.yaml
       ↓
(Stage 3 - Refit):
python scripts/train_best.py outputs/hpo_stage2/best_config.yaml
       ↓
Retrains on train+val, evaluates on test
```

### 5.5 MLflow Integration

**Tracked Metrics:**
- Per-step: training loss
- Per-epoch: val_loss, val_accuracy, val_f1_macro, val_f1_micro
- System: learning_rate, epoch_time, throughput

**Logged Artifacts:**
- Best checkpoint: `checkpoint_best.pt`
- Final model: `model_final.pt`
- Config: `training_config.yaml`

**Backend**: SQLite file store (`mlflow.db`) or custom URI

---

## 6. Augmentation Pipeline

### 6.1 Augmentation Components

**Files:**
- `src/psy_agents_noaug/augmentation/pipeline.py` (156 LOC)
- `src/psy_agents_noaug/augmentation/registry.py` (300+ LOC)
- `src/psy_agents_noaug/augmentation/tfidf_cache.py` (150+ LOC)
- `src/psy_agents_noaug/augmentation/__init__.py` (main exports)

### 6.2 Supported Methods (28+ total)

**nlpaug Methods** (16):
- **Char-level**:
  - KeyboardAug (character typos)
  - OcrAug (OCR errors)
  - RandomCharAug (random char operations)

- **Word-level**:
  - RandomWordAug (random deletion/swap)
  - SpellingAug (misspellings)
  - SynonymAug (synonym replacement)
  - AntonymAug (antonym replacement)
  - ContextualWordEmbsAug (word embeddings)
  - BackTranslationAug (back-translation)
  - ContextualWordEmbsForSentenceAug
  - ContextualWordEmbsForSentenceAug
  - ReservedAug (reserved token protection)
  - TfIdfAug (TF-IDF-based replacement)

**textattack Methods** (12):
- CharSwapAugmenter
- CheckListAugmenter
- DeletionAugmenter
- EasyDataAugmenter
- SwapAugmenter
- SynonymInsertionAugmenter
- WordNetAugmenter
- (Additional recipes available)

### 6.3 AugmenterWrapper

Unified interface for all augmenters:
```python
class AugmenterWrapper:
    def augment_one(text: str) -> str
        # Defensive: returns original text on failure
        # Handles list/string/None outputs uniformly
```

### 6.4 AugConfig (Configuration)

```python
@dataclass
class AugConfig:
    lib: AugLib = "none"              # none|nlpaug|textattack|both
    methods: Sequence[str] = ["all"]  # Method names or "all"
    p_apply: float = 0.15             # Probability per sample
    ops_per_sample: int = 1           # Sequential augmentations
    max_replace_ratio: float = 0.3    # Max replacement %%
    tfidf_model_path: str | None = None
    reserved_map_path: str | None = None
    seed: int = 42
    example_limit: int = 32
```

### 6.5 Pipeline Orchestration (AugmenterPipeline)

```python
class AugmenterPipeline:
    def __init__(config: AugConfig, resources: AugResources)

    def __call__(texts: list[str]) -> list[str]
        # Apply augmentation to batch
        # Respects p_apply probability
        # Returns augmented texts (or originals if not augmented)
```

### 6.6 Current Status: UNUSED

**Why it's disabled:**
1. No augmentation calls in production training paths
2. Dataset classes accept augmenter parameter but rarely use it
3. CLI has augmentation flags but doesn't wire them
4. Test suite has augmentation tests but not in main training loop

**Files that import augmentation:**
- `src/psy_agents_noaug/data/datasets.py` (type hints only)
- `src/psy_agents_noaug/data/augmentation_utils.py` (helper functions)
- `src/psy_agents_noaug/data/classification_loader.py` (collate_fn integration)
- CLI and some HPO runners (argument parsing)

---

## 7. Test Coverage

### 7.1 Test Files (24 total)

**Core Validation Tests:**
1. `test_groundtruth.py` ⭐ - CRITICAL field separation tests
2. `test_loaders.py` - Data loader validation
3. `test_integration.py` - End-to-end workflows

**Training Tests:**
4. `test_training_smoke.py` - Training pipeline smoke tests
5. `test_train_smoke.py` - Alternative training tests
6. `test_arch_shapes.py` - Model architecture output shapes
7. `test_head_space.py` - Head configuration space

**Augmentation Tests:**
8. `test_augmentation_registry.py` - Augmenter registration
9. `test_augmentation_utils.py` - Augmentation utilities
10. `test_pipeline_comprehensive.py` - Full pipeline
11. `test_pipeline_extended.py` - Extended scenarios
12. `test_pipeline_integration.py` - Integration tests
13. `test_pipeline_scope.py` - Scope validation
14. `test_tfidf_cache.py` - TF-IDF caching
15. `test_tfidf_cache_extended.py` - Extended TF-IDF tests

**HPO Tests:**
16. `test_hpo_config.py` - HPO configuration validation
17. `test_hpo_integration.py` - HPO integration

**CLI Tests:**
18. `test_cli_flags.py` - CLI argument parsing

**QA Tests:**
19. `test_qa_null_policy.py` - Q&A null span handling

**Determinism Tests:**
20. `test_seed_determinism.py` - Reproducibility verification

**Other Tests:**
21. `test_smoke.py` - Basic smoke tests
22. `test_perf_contract.py` - Performance contracts
23. `test_mlflow_artifacts.py` - MLflow artifact tracking
24. `test_benchmarks/test_performance_regression.py` - Benchmarks

### 7.2 Test Configuration

**File**: `conftest.py`
- Pytest fixtures
- Device setup (CPU/CUDA detection)
- Temporary directories

**Pytest Config** (pyproject.toml):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "-v --tb=short"
markers = ["slow"]
```

**Coverage Config**:
```toml
[tool.coverage.run]
source = ["src/psy_agents_noaug"]
branch = true

[tool.coverage.report]
exclude_lines = ["pragma: no cover", "raise NotImplementedError"]
precision = 2
show_missing = true
```

**Status**: 67/69 tests passing (97.1%), 31% coverage

---

## 8. Configuration System

### 8.1 Hydra Architecture

**Main Config**: `configs/config.yaml`
```yaml
defaults:
  - data: hf_redsm5        # Data source
  - model: roberta_base    # Model architecture
  - training: default      # Training config
  - task: criteria         # Task definition
  - hpo: stage0_sanity     # HPO stage
```

### 8.2 Configuration Hierarchy

```
configs/
├── config.yaml                          # Main composition
│
├── augmentation/
│   └── default.yaml                     # Augmentation settings
│
├── data/
│   ├── field_map.yaml  ⭐ CRITICAL    # Field validation rules
│   ├── hf_redsm5.yaml                  # HuggingFace source
│   └── local_csv.yaml                  # Local CSV source
│
├── model/
│   ├── bert_base.yaml
│   ├── roberta_base.yaml
│   └── deberta_v3_base.yaml
│
├── task/
│   ├── criteria.yaml                    # Binary classification
│   └── evidence.yaml                    # Span extraction
│
├── training/
│   ├── default.yaml                     # Standard settings
│   ├── optimized.yaml                   # Performance-tuned
│   └── supermax_optimized.yaml          # Super-max HPO config
│
├── hpo/
│   ├── stage0_sanity.yaml              # 8 trials
│   ├── stage1_coarse.yaml              # 20 trials
│   ├── stage2_fine.yaml                # 50 trials
│   ├── stage3_refit.yaml               # Refit on train+val
│   ├── stage_a_baseline.yaml           # Baseline sweep
│   └── stage_b_augmentation.yaml       # Augmentation sweep
│
├── criteria/
│   ├── train.yaml                       # Criteria-specific training
│   └── hpo.yaml                         # Criteria-specific HPO
│
├── evidence/
│   ├── train.yaml
│   └── hpo.yaml
│
├── share/
│   ├── train.yaml
│   └── hpo.yaml
│
├── joint/
│   ├── train.yaml
│   └── hpo.yaml
```

### 8.3 Key Configuration Files

#### field_map.yaml (CRITICAL - Data Validation)

```yaml
# Field mappings
posts:
  post_id: "post_id"
  text: "text"

annotations:
  post_id: "post_id"
  criterion_id: "criterion_id"
  status: "status"        # ← ONLY for criteria
  cases: "cases"          # ← ONLY for evidence

# Status normalization
status_values:
  positive: [positive, present, true, 1, True]
  negative: [negative, absent, false, 0, False]

# Validation rules
validation:
  strict_mode: true
  allow_cross_contamination: false  # MUST be false
  fail_on_invalid_criterion_id: true
  fail_on_missing_post_id: true
```

#### training/default.yaml (Training Config)

```yaml
num_epochs: 3
batch_size: 16
learning_rate: 2e-5
num_workers: 4
pin_memory: true
gradient_accumulation_steps: 1

seed: 42
deterministic: true         # Full reproducibility (slower)
cudnn_benchmark: false      # Required for deterministic

# Mixed precision
amp:
  enabled: true
  dtype: "float16"          # or "bfloat16"

# Early stopping
early_stopping:
  patience: 5
  min_delta: 0.0001
```

#### hpo/stage{0,1,2}.yaml

**Stage 0** (sanity): 8 trials
**Stage 1** (coarse): 20 trials
**Stage 2** (fine): 50 trials

Each defines:
- n_trials
- direction: max/min
- metric: val_loss or val_f1
- sampler: TPE or NSGAIISampler
- pruner: Hyperband or PatientPruner

### 8.4 Override Examples

```bash
# Single override
python -m psy_agents_noaug.cli train task=evidence

# Nested override
python -m psy_agents_noaug.cli train training.batch_size=32 training.learning_rate=3e-5

# Multiple models (multirun)
python -m psy_agents_noaug.cli train -m model=bert_base,roberta_base,deberta_v3_base

# HPO with config
python scripts/run_hpo_stage.py hpo=stage1_coarse task=evidence model=deberta_v3_base
```

---

## 9. Critical Findings

### 9.1 Duplicate Architecture Implementation (904 KB)

**Location**: `src/Project/` vs `src/psy_agents_noaug/architectures/`

**Impact**:
- Code duplication risk
- Maintenance burden
- Potential divergence
- Wasted disk space (376 KB vs 528 KB)

**Current Usage**:
- `src/Project/` - Used by `train_criteria.py`, `eval_criteria.py` (production-ready)
- `src/psy_agents_noaug/architectures/` - Has train/eval engines but NOT used by CLI

**Consolidation Plan**: See `CODEBASE_STRUCTURE_ANALYSIS.md` Phase 1 (2-4 hours)

### 9.2 Unused Augmentation Code (100+ KB)

**Status**: DEAD CODE
- `src/psy_agents_noaug/augmentation/` (32 KB) - Registry + pipeline
- Tests for augmentation (55+ KB) - 9 test files
- Dependencies: nlpaug, textattack (30+ MB installed)

**Why Unused**:
1. No augmentation calls in training paths
2. Dataset classes have augmenter parameter but rarely instantiate it
3. CLI flags exist but don't wire to training
4. HPO runners don't enable augmentation by default

**Impact**: Contradicts project name "NO-AUG", increases pip install time

**Removal Plan**: See `CODEBASE_STRUCTURE_ANALYSIS.md` Phase 1-2 (1-2 hours)

### 9.3 Field Separation (ENFORCED)

**Status**: ✓ Correctly implemented and validated

**Critical Rule** (highest priority):
- Criteria task ONLY uses `status` field for labels
- Evidence task ONLY uses `cases` field for labels
- Assertion fails if violated (test_groundtruth.py validates)

**Implementation**:
```python
# src/psy_agents_noaug/data/groundtruth.py
def _assert_field_usage(field_name, expected_field, operation):
    assert field_name == expected_field, f"STRICT VALIDATION FAILURE..."
```

### 9.4 Production-Ready Components

✓ **Fully Production-Ready**:
1. `src/psy_agents_noaug/data/groundtruth.py` - STRICT validation
2. `src/psy_agents_noaug/training/train_loop.py` - Trainer class with AMP
3. `src/psy_agents_noaug/training/evaluate.py` - Evaluation orchestration
4. `scripts/train_criteria.py` - Standalone training
5. `scripts/eval_criteria.py` - Standalone evaluation
6. `scripts/tune_max.py` - Maximal HPO with NSGAIISampler
7. MLflow integration (artifact + metric tracking)

✓ **Tests**:
- 67/69 tests passing (97.1%)
- Field separation tests comprehensive
- Determinism verification available

### 9.5 HPO System Status

✓ **Multi-Stage HPO** (Progressive Refinement)
- 4 stages: sanity → coarse → fine → refit
- Real data (HuggingFace ReDSM-5)
- Hydra integration
- Config-driven search spaces

✓ **Maximal HPO** (Single Large Run)
- 600-1200 trials per architecture
- Multi-fidelity pruning
- Conditional search spaces
- All 4 architectures supported

✓ **Super-Max HPO** (Ultra-Long)
- 5000-8000 trials per architecture
- 100 epochs per trial
- ~19,000 total trials

---

## 10. Naming Audit - "noaug" References

### 10.1 Summary

**Total Occurrences**: 140 matches across 62 files

**Pattern Analysis**:
- `psy_agents_noaug` - Package name (appears ~100 times in imports)
- `noaug` - Experiment names, storage paths (~20 times)
- `NO-AUG` / `NO_AUG` - Comments, docstrings (~20 times)

### 10.2 Critical Renaming Points

#### Package Name: `psy_agents_noaug`

**All occurrences**:
```
62 files contain: from psy_agents_noaug import ...
```

**Affected**:
- `pyproject.toml`: `name = "noaug-criteria-evidence"` (line 2)
- `pyproject.toml`: `packages = [{include = "psy_agents_noaug", from = "src"}]` (line 42)
- All imports throughout codebase
- `src/psy_agents_noaug/` directory name
- Makefile references
- GitHub Actions workflows

**Rename Strategy**:
1. Rename directory: `src/psy_agents_noaug/` → `src/psy_agents_aug/`
2. Update pyproject.toml: package name + include path
3. Update all imports (62 files)
4. Update Makefile environment variables
5. Update documentation links

**Effort**: 2-3 hours with global find-replace + testing

#### Experiment/Study Names

**Current**:
```
MLflow experiment: "NoAug_Criteria_Evidence"
Optuna storage: "sqlite:///_.optuna/noaug.db"
Optuna study: "noaug-criteria-max"
```

**Recommended Changes**:
```
MLflow experiment: "Criteria_Evidence_Augmented"
Optuna storage: "sqlite:///_.optuna/augmented.db"
Optuna study: "criteria-augmented-max"
```

**Files affected**:
- `scripts/tune_max.py` (line 78)
- `scripts/run_hpo_stage.py` (line 50-51)
- Makefile targets
- Config files

#### Comments & Docstrings

**High-Priority Updates**:
- `CLAUDE.md`: Lines referencing "NO-AUG baseline control" → "augmentation-enabled system"
- `README.md`: Project purpose description
- All docstrings saying "NO-AUG" → "augmentation-enabled"
- Comments in augmentation pipeline activation code

### 10.3 File-by-File Renaming Impact

**Most Critical**:
1. `src/psy_agents_noaug/` - Directory rename required
2. `src/psy_agents_noaug/__init__.py` - Package init
3. `pyproject.toml` - Package metadata
4. `Makefile` - Environment variables

**High Impact** (30+ files):
- All imports in `src/`, `scripts/`, `tests/`
- MLflow experiment setup
- Hydra config composition

**Medium Impact** (documentation):
- `.md` files
- Docstrings
- Comments

### 10.4 Recommended Rename

**FROM**:
- Package: `psy_agents_noaug`
- Project: "PSY Agents NO-AUG"
- Experiment: "NoAug_Criteria_Evidence"

**TO**:
- Package: `psy_agents_aug` or `psy_agents`
- Project: "PSY Agents with Augmentation"
- Experiment: "Criteria_Evidence_Augmented"

**Estimated Effort**: 3-4 hours including testing

---

## Summary Table

| Component | Files | LOC | Status | Notes |
|-----------|-------|-----|--------|-------|
| Core Data Pipeline | 6 | 1.5K | ✓ READY | STRICT field validation enforced |
| Augmentation | 4 | 500 | ✗ UNUSED | Dead code, 28+ methods available |
| Architectures (active) | 20 | 2K | ✓ READY | Criteria production-ready |
| Architectures (duplicate) | 20 | 1K | ⚠️ DUPLICATE | 904 KB duplication with src/Project |
| Training Loop | 4 | 1.1K | ✓ READY | AMP, early stopping, MLflow |
| HPO System | 7 | ~800 | ✓ READY | Multi-stage, maximal, super-max |
| Tests | 24 | ~2K | ✓ READY | 97.1% passing, field validation complete |
| Configs | 27 | ~30 | ✓ READY | Hydra-based, composable |
| Scripts | 17 | ~3K | ✓ READY | 2 production scripts, 5 HPO runners |
| CLI | 1 | 250 | ✓ READY | Typer-based, minimal dependencies |

---

## Next Steps for Production

1. **Consolidate Duplicate Architectures** (src/Project → src/psy_agents_noaug)
2. **Remove Unused Augmentation Code** (or activate it in training paths)
3. **Rename Package** (psy_agents_noaug → psy_agents_aug)
4. **Wire Augmentation to Training** (if using augmentation)
5. **Production Testing** (load testing, stress testing, edge cases)
6. **Documentation Update** (reflect augmentation-enabled status)

---

Generated: 2025-10-26
Maintained by: YuNing Chen
Repository: DataAug_Criteria_Evidence
