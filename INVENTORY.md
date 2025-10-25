# CODEBASE INVENTORY
# PSY Agents NO-AUG: Baseline Clinical Text Analysis
# Production Readiness Audit - Foundation Document
# Generated: 2025-10-25

---

## TABLE OF CONTENTS

1. [Executive Summary](#executive-summary)
2. [Module Tree](#module-tree)
3. [Entry Points](#entry-points)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Augmentation Infrastructure Status](#augmentation-infrastructure-status)
6. [HPO Integration Points](#hpo-integration-points)
7. [Testing Infrastructure](#testing-infrastructure)
8. [Build System](#build-system)
9. [Critical Findings](#critical-findings)
10. [Transformation Roadmap](#transformation-roadmap)

---

## EXECUTIVE SUMMARY

### Repository Stats
- **Total Python Files**: 169
- **Test Files**: 24 (test_*.py)
- **Test Items**: 582 test cases
- **Test Coverage**: 31% (baseline)
- **Documentation Files**: 24 markdown files
- **YAML Configs**: 23 Hydra configuration files

### Architecture Status
- **Purpose**: NO-AUGMENTATION baseline for clinical text analysis
- **Tasks**: Criteria (binary classification) + Evidence (span extraction)
- **Current State**: Production-ready training, DORMANT augmentation code
- **Duplicate Implementations**: 2x architecture copies (src/Project + src/psy_agents_noaug/architectures)
- **Dependencies**: nlpaug (1.1.11+), textattack (0.3.10+) - LISTED but UNUSED
- **Critical Feature**: STRICT field separation (status vs cases) with assertions

### Key Finding for Transformation
**The augmentation infrastructure EXISTS and is COMPLETE but DORMANT.**
- Full augmentation pipeline implemented (src/psy_agents_noaug/augmentation/)
- Registry with 17 augmentation methods ready
- CLI flags present (--aug-lib, --aug-methods, --aug-p-apply)
- Integration hooks exist but set to "none" by default
- Comprehensive tests written (test_augmentation_*.py)

**Transformation Strategy**: ACTIVATE, not BUILD from scratch.

---

## MODULE TREE

### 1. Source Code Structure

```
src/
├── Project/                                 [376KB - STANDALONE IMPLEMENTATIONS]
│   ├── Criteria/                           [Binary classification architecture]
│   │   ├── data/
│   │   │   └── dataset.py                  ← TEXT ENTRY POINT #1 (CriteriaDataset)
│   │   ├── models/
│   │   │   └── model.py                    [Transformer + classification head]
│   │   ├── engine/
│   │   │   ├── trainer.py                  [Training loop]
│   │   │   └── evaluator.py                [Evaluation]
│   │   └── utils/
│   │       ├── optuna_utils.py             ← HPO HOOK #1
│   │       └── log.py
│   │
│   ├── Evidence/                           [Span extraction architecture]
│   │   ├── data/
│   │   │   └── dataset.py                  ← TEXT ENTRY POINT #2 (EvidenceDataset)
│   │   ├── models/
│   │   │   └── model.py                    [Transformer + span head]
│   │   ├── engine/
│   │   │   ├── trainer.py
│   │   │   └── evaluator.py
│   │   └── utils/
│   │       └── optuna_utils.py             ← HPO HOOK #2
│   │
│   ├── Share/                              [Shared encoder + dual heads]
│   │   ├── data/
│   │   │   └── dataset.py                  ← TEXT ENTRY POINT #3
│   │   ├── models/
│   │   ├── engine/
│   │   └── utils/
│   │       └── optuna_utils.py             ← HPO HOOK #3
│   │
│   ├── Joint/                              [Dual encoders + fusion]
│   │   ├── data/
│   │   │   └── dataset.py                  ← TEXT ENTRY POINT #4
│   │   ├── models/
│   │   ├── engine/
│   │   └── utils/
│   │       └── optuna_utils.py             ← HPO HOOK #4
│   │
│   └── utils/
│       └── checkpoint.py                   [Model checkpointing utilities]
│
├── psy_agents_noaug/                       [1.2MB - MAIN PACKAGE]
│   ├── architectures/                      [528KB - DUPLICATE OF Project/]
│   │   ├── criteria/                       ← DUPLICATE (identical models/datasets)
│   │   ├── evidence/                       ← DUPLICATE (identical models/datasets)
│   │   ├── share/                          ← DUPLICATE (identical models/datasets)
│   │   ├── joint/                          ← DUPLICATE (identical models/datasets)
│   │   └── utils/
│   │       ├── outputs.py                  [Output dataclasses]
│   │       └── checkpoint.py               [Checkpoint utils - duplicate]
│   │
│   ├── augmentation/                       [32KB - DORMANT BUT COMPLETE]
│   │   ├── __init__.py                     [Package init with NLTK setup]
│   │   ├── pipeline.py                     [11KB - AugmenterPipeline class]
│   │   │   └── AugmenterPipeline          ← MAIN AUGMENTATION ENGINE
│   │   │       ├── __call__(text: str) → str
│   │   │       ├── set_seed()
│   │   │       ├── stats()                 [Usage statistics]
│   │   │       └── worker_init()           [Multi-GPU seeding]
│   │   │
│   │   ├── registry.py                     [6.6KB - Augmenter registry]
│   │   │   └── REGISTRY                   [17 methods: nlpaug + textattack]
│   │   │       ├── nlpaug/char/* (3 methods: Keyboard, OCR, RandomChar)
│   │   │       ├── nlpaug/word/* (8 methods: Random, Reserved, Spelling, etc.)
│   │   │       └── textattack/* (7 methods: CharSwap, Deletion, Swap, etc.)
│   │   │
│   │   └── tfidf_cache.py                  [2.4KB - TF-IDF model caching]
│   │       └── load_or_fit_tfidf()        [Fit or load cached TF-IDF]
│   │
│   ├── data/                               [DATA PIPELINE - CRITICAL]
│   │   ├── groundtruth.py                  [475 lines - STRICT VALIDATION]
│   │   │   └── _assert_field_usage()      ← FIELD SEPARATION ENFORCER
│   │   │       ├── Criteria: status field ONLY
│   │   │       └── Evidence: cases field ONLY
│   │   │
│   │   ├── loaders.py                      [376 lines - Data loading]
│   │   │   ├── ReDSM5DataLoader           [HuggingFace + local CSV]
│   │   │   ├── load_posts()               ← POST TEXT SOURCE
│   │   │   ├── load_annotations()         [Status + cases fields]
│   │   │   └── group_split_by_post_id()   [Train/val/test split]
│   │   │
│   │   ├── augmentation_utils.py          [124 lines - HOOK PRESENT]
│   │   │   ├── build_evidence_augmenter() ← AUGMENTATION INTEGRATION POINT
│   │   │   │   └── Returns: AugmentationArtifacts | None
│   │   │   └── resolve_methods()          [Method resolution logic]
│   │   │
│   │   ├── classification_loader.py       [Classification-specific loaders]
│   │   └── splits.py                      [Split management]
│   │
│   ├── models/                            [MODEL HEADS - 500+ lines]
│   │   ├── criteria_head.py               [Binary classification head]
│   │   ├── evidence_head.py               [Span prediction head]
│   │   └── encoders.py                    [257 lines - Transformer encoders]
│   │
│   ├── training/                          [TRAINING INFRASTRUCTURE]
│   │   ├── train_loop.py                  [534 lines - Core Trainer]
│   │   │   └── Trainer                    ← MAIN TRAINING LOOP
│   │   │       ├── Mixed precision (AMP)
│   │   │       ├── Gradient accumulation
│   │   │       ├── Early stopping
│   │   │       ├── Checkpoint management
│   │   │       └── MLflow logging         ← METRICS TRACKING
│   │   │
│   │   ├── evaluate.py                    [449 lines - Evaluator]
│   │   └── setup.py                       [Training setup helpers]
│   │
│   ├── hpo/                               [HPO ORCHESTRATION]
│   │   ├── optuna_runner.py               [352 lines - Optuna runner]
│   │   │   └── OptunaRunner               ← HPO COORDINATOR
│   │   │       ├── objective()            [Trial objective function]
│   │   │       ├── run()                  [Execute optimization]
│   │   │       └── save_best_config()     [Export best hyperparams]
│   │   │
│   │   └── __init__.py
│   │
│   ├── utils/                             [UTILITIES - 700+ lines]
│   │   ├── reproducibility.py             [198 lines - Seed + hardware]
│   │   │   ├── set_seed()                 [Global seed setting]
│   │   │   ├── get_device()               [GPU detection]
│   │   │   └── get_optimal_dataloader_kwargs() [Hardware optimization]
│   │   │
│   │   ├── mlflow_utils.py                [341 lines - MLflow integration]
│   │   │   ├── configure_mlflow()         ← EXPERIMENT TRACKING
│   │   │   ├── log_config()               [Config logging]
│   │   │   └── log_metrics()              [Metric logging]
│   │   │
│   │   ├── logging.py                     [Logging utilities]
│   │   ├── logging_config.py              [Logging configuration]
│   │   └── type_aliases.py                [Type hints]
│   │
│   ├── cli.py                             [201 lines - THIN WRAPPER CLI]
│   │   └── Commands:
│   │       ├── train()                    ← ENTRY #1: Training
│   │       │   └── Params: aug_lib, aug_methods, aug_p_apply, etc.
│   │       ├── tune()                     ← ENTRY #2: HPO
│   │       │   └── Params: hpo_augment_only, from_best_of, etc.
│   │       ├── show-best()                [Display top-K trials]
│   │       └── tune-supermax()            [100-epoch HPO runs]
│   │
│   └── __init__.py
```

### 2. Scripts Directory (16 files)

```
scripts/
├── TRAINING DRIVERS (4 files)
│   ├── train_criteria.py                  [PRODUCTION-READY]
│   │   └── Uses: Project/Criteria/        ← PRIMARY TRAINING PATH
│   │       ├── CriteriaDataset           [Dataset from Project/]
│   │       ├── Model                     [Model from Project/]
│   │       └── Trainer                   [From psy_agents_noaug.training]
│   │
│   ├── eval_criteria.py                   [PRODUCTION-READY]
│   │   └── Evaluates trained checkpoints
│   │
│   ├── train_best.py                      [HPO INTEGRATION ROUTER]
│   │   └── Routes to architecture-specific scripts
│   │
│   └── make_groundtruth.py                [Ground truth generation]
│
├── HPO DRIVERS (3 files)                  ← AUGMENTATION HOOK ZONE
│   ├── tune_max.py                        [300+ lines - MAXIMAL HPO]
│   │   └── Orchestrates:                 ← CRITICAL FOR AUGMENTATION
│   │       ├── Model selection (BERT, RoBERTa, DeBERTa)
│   │       ├── Hyperparameter search
│   │       ├── Multi-fidelity pruning
│   │       ├── MLflow logging
│   │       └── [HOOK] Augmentation params (currently unused)
│   │
│   ├── run_hpo_stage.py                   [Multi-stage HPO: Stage 0-3]
│   │   └── Progressive refinement:
│   │       ├── Stage 0: Sanity (8 trials)
│   │       ├── Stage 1: Coarse (20 trials)
│   │       ├── Stage 2: Fine (50 trials)
│   │       └── Stage 3: Refit (best config)
│   │
│   └── run_all_hpo.py                     [NEW - Sequential wrapper]
│       └── Runs HPO for all 4 architectures
│
├── UTILITIES (9 files)
│   ├── audit_security.py                  [Security scanning]
│   ├── bench_dataloader.py                [DataLoader benchmarking]
│   ├── generate_licenses.py               [License compliance]
│   ├── generate_sbom.py                   [SBOM generation]
│   ├── verify_determinism.py              [Reproducibility verification]
│   ├── export_metrics.py                  [MLflow export]
│   ├── validate_installation.py           [Installation check]
│   ├── gpu_utilization.py                 [GPU profiling]
│   └── profile_augmentation.py            [UNUSED - for augmentation profiling]
```

### 3. Configuration System (23 YAML files)

```
configs/
├── config.yaml                            [Main composition file]
│   └── Defaults:
│       ├── data: hf_redsm5               [HuggingFace dataset]
│       ├── model: roberta_base           [Default model]
│       ├── training: default             [Training config]
│       ├── task: criteria                [Task selection]
│       └── hpo: stage0_sanity            [HPO stage]
│
├── data/                                  [DATA CONFIGS - 3 files]
│   ├── field_map.yaml                     ← CRITICAL: Field separation rules
│   │   └── Defines:
│   │       ├── status: used_for=[criteria]
│   │       └── cases: used_for=[evidence]
│   ├── hf_redsm5.yaml                     [HuggingFace source]
│   └── local_csv.yaml                     [Local CSV source]
│
├── model/                                 [MODEL CONFIGS - 3 files]
│   ├── bert_base.yaml                     [bert-base-uncased]
│   ├── roberta_base.yaml                  [roberta-base]
│   └── deberta_v3_base.yaml               [deberta-v3-base]
│
├── training/                              [TRAINING CONFIGS - 2 files]
│   ├── default.yaml                       [Standard settings]
│   │   └── Contains:
│   │       ├── num_epochs: 20
│   │       ├── batch_size: 16
│   │       ├── learning_rate: 2e-5
│   │       ├── optimizer: adamw
│   │       ├── scheduler: linear
│   │       ├── amp: {enabled: true}      [Mixed precision]
│   │       └── [HOOK] augmentation: none  ← CURRENTLY DISABLED
│   │
│   └── optimized.yaml                     [Max performance settings]
│       └── Includes:
│           ├── Hardware optimizations
│           ├── DataLoader tuning
│           ├── Mixed precision (Float16/BFloat16)
│           └── [HOOK] Augmentation params (commented out)
│
├── task/                                  [TASK CONFIGS - 2 files]
│   ├── criteria.yaml                      [Binary classification]
│   └── evidence.yaml                      [Span extraction]
│
├── hpo/                                   [HPO CONFIGS - 4 files]
│   ├── stage0_sanity.yaml                 [8 trials, quick check]
│   ├── stage1_coarse.yaml                 [20 trials, coarse search]
│   ├── stage2_fine.yaml                   [50 trials, fine-tuning]
│   └── stage3_refit.yaml                  [Refit on train+val]
│
└── {criteria,evidence,share,joint}/       [ARCHITECTURE CONFIGS - 8 files]
    ├── train.yaml                         [Architecture-specific training]
    └── hpo.yaml                           [Architecture-specific HPO]
```

---

## ENTRY POINTS

### 1. CLI Entry Points

#### Primary CLI: `src/psy_agents_noaug/cli.py`
```python
# Entry Point: psy-agents command (via Poetry scripts)
# Location: pyproject.toml → [project.scripts] → psy-agents

Commands:
├── train()                                ← TRAINING COMMAND
│   └── Parameters (AUGMENTATION-READY):
│       ├── agent: criteria|evidence|share|joint
│       ├── model_name: str = "bert-base-uncased"
│       ├── epochs: int = 3
│       ├── batch_size: int = 16
│       ├── aug_lib: str = "none"         ← HOOK: Library selection
│       ├── aug_methods: str = "all"      ← HOOK: Method selection
│       ├── aug_p_apply: float = 0.15     ← HOOK: Application probability
│       ├── aug_ops_per_sample: int = 1   ← HOOK: Operations per sample
│       ├── aug_max_replace: float = 0.3  ← HOOK: Max replacement ratio
│       ├── aug_tfidf_model: str = None   ← HOOK: TF-IDF model path
│       ├── aug_reserved_map: str = None  ← HOOK: Reserved tokens
│       ├── loader_workers: int = None
│       └── prefetch_factor: int = None
│
├── tune()                                 ← HPO COMMAND
│   └── Parameters (AUGMENTATION-READY):
│       ├── agent: str
│       ├── study: str
│       ├── n_trials: int = 200
│       ├── parallel: int = 1
│       ├── aug_lib: str = "none"         ← HOOK: Augmentation library
│       ├── aug_methods: str = "all"      ← HOOK: Augmentation methods
│       ├── hpo_augment_only: bool = False ← HOOK: Aug-only search space
│       └── from_best_of: str = None      ← HOOK: Initialize from study
│
├── show-best()                            [Display top-K HPO trials]
└── tune-supermax()                        [100-epoch HPO with EarlyStopping]
```

### 2. Script Entry Points

#### Training Scripts
```python
# 1. CRITERIA TRAINING (PRODUCTION-READY)
scripts/train_criteria.py
├── Uses: Project/Criteria implementation
├── Entry: if __name__ == "__main__": main()
├── Data Flow:
│   └── CriteriaDataset.load()           ← TEXT ENTRY POINT
│       ├── Read CSV: post_id, text, label
│       ├── Tokenize: transformers.AutoTokenizer
│       └── [HOOK] Apply augmentation: NONE (current)
│
└── Training:
    └── Trainer.train()                   [From psy_agents_noaug.training]
        ├── Forward pass
        ├── Loss computation
        ├── Backward pass
        └── MLflow logging

# 2. CRITERIA EVALUATION
scripts/eval_criteria.py
├── Uses: Project/Criteria implementation
└── Loads checkpoint and evaluates on test set

# 3. HPO INTEGRATION ROUTER
scripts/train_best.py
└── Routes to architecture-specific training scripts

# 4. GROUND TRUTH GENERATION
scripts/make_groundtruth.py
└── Generates train/val/test splits with STRICT validation
```

#### HPO Scripts
```python
# 1. MAXIMAL HPO (800-1200 trials)
scripts/tune_max.py
├── Entry: if __name__ == "__main__": main()
├── Arguments:
│   ├── --agent: criteria|evidence|share|joint
│   ├── --study: Study name
│   ├── --n-trials: Number of trials
│   ├── --parallel: Parallel workers
│   └── --outdir: Output directory
│
├── HPO Flow:
│   └── run_training_eval()               ← AUGMENTATION INJECTION POINT
│       ├── Build config from trial.suggest_*()
│       ├── [FUTURE] Suggest augmentation params
│       ├── Train model
│       ├── Evaluate on validation
│       └── Return metric
│
└── Logged to: MLflow + Optuna DB

# 2. MULTI-STAGE HPO (Progressive refinement)
scripts/run_hpo_stage.py
├── Stage 0: Sanity (8 trials)
├── Stage 1: Coarse (20 trials)
├── Stage 2: Fine (50 trials)
└── Stage 3: Refit (best config on train+val)

# 3. SEQUENTIAL HPO WRAPPER
scripts/run_all_hpo.py
└── Runs HPO for all architectures sequentially
```

### 3. Makefile Targets (Key Workflows)

```makefile
# TRAINING
make train TASK=criteria MODEL=roberta_base
  → poetry run python -m psy_agents_noaug.cli train task=criteria model=roberta_base

# HPO (Multi-stage)
make hpo-s0                                [Stage 0: Sanity]
make hpo-s1                                [Stage 1: Coarse]
make hpo-s2                                [Stage 2: Fine]
make refit                                 [Stage 3: Refit]
make full-hpo-all                          [All architectures, multi-stage]

# HPO (Maximal)
make tune-criteria-max                     [800 trials, 6 epochs]
make tune-evidence-max                     [1200 trials, 6 epochs]
make maximal-hpo-all                       [All architectures, maximal]

# HPO (Super-Max) - 100 EPOCHS + EARLYSTOPPING
make tune-criteria-supermax                [5000 trials, 100 epochs, patience=20]
make tune-evidence-supermax                [8000 trials, 100 epochs]
make tune-all-supermax                     [~19K trials total]

# EVALUATION
make eval CHECKPOINT=outputs/checkpoints/best_checkpoint.pt

# UTILITIES
make groundtruth                           [Generate ground truth from HF]
make test                                  [Run all tests]
make test-cov                              [Test with coverage]
```

---

## DATA FLOW ARCHITECTURE

### 1. Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCE                                  │
├─────────────────────────────────────────────────────────────────────┤
│  HuggingFace: irlab-udc/redsm5    OR    Local CSV files           │
│     ├── posts.csv                        ├── data/raw/redsm5/      │
│     │   ├── post_id                      │   ├── posts.csv         │
│     │   └── text ──────────────┐         │   └── annotations.csv   │
│     │                           │         │                          │
│     └── annotations.csv         │         │                          │
│         ├── post_id             │         │                          │
│         ├── criterion_id        │         │                          │
│         ├── status ─────────────┼─────────┼─→ CRITERIA ONLY         │
│         └── cases ──────────────┼─────────┼─→ EVIDENCE ONLY         │
└─────────────────────────────────┼─────────┴──────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│               FIELD MAP VALIDATION                                  │
│         configs/data/field_map.yaml                                 │
├─────────────────────────────────────────────────────────────────────┤
│  status:                                                            │
│    required: true                                                   │
│    used_for: ["criteria"]     ← STRICT SEPARATION ENFORCED         │
│    type: "string"                                                   │
│    normalization:                                                   │
│      positive_values: [present, positive, true, 1]                  │
│      negative_values: [absent, negative, false, 0]                  │
│                                                                      │
│  cases:                                                             │
│    required: true                                                   │
│    used_for: ["evidence"]     ← STRICT SEPARATION ENFORCED         │
│    type: "json"                                                     │
│    structure:                                                       │
│      - text: "string"        ← EVIDENCE TEXT SOURCE                │
│      - start_char: "int"                                            │
│      - end_char: "int"                                              │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│           GROUND TRUTH GENERATION                                   │
│      src/psy_agents_noaug/data/groundtruth.py                       │
├─────────────────────────────────────────────────────────────────────┤
│  _assert_field_usage(field_name, expected_field, operation)        │
│    ├── IF field_name != expected_field: RAISE AssertionError       │
│    └── Prevents data leakage between tasks                         │
│                                                                      │
│  generate_criteria_groundtruth()                                    │
│    ├── ASSERT: uses "status" field ONLY                            │
│    ├── Normalize labels: positive/negative → 0/1                   │
│    └── Output: data/processed/criteria_groundtruth.csv             │
│                                                                      │
│  generate_evidence_groundtruth()                                    │
│    ├── ASSERT: uses "cases" field ONLY                             │
│    ├── Parse JSON: extract text, start_char, end_char              │
│    └── Output: data/processed/evidence_groundtruth.csv             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴──────────────┐
                    ↓                            ↓
┌───────────────────────────┐    ┌──────────────────────────────┐
│   CRITERIA PIPELINE       │    │   EVIDENCE PIPELINE          │
├───────────────────────────┤    ├──────────────────────────────┤
│ CriteriaDataset           │    │ EvidenceDataset              │
│ ├── Load CSV              │    │ ├── Load CSV                 │
│ ├── Read: post_id, text,  │    │ ├── Read: post_id, text,     │
│ │         label           │    │ │         start, end         │
│ ├── Tokenize text         │    │ ├── Tokenize text            │
│ └── [HOOK] Augment: NONE  │    │ ├── [HOOK] Augment: NONE     │
│                           │    │ │   ↓                         │
│ Model (Binary Classifier) │    │ │   Evidence Text ───────┐    │
│ ├── Encoder: Transformer  │    │ │                        │    │
│ └── Head: Classification  │    │ └── Find span positions │    │
│                           │    │                          │    │
│ Output: present/absent    │    │ Model (Span Extractor)   │    │
└───────────────────────────┘    │ ├── Encoder: Transformer │    │
                                  │ ├── Head: Span prediction│    │
                                  │ └── Output: start/end pos│    │
                                  └──────────────────────────┘
```

### 2. Text Entry Points (WHERE AUGMENTATION MUST BE INJECTED)

```python
# LOCATION 1: Project/Criteria/data/dataset.py
class CriteriaDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        text_column: str = "text",        ← TEXT ENTRY POINT
        label_column: str = "label",
        max_length: int = 512,
        # [MISSING] augmenter: AugmenterPipeline | None = None
    ):
        ...

    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.text_column]
        label = self.df.iloc[idx][self.label_column]

        # [INJECTION POINT #1] ─────────────────────────┐
        # if self.augmenter and self.training:          │
        #     text = self.augmenter(text)                │ AUGMENTATION
        # ───────────────────────────────────────────────┘ HOOK

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# LOCATION 2: Project/Evidence/data/dataset.py
class EvidenceDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        text_column: str = "text",        ← TEXT ENTRY POINT
        start_column: str = "start_char",
        end_column: str = "end_char",
        max_length: int = 512,
        # [MISSING] augmenter: AugmenterPipeline | None = None
    ):
        ...

    def __getitem__(self, idx):
        text = self.df.iloc[idx][self.text_column]
        start_char = self.df.iloc[idx][self.start_column]
        end_char = self.df.iloc[idx][self.end_column]

        # [INJECTION POINT #2] ─────────────────────────┐
        # if self.augmenter and self.training:          │
        #     text = self.augmenter(text)                │ AUGMENTATION
        #     # CRITICAL: Adjust span positions!         │ HOOK
        # ───────────────────────────────────────────────┘

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Token-level positions
        start_token = self._char_to_token_position(start_char, encoding)
        end_token = self._char_to_token_position(end_char, encoding)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "start_positions": torch.tensor(start_token, dtype=torch.long),
            "end_positions": torch.tensor(end_token, dtype=torch.long)
        }

# LOCATION 3: Share architecture (dual task)
# Similar injection points for both criteria and evidence heads

# LOCATION 4: Joint architecture (dual encoder)
# Similar injection points for both encoders
```

### 3. Current Dataset/DataLoader Architecture

```python
# CURRENT STATE (NO AUGMENTATION)

# Step 1: Dataset instantiation
from Project.Criteria.data.dataset import CriteriaDataset

dataset = CriteriaDataset(
    csv_path="data/processed/criteria_groundtruth.csv",
    tokenizer=tokenizer,
    text_column="text",
    label_column="label",
    max_length=512
)
# → No augmenter parameter
# → No augmentation in __getitem__

# Step 2: DataLoader creation
from torch.utils.data import DataLoader
from psy_agents_noaug.utils.reproducibility import get_optimal_dataloader_kwargs

dataloader_kwargs = get_optimal_dataloader_kwargs(
    device=device,
    num_workers=8,           # Hardware-optimized
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    **dataloader_kwargs
)
# → No worker_init_fn for augmentation seeding
```

---

## AUGMENTATION INFRASTRUCTURE STATUS

### 1. Current State: DORMANT BUT COMPLETE

```
Status: ████████████░░░░░░░░ 60% Complete (Implementation exists, integration pending)

Components:
├── Registry: ✅ COMPLETE (17 methods)
├── Pipeline: ✅ COMPLETE (AugmenterPipeline class)
├── TF-IDF:   ✅ COMPLETE (Caching + loading)
├── CLI:      ✅ COMPLETE (Flags present)
├── Configs:  ⚠️  PARTIAL (Hooks present, set to "none")
├── Datasets: ❌ MISSING (No augmenter parameter)
├── HPO:      ❌ MISSING (No augmentation search space)
└── Tests:    ✅ COMPLETE (Comprehensive test coverage)
```

### 2. Augmentation Code Location

```
src/psy_agents_noaug/augmentation/
├── __init__.py                            [57 lines - Package initialization]
│   ├── Ensures NLTK WordNet downloaded
│   └── Exports: AugConfig, AugmenterPipeline, REGISTRY, etc.
│
├── pipeline.py                            [336 lines - CORE ENGINE]
│   ├── AugConfig                          [Dataclass: Configuration]
│   │   ├── lib: "none" | "nlpaug" | "textattack" | "both"
│   │   ├── methods: list[str] = ["all"]
│   │   ├── p_apply: float = 0.15          [Application probability]
│   │   ├── ops_per_sample: int = 1        [Operations per sample]
│   │   ├── max_replace_ratio: float = 0.3 [Max replacement ratio]
│   │   ├── tfidf_model_path: str | None
│   │   ├── reserved_map_path: str | None
│   │   ├── seed: int = 42
│   │   └── example_limit: int = 32        [Log examples]
│   │
│   ├── AugResources                       [Dataclass: External resources]
│   │   ├── tfidf_model_path: str | None
│   │   └── reserved_map_path: str | None
│   │
│   ├── AugmenterPipeline                  [MAIN AUGMENTATION CLASS]
│   │   ├── __init__(cfg: AugConfig, resources: AugResources | None)
│   │   │   └── Instantiates augmenters from registry
│   │   │
│   │   ├── __call__(text: str) → str      ← MAIN AUGMENTATION METHOD
│   │   │   ├── Random roll: self._rng.random() > p_apply → skip
│   │   │   ├── Select method: random.choice(self._augmenters)
│   │   │   ├── Apply: wrapper.augment_one(text)
│   │   │   ├── Record stats: self.method_counts[method_name] += 1
│   │   │   └── Return augmented text
│   │   │
│   │   ├── set_seed(seed: int)            [Reset RNG for reproducibility]
│   │   ├── stats() → dict                 [Usage statistics]
│   │   ├── drain_examples() → list        [Logged examples]
│   │   └── close()                        [Resource cleanup]
│   │
│   ├── worker_init(worker_id, base_seed, rank, num_workers_per_rank)
│   │   └── Ensures unique seeds for DataLoader workers (DDP-aware)
│   │
│   └── is_enabled(cfg: AugConfig) → bool  [Check if augmentation active]
│
├── registry.py                            [200 lines - AUGMENTER REGISTRY]
│   ├── REGISTRY: dict[str, RegistryEntry] ← 17 AUGMENTATION METHODS
│   │   │
│   │   ├── nlpaug/char/* (3 methods)
│   │   │   ├── KeyboardAug                [Keyboard typos]
│   │   │   ├── OcrAug                     [OCR errors]
│   │   │   └── RandomCharAug              [Random char operations]
│   │   │
│   │   ├── nlpaug/word/* (8 methods)
│   │   │   ├── RandomWordAug              [Random word operations]
│   │   │   ├── ReservedAug                [Protected token substitution]
│   │   │   ├── SpellingAug                [Spelling errors]
│   │   │   ├── SplitAug                   [Word splitting]
│   │   │   ├── SynonymAug                 [WordNet synonyms]
│   │   │   ├── AntonymAug                 [WordNet antonyms]
│   │   │   └── TfIdfAug                   [TF-IDF-based substitution]
│   │   │
│   │   └── textattack/* (7 methods)
│   │       ├── CharSwapAugmenter          [Swap adjacent chars]
│   │       ├── DeletionAugmenter          [Delete words]
│   │       ├── SwapAugmenter              [Swap word order]
│   │       ├── SynonymInsertionAugmenter  [Insert synonyms]
│   │       ├── EasyDataAugmenter          [EDA recipe]
│   │       ├── CheckListAugmenter         [CheckList recipe]
│   │       └── WordNetAugmenter           [WordNet recipe]
│   │
│   ├── AugmenterWrapper                   [Wrapper class for uniform interface]
│   │   └── augment_one(text: str) → str   [Normalized augmentation]
│   │
│   └── ALL_METHODS, NLPAUG_METHODS, TEXTATTACK_METHODS [Method lists]
│
├── tfidf_cache.py                         [97 lines - TF-IDF MODEL CACHING]
│   ├── TfidfResource                      [Dataclass: TF-IDF model + metadata]
│   │   ├── path: Path                     [Model pickle path]
│   │   ├── num_docs: int                  [Number of documents]
│   │   └── vocab_size: int                [Vocabulary size]
│   │
│   └── load_or_fit_tfidf(texts, model_path) → TfidfResource
│       ├── IF model_path exists: load pickle
│       └── ELSE: fit TF-IDF, save pickle, return
│
└── (Integration point)
    src/psy_agents_noaug/data/augmentation_utils.py
    ├── build_evidence_augmenter()         ← INTEGRATION HELPER
    │   ├── Resolve methods
    │   ├── Fit TF-IDF if needed
    │   ├── Instantiate AugmenterPipeline
    │   └── Return AugmentationArtifacts | None
    │
    └── AugmentationArtifacts              [Dataclass: Bundle pipeline + resources]
        ├── pipeline: AugmenterPipeline
        ├── config: AugConfig
        ├── resources: AugResources
        ├── tfidf: TfidfResource | None
        └── methods: tuple[str, ...]
```

### 3. What Needs Activation

```yaml
# STEP 1: Modify Dataset Classes (4 files)
# Location: src/Project/{Criteria,Evidence,Share,Joint}/data/dataset.py
Changes:
  - Add augmenter parameter to __init__()
  - Add training mode flag
  - Inject augmentation call in __getitem__()
  - Handle span position adjustment (Evidence only)

# STEP 2: Modify Training Scripts (4 files)
# Location: scripts/train_{criteria,evidence,share,joint}.py
Changes:
  - Import: from psy_agents_noaug.augmentation import AugConfig, AugmenterPipeline
  - Build augmenter: build_evidence_augmenter(cfg, train_texts)
  - Pass to dataset: CriteriaDataset(..., augmenter=pipeline)
  - Add worker_init_fn to DataLoader

# STEP 3: Modify HPO Scripts (1 file)
# Location: scripts/tune_max.py
Changes:
  - Add augmentation parameters to search space:
    * lib: trial.suggest_categorical("aug_lib", ["none", "nlpaug", "textattack", "both"])
    * methods: trial.suggest_categorical("aug_methods", ["all", ...])
    * p_apply: trial.suggest_float("aug_p_apply", 0.0, 0.3)
    * ops_per_sample: trial.suggest_int("aug_ops_per_sample", 1, 2)
    * max_replace: trial.suggest_float("aug_max_replace", 0.1, 0.5)
  - Pass augmentation config to training function

# STEP 4: Update Configs (2 files)
# Location: configs/training/{default,optimized}.yaml
Changes:
  - Add augmentation section:
    augmentation:
      lib: "nlpaug"              # Changed from "none"
      methods: ["all"]
      p_apply: 0.15
      ops_per_sample: 1
      max_replace_ratio: 0.3

# STEP 5: Add Missing Tests (Estimated: 3-5 new files)
# Location: tests/
New Tests:
  - test_augmentation_integration.py    [Dataset + augmenter integration]
  - test_augmentation_hpo.py            [HPO search space for augmentation]
  - test_augmentation_evidence_spans.py [Span position preservation]
  - test_augmentation_determinism.py    [Worker seeding + reproducibility]
```

### 4. Augmentation Readiness Matrix

| Component                  | Status      | Lines to Add | Complexity | Priority |
|----------------------------|-------------|--------------|------------|----------|
| Registry                   | ✅ Complete | 0            | N/A        | N/A      |
| Pipeline                   | ✅ Complete | 0            | N/A        | N/A      |
| TF-IDF Cache               | ✅ Complete | 0            | N/A        | N/A      |
| CLI Flags                  | ✅ Complete | 0            | N/A        | N/A      |
| Criteria Dataset           | ❌ Missing  | ~15          | Low        | HIGH     |
| Evidence Dataset           | ❌ Missing  | ~30          | Medium     | HIGH     |
| Share Dataset              | ❌ Missing  | ~25          | Medium     | MEDIUM   |
| Joint Dataset              | ❌ Missing  | ~25          | Medium     | MEDIUM   |
| Criteria Training Script   | ❌ Missing  | ~20          | Low        | HIGH     |
| Evidence Training Script   | ⚠️  Partial | ~20          | Low        | HIGH     |
| HPO Search Space           | ❌ Missing  | ~40          | Medium     | HIGH     |
| Config Defaults            | ⚠️  Partial | ~10          | Low        | MEDIUM   |
| Integration Tests          | ❌ Missing  | ~200         | Medium     | HIGH     |
| Documentation              | ⚠️  Partial | ~500         | Low        | MEDIUM   |

**Estimated Total Effort**: 385 lines + tests + docs = ~1000 lines
**Estimated Time**: 8-12 hours for full integration + testing

---

## HPO INTEGRATION POINTS

### 1. HPO Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                     HPO SYSTEM ARCHITECTURE                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TWO HPO MODES (both support augmentation params):              │
│                                                                  │
│  1. MULTI-STAGE HPO (Progressive refinement)                    │
│     ├── Stage 0: Sanity (8 trials)       ← Quick check          │
│     ├── Stage 1: Coarse (20 trials)      ← Explore space        │
│     ├── Stage 2: Fine (50 trials)        ← Refine best          │
│     └── Stage 3: Refit (1 run)           ← Train on train+val   │
│                                                                  │
│  2. MAXIMAL HPO (Single large run)                              │
│     ├── Criteria: 800 trials, 6 epochs                          │
│     ├── Evidence: 1200 trials, 6 epochs                         │
│     ├── Share: 600 trials, 6 epochs                             │
│     └── Joint: 600 trials, 6 epochs                             │
│                                                                  │
│  3. SUPER-MAXIMAL HPO (100-epoch trials with EarlyStopping)     │
│     ├── Criteria: 5000 trials, 100 epochs, patience=20          │
│     ├── Evidence: 8000 trials, 100 epochs, patience=20          │
│     ├── Share: 3000 trials, 100 epochs, patience=20             │
│     └── Joint: 3000 trials, 100 epochs, patience=20             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 2. HPO Search Space (scripts/tune_max.py)

```python
# CURRENT SEARCH SPACE (NO AUGMENTATION)

def suggest_config(trial: optuna.Trial, agent: str) -> dict:
    """Build trial configuration."""

    # ===== MODEL SELECTION =====
    model_name = trial.suggest_categorical(
        "model_name",
        [
            "bert-base-uncased",
            "bert-large-uncased",
            "roberta-base",
            "roberta-large",
            "microsoft/deberta-v3-base",
            "microsoft/deberta-v3-large",
        ]
    )

    # ===== OPTIMIZER =====
    optimizer = trial.suggest_categorical("optimizer", ["adamw", "adafactor"])
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)

    # ===== SCHEDULER =====
    scheduler = trial.suggest_categorical(
        "scheduler", ["linear", "cosine", "polynomial"]
    )
    warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)

    # ===== TRAINING =====
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2, 4])
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0)

    # ===== ARCHITECTURE (agent-specific) =====
    if agent == "criteria":
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        # ... criteria-specific params
    elif agent == "evidence":
        span_loss_weight = trial.suggest_float("span_loss_weight", 0.5, 2.0)
        # ... evidence-specific params

    # ===== [MISSING] AUGMENTATION SEARCH SPACE =====
    # aug_lib = trial.suggest_categorical("aug_lib", ["none", "nlpaug", "textattack", "both"])
    # aug_methods = ...
    # aug_p_apply = trial.suggest_float("aug_p_apply", 0.0, 0.3)
    # aug_ops_per_sample = trial.suggest_int("aug_ops_per_sample", 1, 2)
    # aug_max_replace = trial.suggest_float("aug_max_replace", 0.1, 0.5)

    return {
        "model_name": model_name,
        "optimizer": optimizer,
        "lr": lr,
        # ...
        # "augmentation": {
        #     "lib": aug_lib,
        #     "methods": aug_methods,
        #     "p_apply": aug_p_apply,
        #     # ...
        # }
    }
```

### 3. Augmentation Injection Points in HPO

```python
# LOCATION: scripts/tune_max.py

def run_training_eval(cfg: dict, callbacks: list) -> dict:
    """
    Train and evaluate with given config.

    This is THE CRITICAL INTEGRATION POINT for augmentation.

    Current flow:
    1. Build model from cfg["model_name"]
    2. Build optimizer from cfg["optimizer"], cfg["lr"]
    3. Build scheduler from cfg["scheduler"]
    4. Load datasets (NO AUGMENTATION)
    5. Train for N epochs
    6. Evaluate on validation set
    7. Return metrics

    REQUIRED CHANGES for augmentation:
    """

    # [STEP 1] Extract augmentation config
    aug_cfg = AugConfig(
        lib=cfg.get("aug_lib", "none"),
        methods=cfg.get("aug_methods", ["all"]),
        p_apply=cfg.get("aug_p_apply", 0.15),
        ops_per_sample=cfg.get("aug_ops_per_sample", 1),
        max_replace_ratio=cfg.get("aug_max_replace", 0.3),
        seed=cfg.get("seed", 42),
    )

    # [STEP 2] Build augmenter if enabled
    augmenter = None
    if is_enabled(aug_cfg):
        # For evidence task, need to fit TF-IDF on training data
        if agent in ["evidence", "share", "joint"]:
            train_texts = [sample["text"] for sample in train_dataset]
            artifacts = build_evidence_augmenter(
                aug_cfg,
                train_texts,
                tfidf_dir=f"_artifacts/tfidf/{agent}"
            )
            augmenter = artifacts.pipeline if artifacts else None

    # [STEP 3] Pass augmenter to dataset
    train_dataset = CriteriaDataset(
        csv_path=train_path,
        tokenizer=tokenizer,
        augmenter=augmenter,  # ← NEW PARAMETER
        training=True         # ← ENABLE AUGMENTATION
    )

    # [STEP 4] Add worker_init_fn for multi-GPU determinism
    from psy_agents_noaug.augmentation import worker_init

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        worker_init_fn=lambda worker_id: worker_init(
            worker_id,
            base_seed=cfg["seed"],
            rank=0,  # Single-GPU for now
            num_workers_per_rank=cfg.get("num_workers", 4)
        )
    )

    # ... rest of training ...

    # [STEP 5] Log augmentation stats to MLflow
    if augmenter:
        stats = augmenter.stats()
        mlflow.log_metrics({
            "aug_total": stats["total"],
            "aug_applied": stats["applied"],
            "aug_skipped": stats["skipped"],
            "aug_rate": stats["applied"] / max(stats["total"], 1)
        })
        mlflow.log_dict(stats["method_counts"], "augmentation_method_counts.json")

        # Log augmentation examples
        examples = augmenter.drain_examples()
        mlflow.log_dict(examples[:10], "augmentation_examples.json")
```

### 4. HPO Metrics & Logging

```python
# MLflow Tracking Structure

┌─────────────────────────────────────────────────────────────┐
│                    MLflow Experiment                        │
│               "NoAug_Criteria_Evidence"                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Run ID: <uuid>                                             │
│  ├── Parameters                                             │
│  │   ├── model_name: "roberta-base"                        │
│  │   ├── lr: 2.5e-5                                        │
│  │   ├── batch_size: 16                                    │
│  │   ├── optimizer: "adamw"                                │
│  │   ├── scheduler: "linear"                               │
│  │   ├── aug_lib: "nlpaug"            ← NEW                │
│  │   ├── aug_methods: "all"           ← NEW                │
│  │   ├── aug_p_apply: 0.15            ← NEW                │
│  │   ├── aug_ops_per_sample: 1        ← NEW                │
│  │   └── aug_max_replace: 0.3         ← NEW                │
│  │                                                          │
│  ├── Metrics (per epoch)                                   │
│  │   ├── train_loss: [0.45, 0.32, 0.25, ...]              │
│  │   ├── val_loss: [0.50, 0.38, 0.30, ...]                │
│  │   ├── val_accuracy: [0.75, 0.82, 0.85, ...]            │
│  │   ├── val_f1_macro: [0.70, 0.78, 0.82, ...]            │
│  │   ├── aug_total: 12000             ← NEW                │
│  │   ├── aug_applied: 1800            ← NEW                │
│  │   ├── aug_skipped: 10200           ← NEW                │
│  │   └── aug_rate: 0.15               ← NEW                │
│  │                                                          │
│  ├── Artifacts                                             │
│  │   ├── config.yaml                                       │
│  │   ├── model_checkpoint.pt                               │
│  │   ├── augmentation_method_counts.json  ← NEW            │
│  │   │   {                                                 │
│  │   │     "nlpaug/word/SynonymAug": 450,                  │
│  │   │     "nlpaug/word/RandomWordAug": 380,               │
│  │   │     ...                                             │
│  │   │   }                                                 │
│  │   └── augmentation_examples.json    ← NEW               │
│  │       [                                                 │
│  │         {                                               │
│  │           "original": "Patient reports...",             │
│  │           "augmented": "Patient describes...",          │
│  │           "methods": ["nlpaug/word/SynonymAug"],        │
│  │           "timestamp": 1698765432.123                   │
│  │         },                                              │
│  │         ...                                             │
│  │       ]                                                 │
│  │                                                          │
│  └── Tags                                                   │
│      ├── agent: "criteria"                                 │
│      ├── study: "noaug-criteria-max"                       │
│      └── trial_number: 42                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

# Optuna Database Structure

Optuna SQLite DB: ./_optuna/noaug.db
├── Studies
│   ├── noaug-criteria-max
│   ├── noaug-evidence-max
│   ├── noaug-share-max
│   └── noaug-joint-max
│
└── Trials (per study)
    ├── Trial #0
    │   ├── State: COMPLETE
    │   ├── Value: 0.8523 (val_f1_macro)
    │   ├── Params:
    │   │   ├── model_name: "roberta-base"
    │   │   ├── lr: 2.5e-5
    │   │   ├── aug_lib: "nlpaug"           ← NEW
    │   │   ├── aug_p_apply: 0.15           ← NEW
    │   │   └── ...
    │   └── User Attrs:
    │       ├── mlflow_run_id: <uuid>
    │       ├── aug_total: 12000             ← NEW
    │       ├── aug_applied: 1800            ← NEW
    │       └── aug_rate: 0.15               ← NEW
    │
    ├── Trial #1
    │   └── ...
    │
    └── Trial #N
```

---

## TESTING INFRASTRUCTURE

### 1. Test Organization

```
tests/                                     [24 test files, 582 test cases]
├── CRITICAL VALIDATION
│   └── test_groundtruth.py                [Field separation enforcement]
│       ├── test_criteria_uses_status_only()
│       ├── test_evidence_uses_cases_only()
│       ├── test_field_mixing_raises_error()
│       └── test_normalization_logic()
│
├── ARCHITECTURE TESTS
│   ├── test_arch_shapes.py                [Output shape validation]
│   │   ├── test_criteria_output_shape()
│   │   ├── test_evidence_output_shape()
│   │   ├── test_share_output_shape()
│   │   └── test_joint_output_shape()
│   │
│   └── test_head_space.py                 [Head implementation space]
│
├── HPO TESTS
│   ├── test_hpo_config.py                 [HPO configuration validation]
│   │   ├── test_stage0_config()
│   │   ├── test_stage1_config()
│   │   ├── test_stage2_config()
│   │   └── test_optuna_sampler_initialization()
│   │
│   └── test_hpo_integration.py            [HPO workflow validation]
│       ├── test_multi_stage_hpo()
│       ├── test_maximal_hpo()
│       └── test_best_config_export()
│
├── TRAINING TESTS
│   ├── test_training_smoke.py             [Training pipeline smoke tests]
│   │   ├── test_criteria_training()
│   │   ├── test_evidence_training()
│   │   ├── test_mixed_precision()
│   │   └── test_early_stopping()
│   │
│   └── test_train_smoke.py                [Alternative smoke tests]
│
├── DATA PIPELINE TESTS
│   ├── test_loaders.py                    [DataLoader tests]
│   │   ├── test_redsm5_loader_hf()
│   │   ├── test_redsm5_loader_local()
│   │   ├── test_field_map_validation()
│   │   └── test_group_split_by_post_id()
│   │
│   ├── test_pipeline_comprehensive.py     [End-to-end pipeline]
│   ├── test_pipeline_extended.py          [Extended pipeline tests]
│   ├── test_pipeline_integration.py       [Integration tests]
│   └── test_pipeline_scope.py             [Pipeline scope validation]
│
├── AUGMENTATION TESTS (Ironic for NO-AUG)  ← ALREADY EXIST!
│   ├── test_augmentation_registry.py      [12KB - Registry tests]
│   │   ├── test_all_methods_in_registry()
│   │   ├── test_nlpaug_methods()
│   │   ├── test_textattack_methods()
│   │   ├── test_augmenter_wrapper()
│   │   ├── test_reserved_aug()
│   │   └── test_tfidf_aug()
│   │
│   ├── test_augmentation_utils.py         [19KB - Utils tests]
│   │   ├── test_build_evidence_augmenter()
│   │   ├── test_resolve_methods()
│   │   ├── test_augmentation_artifacts()
│   │   └── test_tfidf_fitting()
│   │
│   ├── test_tfidf_cache.py                [771B - Basic cache tests]
│   │   ├── test_tfidf_cache_save_load()
│   │   └── test_tfidf_cache_metadata()
│   │
│   ├── test_tfidf_cache_extended.py       [11KB - Extended cache tests]
│   │   ├── test_tfidf_concurrent_access()
│   │   ├── test_tfidf_cache_invalidation()
│   │   └── test_tfidf_resource_cleanup()
│   │
│   └── test_seed_determinism.py           [Worker seeding tests]
│       ├── test_worker_init()
│       ├── test_ddp_seeding()
│       └── test_augmentation_determinism()
│
├── PERFORMANCE TESTS
│   ├── test_perf_contract.py              [Performance contract validation]
│   ├── test_benchmarks/
│   │   └── test_performance_regression.py [Performance regression tests]
│   │
│   └── test_mlflow_artifacts.py           [MLflow artifact validation]
│
├── INTEGRATION TESTS
│   ├── test_integration.py                [End-to-end integration]
│   └── test_smoke.py                      [Basic smoke tests]
│
└── MISC
    ├── test_cli_flags.py                  [CLI flag validation]
    ├── test_qa_null_policy.py             [QA null policy tests]
    └── conftest.py                        [Pytest fixtures]
```

### 2. Test Coverage Analysis

```yaml
Current Coverage: 31%

Breakdown by Module:
  src/psy_agents_noaug/augmentation/:
    - pipeline.py: 85% ✅ (HIGH - Well-tested)
    - registry.py: 78% ✅ (HIGH - Well-tested)
    - tfidf_cache.py: 82% ✅ (HIGH - Well-tested)

  src/psy_agents_noaug/data/:
    - groundtruth.py: 68% ⚠️ (MEDIUM - Needs more edge case tests)
    - loaders.py: 55% ⚠️ (MEDIUM - Needs HF dataset tests)
    - augmentation_utils.py: 0% ❌ (CRITICAL - No integration tests)

  src/psy_agents_noaug/training/:
    - train_loop.py: 42% ⚠️ (MEDIUM - Missing edge cases)
    - evaluate.py: 38% ⚠️ (MEDIUM - Missing edge cases)

  src/psy_agents_noaug/hpo/:
    - optuna_runner.py: 25% ❌ (LOW - Needs more tests)

  src/Project/:
    - Criteria/: 15% ❌ (LOW - Minimal test coverage)
    - Evidence/: 12% ❌ (LOW - Minimal test coverage)
    - Share/: 8% ❌ (LOW - Minimal test coverage)
    - Joint/: 8% ❌ (LOW - Minimal test coverage)

  scripts/:
    - train_criteria.py: 0% ❌ (NONE - No tests)
    - tune_max.py: 0% ❌ (NONE - No tests)
    - run_hpo_stage.py: 0% ❌ (NONE - No tests)

Target Coverage for Production: 80%+

Missing Critical Tests:
  1. ❌ Dataset + Augmenter Integration (test_augmentation_integration.py)
     - Test: CriteriaDataset with augmenter parameter
     - Test: Evidence span position preservation after augmentation
     - Test: Augmentation statistics tracking

  2. ❌ HPO Augmentation Search Space (test_augmentation_hpo.py)
     - Test: Augmentation params in trial.suggest_*()
     - Test: HPO with aug_lib="nlpaug", "textattack", "both"
     - Test: Optuna pruning with augmentation

  3. ❌ Multi-GPU Worker Seeding (test_augmentation_worker_seeding.py)
     - Test: DDP-aware worker_init()
     - Test: Unique augmentation per worker
     - Test: Reproducibility across runs

  4. ❌ MLflow Augmentation Logging (test_mlflow_augmentation.py)
     - Test: Augmentation params logged
     - Test: Augmentation stats logged
     - Test: Augmentation examples logged

  5. ⚠️ Evidence Span Preservation (extend test_augmentation_utils.py)
     - Test: Span positions remain valid after augmentation
     - Test: Span text matches original meaning
     - Test: Boundary cases (start=0, end=len(text))
```

### 3. Testing Priorities for Augmentation Integration

| Priority | Test Category                | Files to Create/Modify | Est. Lines | Complexity |
|----------|------------------------------|-------------------------|------------|------------|
| P0       | Dataset Integration          | test_augmentation_integration.py | ~250 | Medium     |
| P0       | HPO Search Space             | test_augmentation_hpo.py | ~200 | Medium     |
| P0       | Span Position Preservation   | test_evidence_spans.py | ~180 | High       |
| P1       | Multi-GPU Worker Seeding     | test_worker_seeding_ddp.py | ~150 | Medium     |
| P1       | MLflow Logging               | test_mlflow_augmentation.py | ~120 | Low        |
| P2       | Performance Regression       | test_perf_augmentation.py | ~100 | Medium     |
| P2       | Determinism                  | Extend test_seed_determinism.py | ~80 | Low        |

**Estimated Testing Effort**: ~1080 lines + fixtures + docs = ~1500 lines total
**Estimated Time**: 12-16 hours for comprehensive test suite

---

## BUILD SYSTEM

### 1. Dependency Management (pyproject.toml)

```toml
[project]
name = "noaug-criteria-evidence"
requires-python = ">=3.10"

dependencies = [
  # Core ML
  "torch>=2.2",                            # PyTorch 2.2+
  "transformers>=4.44",                    # HuggingFace Transformers
  "tokenizers>=0.15",                      # Fast tokenizers

  # Datasets
  "datasets>=2.20",                        # HuggingFace Datasets (for redsm5)
  "pandas>=2.0",                           # DataFrame operations
  "numpy>=1.26",                           # Numerical operations
  "scikit-learn>=1.4",                     # Metrics + splitting

  # HPO & Tracking
  "optuna~=4.5.0",                         # Hyperparameter optimization
  "mlflow>=2.14",                          # Experiment tracking

  # Augmentation (LISTED but UNUSED)       ← CRITICAL FOR TRANSFORMATION
  "nlpaug>=1.1.11",                        # ✅ READY: NLP augmentation
  "textattack>=0.3.10",                    # ✅ READY: Text attack recipes
  "nltk>=3.8",                             # ✅ READY: WordNet, tokenization

  # CLI & Config
  "typer>=0.12",                           # CLI framework
  "pydantic>=2.8",                         # Config validation
  "evaluate>=0.4",                         # Evaluation metrics

  # System Monitoring
  "psutil>=7.1.1,<8.0.0",                  # CPU/memory monitoring
  "pynvml>=13.0.1,<14.0.0",                # GPU monitoring (NVIDIA)
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",                           # Testing framework
  "pytest-cov>=4.1",                       # Coverage reports
  "ruff>=0.6",                             # Fast linter
  "black>=24.0",                           # Code formatter
  "mypy>=1.9",                             # Type checking
]

[tool.poetry.dependencies]
python = "^3.10,<3.12"
# ... (same as above, with version constraints)

[tool.poetry.group.dev.dependencies]
ruff = ">=0.6,<0.8"
black = ">=24.0,<25.0"
isort = ">=5.12.0,<6.0"
pytest = ">=8.0,<9.0"
pytest-cov = ">=4.1.0,<6.0"
pytest-mock = ">=3.11.1,<4.0"
pre-commit = ">=3.3.0,<4.0"
bandit = ">=1.7.5,<2.0"                    # Security linter
mypy = ">=1.9,<2.0"
pip-audit = ">=2.6,<3.0"                   # Vulnerability scanning
pip-licenses = ">=4.3,<5.0"                # License compliance
pipdeptree = ">=2.13,<3.0"                 # Dependency tree
hypothesis = ">=6.98,<7.0"                 # Property-based testing

[project.scripts]
psy-agents = "psy_agents_noaug.cli:main"   # CLI entry point

[tool.poetry.scripts]
psy-agents = "psy_agents_noaug.cli:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### 2. Makefile Targets (Critical Workflows)

```makefile
#==============================================================================
# Key Targets for Augmentation Transformation
#==============================================================================

# DATA PREPARATION
make groundtruth                           # Generate ground truth from HF
  → poetry run python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5

# TRAINING (will need augmentation integration)
make train TASK=criteria MODEL=roberta_base
  → poetry run python -m psy_agents_noaug.cli train task=criteria model=roberta_base
  [NEEDS: augmentation params passed via CLI]

# HPO - MULTI-STAGE (Progressive refinement)
make hpo-s0 HPO_TASK=criteria              # Stage 0: Sanity (8 trials)
  → poetry run python -m psy_agents_noaug.cli hpo hpo=stage0_sanity task=criteria
  [NEEDS: augmentation search space in optuna_runner.py]

make hpo-s1 HPO_TASK=criteria              # Stage 1: Coarse (20 trials)
make hpo-s2 HPO_TASK=criteria              # Stage 2: Fine (50 trials)
make refit HPO_TASK=criteria               # Stage 3: Refit
make full-hpo HPO_TASK=criteria            # Run all stages (0→1→2→3)
make full-hpo-all                          # All architectures (criteria+evidence+share+joint)

# HPO - MAXIMAL (Single large run)
make tune-criteria-max                     # 800 trials, 6 epochs
  → python scripts/tune_max.py --agent criteria --study noaug-criteria-max --n-trials 800
  [NEEDS: augmentation search space in tune_max.py]

make tune-evidence-max                     # 1200 trials, 6 epochs
make tune-share-max                        # 600 trials, 6 epochs
make tune-joint-max                        # 600 trials, 6 epochs
make maximal-hpo-all                       # All architectures sequentially
  → python scripts/run_all_hpo.py --mode maximal

# HPO - SUPER-MAXIMAL (100 epochs + EarlyStopping)
make tune-criteria-supermax                # 5000 trials, 100 epochs, patience=20
  → HPO_EPOCHS=100 HPO_PATIENCE=20 python scripts/tune_max.py ...
make tune-all-supermax                     # All architectures (~19K trials)

# TESTING
make test                                  # Run all tests
  → poetry run pytest tests/ -v
make test-cov                              # Tests with coverage
  → poetry run pytest tests/ -v --cov=src/psy_agents_noaug --cov-report=html
make test-groundtruth                      # Field separation tests only
  → poetry run pytest tests/test_groundtruth.py -v

# SECURITY & COMPLIANCE
make audit                                 # Security vulnerability scan
  → poetry run python scripts/audit_security.py --severity medium
make sbom                                  # Generate SBOM
  → poetry run python scripts/generate_sbom.py --format json --output sbom.json
make licenses                              # Generate license report
  → poetry run python scripts/generate_licenses.py --format markdown
make compliance                            # All compliance checks (audit + sbom + licenses)

# PERFORMANCE
make bench                                 # DataLoader performance benchmarks
  → poetry run python scripts/bench_dataloader.py --num-batches 50
make verify-determinism                    # Verify reproducibility
  → poetry run python scripts/verify_determinism.py

# DEVELOPMENT
make lint                                  # Run ruff + black --check
make format                                # Format code (ruff --fix + black)
make typecheck                             # Run mypy
make pre-commit-run                        # Run all pre-commit hooks
```

### 3. Docker & Dev Container

```yaml
# .devcontainer/devcontainer.json
{
  "name": "PSY Agents NO-AUG Dev Container",
  "build": {
    "dockerfile": "Dockerfile",
    "context": "..",
    "args": {
      "VARIANT": "3.10"
    }
  },
  "runArgs": ["--gpus", "all"],           # CUDA support
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff"
      ]
    }
  }
}

# .devcontainer/Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    git curl build-essential
RUN pip3 install --upgrade pip poetry
WORKDIR /workspace
COPY pyproject.toml poetry.lock ./
RUN poetry install --with dev
```

### 4. CI/CD Pipelines

```yaml
# .github/workflows/test.yml (Excerpts)

name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install --with dev

      - name: Run linters
        run: |
          poetry run ruff check src/ tests/
          poetry run black --check src/ tests/

      - name: Run tests
        run: |
          poetry run pytest tests/ -v --cov=src/psy_agents_noaug --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run security audit
        run: |
          poetry install --with dev
          poetry run python scripts/audit_security.py --severity critical

      - name: Generate SBOM
        run: poetry run python scripts/generate_sbom.py --format json
```

---

## CRITICAL FINDINGS

### 1. Duplicate Architecture Implementations

**Issue**: Two complete, parallel implementations of all 4 architectures exist:
- `src/Project/` (376KB): Used by standalone training scripts
- `src/psy_agents_noaug/architectures/` (528KB): Extended features, unused

**Impact**:
- Code maintenance: Changes must be made in 2 places
- Testing burden: Need to test both implementations
- Confusion: Developers unsure which to use
- Disk space: ~900KB of redundant code

**Recommendation**:
- **Priority**: MEDIUM (after augmentation integration)
- **Action**: Consolidate to single implementation
- **Estimated Effort**: 2-4 hours
- **See**: `OPTIMIZATION_SUMMARY.md` for consolidation plan

### 2. Augmentation Infrastructure is DORMANT

**Issue**: Complete augmentation system exists but is disabled:
- CLI flags present (--aug-lib, --aug-methods, etc.)
- Registry with 17 methods implemented
- Pipeline class fully functional
- Comprehensive tests written
- BUT: No dataset integration, no HPO integration

**Impact**:
- Wasted infrastructure: ~2000 lines of unused code
- Misleading repository name: "NO-AUG" suggests no augmentation code exists
- Testing overhead: Tests for unused features

**Recommendation**:
- **Priority**: HIGH (primary transformation goal)
- **Action**: ACTIVATE augmentation by integrating into datasets + HPO
- **Estimated Effort**: 8-12 hours (see Transformation Roadmap)

### 3. Low Test Coverage (31%)

**Issue**: Test coverage is below production standards (target: 80%+):
- Augmentation code: 85% ✅ (well-tested)
- Data pipeline: 55% ⚠️ (medium)
- Training: 42% ⚠️ (medium)
- HPO: 25% ❌ (low)
- Project/ architectures: 15% ❌ (very low)
- Scripts: 0% ❌ (none)

**Impact**:
- Regression risk: Changes may break untested code
- Integration risk: Augmentation integration may introduce bugs
- Production readiness: Not suitable for production deployment

**Recommendation**:
- **Priority**: HIGH (concurrent with augmentation integration)
- **Action**: Write integration tests during augmentation activation
- **Estimated Effort**: 12-16 hours for comprehensive test suite

### 4. Missing HPO Augmentation Search Space

**Issue**: HPO system supports augmentation parameters but doesn't search over them:
- CLI accepts --hpo-augment-only flag (unused)
- No trial.suggest_*() calls for augmentation params
- No logging of augmentation metrics

**Impact**:
- Suboptimal performance: Augmentation hyperparams not tuned
- Manual tuning required: Defeats purpose of HPO
- Incomplete experiment tracking: Missing augmentation metadata

**Recommendation**:
- **Priority**: HIGH (part of augmentation integration)
- **Action**: Add augmentation search space to scripts/tune_max.py
- **Estimated Effort**: 4-6 hours

### 5. Span Position Tracking in Evidence Task

**Issue**: Evidence task extracts text spans, but augmentation will modify text:
- Current: (start_char, end_char) positions fixed
- With augmentation: Positions may become invalid
- No mechanism to track span position changes

**Impact**:
- Data corruption: Spans may point to wrong text after augmentation
- Training failure: Invalid spans → incorrect supervision signal
- Evaluation error: Metrics computed on wrong spans

**Recommendation**:
- **Priority**: CRITICAL (blocker for evidence augmentation)
- **Action**: Implement span position tracking or disable augmentation for spans
- **Estimated Effort**: 6-8 hours
- **Options**:
  1. Only augment text outside spans (conservative)
  2. Track character position changes (complex)
  3. Use token-level augmentation (simplest)

---

## TRANSFORMATION ROADMAP

### Phase 1: Foundation (Week 1)

**Goal**: Integrate augmentation into Criteria architecture (simplest task)

```yaml
Tasks:
  1. Modify CriteriaDataset (2-3 hours)
     - File: src/Project/Criteria/data/dataset.py
     - Add augmenter parameter to __init__()
     - Add training mode flag
     - Inject augmentation call in __getitem__()
     - Write unit tests (test_criteria_augmentation.py)

  2. Modify train_criteria.py (2-3 hours)
     - File: scripts/train_criteria.py
     - Import AugConfig, AugmenterPipeline
     - Build augmenter from config
     - Pass to CriteriaDataset
     - Add worker_init_fn to DataLoader
     - Test with --aug-lib=nlpaug

  3. Add Criteria Augmentation Tests (3-4 hours)
     - File: tests/test_augmentation_integration.py
     - Test: CriteriaDataset with augmenter
     - Test: Augmentation statistics tracking
     - Test: Worker seeding with DataLoader
     - Test: Determinism across runs

  4. Update Configs (1 hour)
     - File: configs/training/default.yaml
     - Add augmentation section with defaults
     - File: configs/criteria/train.yaml
     - Add architecture-specific augmentation config

Deliverables:
  - ✅ Criteria training with augmentation functional
  - ✅ Unit tests passing
  - ✅ Integration tests passing
  - ✅ Documentation updated

Estimated Time: 8-11 hours
```

### Phase 2: Evidence Architecture (Week 2)

**Goal**: Integrate augmentation into Evidence architecture (span tracking)

```yaml
Tasks:
  1. Solve Span Position Problem (4-6 hours)
     - Design: Choose span tracking approach
       * Option A: Token-level augmentation only (recommended)
       * Option B: Augment non-span regions only
       * Option C: Character position tracking (complex)
     - Implement: Span position adjustment logic
     - Test: Extensive unit tests for boundary cases

  2. Modify EvidenceDataset (3-4 hours)
     - File: src/Project/Evidence/data/dataset.py
     - Add augmenter parameter
     - Implement span-aware augmentation
     - Validate span positions after augmentation
     - Write unit tests (test_evidence_augmentation.py)

  3. Add Evidence Integration Tests (3-4 hours)
     - File: tests/test_evidence_spans.py
     - Test: Span positions remain valid
     - Test: Span text matches original meaning
     - Test: Boundary cases (start=0, end=len(text))
     - Test: Multiple spans per text

  4. Update Evidence Training Script (2 hours)
     - File: scripts/train_evidence.py (NEW)
     - Mirror train_criteria.py structure
     - Add evidence-specific augmentation logic

Deliverables:
  - ✅ Evidence training with augmentation functional
  - ✅ Span position tracking validated
  - ✅ Comprehensive tests passing
  - ✅ Documentation on span tracking approach

Estimated Time: 12-16 hours
```

### Phase 3: HPO Integration (Week 3)

**Goal**: Add augmentation hyperparameters to Optuna search space

```yaml
Tasks:
  1. Modify tune_max.py (4-6 hours)
     - File: scripts/tune_max.py
     - Add augmentation search space:
       * aug_lib: categorical(["none", "nlpaug", "textattack", "both"])
       * aug_methods: categorical(["all", subset options])
       * aug_p_apply: float(0.0, 0.3)
       * aug_ops_per_sample: int(1, 2)
       * aug_max_replace: float(0.1, 0.5)
     - Pass augmentation config to training function
     - Log augmentation stats to MLflow

  2. Add MLflow Augmentation Logging (2-3 hours)
     - File: scripts/tune_max.py (extend run_training_eval)
     - Log: augmentation parameters
     - Log: augmentation statistics (total, applied, skipped)
     - Log: method counts (JSON artifact)
     - Log: augmentation examples (JSON artifact)

  3. Write HPO Tests (3-4 hours)
     - File: tests/test_augmentation_hpo.py
     - Test: Augmentation params in trial.suggest_*()
     - Test: HPO with aug_lib="nlpaug", "textattack", "both"
     - Test: Optuna pruning with augmentation
     - Test: MLflow logging of augmentation metrics

  4. Run Pilot HPO Sweep (2-3 hours)
     - Run: make hpo-s0 HPO_TASK=criteria (with augmentation)
     - Validate: Augmentation params varied across trials
     - Validate: MLflow artifacts logged correctly
     - Analyze: Impact on validation metrics

Deliverables:
  - ✅ HPO searches over augmentation params
  - ✅ MLflow logs augmentation metadata
  - ✅ HPO tests passing
  - ✅ Pilot sweep results analyzed

Estimated Time: 11-16 hours
```

### Phase 4: Share & Joint Architectures (Week 4)

**Goal**: Extend augmentation to multi-task architectures

```yaml
Tasks:
  1. Share Architecture (3-4 hours)
     - File: src/Project/Share/data/dataset.py
     - Add dual-task augmentation (criteria + evidence)
     - Handle span tracking for evidence head
     - Write unit tests

  2. Joint Architecture (3-4 hours)
     - File: src/Project/Joint/data/dataset.py
     - Add dual-encoder augmentation
     - Ensure consistency across encoders
     - Write unit tests

  3. Update Training Scripts (2-3 hours)
     - Create: scripts/train_share.py
     - Create: scripts/train_joint.py
     - Mirror criteria/evidence script structure

  4. Integration Testing (3-4 hours)
     - File: tests/test_multi_task_augmentation.py
     - Test: Share architecture with augmentation
     - Test: Joint architecture with augmentation
     - Test: Cross-architecture consistency

  5. Run HPO for All Architectures (2-3 hours)
     - Run: make full-hpo-all (with augmentation)
     - Validate: All 4 architectures functional
     - Compare: Performance with vs without augmentation

Deliverables:
  - ✅ All 4 architectures support augmentation
  - ✅ Training scripts for all architectures
  - ✅ Integration tests passing
  - ✅ HPO sweep results for all architectures

Estimated Time: 13-18 hours
```

### Phase 5: Documentation & Production (Week 5)

**Goal**: Production-ready augmentation system

```yaml
Tasks:
  1. Comprehensive Documentation (4-6 hours)
     - Update: CLAUDE.md (augmentation section)
     - Create: AUGMENTATION_GUIDE.md
       * Architecture overview
       * Integration guide
       * HPO guide
       * Troubleshooting
     - Update: README.md (remove "NO-AUG" emphasis)
     - Create: TRANSFORMATION_SUMMARY.md

  2. Test Coverage Push (4-6 hours)
     - Target: 80%+ coverage for augmentation modules
     - Add: Missing edge case tests
     - Add: Performance regression tests
     - Add: Multi-GPU worker seeding tests

  3. Performance Benchmarking (2-3 hours)
     - File: scripts/bench_augmentation.py (NEW)
     - Benchmark: Augmentation overhead per method
     - Benchmark: DataLoader throughput with augmentation
     - Benchmark: Memory usage
     - Document: Performance characteristics

  4. Security & Compliance (2-3 hours)
     - Run: make audit (ensure no new vulnerabilities)
     - Run: make sbom (update SBOM with augmentation deps)
     - Run: make licenses (verify augmentation lib licenses)
     - Update: THIRD_PARTY_LICENSES.md

  5. Repository Rename (1-2 hours)
     - Rename: "NoAug_Criteria_Evidence" → "Criteria_Evidence_Augmentation"
     - Update: All references in docs, configs, scripts
     - Update: pyproject.toml (project name)
     - Update: Docker image names

Deliverables:
  - ✅ Comprehensive documentation
  - ✅ 80%+ test coverage
  - ✅ Performance benchmarks documented
  - ✅ Security & compliance validated
  - ✅ Repository renamed

Estimated Time: 13-20 hours
```

### Total Transformation Effort

```
Phase 1: Foundation                  8-11 hours
Phase 2: Evidence                   12-16 hours
Phase 3: HPO Integration            11-16 hours
Phase 4: Share & Joint              13-18 hours
Phase 5: Documentation              13-20 hours
─────────────────────────────────────────────
TOTAL:                              57-81 hours

Estimated Calendar Time:
- Full-time (8 hours/day):    7-10 days
- Part-time (4 hours/day):   14-20 days
- Realistic with testing:     3-4 weeks
```

### Success Metrics

```yaml
Technical Metrics:
  - ✅ All 4 architectures support augmentation
  - ✅ 80%+ test coverage for augmentation modules
  - ✅ HPO searches over augmentation hyperparameters
  - ✅ MLflow logs augmentation metadata
  - ✅ Span position tracking validated for Evidence task
  - ✅ Determinism verified (reproducible results)
  - ✅ No performance degradation (>10% slowdown acceptable)

Quality Metrics:
  - ✅ All tests passing (pytest)
  - ✅ No linting errors (ruff + black)
  - ✅ No type errors (mypy)
  - ✅ No security vulnerabilities (pip-audit)
  - ✅ Documentation comprehensive and accurate

Performance Metrics:
  - ✅ Augmentation overhead: <20% training time increase
  - ✅ Memory overhead: <15% increase
  - ✅ DataLoader throughput: >80% of baseline

Science Metrics:
  - ✅ Augmentation improves validation metrics (F1, accuracy)
  - ✅ Optimal augmentation params found via HPO
  - ✅ Generalization gap reduced (train-test gap)
```

---

## END OF INVENTORY

**Document Version**: 1.0
**Generated**: 2025-10-25
**Purpose**: Production readiness audit foundation
**Status**: ✅ COMPLETE

**Next Steps**:
1. Review this inventory with team
2. Prioritize transformation phases
3. Begin Phase 1: Criteria augmentation integration
4. Track progress against success metrics

**For Questions/Updates**: See `CLAUDE.md` for development guidelines
