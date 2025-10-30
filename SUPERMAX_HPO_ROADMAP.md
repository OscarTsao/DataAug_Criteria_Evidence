# SUPERMAX HPO Implementation Roadmap

**Created**: 2025-10-30
**Status**: Planning → Implementation
**Goal**: Transform baseline HPO system into enterprise-grade, production-ready hyperparameter optimization infrastructure

---

## Executive Summary

This document provides a comprehensive, phased implementation plan for expanding the PSY Agents HPO system from its current baseline (78 trials/agent, 3 architectures, 4 optimizers) to a supermax configuration (900-2,400 trials/agent, 8+ architectures, 5+ advanced optimizers, multi-objective optimization).

**Key Insight**: ~70% of requested infrastructure ALREADY EXISTS. The expansion is primarily about:
1. Adding 5 new model architectures (electra, albert, distilbert, convbert, xlnet)
2. Integrating 1-2 new optimizers (LAMB, AdamW-8bit)
3. Wiring up existing augmentation per-method tuning
4. Enhancing test coverage from 31% → 90%
5. Documentation and validation

**Estimated Total Effort**: 40-60 hours (1-2 weeks full-time)

---

## Current State Audit (2025-10-30)

### ✅ Infrastructure Already Present

#### 1. Augmentation System (100% Complete)
- **17-method allowlist** ✅ PRESENT in `src/psy_agents_noaug/augmentation/registry.py:26-47`
  - nlpaug: 10 methods (char: 3, word: 7)
  - TextAttack: 7 methods
- **TF-IDF caching** ✅ IMPLEMENTED in `src/psy_agents_noaug/augmentation/tfidf_cache.py`
  - `load_or_fit_tfidf()` function with joblib persistence
  - Generates `tfidfaug_w2idf.txt` and `tfidfaug_w2tfidf.txt` for nlpaug
- **Reserved token support** ✅ IMPLEMENTED in `registry.py:127-132`
- **Antonym guard logic** ✅ IMPLEMENTED (AntonymAug in allowlist)
- **Pipeline integration** ✅ PRESENT in `src/psy_agents_noaug/augmentation/pipeline.py`

**Status**: Augmentation infrastructure is production-ready. Only needs HPO integration.

#### 2. HPO Core Infrastructure (90% Complete)
- **Multi-objective optimization** ✅ IMPLEMENTED
  - NSGA-II sampler in `src/psy_agents_noaug/hpo/samplers.py:18-19`
  - Dual-objective returns in `objectives.py:127-129` (F1-macro, ECE)
- **Advanced search space** ✅ IMPLEMENTED in `src/psy_agents_noaug/hpo/spaces.py`
  - 4 optimizers: adamw, adam, adafactor, lion (line 113)
  - 5 schedulers: linear, cosine, cosine_restart, polynomial, one_cycle (line 127)
  - Architecture-aware pooling: cls, mean, max, attention (line 82)
  - Null strategy: none, threshold, ratio, calibrated (line 160)
- **MLflow integration** ✅ PRODUCTION-READY
  - Automatic experiment tracking
  - Artifact logging
  - Comprehensive callbacks in `src/psy_agents_noaug/hpo/callbacks.py`
- **Performance tracking** ✅ IMPLEMENTED
  - `TrialTimer` in `src/psy_agents_noaug/hpo/utils.py`
  - Runtime metrics logged per trial
  - Top-K trial storage with `TopKStore`

**Status**: HPO core is production-ready. Only needs optimizer expansion and Stage-B/C implementation.

#### 3. Model Infrastructure (60% Complete)
- **Current architectures** ✅ 3/8 present:
  - bert-base-uncased (`configs/model/bert_base.yaml`)
  - roberta-base (`configs/model/roberta_base.yaml`)
  - microsoft/deberta-v3-base (`configs/model/deberta_v3_base.yaml`)
- **Missing architectures** ⏳ 5 to add:
  - google/electra-base-discriminator
  - albert-base-v2
  - distilbert-base-uncased
  - YituTech/conv-bert-base
  - xlnet-base-cased

**Status**: 60% complete. Need 5 new YAML configs (15-30 min work).

#### 4. Testing Infrastructure (Surprisingly Good)
- **Current test suite**:
  - 27 test files (6,897 lines)
  - 452 test functions
  - Coverage: 31% (67/69 tests passing)
- **Strong areas**:
  - HPO smoke tests (`test_hpo_max_smoke.py`, `test_hpo_stage_smoke.py`)
  - Augmentation tests (`test_augmentation_registry.py`, `test_tfidf_cache.py`)
  - Integration tests (`test_pipeline_comprehensive.py`)
  - Performance contracts (`test_perf_contract.py`, `test_performance_regression.py`)
- **Gaps**:
  - Stage-B/C HPO tests (will add)
  - Multi-objective optimization validation (will add)
  - Per-method augmentation tests (will add)
  - Edge case coverage

**Status**: 70% complete. Need targeted additions for new features.

---

## Gap Analysis: What's Missing vs. Requested

| Feature | Requested | Current | Gap | Effort |
|---------|-----------|---------|-----|--------|
| **Augmentation Methods** | 17 | 17 ✅ | None | 0h |
| **TF-IDF Caching** | Yes | Yes ✅ | None | 0h |
| **Per-Method Tuning** | Yes | Infra exists ⚠️ | Wire to HPO | 2-3h |
| **Model Architectures** | 8 | 3 | 5 configs | 1-2h |
| **Advanced Optimizers** | 5 | 4 | 1-2 (LAMB, 8bit) | 2-4h |
| **NSGA-II Sampler** | Yes | Yes ✅ | None | 0h |
| **Multi-Objective** | Yes | Yes ✅ | None | 0h |
| **Stage-A (900-1200)** | Yes | No | Implement | 4-6h |
| **Stage-B (1200-2400)** | Yes | No | Implement | 6-8h |
| **Stage-C (300-600)** | Yes | No | Implement | 3-4h |
| **K-Fold CV** | Yes | No | Implement | 4-6h |
| **EMA Tracking** | Yes | Partial | Enhance | 2-3h |
| **90% Coverage** | Yes | 31% | Add tests | 8-12h |
| **mypy --strict** | Yes | Standard | Fix errors | 3-5h |
| **Documentation** | Comprehensive | Good | Expand | 4-6h |

**Total Gap**: ~40-60 hours of focused work

---

## Implementation Strategy

### Principle: Incremental Value Delivery

Each phase delivers immediate, testable value while building toward the final system.

### Phases

1. **Phase 0: Foundation** (2-3 hours) - IMMEDIATE VALUE
2. **Phase 1: Augmentation Integration** (4-6 hours) - HIGH IMPACT
3. **Phase 2: Architecture & Optimizer Expansion** (6-8 hours) - SCALING
4. **Phase 3: Advanced HPO Stages** (12-16 hours) - SOPHISTICATION
5. **Phase 4: Quality Infrastructure** (12-18 hours) - ROBUSTNESS
6. **Phase 5: Production Validation** (4-6 hours) - CONFIDENCE

Total: **40-60 hours** (1-2 weeks full-time, 2-4 weeks part-time)

---

## Phase 0: Foundation (2-3 hours) 🏗️

**Goal**: Set up tracking, documentation, and baseline validation

### Tasks

1. **Create tracking infrastructure** (30 min)
   - This roadmap document ✅ DONE
   - Implementation progress tracker
   - Issue/blocker log

2. **Baseline performance benchmark** (1 hour)
   - Run current HPO on all 4 agents (criteria, evidence, share, joint)
   - Document baseline metrics (F1, ECE, runtime)
   - Establish performance regression thresholds

3. **Environment validation** (30 min)
   - Verify all dependencies installed
   - Test CUDA availability
   - Validate Optuna + MLflow integration

4. **Code audit report** (1 hour)
   - Document current architecture choices
   - Identify technical debt to address
   - Flag any breaking changes needed

**Deliverables**:
- ✅ SUPERMAX_HPO_ROADMAP.md (this document)
- 📊 BASELINE_PERFORMANCE_REPORT.md
- 🔍 CODE_AUDIT_PHASE0.md
- ✅ Environment validated

**Success Criteria**: Clear baseline, validated environment, documented starting point

---

## Phase 1: Augmentation Integration (4-6 hours) 🎯

**Goal**: Wire existing augmentation infrastructure into HPO search space

**Impact**: HIGHEST - Enables on-the-fly augmentation during HPO with minimal risk

### Tasks

#### 1.1: Add Augmentation Parameters to Search Space (2 hours)

**File**: `src/psy_agents_noaug/hpo/spaces.py`

```python
# Add to SearchSpace.sample() method (after line 176)

# Augmentation strategy
params["aug.enabled"] = trial.suggest_categorical(
    "aug.enabled",
    constraints.merged_choices("aug.enabled", [False, True])
)

if params["aug.enabled"]:
    # Probability of applying augmentation to a sample
    prob_low, prob_high = constraints.merged_float("aug.p_apply", 0.05, 0.30)
    params["aug.p_apply"] = trial.suggest_float(
        "aug.p_apply", prob_low, prob_high, step=0.05
    )

    # Number of augmentation operations per sample
    params["aug.ops_per_sample"] = trial.suggest_int(
        "aug.ops_per_sample",
        *constraints.merged_int("aug.ops_per_sample", 1, 3)
    )

    # Maximum fraction of tokens to replace
    replace_low, replace_high = constraints.merged_float("aug.max_replace", 0.1, 0.4)
    params["aug.max_replace"] = trial.suggest_float(
        "aug.max_replace", replace_low, replace_high, step=0.05
    )

    # Antonym guard strategy
    params["aug.antonym_guard"] = trial.suggest_categorical(
        "aug.antonym_guard",
        constraints.merged_choices("aug.antonym_guard", ["off", "on_low_weight"])
    )

    # Method selection strategy
    params["aug.method_strategy"] = trial.suggest_categorical(
        "aug.method_strategy",
        constraints.merged_choices("aug.method_strategy", ["all", "nlpaug", "textattack", "custom"])
    )
else:
    # No augmentation - set defaults
    params["aug.p_apply"] = 0.0
    params["aug.ops_per_sample"] = 0
    params["aug.max_replace"] = 0.0
    params["aug.antonym_guard"] = "off"
    params["aug.method_strategy"] = "none"

return params
```

**Testing**:
```python
# tests/test_hpo_aug_space.py
def test_augmentation_params_in_search_space():
    space = SearchSpace("criteria")
    study = optuna.create_study()
    trial = study.ask()
    params = space.sample(trial)
    assert "aug.enabled" in params
    if params["aug.enabled"]:
        assert "aug.p_apply" in params
        assert 0.05 <= params["aug.p_apply"] <= 0.30
```

#### 1.2: TF-IDF Pre-fitting Pipeline (1.5 hours)

**File**: `scripts/prepare_tfidf_cache.py` (NEW)

```python
"""Pre-fit TF-IDF resources for augmentation HPO."""

from pathlib import Path
from psy_agents_noaug.augmentation.tfidf_cache import load_or_fit_tfidf
from psy_agents_noaug.data.loaders import load_groundtruth

def main():
    # Load all training texts
    gt = load_groundtruth("criteria")  # or evidence
    texts = [row["post_text"] for row in gt["data"]]

    # Fit TF-IDF
    cache_dir = Path("data/augmentation_cache/tfidf")
    resource = load_or_fit_tfidf(
        train_texts=texts,
        model_path=cache_dir,
        max_features=40000,
        ngram_range=(1, 2),
    )

    print(f"TF-IDF fitted in {resource.build_time_sec:.2f}s")
    print(f"Cached to: {resource.path}")

if __name__ == "__main__":
    main()
```

**Makefile target**:
```makefile
prepare-tfidf:
	@echo "$(BLUE)Pre-fitting TF-IDF cache for augmentation$(NC)"
	poetry run python scripts/prepare_tfidf_cache.py
```

#### 1.3: Integration with Evaluation Pipeline (1 hour)

**File**: `src/psy_agents_noaug/hpo/evaluation.py`

Add augmentation config to `run_experiment()` function:
```python
def run_experiment(agent, params, epochs, seeds, patience, max_samples):
    # Extract augmentation params
    aug_config = None
    if params.get("aug.enabled", False):
        aug_config = {
            "enabled": True,
            "p_apply": params["aug.p_apply"],
            "ops_per_sample": params["aug.ops_per_sample"],
            "max_replace": params["aug.max_replace"],
            "antonym_guard": params["aug.antonym_guard"],
            "method_strategy": params["aug.method_strategy"],
            "tfidf_model": "data/augmentation_cache/tfidf",
        }

    # Pass to dataset/training loop
    # ... existing code ...
```

#### 1.4: Testing Suite (30 min)

**Files**:
- `tests/test_hpo_aug_space.py` - Search space validation
- `tests/test_tfidf_hpo_integration.py` - TF-IDF caching integration
- `tests/test_aug_hpo_smoke.py` - Smoke test with augmentation enabled

**Deliverables**:
- ✅ Augmentation parameters in HPO search space
- ✅ TF-IDF pre-fitting pipeline and Makefile target
- ✅ Integration with evaluation pipeline
- ✅ 3 new test files validating augmentation HPO
- 📊 Smoke test results showing augmentation working

**Success Criteria**: HPO run with `aug.enabled=True` completes successfully and shows augmentation metrics in MLflow

**Risk**: LOW - Infrastructure exists, just wiring together

---

## Phase 2: Architecture & Optimizer Expansion (6-8 hours) 📈

**Goal**: Expand model and optimizer options to support large-scale exploration

### Tasks

#### 2.1: Add 5 New Model Architectures (2 hours)

Create YAML configs in `configs/model/`:

**electra_base.yaml**:
```yaml
name: "google/electra-base-discriminator"
hidden_size: 768
num_attention_heads: 12
num_hidden_layers: 12
max_position_embeddings: 512
gradient_checkpointing: false
```

**albert_base.yaml**:
```yaml
name: "albert-base-v2"
hidden_size: 768
num_attention_heads: 12
num_hidden_layers: 12
max_position_embeddings: 512
gradient_checkpointing: false
```

**distilbert_base.yaml**:
```yaml
name: "distilbert-base-uncased"
hidden_size: 768
num_attention_heads: 12
num_hidden_layers: 6  # Distilled - fewer layers
max_position_embeddings: 512
gradient_checkpointing: false
```

**convbert_base.yaml**:
```yaml
name: "YituTech/conv-bert-base"
hidden_size: 768
num_attention_heads: 12
num_hidden_layers: 12
max_position_embeddings: 512
gradient_checkpointing: false
```

**xlnet_base.yaml**:
```yaml
name: "xlnet-base-cased"
hidden_size: 768
num_attention_heads: 12
num_hidden_layers: 12
max_position_embeddings: 512
gradient_checkpointing: false
```

**Update search space** in `src/psy_agents_noaug/hpo/spaces.py:50-54`:
```python
self.backbones = [
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base",
    "google/electra-base-discriminator",
    "albert-base-v2",
    "distilbert-base-uncased",
    "YituTech/conv-bert-base",
    "xlnet-base-cased",
]
```

**Testing**:
```python
# tests/test_model_architectures.py
@pytest.mark.parametrize("arch", [
    "google/electra-base-discriminator",
    "albert-base-v2",
    "distilbert-base-uncased",
    "YituTech/conv-bert-base",
    "xlnet-base-cased",
])
def test_new_architecture_loads(arch):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(arch)
    assert model.config.hidden_size == 768
```

#### 2.2: Add Advanced Optimizers (2-3 hours)

**Option A: LAMB Optimizer** (PyTorch built-in since 2.0)
```python
# src/psy_agents_noaug/training/optimizers.py (NEW)

from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer(name, model_params, lr, weight_decay, **kwargs):
    if name == "adamw":
        return AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif name == "lamb":
        # LAMB = Layer-wise Adaptive Moments optimizer for Batch training
        # PyTorch 2.0+ has experimental LAMB support
        try:
            from torch.optim import LAMB  # Requires torch >= 2.1
            return LAMB(model_params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            # Fallback: use AdamW with LAMB-like settings
            return AdamW(
                model_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-6,
            )
    elif name == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=False,
            relative_step=False,
        )
    elif name == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(model_params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("lion-pytorch not installed: pip install lion-pytorch")
    else:
        raise ValueError(f"Unknown optimizer: {name}")
```

**Option B: 8-bit AdamW** (via bitsandbytes)
```python
    elif name == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model_params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("bitsandbytes not installed: pip install bitsandbytes")
```

**Update search space**:
```python
# src/psy_agents_noaug/hpo/spaces.py:111-114
params["optim.name"] = trial.suggest_categorical(
    "optim.name",
    constraints.merged_choices("optim.name", [
        "adamw", "adam", "adafactor", "lion", "lamb", "adamw_8bit"
    ]),
)
```

**Dependencies** (add to `pyproject.toml`):
```toml
[tool.poetry.dependencies]
lion-pytorch = { version = "^0.1.2", optional = true }
bitsandbytes = { version = "^0.41.0", optional = true }

[tool.poetry.extras]
optimizers = ["lion-pytorch", "bitsandbytes"]
```

**Testing**:
```python
# tests/test_optimizers.py
@pytest.mark.parametrize("optim_name", ["adamw", "adam", "adafactor", "lion", "lamb"])
def test_optimizer_creation(optim_name):
    model = torch.nn.Linear(10, 2)
    opt = get_optimizer(optim_name, model.parameters(), lr=1e-3, weight_decay=0.01)
    assert opt is not None
```

#### 2.3: Validation and Smoke Tests (1-2 hours)

Run HPO with expanded architectures/optimizers:
```bash
# Quick validation (2 trials each)
make hpo-validate-architectures
make hpo-validate-optimizers
```

**Deliverables**:
- ✅ 5 new model YAML configs
- ✅ 2 new optimizers (LAMB, AdamW-8bit)
- ✅ Updated search space
- ✅ Optimizer factory function
- ✅ Test suite for all combinations
- 📊 Validation report

**Success Criteria**: HPO can sample and train with all 8 architectures and 6 optimizers

**Risk**: MEDIUM - New dependencies may have compatibility issues

---

## Phase 3: Advanced HPO Stages (12-16 hours) 🚀

**Goal**: Implement Stage-A/B/C multi-stage progressive refinement

### Tasks

#### 3.1: Stage-A Implementation (4-6 hours)

**Goal**: Baseline exploration with 900-1,200 trials

**Strategy**:
- Large search space
- Standard single-objective (F1-macro)
- Early stopping with ASHA pruner
- Export Pareto-optimal candidates for Stage-B

**File**: `scripts/run_stage_a.py` (NEW)

```python
"""Stage-A: Baseline exploration across full search space."""

import argparse
from pathlib import Path
import optuna
from psy_agents_noaug.hpo import (
    SearchSpace,
    SpaceConstraints,
    ObjectiveBuilder,
    ObjectiveSettings,
    create_sampler,
    create_pruner,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, choices=["criteria", "evidence", "share", "joint"])
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--storage", default="sqlite:///./_optuna/supermax.db")
    args = parser.parse_args()

    study_name = f"{args.agent}-stage-a-baseline"

    # Create study with TPE sampler
    sampler = create_sampler(multi_objective=False, seed=2025, sampler="tpe")
    pruner = create_pruner("asha", min_resource=2, max_resource=20, reduction_factor=3)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Build objective
    space = SearchSpace(args.agent)
    settings = ObjectiveSettings(
        agent=args.agent,
        study=study_name,
        outdir=Path("outputs/stage_a"),
        epochs=20,  # Moderate epochs for exploration
        seeds=[1, 2, 3],  # Multiple seeds for robustness
        patience=5,
        max_samples=None,
        multi_objective=False,
        topk=50,  # Keep top 50 for Stage-B
        mlflow_uri="sqlite:///mlflow.db",
        mlflow_experiment=f"stage-a-{args.agent}",
    )

    objective = ObjectiveBuilder(
        space=space,
        settings=settings,
        constraints=SpaceConstraints(),  # No constraints - full space
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    # Export top-K for Stage-B
    trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
    top_configs = [t.params for t in sorted_trials[:50]]

    export_path = Path("outputs/stage_a") / f"{args.agent}_top50_configs.json"
    import json
    export_path.write_text(json.dumps(top_configs, indent=2))

    print(f"Stage-A complete. Top 50 configs exported to {export_path}")

if __name__ == "__main__":
    main()
```

**Makefile target**:
```makefile
stage-a:
	@echo "$(BLUE)Running Stage-A: Baseline Exploration (1000 trials)$(NC)"
	poetry run python scripts/run_stage_a.py \
		--agent $(HPO_TASK) \
		--n-trials 1000 \
		--storage sqlite:///./_optuna/supermax.db
```

#### 3.2: Stage-B Implementation (6-8 hours)

**Goal**: Multi-objective refinement with NSGA-II on top candidates

**Strategy**:
- Seed from Stage-A top-50 configurations
- Multi-objective: maximize F1-macro, minimize ECE
- NSGA-II sampler for Pareto front
- 1,200-2,400 trials

**File**: `scripts/run_stage_b.py` (NEW)

```python
"""Stage-B: Multi-objective refinement with NSGA-II."""

import argparse
import json
from pathlib import Path
import optuna
from psy_agents_noaug.hpo import (
    SearchSpace,
    SpaceConstraints,
    ObjectiveBuilder,
    ObjectiveSettings,
    create_sampler,
    create_pruner,
)

def load_stage_a_configs(agent):
    """Load top-K configurations from Stage-A."""
    config_path = Path("outputs/stage_a") / f"{agent}_top50_configs.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Stage-A configs not found: {config_path}")
    return json.loads(config_path.read_text())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--n-trials", type=int, default=1500)
    parser.add_argument("--storage", default="sqlite:///./_optuna/supermax.db")
    args = parser.parse_args()

    study_name = f"{args.agent}-stage-b-multiobjective"

    # Multi-objective with NSGA-II
    sampler = create_sampler(multi_objective=True, seed=2025, sampler="nsga2")
    pruner = create_pruner("asha", min_resource=3, max_resource=30, reduction_factor=3)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        directions=["maximize", "minimize"],  # F1-macro (max), ECE (min)
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Load Stage-A configs to seed search
    stage_a_configs = load_stage_a_configs(args.agent)
    print(f"Loaded {len(stage_a_configs)} Stage-A configurations for seeding")

    # Build objective
    space = SearchSpace(args.agent)

    # Narrow constraints based on Stage-A results
    # (Could analyze Stage-A to find promising ranges)
    constraints = SpaceConstraints()

    settings = ObjectiveSettings(
        agent=args.agent,
        study=study_name,
        outdir=Path("outputs/stage_b"),
        epochs=30,  # More epochs for refinement
        seeds=[1, 2, 3, 4],  # More seeds for stability
        patience=8,
        max_samples=None,
        multi_objective=True,
        topk=20,  # Keep top 20 Pareto-optimal
        mlflow_uri="sqlite:///mlflow.db",
        mlflow_experiment=f"stage-b-{args.agent}",
    )

    objective = ObjectiveBuilder(
        space=space,
        settings=settings,
        constraints=constraints,
    )

    # Enqueue Stage-A top configs as initial trials
    for config in stage_a_configs[:20]:  # Seed with top 20
        study.enqueue_trial(config)

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        catch=(RuntimeError,),
    )

    # Export Pareto front for Stage-C
    pareto_trials = study.best_trials  # NSGA-II Pareto front
    pareto_configs = [t.params for t in pareto_trials]

    export_path = Path("outputs/stage_b") / f"{args.agent}_pareto_configs.json"
    export_path.write_text(json.dumps(pareto_configs, indent=2))

    print(f"Stage-B complete. Pareto front ({len(pareto_configs)} configs) exported to {export_path}")

if __name__ == "__main__":
    main()
```

**Makefile target**:
```makefile
stage-b:
	@echo "$(BLUE)Running Stage-B: Multi-Objective NSGA-II (1500 trials)$(NC)"
	poetry run python scripts/run_stage_b.py \
		--agent $(HPO_TASK) \
		--n-trials 1500 \
		--storage sqlite:///./_optuna/supermax.db
```

#### 3.3: Stage-C Implementation (3-4 hours)

**Goal**: Joint refinement with K-fold cross-validation

**Strategy**:
- Select 5-10 candidates from Stage-B Pareto front
- Full K-fold CV (k=5) for robust evaluation
- Final model selection
- 300-600 trials (per-fold evaluations)

**File**: `scripts/run_stage_c.py` (NEW)

```python
"""Stage-C: Joint refinement with K-fold cross-validation."""

import argparse
import json
from pathlib import Path
from sklearn.model_selection import KFold
import optuna

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--k-folds", type=int, default=5)
    args = parser.parse_args()

    # Load Pareto front from Stage-B
    pareto_path = Path("outputs/stage_b") / f"{args.agent}_pareto_configs.json"
    pareto_configs = json.loads(pareto_path.read_text())[:args.n_candidates]

    print(f"Stage-C: Evaluating {len(pareto_configs)} candidates with {args.k_folds}-fold CV")

    results = []
    for idx, config in enumerate(pareto_configs):
        print(f"Candidate {idx+1}/{len(pareto_configs)}: {config['model.name']}")

        # Run K-fold CV
        fold_metrics = []
        kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(1000))):  # Placeholder
            print(f"  Fold {fold_idx+1}/{args.k_folds}")

            # TODO: Integrate with actual training loop
            # metrics = run_experiment_with_fold(config, train_idx, val_idx)
            # fold_metrics.append(metrics)

        # Average metrics across folds
        avg_f1 = sum(m["f1_macro"] for m in fold_metrics) / len(fold_metrics)
        avg_ece = sum(m["ece"] for m in fold_metrics) / len(fold_metrics)

        results.append({
            "config": config,
            "cv_f1_macro": avg_f1,
            "cv_ece": avg_ece,
            "fold_metrics": fold_metrics,
        })

    # Export final results
    export_path = Path("outputs/stage_c") / f"{args.agent}_final_results.json"
    export_path.write_text(json.dumps(results, indent=2))

    best = max(results, key=lambda r: r["cv_f1_macro"])
    print(f"Best config: F1={best['cv_f1_macro']:.4f}, ECE={best['cv_ece']:.4f}")

if __name__ == "__main__":
    main()
```

**Makefile targets**:
```makefile
stage-c:
	@echo "$(BLUE)Running Stage-C: K-Fold CV Refinement$(NC)"
	poetry run python scripts/run_stage_c.py \
		--agent $(HPO_TASK) \
		--n-candidates 5 \
		--k-folds 5

supermax-all:
	@echo "$(BLUE)Running SUPERMAX HPO: Stage A → B → C$(NC)"
	$(MAKE) stage-a HPO_TASK=criteria
	$(MAKE) stage-b HPO_TASK=criteria
	$(MAKE) stage-c HPO_TASK=criteria
	$(MAKE) stage-a HPO_TASK=evidence
	$(MAKE) stage-b HPO_TASK=evidence
	$(MAKE) stage-c HPO_TASK=evidence
	$(MAKE) stage-a HPO_TASK=share
	$(MAKE) stage-b HPO_TASK=share
	$(MAKE) stage-c HPO_TASK=share
	$(MAKE) stage-a HPO_TASK=joint
	$(MAKE) stage-b HPO_TASK=joint
	$(MAKE) stage-c HPO_TASK=joint
```

**Deliverables**:
- ✅ Stage-A script (baseline exploration)
- ✅ Stage-B script (multi-objective NSGA-II)
- ✅ Stage-C script (K-fold CV refinement)
- ✅ Makefile targets for each stage
- ✅ Config export/import pipeline
- 📊 End-to-end validation results

**Success Criteria**: Complete Stage A→B→C pipeline for one agent (criteria)

**Risk**: HIGH - Complex multi-stage coordination, long runtime

---

## Phase 4: Quality Infrastructure (12-18 hours) 🧪

**Goal**: Achieve 90% code coverage and mypy --strict compliance

### Tasks

#### 4.1: Add 8 New Test Files (8-10 hours)

**1. tests/test_stage_a_smoke.py** (1 hour)
```python
"""Smoke tests for Stage-A baseline exploration."""
import pytest
import optuna
from scripts.run_stage_a import main as run_stage_a

def test_stage_a_runs_single_trial(tmp_path):
    """Verify Stage-A completes 1 trial."""
    # Configure with tmp storage
    storage = f"sqlite:///{tmp_path}/test.db"
    # Run with --n-trials 1
    # Assert study completes
```

**2. tests/test_stage_b_multiobjective.py** (1.5 hours)
```python
"""Tests for Stage-B multi-objective optimization."""

def test_nsga2_sampler_creates_pareto_front():
    """Verify NSGA-II produces Pareto-optimal solutions."""
    pass

def test_stage_b_seeds_from_stage_a():
    """Verify Stage-B loads and enqueues Stage-A configs."""
    pass
```

**3. tests/test_stage_c_kfold.py** (1.5 hours)
```python
"""Tests for Stage-C K-fold cross-validation."""

def test_kfold_splits_data_correctly():
    pass

def test_stage_c_averages_fold_metrics():
    pass
```

**4. tests/test_augmentation_hpo_integration.py** (2 hours)
```python
"""Integration tests for augmentation during HPO."""

def test_aug_params_affect_metrics():
    """Run 2 trials: aug_enabled=True vs False, verify metrics differ."""
    pass

def test_tfidf_cache_loads_during_hpo():
    """Verify TF-IDF cache is used when aug.enabled=True."""
    pass

def test_per_method_augmentation():
    """Verify individual augmentation methods work."""
    pass
```

**5. tests/test_optimizer_advanced.py** (1 hour)
```python
"""Tests for LAMB and 8-bit optimizers."""

@pytest.mark.skipif(not has_bitsandbytes, reason="bitsandbytes not installed")
def test_adamw_8bit_optimizer():
    pass

def test_lamb_optimizer_convergence():
    pass
```

**6. tests/test_model_architectures_extended.py** (1 hour)
```python
"""Tests for 5 new model architectures."""

@pytest.mark.parametrize("arch", NEW_ARCHITECTURES)
def test_architecture_forward_pass(arch):
    """Verify each new architecture performs forward pass."""
    pass
```

**7. tests/test_performance_regression_supermax.py** (1.5 hours)
```python
"""Performance regression tests for SUPERMAX HPO."""

def test_stage_a_trial_runtime_under_threshold():
    """Verify Stage-A trials complete within expected time."""
    pass

def test_memory_usage_within_limits():
    """Verify GPU memory stays below 20GB per trial."""
    pass
```

**8. tests/test_mlflow_supermax_tracking.py** (1.5 hours)
```python
"""MLflow tracking validation for multi-stage HPO."""

def test_stage_a_logs_to_mlflow():
    pass

def test_pareto_front_logged_to_artifacts():
    pass
```

#### 4.2: Increase Coverage to 90% (4-6 hours)

Current coverage: 31% (67/69 tests passing)

**Strategy**:
1. Run coverage report: `poetry run pytest --cov=src/psy_agents_noaug --cov-report=html`
2. Identify uncovered lines (target: augmentation, HPO, evaluation modules)
3. Add targeted tests for edge cases
4. Focus on critical paths (model loading, optimizer creation, augmentation pipeline)

**Target files for coverage**:
- `src/psy_agents_noaug/hpo/objectives.py` (currently likely ~50%)
- `src/psy_agents_noaug/hpo/evaluation.py` (likely ~40%)
- `src/psy_agents_noaug/augmentation/pipeline.py` (likely ~30%)
- `src/psy_agents_noaug/augmentation/registry.py` (likely ~50%)

#### 4.3: mypy --strict Compliance (3-4 hours)

Current status: mypy passing but not with --strict

**Tasks**:
1. Enable `strict = true` in `pyproject.toml`
2. Fix type errors incrementally by file
3. Add missing type annotations
4. Use `typing.cast()` where necessary
5. Add `# type: ignore[error-code]` comments with justification for unavoidable cases

**Priority files**:
- `src/psy_agents_noaug/hpo/*.py` (10 files)
- `src/psy_agents_noaug/augmentation/*.py` (5 files)
- `scripts/run_stage_*.py` (3 new files)

**Deliverables**:
- ✅ 8 new test files
- ✅ 90%+ code coverage
- ✅ mypy --strict compliance
- 📊 Coverage report
- 📊 Type checking report

**Success Criteria**: All tests pass, coverage ≥90%, mypy --strict clean

**Risk**: MEDIUM - Coverage increase may reveal bugs

---

## Phase 5: Production Validation (4-6 hours) ✅

**Goal**: End-to-end validation and performance benchmarking

### Tasks

#### 5.1: Full Pipeline Run (2 hours per agent)

Run complete SUPERMAX pipeline for one agent:
```bash
time make supermax-all HPO_TASK=criteria
```

**Monitoring**:
- GPU utilization (should be near 100%)
- Memory usage (should not exceed 20GB)
- Trial throughput (trials/hour)
- MLflow logging (all metrics present)

**Expected timeline**:
- Stage-A: ~24-48 hours (1000 trials × 20 epochs × 3 seeds)
- Stage-B: ~36-72 hours (1500 trials × 30 epochs × 4 seeds)
- Stage-C: ~6-12 hours (5 candidates × 5 folds × 40 epochs)
- **Total per agent**: ~3-5 days on single GPU

#### 5.2: Performance Validation (1 hour)

**Metrics to validate**:
- F1-macro improvement over baseline (expect +2-5%)
- ECE calibration (expect <0.10)
- Runtime per trial (expect <10 min/trial on average)
- Memory efficiency (expect <20GB GPU RAM)

**Comparison**:
| Metric | Baseline (78 trials) | SUPERMAX (3000+ trials) | Delta |
|--------|---------------------|------------------------|-------|
| F1-macro | 0.XX | 0.XX | +X.X% |
| ECE | 0.XX | 0.XX | -X.X% |
| Best trial time | XXm | XXm | ±X% |
| GPU memory | XGB | XGB | ±X% |

#### 5.3: Documentation Finalization (1-2 hours)

**Documents to create/update**:
1. **SUPERMAX_USAGE_GUIDE.md** - How to run SUPERMAX HPO
2. **SUPERMAX_RESULTS.md** - Benchmark results and analysis
3. **ARCHITECTURE_GUIDE.md** - New model architectures
4. **OPTIMIZER_GUIDE.md** - Advanced optimizer selection
5. Update **CLAUDE.md** with SUPERMAX commands
6. Update **README.md** with new features

**Deliverables**:
- ✅ Full pipeline validated on 1 agent
- 📊 Performance benchmark report
- 📊 Comparison vs. baseline
- 📚 Complete documentation suite
- ✅ User-facing guides

**Success Criteria**: SUPERMAX pipeline runs end-to-end successfully

**Risk**: LOW - Validation only, no new development

---

## Timeline & Resource Estimates

### Development Timeline (Sequential)

| Phase | Duration | Dependencies | Can Start |
|-------|----------|--------------|-----------|
| Phase 0 | 2-3 hours | None | Immediately |
| Phase 1 | 4-6 hours | Phase 0 | After P0 |
| Phase 2 | 6-8 hours | Phase 0 | Parallel with P1 |
| Phase 3 | 12-16 hours | Phase 1, 2 | After P1+P2 |
| Phase 4 | 12-18 hours | Phase 3 | After P3 (can overlap) |
| Phase 5 | 4-6 hours | Phase 3, 4 | After P3+P4 |

**Total**: 40-57 hours

### Execution Timeline (Parallelized)

**Week 1** (Full-time):
- Day 1: Phase 0 (3h) + Phase 1 (5h)
- Day 2: Phase 2 (7h)
- Day 3: Phase 3 Stage-A (5h) + Stage-B start (3h)
- Day 4: Phase 3 Stage-B finish (5h) + Stage-C (3h)
- Day 5: Phase 4 Testing (8h)

**Week 2** (Full-time):
- Day 6: Phase 4 Coverage + mypy (8h)
- Day 7: Phase 5 Validation (full day)

**Total**: 10-12 working days if part-time (4-6 hours/day)

### Computational Resources

**Development/Testing**:
- 1× GPU (RTX 4090 or equivalent)
- 64GB RAM (system)
- 500GB storage

**Production SUPERMAX Run** (per agent):
- 3-5 days continuous GPU time
- ~100-200 GB storage for checkpoints + artifacts
- MLflow tracking database (~10GB)

**Estimated costs** (if using cloud):
- AWS p3.2xlarge: ~$3/hour × 120 hours = ~$360/agent
- 4 agents × $360 = **~$1,440 total** for full SUPERMAX run

---

## Risk Assessment & Mitigation

### High-Risk Items

**Risk 1: Long Runtime for Stage-A/B**
- **Impact**: 3-5 days per agent means 12-20 days for all 4 agents
- **Mitigation**:
  - Run agents in parallel if multi-GPU available
  - Reduce n_trials for testing (100 instead of 1000)
  - Use smaller model (distilbert) for validation runs

**Risk 2: OOM Errors with New Architectures**
- **Impact**: xlnet-base-cased has large memory footprint
- **Mitigation**:
  - Add OOM handling in objective (already present: objectives.py:94-100)
  - Set batch_size constraints for large models
  - Use gradient checkpointing

**Risk 3: Optimizer Compatibility**
- **Impact**: LAMB/8-bit may not work with all model types
- **Mitigation**:
  - Fallback to AdamW if import fails
  - Test each optimizer + architecture combination
  - Add compatibility matrix to docs

### Medium-Risk Items

**Risk 4: Stage-B Config Seeding**
- **Impact**: Enqueuing Stage-A configs may fail if schema changed
- **Mitigation**:
  - Validate config schema before enqueuing
  - Add version metadata to exported configs
  - Test backward compatibility

**Risk 5: Test Coverage Gaps**
- **Impact**: 90% coverage may miss critical edge cases
- **Mitigation**:
  - Focus on critical paths (model loading, training loop)
  - Add property-based tests for search space
  - Manual testing of failure modes

### Low-Risk Items

**Risk 6: Documentation Drift**
- **Impact**: Docs may not reflect final implementation
- **Mitigation**:
  - Update docs incrementally after each phase
  - Auto-generate CLI help text
  - Add examples to docstrings

---

## Success Metrics

### Technical Metrics

1. **Functionality**:
   - ✅ All 8 architectures loadable and trainable
   - ✅ All 6 optimizers work with all architectures
   - ✅ Augmentation integration produces different metrics than baseline
   - ✅ Stage-A/B/C pipeline completes end-to-end

2. **Quality**:
   - ✅ 90%+ code coverage
   - ✅ mypy --strict passes
   - ✅ All tests pass (target: 100% pass rate)
   - ✅ No regressions in existing functionality

3. **Performance**:
   - ✅ F1-macro improvement: +2-5% over baseline
   - ✅ ECE calibration: <0.10 (better than baseline)
   - ✅ Trial throughput: ≥6 trials/hour
   - ✅ GPU utilization: >90% during training

### User Experience Metrics

1. **Usability**:
   - ✅ Single command to run SUPERMAX: `make supermax-all`
   - ✅ Clear progress indicators during long runs
   - ✅ Informative error messages
   - ✅ Easy result inspection (MLflow UI)

2. **Documentation**:
   - ✅ Comprehensive usage guide
   - ✅ Architecture selection guide
   - ✅ Optimizer selection guide
   - ✅ Troubleshooting guide

3. **Transparency**:
   - ✅ All HPO results logged to MLflow
   - ✅ Pareto fronts visualized
   - ✅ Config export at each stage
   - ✅ Runtime estimates provided

---

## Post-Implementation: Production Readiness Checklist

### Before First Production Run

- [ ] Validate full pipeline on small dataset (10% of data, 10 trials per stage)
- [ ] Establish performance baselines
- [ ] Set up monitoring (GPU, memory, disk)
- [ ] Configure automatic checkpointing
- [ ] Test resume-from-checkpoint
- [ ] Document expected runtime
- [ ] Set up alert thresholds (OOM, trial failure rate)

### During Production Run

- [ ] Monitor GPU utilization (should be >90%)
- [ ] Monitor trial failure rate (should be <5%)
- [ ] Check MLflow logs every 12 hours
- [ ] Verify Pareto front is expanding (Stage-B)
- [ ] Check disk space (artifacts can be large)

### After Production Run

- [ ] Export final configs
- [ ] Generate performance report
- [ ] Compare to baseline
- [ ] Identify best single config vs. ensemble candidates
- [ ] Archive artifacts to long-term storage
- [ ] Update documentation with results

---

## Appendix A: Command Reference

### Development Commands

```bash
# Phase 0: Setup
make baseline-benchmark

# Phase 1: Augmentation
make prepare-tfidf
make test-aug-hpo

# Phase 2: Architectures/Optimizers
make test-architectures
make test-optimizers

# Phase 3: SUPERMAX Stages
make stage-a HPO_TASK=criteria
make stage-b HPO_TASK=criteria
make stage-c HPO_TASK=criteria
make supermax-all  # All agents, all stages

# Phase 4: Testing
make test-cov
make lint
poetry run mypy src --strict

# Phase 5: Validation
make validate-supermax
```

### Monitoring Commands

```bash
# GPU monitoring
nvidia-smi -l 1

# MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Optuna dashboard
optuna-dashboard sqlite:///./_optuna/supermax.db

# Log tailing
tail -f outputs/stage_a/*.log
```

---

## Appendix B: Estimated Costs

### Development Time

| Role | Rate | Hours | Cost |
|------|------|-------|------|
| Senior ML Engineer | $150/hr | 50 | $7,500 |
| Testing Engineer | $100/hr | 20 | $2,000 |
| **Total** | | **70** | **$9,500** |

### Compute Resources (Cloud)

| Resource | Config | Hours | Rate | Cost |
|----------|--------|-------|------|------|
| Development GPU | p3.2xlarge | 50 | $3/hr | $150 |
| Stage-A (all agents) | p3.2xlarge × 4 | 192 | $3/hr | $576 |
| Stage-B (all agents) | p3.2xlarge × 4 | 288 | $3/hr | $864 |
| Stage-C (all agents) | p3.2xlarge × 4 | 48 | $3/hr | $144 |
| Storage | 500GB EBS | 30 days | $0.10/GB | $15 |
| **Total** | | | | **$1,749** |

### On-Premise (User's Setup)

**Cost**: $0 (using existing RTX 4090)
**Time**: 12-20 days continuous runtime
**Electricity**: ~$50-100 (assuming $0.12/kWh, 350W GPU, 480 hours)

**Total estimated cost**: ~$10,000 (development) + ~$1,700 (cloud compute) = **~$11,700**

**Or**: ~$10,000 (development) + ~$100 (on-premise electricity) = **~$10,100**

---

## Next Steps

1. **User Review** (30 min):
   - Review this roadmap
   - Confirm priorities
   - Adjust scope if needed
   - Approve Phase 0 start

2. **Phase 0 Kickoff** (Immediate):
   - Commit this roadmap
   - Run baseline benchmark
   - Validate environment
   - Create tracking issues

3. **Begin Implementation** (After approval):
   - Start Phase 1: Augmentation integration
   - Target completion: 4-6 hours
   - Deliverable: Working aug-enabled HPO

**Questions for User**:
1. Is the 40-60 hour timeline acceptable?
2. Should we prioritize certain agents (e.g., criteria first)?
3. Do you want to run SUPERMAX in cloud or on-premise?
4. Should we reduce trial counts for faster iteration?
5. Any specific performance targets (e.g., F1 > 0.85)?

---

**Document Status**: ✅ COMPLETE - Ready for review and implementation

**Last Updated**: 2025-10-30 22:30 UTC
**Author**: Claude Code (Anthropic)
**Version**: 1.0
