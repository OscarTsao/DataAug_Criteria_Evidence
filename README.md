# PSY Agents NO-AUG

## Quick Start

# setup
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

# sanity
pytest -q

# train (stub; wire to your trainer as needed)
psy-agents train --agent criteria --model-name bert-base-uncased --epochs 1 --outdir ./_runs

# HPO (Stage A global sweep)
HPO_EPOCHS=6 make tune-criteria-max
HPO_EPOCHS=6 make tune-evidence-max

# Show winners
psy-agents show-best --agent criteria --study noaug-criteria-max --topk 5
psy-agents show-best --agent evidence --study noaug-evidence-max --topk 5

> Notes:
> - MLflow logs to ./_runs/mlruns by default (file URI).
> - Optuna storage defaults to ./_optuna/noaug.db (sqlite).
> - No augmentation: ensure criteria uses "status", evidence uses "cases" only.

## 🚀 Data Augmentation (Infrastructure Ready)

**Status:** Augmentation infrastructure is 60% complete and ready for activation.

### Infrastructure Already in Place

- ✅ **17 CPU-light augmenters** in `src/psy_agents_noaug/augmentation/registry.py`
  - nlpaug: Synonym, Spelling, Keyboard, OCR, Random, Split, TF-IDF, Reserved
  - TextAttack: CharSwap, Deletion, Swap, Synonym Insertion, EDA, CheckList, WordNet
- ✅ **Pipeline class** with deterministic seeding (`augmentation/pipeline.py`)
- ✅ **Dataset hooks** already implemented in `data/datasets.py`
- ✅ **Worker initialization** for multi-GPU determinism
- ✅ **Configuration system** (`configs/augmentation/default.yaml`)

### Quick Start with Augmentation

```bash
# Enable augmentation (currently dormant by default)
psy-agents train --agent criteria augmentation.enabled=true

# Configure augmentation scope
psy-agents train augmentation.scope=train_only  # Recommended
psy-agents train augmentation.scope=all         # All splits
psy-agents train augmentation.scope=none        # Disabled

# Set augmentation probability and operations
psy-agents train augmentation.p_apply=0.15 augmentation.ops_per_sample=1
```

### Transformation Roadmap

The project is transitioning from NO-AUG baseline to production-ready AUG system:

📋 **Foundation Documents** (Completed):
- `INVENTORY.md` - Complete codebase mapping (1,954 lines)
- `QUALITY-GATES.md` - 10 production quality gates (351 lines)
- `PR_PLAN.md` - 5-PR transformation roadmap (576 lines)

⏳ **5 Sequential PRs** (54-78 hours, 3 weeks):
1. PR#1: Quality Gates & CI Infrastructure (8-12h)
2. PR#2: Augmentation Integration & Tests (16-24h)
3. PR#3: HPO Integration & Observability (12-16h)
4. PR#4: Packaging, Docker & Security (10-14h)
5. PR#5: Documentation & Release (8-12h)

### Key Files

- `INVENTORY.md` - Codebase analysis and augmentation status
- `QUALITY-GATES.md` - Production readiness criteria
- `PR_PLAN.md` - Detailed transformation plan
- `CHANGELOG.md` - Version history and transformation progress
- `configs/augmentation/default.yaml` - Augmentation configuration

### Next Steps

1. Review foundation documents (`INVENTORY.md`, `QUALITY-GATES.md`, `PR_PLAN.md`)
2. Follow PR sequence for full activation
3. Run quality gates to ensure production readiness
4. Monitor `CHANGELOG.md` for transformation progress

## Recent Updates (October 2025)

**Production-Ready HPO System:**
- ✅ Fixed Optuna 4.5.0 compatibility (MOTPESampler → NSGAIISampler)
- ✅ Implemented functional training bridge with synthetic data
- ✅ Comprehensive HEAD search (pooling/layers/hidden/activation/dropout)
- ✅ QA null policy search (threshold/ratio/calibrated)
- ✅ 9 model backbones, 5 schedulers, 4 optimizers, regularization knobs

**Interface Parity Achieved:**
- ✅ All 4 models in `src/Project/` now accept `head_cfg` and `task_cfg`
- ✅ Fixed output keys: Share/Joint return `"logits"` (was `"criteria_logits"`)
- ✅ Backward compatible with direct parameter passing

**Code Quality:**
- ✅ Updated to PyTorch 2.x AMP API (torch.amp instead of torch.cuda.amp)
- ✅ Registered pytest markers (eliminates warnings)
- ✅ 67/69 tests passing (97.1%), 31% code coverage

**Verified Smoke Tests:**
- 3-trial HPO run completes successfully
- HEAD parameters correctly logged to MLflow
- Top-K JSON export functional

All project documentation now resides in the [`docs/`](docs/) directory.

Primary entry points:
- `docs/README.md` – project overview and repository structure
- `docs/QUICK_START.md` – quick setup and usage guide
- `docs/CI_CD_SETUP.md` – CI/CD pipeline reference
- `docs/TESTING.md` – testing strategy and commands

Additional guides are available alongside these files for setup, training
infrastructure, data pipeline details, and CLI/Makefile usage.

Model implementations for the four supported architectures live in
`src/psy_agents_noaug/architectures/` (`criteria`, `evidence`, `share`, `joint`).
