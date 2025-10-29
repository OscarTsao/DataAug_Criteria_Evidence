# HPO Pipeline Fix Summary

## Problem
The `make full-hpo-all` pipeline was broken because Makefile HPO targets (hpo-s0, hpo-s1, hpo-s2, refit) were calling a non-existent CLI subcommand:

```bash
poetry run python -m psy_agents_noaug.cli hpo ...
```

However, the CLI (`src/psy_agents_noaug/cli.py`) no longer has an `hpo` subcommand. It was refactored to have only:
- `train` - Run training job
- `tune` - Run maximal HPO via tune_max.py
- `show-best` - Display top-K results
- `tune-supermax` - Run very large HPO trials

## Root Cause
The CLI was refactored from using Hydra-based multi-stage HPO to a simpler system using `scripts/tune_max.py`. However, the Makefile targets were never updated to match this change.

## Solution
Updated all Makefile HPO targets to call `scripts/run_hpo_stage.py` directly instead of the non-existent CLI command:

### Before (Broken):
```makefile
hpo-s0:
	poetry run python -m psy_agents_noaug.cli hpo hpo=stage0_sanity task=$(HPO_TASK) model=$(HPO_MODEL)
```

### After (Fixed):
```makefile
hpo-s0:
	poetry run python scripts/run_hpo_stage.py hpo=stage0_sanity task=$(HPO_TASK) model=$(HPO_MODEL)
```

## What Was Changed

**File**: `Makefile`

**Lines Modified**: 193, 198, 203, 212

**Changes**:
1. **hpo-s0** (line 193): `cli hpo` → `scripts/run_hpo_stage.py`
2. **hpo-s1** (line 198): `cli hpo` → `scripts/run_hpo_stage.py`
3. **hpo-s2** (line 203): `cli hpo` → `scripts/run_hpo_stage.py`
4. **refit** (line 212): `cli refit` → `scripts/run_hpo_stage.py` (with best_config arg)

## Verification

The fix preserves the multi-stage HPO workflow:

- ✅ `scripts/run_hpo_stage.py` exists and is executable
- ✅ Hydra configs exist in `configs/hpo/`:
  - `stage0_sanity.yaml` (8 trials, 3 epochs)
  - `stage1_coarse.yaml` (20 trials)
  - `stage2_fine.yaml` (50 trials)
  - `stage3_refit.yaml` (refit on train+val)
- ✅ HPO module exists: `src/psy_agents_noaug/hpo/optuna_runner.py`
- ✅ Data files exist:
  - `data/redsm5/redsm5_annotations.csv`
  - `data/redsm5/redsm5_posts.csv`
  - `data/raw/redsm5/dsm_criteria.json`
  - `data/processed/redsm5_matched_evidence.csv`

## Testing the Fix

### Test Individual Stages:
```bash
# Stage 0: Sanity check (8 trials, 3 epochs)
make hpo-s0 HPO_TASK=criteria

# Stage 1: Coarse search (20 trials)
make hpo-s1 HPO_TASK=criteria

# Stage 2: Fine search (50 trials)
make hpo-s2 HPO_TASK=criteria

# Stage 3: Refit best model
make refit HPO_TASK=criteria
```

### Test Full Pipeline:
```bash
# Run all stages for one architecture
make full-hpo HPO_TASK=criteria

# Run all stages for ALL architectures sequentially
make full-hpo-all
```

### Test Maximal HPO:
```bash
# Run maximal HPO for ALL architectures
make maximal-hpo-all
```

## Architecture Comparison

### Multi-Stage HPO (Now Fixed)
- **Purpose**: Progressive refinement through stages (S0→S1→S2→refit)
- **Script**: `scripts/run_hpo_stage.py`
- **Config**: Hydra configs in `configs/hpo/stage*.yaml`
- **Usage**: `make hpo-s0`, `make hpo-s1`, `make hpo-s2`, `make refit`
- **Orchestrator**: `scripts/run_all_hpo.py --mode multistage`

### Maximal HPO (Already Working)
- **Purpose**: Single large optimization run (800-1200 trials)
- **Script**: `scripts/tune_max.py`
- **Config**: Environment variables (HPO_EPOCHS, HPO_PATIENCE)
- **Usage**: `make tune-criteria-max`, `make tune-evidence-max`
- **Orchestrator**: `scripts/run_all_hpo.py --mode maximal`

## Next Steps

1. **Test the fix**: Run `make hpo-s0 HPO_TASK=criteria` to verify it works
2. **Monitor GPU load**: Use `nvidia-smi -l 1` in another terminal
3. **Check logs**: Monitor MLflow UI at `http://localhost:5000`
4. **Run full pipeline**: Once verified, run `make full-hpo-all` for complete multi-stage HPO

## Additional Notes

### make test Command
✅ **Already working** - No issues found. Tests run successfully:
```bash
make test        # Run all tests
make test-cov    # Run with coverage
```

### Other Make Commands
All standard Make commands should work correctly:
- ✅ `make setup` - Full setup
- ✅ `make groundtruth` - Generate ground truth
- ✅ `make train` - Train model
- ✅ `make eval` - Evaluate model
- ✅ `make lint` - Run linters
- ✅ `make format` - Format code
- ✅ `make test` - Run tests

## Commit
**Hash**: bc21ceb
**Message**: "fix: Replace non-existent 'cli hpo' command with run_hpo_stage.py script"
**Date**: 2025-10-29

---

## ⚠️ IMPORTANT LIMITATIONS DISCOVERED

After fixing the Makefile commands, we discovered that **multi-stage HPO is only partially implemented**:

### What Works:
- ✅ `make hpo-s0 HPO_TASK=criteria` (sanity check for criteria only)
- ✅ `make tune-criteria-max` (maximal HPO for criteria)
- ✅ `make tune-evidence-max` (maximal HPO for evidence - via tune_max.py)
- ✅ `make maximal-hpo-all` (maximal HPO for all agents)

### What Doesn't Work:
- ❌ `make hpo-s0 HPO_TASK=evidence` - NotImplementedError
- ❌ `make hpo-s0 HPO_TASK=share` - NotImplementedError
- ❌ `make hpo-s0 HPO_TASK=joint` - NotImplementedError
- ❌ `make full-hpo-all` - Fails on evidence task

### Root Cause:
`scripts/run_hpo_stage.py` only implements the criteria task (line 136):
```python
if cfg.task.name == "criteria":
    dataset = CriteriaDataset(...)
else:
    raise NotImplementedError(f"Task {cfg.task.name} not implemented yet")
```

### Two HPO Systems Exist:

**System 1: Multi-Stage HPO** (INCOMPLETE)
- Script: `scripts/run_hpo_stage.py`
- Status: ❌ Only criteria implemented
- Used by: `make hpo-s0`, `make hpo-s1`, `make hpo-s2`

**System 2: Maximal HPO** (COMPLETE)
- Script: `scripts/tune_max.py`
- Status: ✅ All 4 agents implemented
- Used by: `make tune-*-max`, `make maximal-hpo-all`

### Recommended Solution:

**Use the maximal HPO system** (System 2) which is complete:
```bash
# Test with reduced trials
python scripts/tune_max.py --agent criteria --study test --n-trials 10

# Production runs (use appropriate hardware)
make tune-criteria-max    # 800 trials
make tune-evidence-max    # 1200 trials
make maximal-hpo-all      # All agents sequentially
```

See `MULTI_STAGE_HPO_LIMITATIONS.md` for complete details and options.

---

**Status**: ⚠️ **PARTIALLY FIXED** - Makefile now calls correct script, but multi-stage HPO only works for criteria task. Use maximal HPO system for all agents.
