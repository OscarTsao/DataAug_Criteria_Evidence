"""Checkpoint management for resumable HPO studies.

This module provides robust checkpointing to enable:
- Resume interrupted HPO runs
- Recovery from crashes
- Incremental study updates
- Backup and restore
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna

LOGGER = logging.getLogger(__name__)


class CheckpointManager:
    """Manages study checkpoints for resume capability."""

    def __init__(
        self,
        checkpoint_dir: Path,
        study_name: str,
        backup_count: int = 5,
        checkpoint_interval: int = 10,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            study_name: Name of the study
            backup_count: Number of backup checkpoints to keep
            checkpoint_interval: Save checkpoint every N trials
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.study_name = study_name
        self.backup_count = backup_count
        self.checkpoint_interval = checkpoint_interval

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._trial_count = 0
        self._last_checkpoint = 0

    def get_checkpoint_path(self, backup_id: int | None = None) -> Path:
        """Get path to checkpoint file.

        Args:
            backup_id: Optional backup number (for rotated backups)

        Returns:
            Path to checkpoint file
        """
        if backup_id is None:
            return self.checkpoint_dir / f"{self.study_name}_checkpoint.json"
        else:
            return self.checkpoint_dir / f"{self.study_name}_checkpoint_{backup_id}.json"

    def should_checkpoint(self) -> bool:
        """Check if it's time to save a checkpoint.

        Returns:
            True if checkpoint should be saved
        """
        return (self._trial_count - self._last_checkpoint) >= self.checkpoint_interval

    def save_checkpoint(
        self,
        study: optuna.Study,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save study checkpoint.

        Args:
            study: Optuna study to checkpoint
            metadata: Optional metadata to include

        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "best_value": study.best_value if study.best_trial else None,
            "best_trial_number": study.best_trial.number if study.best_trial else None,
            "directions": [str(d) for d in study.directions],
            "metadata": metadata or {},
        }

        # Save trials data
        trials_data = []
        for trial in study.get_trials(deepcopy=False):
            trial_dict = {
                "number": trial.number,
                "state": str(trial.state),
                "value": trial.value,
                "values": trial.values,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "system_attrs": trial.system_attrs,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            }
            trials_data.append(trial_dict)

        checkpoint_data["trials"] = trials_data

        # Rotate backups
        self._rotate_backups()

        # Save new checkpoint
        checkpoint_path = self.get_checkpoint_path()
        with checkpoint_path.open("w") as f:
            json.dump(checkpoint_data, f, indent=2)

        self._last_checkpoint = self._trial_count

        LOGGER.info("Saved checkpoint: %s (%d trials)", checkpoint_path, len(study.trials))
        return checkpoint_path

    def _rotate_backups(self) -> None:
        """Rotate checkpoint backups."""
        current = self.get_checkpoint_path()

        if not current.exists():
            return

        # Shift existing backups
        for i in range(self.backup_count - 1, 0, -1):
            old_backup = self.get_checkpoint_path(i)
            new_backup = self.get_checkpoint_path(i + 1)

            if old_backup.exists():
                if new_backup.exists():
                    new_backup.unlink()
                shutil.move(str(old_backup), str(new_backup))

        # Move current to backup 1
        backup1 = self.get_checkpoint_path(1)
        if backup1.exists():
            backup1.unlink()
        shutil.copy2(str(current), str(backup1))

    def load_checkpoint(self) -> dict[str, Any] | None:
        """Load latest checkpoint.

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        checkpoint_path = self.get_checkpoint_path()

        if not checkpoint_path.exists():
            LOGGER.info("No checkpoint found: %s", checkpoint_path)
            return None

        try:
            with checkpoint_path.open("r") as f:
                checkpoint_data = json.load(f)

            LOGGER.info(
                "Loaded checkpoint: %s (%d trials, saved %s)",
                checkpoint_path,
                checkpoint_data.get("n_trials", 0),
                checkpoint_data.get("timestamp", "unknown"),
            )
            return checkpoint_data

        except Exception as e:
            LOGGER.error("Failed to load checkpoint %s: %s", checkpoint_path, e)

            # Try to load from backup
            for i in range(1, self.backup_count + 1):
                backup_path = self.get_checkpoint_path(i)
                if backup_path.exists():
                    try:
                        with backup_path.open("r") as f:
                            checkpoint_data = json.load(f)
                        LOGGER.warning("Loaded from backup %d: %s", i, backup_path)
                        return checkpoint_data
                    except Exception as backup_e:
                        LOGGER.error("Backup %d also failed: %s", i, backup_e)
                        continue

            return None

    def on_trial_complete(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Callback when trial completes.

        Args:
            study: Optuna study
            trial: Completed trial
        """
        self._trial_count += 1

        if self.should_checkpoint():
            self.save_checkpoint(study)

    def finalize(self, study: optuna.Study) -> None:
        """Save final checkpoint.

        Args:
            study: Optuna study
        """
        LOGGER.info("Saving final checkpoint...")
        self.save_checkpoint(study, metadata={"final": True})


def save_study_checkpoint(
    study: optuna.Study,
    checkpoint_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save study checkpoint to file.

    Args:
        study: Optuna study to save
        checkpoint_path: Path to checkpoint file
        metadata: Optional metadata
    """
    manager = CheckpointManager(
        checkpoint_dir=checkpoint_path.parent,
        study_name=study.study_name,
    )
    manager.save_checkpoint(study, metadata=metadata)


def load_study_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load study checkpoint from file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Checkpoint data or None
    """
    if not checkpoint_path.exists():
        return None

    try:
        with checkpoint_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.error("Failed to load checkpoint: %s", e)
        return None


def create_checkpoint_callback(
    manager: CheckpointManager,
) -> optuna.study.StudyCallback:
    """Create Optuna callback for automatic checkpointing.

    Args:
        manager: Checkpoint manager

    Returns:
        Optuna callback function
    """

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Callback invoked after each trial."""
        manager.on_trial_complete(study, trial)

    return callback
