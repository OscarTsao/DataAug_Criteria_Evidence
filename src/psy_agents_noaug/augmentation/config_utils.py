"""Utilities for converting HPO config dicts to AugConfig objects."""

from __future__ import annotations

import logging
from typing import Any

from .pipeline import AugConfig, AugLib

LOGGER = logging.getLogger(__name__)


def hpo_config_to_aug_config(
    config: dict[str, Any],
    global_seed: int = 42,
) -> AugConfig | None:
    """Convert HPO augmentation config dict to AugConfig object.

    Args:
        config: HPO configuration dictionary (may contain "augmentation" key)
        global_seed: Global random seed (used if augmentation.seed is None)

    Returns:
        AugConfig object if augmentation is enabled, None otherwise

    Examples:
        >>> cfg = {
        ...     "augmentation": {
        ...         "enabled": True,
        ...         "lib": "nlpaug",
        ...         "methods": ["SynonymAug", "RandomWordAug"],
        ...         "p_apply": 0.15,
        ...         "ops_per_sample": 1,
        ...         "max_replace_ratio": 0.3,
        ...         "scope": "train_only",
        ...         "seed": None,
        ...     }
        ... }
        >>> aug_cfg = hpo_config_to_aug_config(cfg, global_seed=42)
        >>> aug_cfg.lib
        'nlpaug'
        >>> aug_cfg.methods
        ['SynonymAug', 'RandomWordAug']
    """
    # Check if augmentation config exists
    aug_dict = config.get("augmentation")
    if not aug_dict:
        LOGGER.debug("No 'augmentation' key in config, augmentation disabled")
        return None

    # Check if augmentation is enabled
    if not aug_dict.get("enabled", False):
        LOGGER.debug("Augmentation disabled (enabled=False)")
        return None

    # Extract augmentation parameters with defaults
    lib = aug_dict.get("lib", "none")
    if lib == "none":
        LOGGER.debug("Augmentation lib is 'none', skipping")
        return None

    # Validate lib value
    valid_libs: list[AugLib] = ["nlpaug", "textattack", "both"]
    if lib not in valid_libs:
        LOGGER.warning(f"Invalid augmentation lib '{lib}', must be one of {valid_libs}")
        return None

    # Extract methods - convert to list if needed
    methods = aug_dict.get("methods", ["all"])
    if isinstance(methods, str):
        methods = [methods]

    # Extract other parameters
    p_apply = float(aug_dict.get("p_apply", 0.15))
    ops_per_sample = int(aug_dict.get("ops_per_sample", 1))
    max_replace_ratio = float(aug_dict.get("max_replace_ratio", 0.3))

    # Seed handling: use augmentation.seed if provided, else global_seed
    seed = aug_dict.get("seed")
    if seed is None:
        seed = global_seed

    # Optional resource paths
    tfidf_model_path = aug_dict.get("tfidf_model_path")
    reserved_map_path = aug_dict.get("reserved_map_path")

    # Method-specific kwargs (optional)
    method_kwargs = aug_dict.get("method_kwargs", {})

    # Build AugConfig
    aug_config = AugConfig(
        lib=lib,  # type: ignore[arg-type]
        methods=methods,
        p_apply=p_apply,
        ops_per_sample=ops_per_sample,
        max_replace_ratio=max_replace_ratio,
        tfidf_model_path=tfidf_model_path,
        reserved_map_path=reserved_map_path,
        seed=seed,
        method_kwargs=method_kwargs,
    )

    LOGGER.info(
        f"Augmentation enabled: lib={lib}, methods={methods}, "
        f"p_apply={p_apply:.3f}, ops_per_sample={ops_per_sample}, "
        f"max_replace_ratio={max_replace_ratio:.3f}, seed={seed}"
    )

    return aug_config


def should_apply_augmentation(scope: str, is_training: bool) -> bool:
    """Determine if augmentation should be applied based on scope and training mode.

    Args:
        scope: Augmentation scope ("train_only", "all", or "none")
        is_training: Whether currently in training mode (True) or eval mode (False)

    Returns:
        True if augmentation should be applied, False otherwise

    Examples:
        >>> should_apply_augmentation("train_only", is_training=True)
        True
        >>> should_apply_augmentation("train_only", is_training=False)
        False
        >>> should_apply_augmentation("all", is_training=False)
        True
    """
    if scope == "none":
        return False
    if scope == "train_only":
        return is_training
    if scope == "all":
        return True

    # Default to train_only for unknown scopes
    LOGGER.warning(f"Unknown augmentation scope '{scope}', defaulting to 'train_only'")
    return is_training


__all__ = [
    "hpo_config_to_aug_config",
    "should_apply_augmentation",
]
