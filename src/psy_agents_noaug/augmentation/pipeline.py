"""Augmentation pipeline orchestration and deterministic seeding helpers."""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .registry import (
    ALL_METHODS,
    NLPAUG_METHODS,
    REGISTRY,
    TEXTATTACK_METHODS,
    AugmenterWrapper,
    RegistryEntry,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

LOGGER = logging.getLogger(__name__)

AugLib = Literal["none", "nlpaug", "textattack", "both"]


@dataclass
class AugConfig:
    """Run-time augmentation configuration."""

    lib: AugLib = "none"
    methods: Sequence[str] = field(default_factory=lambda: ["all"])
    p_apply: float = 0.15
    ops_per_sample: int = 1
    max_replace_ratio: float = 0.3
    tfidf_model_path: str | None = None
    reserved_map_path: str | None = None
    seed: int = 42
    method_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)
    example_limit: int = 32


@dataclass
class AugResources:
    """External resources required by certain augmenters."""

    tfidf_model_path: str | None = None
    reserved_map_path: str | None = None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_methods(lib: AugLib, methods: Sequence[str]) -> list[str]:
    """Resolve effective augmentation methods given library selection."""
    if lib == "none":
        return []

    lib_to_methods = {
        "nlpaug": NLPAUG_METHODS,
        "textattack": TEXTATTACK_METHODS,
        "both": ALL_METHODS,
    }
    allowed = lib_to_methods.get(lib, [])

    resolved: list[str] = []
    if isinstance(methods, list | tuple):
        declared: list[str] = list(methods)
    else:
        declared = [methods]  # type: ignore[list-item]
    for method in declared:
        if method == "all":
            resolved.extend(allowed)
            continue
        if method not in REGISTRY:
            raise KeyError(f"Unknown augmentation method: {method}")
        if lib != "both" and method not in allowed:
            LOGGER.debug("Skipping method %s not in lib %s", method, lib)
            continue
        resolved.append(method)

    # Preserve order while removing duplicates
    seen: set[str] = set()
    unique: list[str] = []
    for method in resolved:
        if method in seen:
            continue
        seen.add(method)
        unique.append(method)
    return unique


def _ratio_kwargs(name: str, ratio: float) -> dict[str, Any]:
    """Default intensity controls per method."""
    ratio = _clamp(ratio, 0.0, 1.0)
    if name.startswith("nlpaug/char/"):
        return {"aug_char_p": ratio}
    if name.startswith("nlpaug/word/"):
        return {"aug_p": ratio}

    mapping = {
        "textattack/CharSwapAugmenter": "pct_characters_to_swap",
        "textattack/DeletionAugmenter": "pct_words_to_delete",
        "textattack/SwapAugmenter": "pct_words_to_swap",
        "textattack/SynonymInsertionAugmenter": "pct_words_to_swap",
        "textattack/EasyDataAugmenter": "pct_words_to_swap",
        "textattack/CheckListAugmenter": "pct_words_to_swap",
        "textattack/WordNetAugmenter": "pct_words_to_swap",
    }
    param = mapping.get(name)
    return {param: ratio} if param else {}


def _merge_kwargs(
    base: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def _builder_kwargs(
    name: str,
    cfg: AugConfig,
    resources: AugResources | None,
) -> dict[str, Any]:
    overrides = cfg.method_kwargs.get(name, {})
    ratio_kwargs = _ratio_kwargs(name, cfg.max_replace_ratio)
    kwargs = _merge_kwargs(ratio_kwargs, overrides)

    if name == "nlpaug/char/RandomCharAug" and "action" not in kwargs:
        kwargs["action"] = "substitute"
    if name == "nlpaug/word/RandomWordAug" and "action" not in kwargs:
        kwargs["action"] = "swap"

    if name == "nlpaug/word/TfIdfAug":
        path = kwargs.get("model_path") or cfg.tfidf_model_path
        if resources and not path:
            path = resources.tfidf_model_path
        if path is None:
            raise ValueError("TfIdfAug requires a fitted model_path")
        kwargs["model_path"] = path
        kwargs.setdefault("action", "substitute")
        kwargs.setdefault("device", "cpu")

    if name == "nlpaug/word/ReservedAug":
        path = kwargs.get("reserved_map_path") or cfg.reserved_map_path
        if resources and not path:
            path = resources.reserved_map_path
        if path is None:
            raise ValueError("ReservedAug requires reserved_map_path")
        kwargs["reserved_map_path"] = path
        kwargs.setdefault("action", "substitute")

    return kwargs


class AugmenterPipeline:
    """Deterministic augmentation pipeline for evidence text.

    Note: Workers inherit identical RNG states after fork. Re-seed in worker_init.
    """

    def __init__(
        self,
        cfg: AugConfig,
        resources: AugResources | None = None,
    ) -> None:
        if cfg.lib == "none":
            raise ValueError("Cannot instantiate AugmenterPipeline with lib='none'")

        self.cfg = cfg
        self.resources = resources or AugResources()
        self.methods = _resolve_methods(cfg.lib, cfg.methods)
        self.p_apply = _clamp(cfg.p_apply, 0.0, 1.0)
        self.ops_per_sample = max(1, min(2, int(cfg.ops_per_sample)))
        self.max_replace_ratio = _clamp(cfg.max_replace_ratio, 0.0, 1.0)
        # Global RNG not thread-safe; use instance RNG and re-seed in workers
        self._rng = random.Random(cfg.seed)  # Instance-specific RNG

        self._augmenters: list[tuple[str, AugmenterWrapper]] = []
        for name in self.methods:
            entry: RegistryEntry = REGISTRY[name]
            kwargs = _builder_kwargs(name, cfg, self.resources)
            try:
                wrapper = entry.factory(**kwargs)
            except TypeError as exc:
                LOGGER.debug(
                    "Retrying augmenter %s without ratio kwargs due to %s", name, exc
                )
                clean_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in {"aug_p", "aug_char_p"} and not k.startswith("pct_")
                }
                wrapper = entry.factory(**clean_kwargs)
            self._augmenters.append((name, wrapper))

        self.method_counts: Counter[str] = Counter()
        self.applied_count = 0
        self.skipped_count = 0
        self.total_count = 0
        self.examples: list[dict[str, Any]] = []
        self.example_limit = max(0, int(cfg.example_limit))

    def close(self) -> None:
        """Release augmenter resources (WordNet, spaCy models, etc.)."""
        if hasattr(self, "_augmenters"):
            # Clear augmenter references to allow garbage collection
            del self._augmenters

    def __enter__(self) -> AugmenterPipeline:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit with resource cleanup."""
        self.close()

    def set_seed(self, seed: int) -> None:
        """
        Reset RNG state for deterministic worker behaviour.

        Note: This resets the instance RNG. In multiprocessing environments,
        workers inherit parent RNG state after fork and must call worker_init().
        """
        self._rng.seed(seed)

    def __call__(self, text: str) -> str:
        """Apply augmentation and record statistics."""
        self.total_count += 1
        if not self._augmenters or self.p_apply <= 0.0:
            self.skipped_count += 1
            return text

        if self._rng.random() > self.p_apply:
            self.skipped_count += 1
            return text

        original = text
        applied_methods: list[str] = []
        augmented = text

        for _ in range(self.ops_per_sample):
            method_name, augmenter = self._rng.choice(self._augmenters)
            candidate = augmenter.augment_one(augmented)
            if not isinstance(candidate, str) or not candidate:
                continue
            augmented = candidate
            applied_methods.append(method_name)

        if not applied_methods:
            self.skipped_count += 1
            return text

        self.applied_count += 1
        for method_name in applied_methods:
            self.method_counts[method_name] += 1

        if len(self.examples) < self.example_limit:
            self.examples.append(
                {
                    "original": original,
                    "augmented": augmented,
                    "methods": applied_methods,
                    "timestamp": time.time(),
                }
            )

        return augmented

    def stats(self) -> dict[str, Any]:
        """Return snapshot of usage statistics."""
        return {
            "total": self.total_count,
            "applied": self.applied_count,
            "skipped": self.skipped_count,
            "method_counts": dict(self.method_counts),
        }

    def drain_examples(self) -> list[dict[str, Any]]:
        """Return and clear collected augmentation examples."""
        data = list(self.examples)
        self.examples.clear()
        return data


def worker_init(
    worker_id: int, base_seed: int, rank: int = 0, num_workers_per_rank: int = 1
) -> int:
    """
    Initialize random seeds for DataLoader workers with DDP support.

    In multi-GPU setups (DDP), workers across different ranks must have unique seeds
    to avoid duplicate augmentations. Seed derivation accounts for:
    - rank: GPU/process ID in distributed training
    - num_workers_per_rank: Number of DataLoader workers per GPU
    - worker_id: Worker ID within current rank (0-indexed)

    Args:
        worker_id: Worker ID within current DataLoader (0, 1, 2, ...)
        base_seed: Global base seed
        rank: DDP process rank (default: 0 for single-GPU)
        num_workers_per_rank: Workers per GPU (default: 1)

    Returns:
        The derived worker seed.

    Note: Workers inherit identical RNG states after fork in multiprocessing.
          This function MUST be called to re-seed each worker's RNG.
    """
    # Ensure unique seeds: rank_offset + worker_offset + 1
    seed = int(base_seed) + (rank * num_workers_per_rank) + worker_id + 1
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:  # pragma: no cover - torch optional in tests
        import torch

        torch.manual_seed(seed)
    except Exception:
        LOGGER.debug("Torch not available for worker seeding", exc_info=True)
    return seed


def is_enabled(cfg: AugConfig) -> bool:
    """Check whether augmentation is active."""
    return cfg.lib in {"nlpaug", "textattack", "both"} and bool(
        _resolve_methods(cfg.lib, cfg.methods)
    )
