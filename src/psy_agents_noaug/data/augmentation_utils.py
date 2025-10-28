"""Utilities for constructing augmentation pipelines for the evidence task.

This module keeps augmentation concerns isolated from dataset construction. It
decides which methods are active, prepares any external resources (e.g. TF‑IDF
models), and returns a frozen bundle the loader can use safely.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from psy_agents_noaug.augmentation import (
    ALL_METHODS,
    NLPAUG_METHODS,
    TEXTATTACK_METHODS,
    AugConfig,
    AugmenterPipeline,
    AugResources,
    TfidfResource,
    is_enabled,
    load_or_fit_tfidf,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class AugmentationArtifacts:
    """Container bundling augmentation pipeline and fitted resources."""

    pipeline: AugmenterPipeline
    config: AugConfig
    resources: AugResources
    tfidf: TfidfResource | None
    methods: tuple[str, ...]


def _ensure_sequence(methods: Sequence[str] | str | None) -> list[str]:
    """Normalise a possibly scalar methods arg into a list of strings."""
    if methods is None:
        return []
    if isinstance(methods, str):
        return [methods]
    return list(methods)


def resolve_methods(lib: str, methods: Sequence[str] | str | None) -> list[str]:
    """Resolve declared methods to concrete augmenter identifiers.

    Supports the special value ``"all"`` which expands to all methods available
    within the selected library family. Ensures stable ordering and de‑dupes.
    """
    if lib == "none":
        return []

    allowed_map = {
        "nlpaug": NLPAUG_METHODS,
        "textattack": TEXTATTACK_METHODS,
        "both": ALL_METHODS,
    }
    allowed = allowed_map.get(lib, [])

    declared = _ensure_sequence(methods)
    resolved: list[str] = []
    for method in declared:
        if method == "all":
            resolved.extend(allowed)
            continue
        if method not in ALL_METHODS:
            raise KeyError(f"Unknown augmentation method: {method}")
        if lib != "both" and method not in allowed:
            continue
        resolved.append(method)

    unique: list[str] = []
    seen: set[str] = set()
    for method in resolved:
        if method in seen:
            continue
        seen.add(method)
        unique.append(method)
    return unique


def build_evidence_augmenter(
    cfg: AugConfig,
    train_texts: Iterable[str],
    *,
    tfidf_dir: str | Path = "_artifacts/tfidf/evidence",
) -> AugmentationArtifacts | None:
    """Instantiate augmentation pipeline for evidence task if enabled.

    Side effects: may fit/load a TF‑IDF vectoriser when requested.
    """
    if not is_enabled(cfg):
        return None

    cfg_copy = replace(cfg)
    resolved_methods = resolve_methods(cfg_copy.lib, cfg_copy.methods)

    texts = [str(text) for text in train_texts]

    resources = AugResources(
        tfidf_model_path=cfg_copy.tfidf_model_path,
        reserved_map_path=cfg_copy.reserved_map_path,
    )
    tfidf_resource: TfidfResource | None = None

    if any(method.endswith("TfIdfAug") for method in resolved_methods):
        target_dir = Path(tfidf_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = (
            Path(cfg_copy.tfidf_model_path)
            if cfg_copy.tfidf_model_path
            else target_dir / "tfidf.pkl"
        )
        tfidf_resource = load_or_fit_tfidf(texts, model_path)
        resources.tfidf_model_path = str(tfidf_resource.path)
        cfg_copy.tfidf_model_path = str(tfidf_resource.path)

    # Build the pipeline and seed its internal RNG for reproducibility
    pipeline = AugmenterPipeline(cfg_copy, resources=resources)
    pipeline.set_seed(cfg_copy.seed)

    return AugmentationArtifacts(
        pipeline=pipeline,
        config=cfg_copy,
        resources=resources,
        tfidf=tfidf_resource,
        methods=tuple(resolved_methods),
    )


__all__ = ["AugmentationArtifacts", "build_evidence_augmenter", "resolve_methods"]
