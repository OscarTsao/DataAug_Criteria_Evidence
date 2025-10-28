"""Registry of on‑the‑fly text augmenters.

Unifies disparate augmentation libraries (nlpaug, TextAttack) behind a single
wrapper interface. Each registered method builds an ``AugmenterWrapper`` that
exposes ``augment_one(text) -> str``. If an underlying augmenter produces a
list (common in TextAttack), the wrapper returns the first non‑empty string.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from textattack.augmentation.recipes import (
    CharSwapAugmenter,
    CheckListAugmenter,
    DeletionAugmenter,
    EasyDataAugmenter,
    SwapAugmenter,
    SynonymInsertionAugmenter,
    WordNetAugmenter,
)

LOGGER = logging.getLogger(__name__)


class AugmenterWrapper:
    """Normalise augmenter outputs to a single string result."""

    def __init__(self, name: str, augmenter: Any, *, returns_list: bool = False):
        self.name = name
        self._augmenter = augmenter
        self._returns_list = returns_list

    def augment_one(self, text: str) -> str:
        """Apply augmentation safely, returning the original text on failure.

        Defensive behaviour ensures the training loop never breaks due to
        augmentation errors or corner cases – empty lists, None returns, etc.
        """
        try:
            result = self._augmenter.augment(text)
        except Exception as exc:  # pragma: no cover - defensive path
            LOGGER.debug("Augmenter %s failed: %s", self.name, exc)
            return text

        if result is None:
            return text

        if isinstance(result, str):
            return result or text

        if isinstance(result, list):
            if not result:
                return text
            if self._returns_list:
                candidate = result[0]
                return candidate if isinstance(candidate, str) and candidate else text
            for candidate in result:
                if isinstance(candidate, str) and candidate:
                    return candidate
            return text

        return text


def _load_reserved_tokens(reserved_map_path: str | Path) -> dict[str, str] | list[str]:
    """Load reserved tokens for ReservedAug.

    The JSON can be a mapping (e.g., canonical forms) or a list of tokens.
    """
    data_path = Path(reserved_map_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Reserved map not found: {data_path}")

    raw = json.loads(data_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return raw
    raise ValueError(
        f"Reserved map must be dict or list, received {type(raw).__name__}"
    )


AugmenterFactory = Callable[..., AugmenterWrapper]


@dataclass(frozen=True)
class RegistryEntry:
    """Metadata describing a single augmentation method."""

    lib: str
    factory: AugmenterFactory


def _wrap(
    factory: Callable[..., Any], *, returns_list: bool = False, name: str | None = None
) -> AugmenterFactory:
    """Wrap a concrete augmenter factory into an ``AugmenterWrapper`` builder."""

    def _builder(**kwargs: Any) -> AugmenterWrapper:
        augmenter = factory(**kwargs)
        augmenter_name = name or getattr(factory, "__name__", "augmenter")
        return AugmenterWrapper(augmenter_name, augmenter, returns_list=returns_list)  # type: ignore[arg-type]

    return _builder


def _make_reserved(
    reserved_map_path: str | Path | None, **kwargs: Any
) -> AugmenterWrapper:
    """Factory for ``naw.ReservedAug`` that validates required resources."""
    if reserved_map_path is None:
        raise ValueError("reserved_map_path is required for ReservedAug")
    reserved_tokens = _load_reserved_tokens(reserved_map_path)
    augmenter = naw.ReservedAug(reserved_tokens=reserved_tokens, **kwargs)
    return AugmenterWrapper("ReservedAug", augmenter)


def _make_tfidf(model_path: str | Path | None, **kwargs: Any) -> AugmenterWrapper:
    """Factory for ``naw.TfIdfAug`` that validates required resources."""
    if model_path is None:
        raise ValueError("model_path is required for TfIdfAug")
    augmenter = naw.TfIdfAug(model_path=str(model_path), **kwargs)
    return AugmenterWrapper("TfIdfAug", augmenter)


REGISTRY: dict[str, RegistryEntry] = {
    # nlpaug char-level
    "nlpaug/char/KeyboardAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(nac.KeyboardAug, name="KeyboardAug")
    ),
    "nlpaug/char/OcrAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(nac.OcrAug, name="OcrAug")
    ),
    "nlpaug/char/RandomCharAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(nac.RandomCharAug, name="RandomCharAug")
    ),
    # nlpaug word-level
    "nlpaug/word/RandomWordAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(naw.RandomWordAug, name="RandomWordAug")
    ),
    "nlpaug/word/ReservedAug": RegistryEntry(lib="nlpaug", factory=_make_reserved),
    "nlpaug/word/SpellingAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(naw.SpellingAug, name="SpellingAug")
    ),
    "nlpaug/word/SplitAug": RegistryEntry(
        lib="nlpaug", factory=_wrap(naw.SplitAug, name="SplitAug")
    ),
    "nlpaug/word/SynonymAug": RegistryEntry(
        lib="nlpaug",
        factory=_wrap(
            lambda **kw: naw.SynonymAug(aug_src="wordnet", **kw), name="SynonymAug"
        ),
    ),
    "nlpaug/word/AntonymAug": RegistryEntry(
        lib="nlpaug",
        factory=_wrap(
            lambda **kw: naw.AntonymAug(aug_src="wordnet", **kw), name="AntonymAug"
        ),
    ),
    "nlpaug/word/TfIdfAug": RegistryEntry(lib="nlpaug", factory=_make_tfidf),
    # TextAttack recipes
    "textattack/CharSwapAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(CharSwapAugmenter, returns_list=True, name="CharSwapAugmenter"),
    ),
    "textattack/DeletionAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(DeletionAugmenter, returns_list=True, name="DeletionAugmenter"),
    ),
    "textattack/SwapAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(SwapAugmenter, returns_list=True, name="SwapAugmenter"),
    ),
    "textattack/SynonymInsertionAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(
            SynonymInsertionAugmenter,
            returns_list=True,
            name="SynonymInsertionAugmenter",
        ),
    ),
    "textattack/EasyDataAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(EasyDataAugmenter, returns_list=True, name="EasyDataAugmenter"),
    ),
    "textattack/CheckListAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(CheckListAugmenter, returns_list=True, name="CheckListAugmenter"),
    ),
    "textattack/WordNetAugmenter": RegistryEntry(
        lib="textattack",
        factory=_wrap(WordNetAugmenter, returns_list=True, name="WordNetAugmenter"),
    ),
}

ALL_METHODS: list[str] = list(REGISTRY.keys())
NLPAUG_METHODS: list[str] = [
    name for name, entry in REGISTRY.items() if entry.lib == "nlpaug"
]
TEXTATTACK_METHODS: list[str] = [
    name for name, entry in REGISTRY.items() if entry.lib == "textattack"
]
