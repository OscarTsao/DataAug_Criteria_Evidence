"""Utilities to fit and persist TF-IDF resources for augmentation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass
class TfidfResource:
    """Container holding a fitted TF-IDF vectorizer and its storage path."""

    vectorizer: TfidfVectorizer
    path: Path
    fitted: bool
    build_time_sec: float | None = None


def _prepare_texts(texts: Iterable[str]) -> list[str]:
    prepared: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if stripped:
            prepared.append(stripped)
    if not prepared:
        raise ValueError("No non-empty texts provided for TF-IDF fitting")
    return prepared


def load_or_fit_tfidf(
    train_texts: Iterable[str] | Sequence[str],
    model_path: str | Path,
    *,
    max_features: int = 40000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfResource:
    """
    Load an existing TF-IDF vectorizer or fit a new one on the provided texts.

    Args:
        train_texts: Iterable of training evidence texts.
        model_path: File path for the cached model (joblib format).
        max_features: Maximum vocabulary size.
        ngram_range: N-gram range for vectoriser.

    Returns:
        TfidfResource containing the fitted vectoriser and path.
    """
    start = time.time()
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with path.open("rb") as handle:
            vectorizer = joblib.load(handle)
        return TfidfResource(
            vectorizer=vectorizer, path=path, fitted=False, build_time_sec=None
        )

    texts = _prepare_texts(train_texts)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    vectorizer.fit(texts)

    with path.open("wb") as handle:
        joblib.dump(vectorizer, handle)

    return TfidfResource(
        vectorizer=vectorizer,
        path=path,
        fitted=True,
        build_time_sec=time.time() - start,
    )
