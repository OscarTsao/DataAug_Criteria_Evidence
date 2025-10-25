"""Tests for TF-IDF resource caching used by augmentation."""

from pathlib import Path

from psy_agents_noaug.augmentation.tfidf_cache import load_or_fit_tfidf


def test_tfidf_fit_and_reload(tmp_path):
    texts = [
        "Patient reports persistent sadness.",
        "No evidence of psychosis was observed.",
        "Sleep disturbances noted over two weeks.",
    ]
    model_path = Path(tmp_path) / "tfidf.pkl"

    resource = load_or_fit_tfidf(texts, model_path)
    assert resource.fitted is True
    assert resource.path.exists()

    resource_cached = load_or_fit_tfidf(texts, model_path)
    assert resource_cached.fitted is False
    assert resource_cached.path == resource.path

    assert resource.path.is_file()
    assert resource_cached.path.is_file()
