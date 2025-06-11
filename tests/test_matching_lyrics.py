# tests/test_matching_lyrics.py
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
import pytest

from zeroshots_function.lyrics_matching import get_top_k

# ---------- utilitaire commun ----------
def _setup_mocks(tmp_path, monkeypatch, n_rows=3):
    fake_meta = tmp_path / "meta.csv"
    pd.DataFrame({
        "artist": [f"A{i}" for i in range(n_rows)],
        "track_title_clean": [f"Song{i}" for i in range(n_rows)],
        "lyrics_clean": ["sunshine"]*n_rows
    }).to_csv(fake_meta, index=False)

    fake_emb = np.random.randn(n_rows, 768).astype("float32")
    fake_emb_df = pd.DataFrame(fake_emb,
                               columns=[f"embedding_{i}" for i in range(768)])

    # Patch chemin CSV + embeddings
    monkeypatch.setattr("zeroshots_function.lyrics_matching.RAW_CSV", Path(fake_meta))
    monkeypatch.setattr("zeroshots_function.lyrics_matching.ensure_embeddings", lambda: fake_emb_df)

    # Patch encode → vecteur 768 flottant
    patcher = patch("zeroshots_function.lyrics_matching.SentenceTransformer")
    mock_model = patcher.start()
    mock_model.return_value.encode.return_value = np.random.randn(768).astype("float32")

    return patcher


# ---------- Test 1 : k > nb lignes ----------
def test_k_sup_rows_returns_max(tmp_path, monkeypatch):
    patcher = _setup_mocks(tmp_path, monkeypatch, n_rows=3)
    df = get_top_k("soleil", k=10)     # demande 10, n=3
    patcher.stop()
    assert len(df) == 3                # renvoie tout ce qui existe


# ---------- Test 2 : input vide ----------
def test_empty_input_returns_empty_df(tmp_path, monkeypatch):
    patcher = _setup_mocks(tmp_path, monkeypatch, n_rows=3)
    df = get_top_k("", k=5)            # input vide
    patcher.stop()
    assert df.empty                    # renvoie DataFrame vide
