import pytest
import pandas as pd
from zeroshots_function.zeroshot_pipeline import compute_scores, get_zeroshot_score


def test_zeroshot_score_is_float():
    """Test if the zero-shot score is a float and within the expected range."""

    lyrics = "I love dancing under the sun and the blue sky."
    user_input = "Je pars en vacances au soleil"

    score = get_zeroshot_score(lyrics, user_input)

    assert isinstance(score, float), "Le score doit être un float"
    assert 0 <= score <= 1, "Le score doit être entre 0 et 1"


def test_compute_scores_returns_df():
    """Test if compute_scores returns a DataFrame with the expected structure."""

    data = {
        "artist": ["Artist A", "Artist B", "Artist C"],
        "track_title_clean": ["Song A", "Song B", "Song C"],
        "lyrics_clean": [
            "I walk under the sun every morning.",
            "I cry alone in the rain.",
            "Freedom and open roads are my life."
        ]
    }
    df = pd.DataFrame(data)

    user_input = "je veux écouter une chanson joyeuse et ensoleillée"
    result = compute_scores(df, user_input, threshold=0.0, top_n=2)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty, "Le DataFrame retourné ne doit pas être vide"
    assert "artist" in result.columns
    assert "track_title_clean" in result.columns
    assert all(result["match_score"].between(0, 1))
