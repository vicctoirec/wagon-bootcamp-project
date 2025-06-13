import re
from transformers import pipeline
import textwrap



def preprocess_lyrics(lyrics):
    """Preprocess lyrics by cleaning and normalizing the text."""
    lyrics = lyrics.lower()
    lyrics = re.sub(r'[^\w\s]', '', lyrics)
    return lyrics


def chunk_text(text, max_words=250):
    """Chunk text into smaller parts with a maximum number of words."""

    if not isinstance(text, str):
        return []
    return textwrap.wrap(text, max_words)


def get_zeroshot_score(lyrics, user_input):
    """Compute zero-shot classification score for lyrics against a user label."""
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    chunks = chunk_text(lyrics)
    scores = []
    for chunk in chunks:
        try:
            result = classifier(chunk, candidate_labels=[user_input], multi_label=True)
            scores.append(result['scores'][0])
        except:
            continue
    return max(scores) if scores else 0


def compute_scores(df, user_input, threshold=0.8, top_n=10):
    """
    Compute zero-shot scores for lyrics against user input.

    Args:
        df (DataFrame): DataFrame containing lyrics and metadata.
        user_input (str): User input to compare against lyrics.
        threshold (float): Minimum score to consider a match.
        top_n (int): Number of top matches to return.
    Returns:
        DataFrame: Filtered DataFrame with top matches.
    """


    # Drop the rows with NaN values in 'lyrics_clean' first
    df = df.dropna(subset=["lyrics_clean"]).reset_index(drop=True)


    # Clean the lyrics text
    df['lyrics_clean'] = df['lyrics_clean'].apply(preprocess_lyrics)

    # Compute zero-shot scores
    df["match_score"] = df["lyrics_clean"].apply(lambda x: get_zeroshot_score(x, user_input))
    df_filtered = df[df["match_score"] >= threshold].copy()

    df_top = df_filtered.sort_values(by="match_score", ascending=False).head(top_n)

    return df_top[["artist", "track_title_clean", "match_score"]].reset_index(drop=True)
