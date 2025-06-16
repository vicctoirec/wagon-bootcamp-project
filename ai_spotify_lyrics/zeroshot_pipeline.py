import re
from pathlib import Path
import textwrap
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer



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


class ZeroShotLyrics():

    def __init__(self, model_path: str, model_name: str):

        if Path(model_path).exists():
            print('ZeroShot pipeline from local')
            self.pipeline = pipeline("zero-shot-classification", model=model_path, tokenizer=model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.pipeline = pipeline("zero-shot-classification", model=model_name)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)


    def get_zeroshot_score(self, lyrics, user_input):
        """Compute zero-shot classification score for lyrics against a user label."""
        classifier = self.pipeline
        chunks = chunk_text(lyrics)
        scores = []
        for chunk in chunks:
            try:
                result = classifier(chunk, candidate_labels=[user_input], multi_label=True)
                scores.append(result['scores'][0])
            except:
                continue
        return max(scores) if scores else 0


    def compute_scores(self, df, user_input, threshold=0.8, top_n=10):
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
        df["match_score"] = df["lyrics_clean"].apply(lambda x: self.get_zeroshot_score(x, user_input))
        df_filtered = df[df["match_score"] >= threshold].copy()

        df_top = df_filtered.sort_values(by="match_score", ascending=False).head(top_n)

        return df_top[["artist", "track_title_clean", "match_score"]].reset_index(drop=True)
