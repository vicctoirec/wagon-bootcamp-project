from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from ai_spotify_lyrics.dummy_model import initialize_dummy_model
from ai_spotify_lyrics.themes_agent import ThemesAgent, FALLBACK_ARTIST, FALLBACK_THEMES
from ai_spotify_lyrics.lyrics_matching import LyricsMatching
from ai_spotify_lyrics.similar_songs import SimilarSongs
from ai_spotify_lyrics.enrich_agent import EnrichAgent
from ai_spotify_lyrics.zeroshot_pipeline import ZeroShotLyrics
from ai_spotify_lyrics.params import *
import time

app = FastAPI()

app.state.dummy_model = initialize_dummy_model()
app.state.lyricsmatching = LyricsMatching(LYRICS_MATCHING_MODEL_PATH, LYRICS_MATCHING_MODEL_NAME, DATA_CSV_17k_EMBED, DATA_CSV_17k)
app.state.zeroshot = ZeroShotLyrics(ZEROSHOT_MODEL_PATH, ZEROSHOT_MODEL_NAME)
app.state.similar_songs = SimilarSongs(DATA_9K, SIMILAR_SONGS_PIPELINE, SIMILAR_SONGS_MODEL)
app.state.enrich_agent = EnrichAgent()
app.state.themes_agent = ThemesAgent(DATA_CSV_17k)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API AI SPOTIFY LYRICS is running!"
    }


@app.get("/artists")
def artists():
    """ Get a list of available artists """
    results = app.state.themes_agent.get_artists()
    return {
        'results': results
    }

@app.get("/songs")
def songs():
    """ Get a list of available songs """
    results = app.state.themes_agent.get_songs()
    return {
        'results': results
    }

# Endpoint for https://your-domain.com/predict?
@app.get("/predict")
def get_predict(input: str):
    # input is a text prompt
    # For a dummy version, returns random songs
    prediction = app.state.dummy_model.predict(input)
    return {
        'prediction': prediction,
        'inputs': {
            'input': input,
        }
    }


@app.get("/predict-artist-themes")
def get_predict_themes(input: str):
    # input is an artist name
    # For gemini model returns str
    prediction = app.state.themes_agent.get_themes(input)

    retries = 0
    max_retries = 2
    while not prediction and retries < max_retries:
        time.sleep(1)
        prediction = app.state.themes_agent.get_themes(input)

    if not prediction and input == FALLBACK_ARTIST:
        prediction = FALLBACK_THEMES

    return {
        'prediction': prediction,
        'inputs': {
            'input': input,
        }
    }


# ---------- FEATURE 2 endpoint 1 : enrich the prompt --------------------------
@app.get("/enrich_prompt")
def enrich_prompt(user_input: str):

    """
    Endpoint qui retourne une version enrichie du user_input.
    Si l'utilisateur clique sur 'regénérer", le front-end rappelle la même route
    mais avec ?rerun=True pour regénérer une variante
    """

    enriched = app.state.enrich_agent.get_enriched_mood(user_input)

    return {
        "enriched_input": enriched,
        "original_input": user_input
    }

# ---------- FEATURE 2 endpoint 2 : get playlist -------------------------------
@app.get("/predict-mood-songs")
def get_predict_mood_songs(enriched_input: str, k_recall: int = 40, k_final : int = 10):
    """
    Exporte un top-10 de titres basé sur `enriched_input`
    SBERT recall + raffinage Zero-Shot.
    """

    df = app.state.lyricsmatching.refine_top_k(
        enriched_input=enriched_input,
        zeroshot_model=app.state.zeroshot,
        k_recall=k_recall,
        k_final=k_final,
        verbose=False
    )

    return {
        "used_prompt": enriched_input,
        "prediction": df.to_dict(orient="records")
    }

# ---------- FEATURE 3 endpoint 1 : similar songs ------------------------------
@app.get("/predict-similar-songs")
def get_predict_similar_songs(input_song: str, input_artist: str):
    """
    Endpoint qui retourne les chansons similaires au format "title by artist".
    """
    try:
        prediction = app.state.similar_songs.get_top_similar_songs(input_song, input_artist)
    except ValueError:
        return {
            'error': f"Song '{input_song}' by '{input_artist}' not found."
        }

    return {
        'prediction': prediction,
        'inputs': {
            'input_song': input_song,
            'input_artist': input_artist
        }
    }

# ---------- FEATURE 3 endpoint 2 : explain similarities -----------------------
@app.get("/explain-similar-lyrics")
def get_predict_similar_lyrics(input_song: str, input_artist: str):
    """
    Endpoint qui explique en quoi les lyrics des chansons les plus proches sont similaires".
    """
    try:
        explanation = app.state.similar_songs.get_lyrics_explained(input_song, input_artist)
    except ValueError:
        return {
            'error': f"Song '{input_song}' by '{input_artist}' not found."
        }

    return {
        'prediction': explanation,
        'inputs': {
            'input_song': input_song,
            'input_artist': input_artist
        }
    }



@app.get("/similar-songs/artists")
def similar_songs_artists():
    """ Get a list of available artists """
    results = app.state.similar_songs.get_artists()
    return {
        'results': results
    }

@app.get("/similar-songs/songs-by-artist")
def similar_songs_by_artist(input: str):
    """ Get a list of available songs by artist input """
    results = app.state.similar_songs.get_songs_by_artist(input)
    return {
        'results': results
    }
