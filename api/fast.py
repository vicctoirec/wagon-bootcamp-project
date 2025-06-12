# TODO: Import your package, replace this by explicit imports of what you need
# from ai_spotify_lyrics.main import predict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_spotify_lyrics.model import initialize_dummy_model
from ai_spotify_lyrics.model_gemini import get_artists, get_songs, model_gemini

from zeroshots_function.lyrics_matching import get_top_k

app = FastAPI()
app.state.model = initialize_dummy_model()

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
    results = get_artists()
    return {
        'results': results
    }

@app.get("/songs")
def songs():
    """ Get a list of available songs """
    results = get_songs()
    return {
        'results': results
    }

# Endpoint for https://your-domain.com/predict?
@app.get("/predict")
def get_predict(input: str):
    # input is a text prompt
    # For a dummy version, returns random songs
    prediction = app.state.model.predict(input)
    return {
        'prediction': prediction,
        'inputs': {
            'input': input,
        }
    }


@app.get("/predict-artist-themes")
def get_predict_themes(input: str):
    # input is an artist name
    # For a dummy version, returns fixed themes
    # For gemini model returns str
    prediction = model_gemini(input)
    # prediction = ['journey', 'nature', 'universe', 'stars', 'god']
    return {
        'prediction': prediction,
        'inputs': {
            'input': input,
        }
    }

@app.get("/predict-mood-songs")
def get_predict_mood_songs(input: str):
    # input is a prompt
    # For a dummy version, returns fixes themes
    prediction = get_top_k(input, 5)
    return {
        'prediction': prediction.to_dict(),
        'inputs': {
            'input': input,
        }
    }

@app.get("/predict-similar-songs")
def get_predict_similar_songs(input: str):
    # input is a song
    # For a dummy version, returns fixes themes
    prediction = app.state.model.predict(input)
    return {
        'prediction': prediction,
        'inputs': {
            'input': input,
        }
    }
