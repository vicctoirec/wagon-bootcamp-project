import pandas as pd

from ai_spotify_lyrics.params import DATA_CSV_9k

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

import torch
import pandas as pd
import torch.nn.functional as F
import ast

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

df = pd.read_csv(DATA_CSV_9k)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

def pipeline():
    # Categorize columns
    categ_columns = ['genre']
    num_columns = ['popularity', 'year', 'danceability', 'energy',
        'key', 'loudness', 'mode', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
        'time_signature']

    # Call encoders and scalers
    ohe = OneHotEncoder(sparse_output=False)
    minmax = MinMaxScaler()

    # Make encoding pipeline
    pipe = make_column_transformer(
        (ohe, categ_columns),
        (minmax, num_columns),
        remainder='drop'
    ).set_output(transform="pandas")

    return pipe

def preprocess(df):
    pipe = pipeline()
    X_transformed = pipe.fit_transform(df)
    return X_transformed, pipe

def knn_model(df):
    X_transformed, pipe = preprocess(df)
    model_knn = NearestNeighbors(n_neighbors=100, algorithm='auto', metric='euclidean')
    model_knn.fit(X_transformed)
    return model_knn, pipe

model_knn, pipe = knn_model(df)

def find_song(song_name, artist_name, df, model_knn, pipe):
    song_idx = df.index[(df['title_cleaned'] == song_name) & (df['artist'] == artist_name)].tolist()[0]

    song_transformed = pipe.transform(df.iloc[[song_idx]])

    distances, indices = model_knn.kneighbors(song_transformed, n_neighbors=101)

    # Exclude the first index if it is the song itself
    neighbor_indices = indices[0][1:]

    # Retrieve metadata for neighbors
    neighbors_df = df.iloc[neighbor_indices][['artist', 'title_cleaned', 'text', 'embedding']]

    # Add searched song
    searched_song_df = df[(df['title_cleaned'] == song_name) & (df['artist'] == artist_name)][['artist', 'title_cleaned', 'embedding']]
    neighbors_df = pd.concat([searched_song_df, neighbors_df], axis=0)

    return neighbors_df

def get_top_similar_songs(df, song, artist, top_n=3):

    neighbors_df = find_song(song, artist, df, model_knn, pipe)

    # recupérer la chanson
    input_song = neighbors_df[(neighbors_df['title_cleaned'] == song) & (neighbors_df['artist'] == artist)]

    # si chanson non trouvée msg d'erreur
    if input_song.empty:
        raise ValueError("Song not found.")

    # récupérer le cluster de la chanson et son embedding
    ## Recupérer la chanson et l'embedding
    song_embedding = torch.tensor(input_song.iloc[0]['embedding'])

    # récupérer les chansons du mm cluster (sauf la chanson input)
    ## Récuperer le neighbors_df short de la fonction get_song
    label_songs = neighbors_df[~((neighbors_df['title_cleaned'] == song) & (neighbors_df['artist'] == artist))]

    # similarité
    def compute_similarity(row):
        emb = torch.tensor(row['embedding'])
        return F.cosine_similarity(song_embedding, emb, dim=0).item()

    label_songs['similarity'] = label_songs.apply(compute_similarity, axis=1)

    # top n songs similaires
    top_songs = label_songs.sort_values(by='similarity', ascending=False).head(top_n)

    return top_songs

# Get lyrics from dataframe
def get_lyrics_top_songs(song_title : str, artist_name : str) -> str:
    """ Input a song and artist and get the top 5 songs similar in beat and lyrics.
    Use the artist name and song title in the query as artist_name and song_title """

    top_songs = get_top_similar_songs(df, song_title, artist_name, top_n=3)[['artist', 'title_cleaned', 'text']]
    searched_song = df[(df['title_cleaned'] == song_title) & (df['artist'] == artist_name)][['artist', 'title_cleaned', 'text']]

    songs = pd.concat([searched_song, top_songs], axis=0)

    if songs.empty:
        return f"No songs found for this {artist_name} and {song_title}."

    results = []
    for _, row in songs.iterrows():
        results.append(f"Artist: {row['artist']}\nTitle: {row['title_cleaned']}\nLyrics: {row['text']}\n")
    return "\n".join(results)

# Prompt Gemini model
def model_gemini(song_title, artist_name):

    ### Instantiate Gemini model ###
    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    ### Instantiate variables ###
    # Tools
    tools = [get_lyrics_top_songs]

    # Prompt
    system_prompt = """
        With the name of an artist and a song title as an input use the tool to find the 3 most similar songs based on beats and lyrics.
        Then, analyze the lyrics of all songs and explain why the lyrics are similar, in 5-10 lines max. Be specific.
        Make sure that every time you mention a song, you also mention the artist. """

    ### Create agent
    agent = create_react_agent(model, tools, prompt=system_prompt)

    # Input query
    query = f"Find the top 3 similar songs to {artist_name}'s {song_title}. Summarize why the lyrics are similar"

    # Get response
    response = agent.invoke({"messages": [HumanMessage(content=query)]})

    return response["messages"][-1].content
