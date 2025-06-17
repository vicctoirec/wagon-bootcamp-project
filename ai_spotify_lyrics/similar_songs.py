import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage


class SimilarSongs:

    def __init__(self, data: str, pipeline:str, model: str):

        # Chargement deu pipeline preproc et du model KNN
        self.pipe = joblib.load(pipeline)
        self.model_knn = joblib.load(model)

        # Initialisation de l'agent
        self.chat_agent = self.setup_chat_agent()

        # Charement du dataset
        self.df = pd.read_pickle(data)
        self.artists = sorted(self.df["artist"].unique().tolist())
        self.songs = sorted(self.df["title_cleaned"].unique().tolist())
        self.df_sorted = self.df.sort_values(by=['title_cleaned', 'artist'])


    def get_artists(self) -> list:
        return self.artists


    def get_songs(self, by: bool=False, as_option_dict: bool=False):

        if as_option_dict:
            if by:
                return self.df_sorted.apply(lambda row: f"{row['title_cleaned']} by {row['artist']}", axis=1).to_dict()
            return self.df['title_cleaned'].to_dict()
        elif by:
            return self.df_sorted.apply(lambda row: f"{row['title_cleaned']} by {row['artist']}", axis=1).tolist()
        return self.songs


    def get_songs_by_artist(self, artist_name: str, as_option_dict: bool=False):
        if as_option_dict:
            return self.df.apply(lambda row: row['title_cleaned'], axis=1).to_dict()
        return sorted(self.df[self.df['artist'] == artist_name]['title_cleaned'].unique().tolist())


    def setup_chat_agent(self, model: str="gemini-2.0-flash", model_provider:str ="google_genai"):

        model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        tools = [self.get_lyrics_top_songs]
        system_prompt = """
            With the name of an artist and a song title as an input use the tool to find the most similar songs based on beats and lyrics.
            Then, analyze the lyrics of all songs and explain why the lyrics are similar, in 5-10 lines max. Be specific, you can quote songs if necessary.
            Make sure that every time you mention a song, you also mention the artist."""

        return create_react_agent(model, tools, prompt=system_prompt)


    def find_song(self, song_name, artist_name) -> pd.DataFrame:
        song_idx = self.df.index[(self.df['title_cleaned'] == song_name) & (self.df['artist'] == artist_name)].tolist()[0]

        song_transformed = self.pipe.transform(self.df.iloc[[song_idx]])

        distances, indices = self.model_knn.kneighbors(song_transformed, n_neighbors=101)

        # Exclude the first index if it is the song itself
        neighbor_indices = indices[0][1:]

        # Retrieve metadata for neighbors
        neighbors_df = self.df.iloc[neighbor_indices][['artist', 'title_cleaned', 'text', 'embedding']]

        # Add searched song
        searched_song_df = self.df[(self.df['title_cleaned'] == song_name) & (self.df['artist'] == artist_name)][['artist', 'title_cleaned', 'embedding']]
        neighbors_df = pd.concat([searched_song_df, neighbors_df], axis=0)

        return neighbors_df


    def get_top_similar_songs(self, song: str, artist: str) -> list:

        neighbors_df = self.find_song(song, artist)

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

        # top 3 songs similaires
        top_songs = label_songs.sort_values(by='similarity', ascending=False).head(3)

        return [
            (row['artist'],row['title_cleaned'])
            for _, row in top_songs.iterrows()
        ]

    # Get lyrics from dataframe
    def get_lyrics_top_songs(self, song_title : str, artist_name : str) -> str:
        """ Input a song and artist and get the top 5 songs similar in beat and lyrics.
        Use the artist name and song title in the query as artist_name and song_title """

        similar = self.get_top_similar_songs(song_title, artist_name)
        all_songs = [(song_title,artist_name)] + similar

        # Récupérer les paroles
        results = []
        for title, artist in all_songs:
            try:
                row = self.df[
                    (self.df['title_cleaned'] == title) & (self.df['artist'] == artist)
                ].iloc[0]
                lyrics = row['text']
                results.append(f"Artist: {artist}\nTitle: {title}\nLyrics: {lyrics}\n")
            except IndexError:
                results.append(f"Paroles non trouvées pour: {title} - {artist}\n")

        return "\n".join(results)

    # Prompt agent
    def get_lyrics_explained(self, song_title: str, artist_name: str):

        # Input query
        query = f"Find the top 3 similar songs to {artist_name}'s {song_title}. Summarize why the lyrics are similar"

        # Get response
        response = self.chat_agent.invoke({"messages": [HumanMessage(content=query)]})

        return response["messages"][-1].content
