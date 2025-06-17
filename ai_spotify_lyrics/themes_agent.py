import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage


class ThemesAgent:

    def __init__(self, data: pd.DataFrame, model: str="gemini-2.0-flash", model_provider: str="google_genai"):

        # read CSV
        self.df = pd.read_csv(data)

        # lists uniques artists and songs
        self.artists = sorted(self.df["artist"].unique().tolist())
        self.songs = sorted(self.df["track_title_clean"].unique().tolist())

        # Instantiate Gemini model
        model = init_chat_model(model, model_provider=model_provider)
        ### Instantiate variables ###
        # Tools
        tools = [self.get_lyrics]

        # Prompt
        system_prompt = """
            With the name of an artist as an input, and the lyrics of their songs from the get_lyrics tool,
            you are tasked to summarize the top 3 themes in their lyrics.

            Below is an example of the format to use for the response, please keep the same format:
                **Theme 1** : one line explanation
                - Quote (song 1)
                - Quote (song 2)
                - Quote (song 3)
        """

        ### Create agent
        self.agent = create_react_agent(model, tools, prompt=system_prompt)


    # Get a sorted list of available artists
    def get_artists(self) -> list:
        """ Get a sorted list of available artists """
        return self.artists


    # Get a sorted list of available songs
    def get_songs(self) -> list:
        """ Get a sorted list of available songs """
        return self.songs


    # Get lyrics from dataframe
    @tool
    def get_lyrics(self, artist_name : str) -> str:
        """ Get song titles and lyrics of a specific artist's name.
        Use the artist name in the query as artist_name """

        songs = self.df[self.df['artist'].isin([artist_name])]

        if songs.empty:
            f"No songs found for this {artist_name}."

        results = []
        for _, row in songs.iterrows():
            results.append(f"Title: {row['track_title_clean']}\nLyrics: {row['lyrics_clean']}\n")
        return "\n".join(results)


    # Prompt Gemini model
    def get_themes(self, artist_name: str) -> str:

        # Input query
        query = f"Summarize the top 3 themes of {artist_name} lyrics."

        # Get response
        response = self.agent.invoke({"messages": [HumanMessage(content=query)]})

        return response["messages"][-1].content
