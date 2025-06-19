import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage
from ai_spotify_lyrics.params import DATA_CSV_17k


FALLBACK_ARTIST = "ABBA"
FALLBACK_THEMES = """
    Here are the 3 main themes in ABBA songs:

    **Love and Relationships**
    ABBA's songs frequently explore the complexities of love, ranging from passionate romance to painful breakups.
    - "Honey, I was stronger then" (Waterloo)
    - "You and I were meant to be for each other" (I Do, I Do, I Do, I Do, I Do)
    - "Breaking up is never easy, I know" (The Winner Takes It All)

    Nostalgia and Memories
    Many ABBA songs reflect on past times, evoking a sense of longing and reminiscence.
    - "Do you remember when we kissed by the old oak tree?" (When I Kissed the Teacher)
    - "Those were the days, my friend, we thought they'd never end" (Fernando)
    - "I was so young then, I never thought of needing anyone" (Mamma Mia)

    Dancing and Celebration
    ABBA is known for their upbeat, danceable tracks that celebrate life and encourage listeners to enjoy the moment.
    - "You can dance, you can jive, having the time of your life" (Dancing Queen)
    - "See that girl, watch that scene, dig in the dancing queen" (Dancing Queen)
    - "Gimme, gimme, gimme a man after midnight" (Gimme! Gimme! Gimme! (A Man After Midnight))
    """


DATA = pd.read_csv(DATA_CSV_17k)

@tool(parse_docstring=True)
def get_lyrics(artist_name : str) -> str:
    """ Get song titles and lyrics of a specific artist's name.

    This function searches lyrics by an artist in a song's DataFrame.
    It returns a string containings titles of songs and the lyrics if
    any were found for the provide artist name, or 'No songs found for {artist_name}.".

    Args:
        artist_name: the artist to search for.

    Returns:
        A string where each song's title is prepended by 'Title: ' followed by the lyrics
        prepended by 'Lyrics: '. If no songs match the artist's name the function returns
        a string 'No songs found for {artist_name}.'
    """

    songs = DATA[DATA['artist'].isin([artist_name])]

    if songs.empty:
        f"No songs found for {artist_name}."

    results = []
    for _, row in songs.iterrows():
        results.append(f"Title: {row['track_title_clean']}\nLyrics: {row['lyrics_clean']}\n")
    return "\n".join(results)



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
        tools = [get_lyrics]

        # Prompt
        system_prompt = """
            With the name of an artist as an input, and the lyrics of their songs from the get_lyrics tool,
            you are tasked to summarize the top 3 themes in their lyrics. If the get_lyrics tool does not
            return lyrics for the artist return tool's output.

            Below is an example of the format to use for the response based on an example for one theme for Adele, please keep the same format:

            Here are the 3 main themes in Adele's songs:
                1. Heartbreak and Loss
                    Adele often sings about the intense pain and sorrow that come with the end of a relationship.
                    “Go easy on me, I was still a child” (Easy On Me)
                    “Baby, let the water wash away all our tears” (Water Under the Bridge)
                    “Never mind, I’ll find someone like you” (Someone Like You)
                2. Regret and Remorse
                    Many of her songs reflect on past mistakes and express regret over things said or done in relationships.
                    “Hello, can you hear me? I’m in California dreaming about who we used to be” (Hello)
                    “I should have treated you right” (Take It All)
                    “I regret the things I never said” (Remedy)
                3. Resilience and Moving On
                    Despite the pain, Adele's lyrics often show a strength and determination to overcome heartbreak and move forward.
                    “I’m gonna make it through” (Make You Feel My Love)
                    “We could have had it all, rolling in the deep” (Rolling in the Deep)
                    “I’ve gotta let go of us” (Love In The Dark)
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
    def get_lyrics(self, artist_name : str) -> str:
        """ Get song titles and lyrics of a specific artist's name.

        This function searches lyrics by an artist in a song's DataFrame.
        It returns a string containings titles of songs and the lyrics if
        any were found for the provide artist name, or 'No songs found for this {artist_name}.".

        Args:
            artist_name: the artist to search for.

        Returns:
            A string where each song's title is prepended by 'Title: ' followed by the lyrics
            prepended by 'Lyrics: '. If no songs match the artist's name the function returns
            a string 'No songs found for {artist_name}.'
        """

        songs = self.df[self.df['artist'].isin([artist_name])]

        if songs.empty:
            f"No songs found for {artist_name}."

        results = []
        for _, row in songs.iterrows():
            results.append(f"Title: {row['track_title_clean']}\nLyrics: {row['lyrics_clean']}\n")
        return "\n".join(results)


    # Prompt Gemini model
    def get_themes(self, artist_name: str) -> str:

        # Input query
        query = f"Summarize the top 3 themes of '{artist_name}' lyrics, explain them in one line. For each theme quote 3 different lyrics and put the name of the song in parentheses"

        # Get response
        response = self.agent.invoke({"messages": [HumanMessage(content=query)]})

        return response["messages"][-1].content
