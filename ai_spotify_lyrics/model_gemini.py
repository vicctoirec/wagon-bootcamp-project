import pandas as pd

from ai_spotify_lyrics.params import DATA_CSV_17k
import os

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage


# Create a dataframe
df = pd.read_csv(DATA_CSV_17k)

# Get lyrics from dataframe
@tool
def get_lyrics(artist_name : str) -> str:
    """ Get song titles and lyrics of a specific artist's name.
    Use the artist name in the query as artist_name """
    songs = df[df['artist'].isin([artist_name])]
    if songs.empty:
        return f"No songs found for this {artist}."

    results = []
    for _, row in songs.iterrows():
        results.append(f"Title: {row['track_title_clean']}\nLyrics: {row['lyrics_clean']}\n")
    return "\n".join(results)

# Prompt Gemini model
def model_gemini(artist):

    ### Instantiate Gemini model ###
    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    ### Instantiate variables ###
    # Tools
    tools = [get_lyrics]

    # Prompt
    system_prompt = """
        With the name of an artist as an input, and the lyrics of their songs from the get_lyrics tool,
        you are tasked to summarize the top 5 themes in their lyrics.

        Below is an example of the format to use for the response, please keep the same format:

            Here are the 5 main themes in Adele's songs:

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

            4. Longing and Yearning
                A strong sense of longing for a lost love or a past relationship is a recurring theme in her music.
                “When we were young, the world was so much brighter” (When We Were Young)
                "Don't forget me, I beg, I remember you said, Sometimes it lasts in love, but sometimes it hurts instead" (Someone Like You)
                “Oh, how the time flies, as we get older” (To Be Loved)

            5. Self-Reflection and Growth
                Adele's songs often involve introspection and a journey of self-discovery and personal growth.
                “I’m not the girl I used to be” (Million Years Ago)
                “I’m trying to find myself” (Send My Love (To Your New Lover))
                “I’ve changed my mind, I’ll live and learn” (Set Fire to the Rain)

            These themes combine to create a rich tapestry of emotions, exploring the complexities of love, loss, and personal growth.
    """

    ### Create agent
    agent = create_react_agent(model, tools, prompt=system_prompt)

    # Input query
    query = f"Summarize the top 5 themes of {artist} lyrics, explain them in one line. For each theme quote 3 different lyrics and put the name of the song in parentheses"

    # Get response
    response = agent.invoke({"messages": [HumanMessage(content=query)]})

    return print(response["messages"][-1].content)
