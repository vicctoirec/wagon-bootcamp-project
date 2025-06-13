from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage

# Prompt Gemini model
def prompt_gemini(user_input):

    ### Instantiate Gemini model ###
    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    ### Import tools --> empty here
    tools = []

    ### Create agent
    agent = create_react_agent(model, tools)

    # Input query
    query = f"Give me a mood in 2 sentences based on the following: {user_input}"

    # Get response
    response = agent.invoke({"messages": [HumanMessage(content=query)]})

    return str(response["messages"][-1].content)
