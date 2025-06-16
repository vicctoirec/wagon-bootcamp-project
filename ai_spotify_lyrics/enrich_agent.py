from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain.schema import HumanMessage



class EnrichAgent:

    def __init__(self, model: str="gemini-2.0-flash", model_provider: str="google_genai", tools: list=[]):

        model = init_chat_model(model, model_provider=model_provider)
        self.agent = create_react_agent(model, tools)


    def get_enriched_mood(self, user_input: str):

        # Input query
        query = f"Give me a mood in 2 sentences based on the following: {user_input}"

        # Get response
        response = self.agent.invoke({"messages": [HumanMessage(content=query)]})

        return str(response["messages"][-1].content)
