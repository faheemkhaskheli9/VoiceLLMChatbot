# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import api_keys

class InternetChatBot():

    def __init__(self):
        # Create the agent
        self.memory = MemorySaver()
        self.model = init_chat_model("mistral-large-latest", model_provider="mistralai")
        self.search = TavilySearchResults(max_results=2)
        self.tools = [self.search]

        self.agent_executor = create_react_agent(self.model, self.tools, checkpointer=self.memory)


    def ask(self, message: str, thread_id: str = "abc123"):
        # Use the agent
        config = {"configurable": {"thread_id": thread_id}}
        response = self.agent_executor.invoke(
            {"messages": [HumanMessage(content=message)]},
            config)

        return response["messages"][-1].content
