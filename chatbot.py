# Import relevant functionality
from typing import List, TypedDict

from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent
import os

os.environ["GROQ_API_KEY"] = "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"

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class ChatBot:
    def __init__(self):
        # Create the agent
        self.memory = MemorySaver()
        # self.model = init_chat_model("mistral-large-latest", model_provider="mistralai")
        # self.model = init_chat_model(model="llama3-8b-8192", model_provider="ollama")
        self.model = init_chat_model("llama3-8b-8192", model_provider="groq")
        # self.model = init_chat_model("grok-2", model_provider="xai")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        self.vector_store = InMemoryVectorStore(self.embeddings)

        self.load_data("data/administratoreecaf3b490e2d43d2e3b50c0c068b5d7.pdf")
        # Define prompt for question-answering
        self.prompt = hub.pull("rlm/rag-prompt")

        # Step 2: Execute the retrieval.
        self.tools = ToolNode([self.retrieve])

        self.agent_executor = create_react_agent(
            self.model, self.tools, checkpointer=self.memory
        )

    def load_data(self, pdf_path):
        # Load and chunk contents of the blog
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(docs)

        # Index chunks
        _ = self.vector_store.add_documents(documents=all_splits)

    # Define application steps
    def retrieve(self, state: State):
        """Retrieve information related to a query."""
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def ask(self, message: str, thread_id: str = "abc123"):
        # Use the agent
        config = {"configurable": {"thread_id": thread_id}}
        response = self.agent_executor.invoke(
            {"messages": [HumanMessage(content=message)]}, config
        )

        return response["messages"][-1].content
