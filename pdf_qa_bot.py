from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class PDFChatbot():
    def __init__(self):
        self.llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vector_store = InMemoryVectorStore(self.embeddings)

        # Define prompt for question-answering
        self.prompt = hub.pull("rlm/rag-prompt")

        # Compile application and test
        self.graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        self.graph_builder.add_edge(START, "retrieve")
        self.graph = self.graph_builder.compile()

    def load_data(self, pdf_path):
        # Load and chunk contents of the blog
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # Index chunks
        _ = self.vector_store.add_documents(documents=all_splits)


    # Define application steps
    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def ask_question(self, question: str):
        # state = self.graph.run({"question": question})
        # return state["answer"]
        result = self.graph.invoke({"question": question})
        return result
