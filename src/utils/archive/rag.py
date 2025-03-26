import json
import os

from pyprojroot.here import here
from typing import List, TypedDict, Dict
from sentence_transformers import CrossEncoder


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain import hub
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate

from langgraph.graph import START, StateGraph

from IPython.display import Image, display

zotero_vectordb = here() / "data/zotero_library"

#%%

template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always cite your sources using the relevant text's author, date, and title from the included metadata.

    {context}

    Question: {question}

    Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")

# Define state for application

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def rerank(query: str, query_results: Dict, top_n: int) -> Dict:
    """Rerank results so most relevant documents are first."""
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # get ranks
    scores = reranker.predict([(query, doc)
                                for doc in query_results['documents'][0]])
    # find the indices of the documents in descending order of relevance
    sorted_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True)

    # reorder documents and metadata
    reranked_docs = [query_results['documents'][0][i]
                        for i in sorted_indices]
    reranked_metadata = [query_results['metadatas'][0][i]
                            for i in sorted_indices]

    return reranked_docs[:top_n], reranked_metadata[:top_n]

# Define application steps+
def retrieve(state: State):
    # get top 50 documents, but rerank and accept the top 10
    retrieved_docs = collection.query(
        query_texts=[state["question"]], n_results=50)
    reranked_docs, reranked_metadata = rerank(query=state["question"],
                                                query_results=retrieved_docs,
                                                top_n=10)

    output = {"documents": reranked_docs,
                "metadatas": reranked_metadata}
    return {"context": output}


def generate(state: State):
    metadata = state['context']['metadatas']
    docs = state["context"]['documents']

    docs_content = "\n\n".join(
        f"Document {idx} Metadata: {json.dumps(meta)}"
        f"\n\nDocument {idx}: {doc}"
        for idx, (meta, doc) in enumerate(zip(metadata, docs))
    )

    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


class CustomEmbeddings(Embeddings):
    """Custom embeddings for langchain-chroma if not using chromadb directly."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformerEmbeddingFunction(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.model.embed_with_retries(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]:
        return self.model.embed_with_retries([query])[0].tolist()
    
# %%

if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.histories.chat_with_agent import chat_with_docs
    from langchain_chroma import Chroma
    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=OPENAI_KEY)

    # build db, collection, and embedding function
    chroma_model = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2")  # "sentence-transformers/msmarco-distilbert-cos-v5"
    client = chromadb.PersistentClient(path=str(zotero_vectordb))
    collection = client.get_collection(name="readings_library",
                                       embedding_function=chroma_model)

    #%%

    # embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # vdb = Chroma(
    #     client=client,
    #     collection_name="readings_library",
    #     embedding_function=embedding_model,
    # )

    #%%
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # display(Image(graph.get_graph().draw_mermaid_png()))

    chat_with_docs(graph)

# %%
