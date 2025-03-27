
import chromadb
import os
from pathlib import Path

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from dotenv import load_dotenv


def get_vdb_index(db_path: str|Path, db_name: str):
    # get client and vdb
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=db_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # setup index
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                                embed_model=embed_model,
                                                storage_context=storage_context)

def get_model(model_name: str, temperature: float = 0, **kwargs):
    """
    Get a model from the environment variables.
    """
    # Load the environment variables
    load_dotenv()

    # Get the model from the environment variables
    if "claude" in model_name:
        from langchain_anthropic import ChatAnthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=temperature, **kwargs)
    elif "gpt" in model_name:
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, temperature=temperature, **kwargs)
    elif 'llama' in model_name:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=temperature, **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")