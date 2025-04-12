
#%%
import os
from pathlib import Path
from typing import List

import nest_asyncio
nest_asyncio.apply()

from chromadb import PersistentClient
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex



#%%
def build_opik_tracer_langchain(workspace, project_name, thread_id=None, tags=None):
    """Build and return an OpikTracer instance."""
    import opik
    from opik import Opik
    from opik.integrations.langchain import OpikTracer
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")

    if not Opik().config.api_key:
        opik.configure(api_key=api_key, 
                    use_local=False,
                    workspace=workspace)
        
    tracer=OpikTracer(project_name=project_name, metadata={'Thread ID': thread_id}, tags=tags if isinstance(tags, list) else [tags] if tags else None)

    return tracer



def get_vdb_index(db_path: str|Path, db_name: str):
    """Get a vector database index from the given path and name.
        Rag types: [base, rerank, hybrid]
        """
    # get client and vdb
    client = PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name=db_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # setup index
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                                embed_model=embed_model,
                                                storage_context=storage_context)


def get_model(model_name: str, temperature: float = 0, streaming: bool = True, **kwargs):
    """
    Get a model from the environment variables.
    """
    from dotenv import load_dotenv
    # Load the environment variables
    load_dotenv()
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

    # Get the model from the environment variables
    if "claude" in model_name:
        from langchain_anthropic import ChatAnthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=temperature, 
                            stream_usage=streaming, streaming=streaming,
                            callbacks=[StreamingStdOutCallbackHandler()], **kwargs)
    elif "gpt" in model_name:
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(model=model_name, temperature=temperature, 
                            stream_usage=streaming, streaming=streaming,
                            callbacks=[StreamingStdOutCallbackHandler()], **kwargs)
    elif 'llama' in model_name:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=temperature, 
                            stream_usage=streaming, streaming=streaming,
                            callbacks=[StreamingStdOutCallbackHandler()],**kwargs)
    elif 'gemini' in model_name:
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, 
                            stream_usage=streaming, streaming=streaming,
                            callbacks=[StreamingStdOutCallbackHandler()], **kwargs)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    

def deduplicate_nodes(nodes):
    seen_ids = set()
    unique_nodes = []
    for node in nodes:
        node_id = node.node.node_id if hasattr(node, 'node') else node.node_id
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            unique_nodes.append(node)
    return unique_nodes

class RetrieverFactory:
    def __init__(self, db_path, db_name, top_k: int = 8):
        self.index = get_vdb_index(db_path=db_path, db_name=db_name)
        self.top_k = top_k

    def get_retriever(self, type: str = "base", retriever_weights: List[float]=[0.5, 0.5]):
        """Get a retriever from the index. Retriever weights [vector_weight, bm25_weight] are only used for hybrid retrievers."""
        if type == "base":
            return BaseRetriever(self.index, self.top_k)
        elif type == "rerank":
            return RerankRetriever(self.index, self.top_k)
        elif type == "hybrid":
            return HybridRetriever(self.index, self.top_k, retriever_weights=retriever_weights)
        elif type == "hybrid_rerank":
            return HybridRerankRetriever(self.index, self.top_k, retriever_weights=retriever_weights)
        elif type == "bm25":
            return BM25Retriever(self.index, self.top_k)
        elif type == "bm25_rerank":
            return BM25RerankRetriever(self.index, self.top_k)
        else:
            raise ValueError(f"Unknown retriever type: {type}. Expected either 'base', 'rerank', or 'hybrid'.")


class BaseRetriever:
    def __init__(self, index: VectorStoreIndex, top_k: int):
        self.name = "base"
        self.retriever = index.as_retriever(similarity_top_k=top_k)
        self.ranker=None
    
    def retrieve(self, query: str):
        return self._retrieve(query)

    def _retrieve(self, query: str):
        return self.retriever.retrieve(query)


class RerankRetriever:
    def __init__(self, index: VectorStoreIndex, top_k: int, device: str = "cuda"):
        from llama_index.core.postprocessor import SentenceTransformerRerank

        self.name = "base_rerank"
        self.retriever = index.as_retriever(similarity_top_k=top_k)
        self.ranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base", # BAAI/bge-reranker-base # cross-encoder/ms-marco-MiniLM-L-2-v2
            top_n=top_k,
            device=device,
        )

    def retrieve(self, query: str):
        return self._retrieve(query)

    def _retrieve(self, query: str):
        nodes = self.retriever.retrieve(query)
        return nodes


class HybridRetriever:
    def __init__(self, index: VectorStoreIndex, top_k: int, retriever_weights: List[float] = [0.5, 0.5]):
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever

        self.name = "hybrid"
        self.vector_retriever = index.as_retriever(similarity_top_k=top_k)

        # Use the same index to get the source nodes for BM25 retriever
        # This is a workaround to get the source nodes from the index
        source_nodes = index.as_retriever(similarity_top_k=10000).retrieve(" ")
        nodes = [x.node for x in source_nodes]
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes,
                                                          similarity_top_k=top_k)

        self.retriever = QueryFusionRetriever(
            [self.vector_retriever, self.bm25_retriever],
            similarity_top_k=top_k, 
            num_queries=1,  # set this to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            retriever_weights=retriever_weights,
        )
        self.ranker = None

    def retrieve(self, query: str):
        return self._retrieve(query)

    def _retrieve(self, query: str):
        nodes = self.retriever.retrieve(query)
        return deduplicate_nodes(nodes)
    

class HybridRerankRetriever:
    def __init__(self, index: VectorStoreIndex, top_k: int, 
                 retriever_weights: List[float] = [0.5, 0.5],
                 device: str = "cuda"):
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.postprocessor import SentenceTransformerRerank

        self.name = "hybrid_rerank"
        self.vector_retriever = index.as_retriever(similarity_top_k=top_k)

        # Use the same index to get the source nodes for BM25 retriever as a workaround
        source_nodes = index.as_retriever(similarity_top_k=10000).retrieve(" ")
        nodes = [x.node for x in source_nodes]
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes,
                                                          similarity_top_k=top_k)

        self.retriever = QueryFusionRetriever(
            [self.vector_retriever, self.bm25_retriever],
            similarity_top_k=top_k,
            num_queries=1,  # set this to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            retriever_weights=retriever_weights,
        )
        
        self.ranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base", # BAAI/bge-reranker-base # cross-encoder/ms-marco-MiniLM-L-2-v2
            top_n=top_k,
            device=device,
        )

    def retrieve(self, query: str):
        return self._retrieve(query)

    def _retrieve(self, query: str):
        nodes = self.retriever.retrieve(query)
        return deduplicate_nodes(nodes)
    

class BM25Retriever:
    def __init__(self, index: VectorStoreIndex, top_k: int):
        from llama_index.retrievers.bm25 import BM25Retriever

        self.name = "bm25"

        # Use the same index to get the source nodes for BM25 retriever as a workaround
        source_nodes = index.as_retriever(similarity_top_k=10000).retrieve(" ")
        nodes = [x.node for x in source_nodes]
        self.retriever = BM25Retriever.from_defaults(nodes=nodes,
                                                          similarity_top_k=top_k)
        
        self.ranker = None

    def retrieve(self, query: str):
        return self._retrieve(query)

    def _retrieve(self, query: str):
        return self.retriever.retrieve(query)
    

class BM25RerankRetriever:
    def __init__(self, index: VectorStoreIndex, top_k: int, device: str = "cuda"):
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.core.postprocessor import SentenceTransformerRerank

        self.name = "bm25_rerank"

        # Use the same index to get the source nodes for BM25 retriever as a workaround
        source_nodes = index.as_retriever(similarity_top_k=10000).retrieve(" ")
        nodes = [x.node for x in source_nodes]
        self.retriever = BM25Retriever.from_defaults(nodes=nodes,
                                                          similarity_top_k=top_k)
        
        self.ranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base", # BAAI/bge-reranker-base # cross-encoder/ms-marco-MiniLM-L-2-v2
            top_n=top_k,
            device=device,
        )

    def retrieve(self, query: str):
        return self._retrieve(query)

    def _retrieve(self, query: str):
        return self.retriever.retrieve(query)





if __name__ == "__main__":
    from pprint import pprint
    from llama_index.core.query_engine import RetrieverQueryEngine
    db_path = Path("data/vdb")
    db_name = "coi"
 
    retriever_factory = RetrieverFactory(db_path=db_path, db_name=db_name, top_k=5)
    base_retriever = retriever_factory.get_retriever(type="base")
    rerank_retriever = retriever_factory.get_retriever(type="rerank")
    hybrid_retriever = retriever_factory.get_retriever(type="hybrid")
    hybrid_rerank_retriever = retriever_factory.get_retriever(type="hybrid_rerank")
    bm25_retriever = retriever_factory.get_retriever(type="bm25")
    bm25_rerank_retriever = retriever_factory.get_retriever(type="bm25_rerank")


    query = "What should I major in if I want to be a pilot?"
    
    base_results = base_retriever.retrieve(query)
    rerank_results = rerank_retriever.ranker.postprocess_nodes(rerank_retriever.retrieve(query), query_str=query)
    hybrid_results = hybrid_retriever.retrieve(query)
    hybrid_rerank_results = hybrid_rerank_retriever.ranker.postprocess_nodes(hybrid_rerank_retriever.retrieve(query), query_str=query)
    bm25_results = bm25_retriever.retrieve(query)
    bm25_rerank_results = bm25_rerank_retriever.ranker.postprocess_nodes(bm25_rerank_retriever.retrieve(query), query_str=query)

    print(f"Base Results: {[result.text for result in base_results]}\n\n")
    print(f"Rerank Results: {[result.text for result in rerank_results]}")
    print("Hybrid Results:", {(result.score, result.text) for result in hybrid_results})
    print("Hybrid Rerank Results:", {(result.score, result.text) for result in hybrid_rerank_results})
    print("BM25 Results:", {(result.score, result.text) for result in bm25_results})
    print("BM25 Rerank Results:", {(result.score, result.text) for result in bm25_rerank_results})


    #%%
    query_engine = RetrieverQueryEngine.from_args(bm25_rerank_retriever, node_postprocessors=[bm25_rerank_retriever.ranker])
    response = query_engine.query(query)

# %%
