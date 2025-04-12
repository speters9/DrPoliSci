#%%
import os
import opik
from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Hallucination, AnswerRelevance, ContextRecall, ContextPrecision

from opik.integrations.llama_index import LlamaIndexCallbackHandler
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_engine import RetrieverQueryEngine

from src.utils.rag_tools import RetrieverFactory
from src.utils.chatbot import ChatbotGraph

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


from dotenv import load_dotenv
from pyprojroot.here import here

from src.utils.global_helpers import load_config

config = load_config()
MAX_IDLE = config.timeouts.idle_seconds
model_args = config.model.to_dict()

load_dotenv()
api_key = os.getenv("COMET_API_KEY")
EVAL_MODEL = "gpt-4o-mini"
#%%

# setup opik
opik.configure(api_key=api_key, 
                use_local=False,
                workspace="llm-testing")
opik_client = Opik()
opik_callback_handler = LlamaIndexCallbackHandler(project_name="polsci-advisor-testing",)
Settings.callback_manager = CallbackManager([opik_callback_handler])


# setup dataset
dataset = opik_client.get_or_create_dataset(
    name="polsci-advisor-testset-short"
)


#%%
# using GPT 3.5, use GPT 4 / 4-turbo for better accuracy
evaluator_llm = OpenAI(model=EVAL_MODEL)

metrics = [
    Hallucination(model=EVAL_MODEL),
    AnswerRelevance(model=EVAL_MODEL),
    ContextPrecision(model=EVAL_MODEL),
    ContextRecall(model=EVAL_MODEL),
    ]



#%%

# index = get_vdb_index(
#     db_path=here("data/vdb"),
#     db_name="coi",
# )
# query_engine = index.as_query_engine(similarity_top_k=8) # 15 is chatbot fallback
retriever_factory = RetrieverFactory(db_path=here("data/vdb"), db_name="coi", top_k=10)

base_retriever = retriever_factory.get_retriever(type="base")
rerank_retriever = retriever_factory.get_retriever(type="rerank")
hybrid_retriever = retriever_factory.get_retriever(type="hybrid")
hybrid_rerank_retriever = retriever_factory.get_retriever(type="hybrid_rerank")
bm25_retriever = retriever_factory.get_retriever(type="bm25")
bm25_rerank_retriever = retriever_factory.get_retriever(type="bm25_rerank")


retrievers = [base_retriever, rerank_retriever, 
              hybrid_retriever, hybrid_rerank_retriever,
              bm25_retriever, bm25_rerank_retriever]


#%%

def evaluation_task(x: dict) -> dict:
    result = query_engine.query(x["input"])
                                
    if isinstance(result.response, str):
        answer = result.response
    elif isinstance(result.response, dict):
        answer = result.response.get("blocks")[0].get("text")
    else:
        answer = result.response.get("text", "Bad Response")
  
    return {
        "input": x["input"],
        'output': answer,
        "expected_output": x["expected_output"],
        "context": x["context"],
        "reference": x["context"],
    }

#%%
for retriever in retrievers:
    # use reranker if available
    query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[retriever.ranker] if retriever.ranker else None)

    result = evaluate(
        dataset=dataset,
        task = evaluation_task,
        scoring_metrics=metrics,
        experiment_name=f"test-rag-{retriever.name}",
        task_threads=2,
        experiment_config={
            "model": EVAL_MODEL,
            "metrics": [metric.name for metric in metrics],
            "dataset": dataset.name,
            "eval_type": "full rag",
            "retriever": retriever.name
        }
    )

#%%
import itertools
# test hybrid retriever weights
set1 = range(0,10,2)
set2 = range(0,10,2)
rangelist = list(itertools.product(set1, set2))

#weights_list = [(r[0]/10, r[1]/10) for r in rangelist if r[0] + r[1] == 10]
factory = RetrieverFactory(db_path=here("data/vdb"), db_name="coi", top_k=10)


weights_list = [[0.3, 0.7], [0.7, 0.3]]
retrievers_to_test = ['hybrid', 'hybrid_rerank']
for ret in retrievers_to_test:
    for balance in weights_list:
        retriever = factory.get_retriever(type=ret, retriever_weights=list(balance)) 
        query_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[retriever.ranker] if retriever.ranker else None)

        result = evaluate(
            dataset=dataset,
            task = evaluation_task,
            scoring_metrics=metrics,
            experiment_name=f"test-rag-{retriever.name}-weights-{balance}",
            task_threads=4,
            experiment_config={
                "model": EVAL_MODEL,
                "metrics": [metric.name for metric in metrics],
                "dataset": dataset.name,
                "eval_type": "hybrid rag",
                "retriever": retriever.name,
                "retriever-weights": balance
            }
        )

# %%
