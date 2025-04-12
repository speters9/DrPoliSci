#%%
import asyncio
import nest_asyncio
nest_asyncio.apply()

from ragas.metrics import (
    AnswerRelevancy, Faithfulness, ContextPrecision, ContextRecall
)
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from opik.evaluation.metrics import base_metric, score_result

from dotenv import load_dotenv
load_dotenv()
from src.utils.global_helpers import load_config

config = load_config()
MAX_IDLE = config.timeouts.idle_seconds
model_args = config.model.to_dict()

# Shared LLM and embedding setup
llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo", 
                                     temperature=0))
emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

#%%
# Initialize Ragas metrics
ragas_answer_relevancy = AnswerRelevancy(llm=llm, embeddings=emb)
ragas_faithfulness = Faithfulness(llm=llm)
ragas_context_precision = ContextPrecision(llm=llm)
ragas_context_recall = ContextRecall(llm=llm)

# Answer Relevancy Wrapper
class AnswerRelevancyWrapper(base_metric.BaseMetric):
    def __init__(self, metric):
        self.name = "answer_relevancy_metric"
        self.metric = metric

    async def get_score(self, row):
        row = SingleTurnSample(**row)
        score = await self.metric.single_turn_ascore(row)
        return score

    def score(self, user_input, response, **ignored_kwargs):
        loop = asyncio.get_event_loop()
        row = {"user_input": user_input, "reference": response}
        result = loop.run_until_complete(self.get_score(row))
        return score_result.ScoreResult(value=result, name=self.name)

# Faithfulness Wrapper
class FaithfulnessWrapper(base_metric.BaseMetric):
    def __init__(self, metric):
        self.name = "faithfulness_metric"
        self.metric = metric

    async def get_score(self, row):
        row = SingleTurnSample(**row)
        score = await self.metric.single_turn_ascore(row)
        return score

    def score(self, user_input, response, **ignored_kwargs):
        loop = asyncio.get_event_loop()
        row = {"user_input": user_input, "reference": response}
        result = loop.run_until_complete(self.get_score(row))
        return score_result.ScoreResult(value=result, name=self.name)

# Context Precision Wrapper
class ContextPrecisionWrapper(base_metric.BaseMetric):
    def __init__(self, metric):
        self.name = "context_precision_metric"
        self.metric = metric

    async def get_score(self, row):
        row = SingleTurnSample(**row)
        score = await self.metric.single_turn_ascore(row)
        return score

    def score(self, user_input, response, **ignored_kwargs):
        loop = asyncio.get_event_loop()
        row = {"user_input": user_input, "reference": response}
        result = loop.run_until_complete(self.get_score(row))
        return score_result.ScoreResult(value=result, name=self.name)

# Context Recall Wrapper
class ContextRecallWrapper(base_metric.BaseMetric):
    def __init__(self, metric):
        self.name = "context_recall_metric"
        self.metric = metric

    async def get_score(self, row):
        row = SingleTurnSample(**row)
        score = await self.metric.single_turn_ascore(row)
        return score

    def score(self, user_input, response, **ignored_kwargs):
        loop = asyncio.get_event_loop()
        row = {"user_input": user_input, "reference": response}
        result = loop.run_until_complete(self.get_score(row))
        return score_result.ScoreResult(value=result, name=self.name)

#%%
# Instantiate wrappers
answer_relevancy_opik = AnswerRelevancyWrapper(ragas_answer_relevancy)
faithfulness_opik = FaithfulnessWrapper(ragas_faithfulness)
context_precision_opik = ContextPrecisionWrapper(ragas_context_precision)
context_recall_opik = ContextRecallWrapper(ragas_context_recall)


# %%
