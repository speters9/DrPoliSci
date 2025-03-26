#%%
import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate)

from src.utils.helpers import build_opik_tracer


load_dotenv()

OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

COMET_API_KEY = os.getenv('comet_api_key')


tracer = build_opik_tracer(api_key=COMET_API_KEY, 
                            workspace="llm-testing", 
                            project_name="opik-test")


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,
    organization=OPENAI_ORG,
)

def get_response(llm, query, parser = StrOutputParser()) -> Dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "For every human message, your job is to answer 'yes'"
                )
            ),
            HumanMessagePromptTemplate.from_template(
                template=f"Answer the following question: {query}"
            ),
        ]
    )
    chain = prompt | llm | parser

    response = chain.invoke({"query":query}, config={"callbacks": [tracer]})


    return response

get_response(llm, 
                "How tall is the Eiffel Tower?")
