#%%
"""following the (slightly modified) steps outlined in https://medium.com/@med.el.harchaoui/rag-evaluation-generate-datasets-with-your-domain-expertise-data-a116cd023dc8

1. Load docs and clean in same manner as in 01_create_vdb.py
2. Prompt model with docs on content richness
3. For the 'yes' responses, prompt model with the docs and ask for a question

"""
import uuid
import re
import json
import os
import pandas as pd

from pathlib import Path
from typing_extensions import Annotated, TypedDict

from src.utils.rag_tools import get_model, get_vdb_index
from src.testset_utils.testset_prompts import eval_prompts, qa_prompts, val_prompts, groundedness, relevance, standalone

from pyprojroot.here import here
from dotenv import load_dotenv
from tqdm import tqdm

from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

# from langchain_core.output_parsers.json import JsonOutputParser
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter

from llama_index.core import SimpleDirectoryReader

from pydantic import BaseModel, Field

load_dotenv()
source_path = here() / "data/testset_source"
output_path = here() / "data/processed"



llm = get_model(model_name="gpt-4o-mini")

#%%

# remove urls, emails, and proper names from the text
def clean_text(text: str) -> str:
    """Remove urls and emails from text."""
    pattern = r'(?:[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|(?:https?://[^\s]+)'
    return re.sub(pattern, '[url]', text)

def remove_names(doc_chunk) -> str:
    original_text = doc_chunk.page_content
    sentences = splitter.split(original_text)
    tagger.predict(sentences)
    sentence_list = []
    return_as_is = True

    for sentence in sentences:
        sent_dict = sentence.to_dict()
        text = sent_dict['text']
        if sent_dict.get('entities'):
            for entity in sent_dict['entities']:
                # Check if the entity is a person (PER)
                if any(label.get('value') == 'PER' for label in entity['labels']):
                    return_as_is = False
                    # Use re.escape to safely use the entity text as a regex pattern
                    pattern = re.escape(entity['text'])
                    text = re.sub(pattern, '[NAME]', text)
        sentence_list.append(text)
    if return_as_is:
        return original_text
    else:
        return " ".join(sentence_list)

def combine_subsection(doc_chunk):
    if 'Subsection' in doc_chunk.metadata.keys():
        doc_chunk.page_content = f"{doc_chunk.metadata.get("Subsection")}\n\n{doc_chunk.page_content}"
    return doc_chunk


#%%

if Path(source_path / "cleaned_documents.json").exists():
    with open(source_path / "cleaned_documents.json", "r", encoding="utf-8") as f:
        serialized_docs = json.load(f)
        print(f"Loaded {len(serialized_docs)} documents.")
else:
    # load documents
    reader = SimpleDirectoryReader(input_dir=str(output_path))
    documents = reader.load_data()

    splitter = SegtokSentenceSplitter()
    tagger = Classifier.load('ner')

    # split markdown text into headers and add unique identifiers
    headers_to_split_on = [
        ("#", "Document"),
        ("##", "Chapter"),
        ("###", "Section"),
        ("####", "Subsection"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

    split_docs = []
    for doc in documents:
        split_docs.extend(markdown_splitter.split_text(doc.text))

    for split_doc in tqdm(split_docs):
        # get doc ids and clean text
        if not split_doc.model_dump().get('id', None):
            split_doc.id = str(uuid.uuid4())
        
        # remove person names from text
        names_removed = remove_names(split_doc)

        # remove urls and emails from text
        cleaned_text = clean_text(names_removed)

        # update document text
        split_doc.page_content = cleaned_text

        # combine subsections (only exist if they describe academic classes)
        split_doc = combine_subsection(split_doc)

    serialized_docs = [doc.model_dump() for doc in split_docs]

    with open(source_path/"cleaned_documents.json", "w", encoding="utf-8") as f:
        json.dump(serialized_docs, f, indent=2)


#%%

# -------------------------------------------------------------------------

######## generate a test set for the RAG model from the documents #########

# -------------------------------------------------------------------------


#%%

class ContextEval(TypedDict):
    reasoning = Annotated[str, ..., "Explanation of the reasoning behind the evaluation"]
    evaluation = Annotated[str, ..., "Yes or No"]


eval_llm = llm.with_structured_output(ContextEval)
eval_prompt = PromptTemplate.from_template(eval_prompts.prompt)

eval_chain = eval_prompt | eval_llm 

evaluated_docs = [] 
for doc in tqdm(serialized_docs):
    response = eval_chain.invoke({"context": doc['page_content'], 
                                "rules": eval_prompts.rules, 
                                "guidelines": eval_prompts.guidelines, 
                                "examples": eval_prompts.examples, 
                                "format": eval_prompts.json_format})
    doc['use_for_testset'] = response['evaluation']
    doc['use_for_testset_reasoning'] = response['reasoning']
    evaluated_docs.append(doc)

with open(source_path/"evaluated_documents.json", "w", encoding="utf-8") as f:
    json.dump(evaluated_docs, f, indent=2)


# %%

# -------------------------------------------------------------------------

# generate a test set for the QA model from the documents

# -------------------------------------------------------------------------

#%%

# reload the documents with the evaluation results
assert os.path.exists(source_path / "evaluated_documents.json"), "Make sure evaluated documents are saved before proceeding."
with open(source_path / "evaluated_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)
    print(f"Loaded {len(documents)} documents.")

# %%

class QAGen(TypedDict):
    question = Annotated[str, ..., "Question generated from the context"]
    answer = Annotated[str, ..., "Answer to the question"]

qa_llm = llm.with_structured_output(QAGen)
qa_prompt = PromptTemplate.from_template(qa_prompts.prompt)

qa_chain = qa_prompt | qa_llm 

qa_docs = [] 
for doc in tqdm(documents):
    if doc.get('use_for_testset') in ["No", "no", "NO", "n", "N"]:
        continue
    response = qa_chain.invoke({"context": doc['page_content'], 
                                "rules": qa_prompts.rules, 
                                "guidelines": qa_prompts.guidelines, 
                                "examples": qa_prompts.examples, 
                                "format": qa_prompts.json_format})
    doc['generated_question'] = response['question']
    doc['answer'] = response['answer']
    qa_docs.append(doc)

with open(source_path/"qa_documents.json", "w", encoding="utf-8") as f:
    json.dump(qa_docs, f, indent=2)


#%%

# -------------------------------------------------------------------------

# Evaluate the generated questions

# -------------------------------------------------------------------------

#%%

# reload the documents with the qa results
assert os.path.exists(source_path / "qa_documents.json"), "Make sure documents with questions are saved before proceeding."
with open(source_path / "qa_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)
    print(f"Loaded {len(documents)} documents.")

#%%

class LLMJudge(TypedDict):
    evaluation = Annotated[str, ..., "Your written rationale for the rating given."]
    score = Annotated[str, ..., "Your rating, as a integer between 1 and 5"]

judge_llm = llm.with_structured_output(LLMJudge)
judge_prompt = PromptTemplate.from_template(val_prompts.prompt)

judge_chain = judge_prompt | judge_llm 

metrics = [groundedness, relevance, standalone]

val_docs = [] 
for doc in tqdm(documents):
    for metric in metrics:
        response = judge_chain.invoke({"context": doc['page_content'], 
                                    "question": doc['generated_question'],
                                    "task": metric.task,
                                    "evaluation_criteria": metric.evaluation_criteria,
                                    "evaluation_steps": metric.evaluation_steps, 
                                    "format": val_prompts.json_format})
        doc[metric.name] = response['score']
        doc[f'{metric.name}_eval'] = response['evaluation']
    val_docs.append(doc)

with open(source_path/"validated_documents.json", "w", encoding="utf-8") as f:
    json.dump(val_docs, f, indent=2)



# %%

assert os.path.exists(source_path / "validated_documents.json"), "Make sure documents with llm-as-a-judge feedback are saved before proceeding."
with open(source_path / "validated_documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)
    print(f"Loaded {len(documents)} documents.")


#%%

df = pd.DataFrame(documents)

# filter out documents where groundedness, relevance, or standalone are >= 4

testset_full = (df[['id', 'page_content', 'generated_question', 'answer', 'groundedness', 'relevance', 'standalone']]
            .assign(mean_score = lambda x: x[['groundedness', 'relevance', 'standalone']].mean(axis=1),
                    too_long = lambda x: x['answer'].apply(lambda y: len(y) > 300))
            .sort_values(by='mean_score', ascending=False)
            .rename(columns={#"id": "question_id",
                                    "page_content": "context",
                                    "generated_question": "input",
                                    "answer": "expected_output",
                                    "groundedness": "groundedness_score",
                                    "relevance": "relevance_score",
                                    "standalone": "standalone_score"})
    ).copy()

testset_short = testset_full[(testset_full['groundedness_score'] >= 4) & 
                  (testset_full['relevance_score'] >= 4) & 
                  (testset_full['standalone_score'] >= 4)].reset_index(drop=True).copy()

testset_full = testset_full[testset_full['mean_score'] >= 4].reset_index(drop=True)

print(f'{testset_short.shape=}')
print(f'{testset_short['too_long'].sum()=}')


print(f'{testset_full.shape=}')
print(f'{testset_full['too_long'].sum()=}')


columns_to_keep = ['input', 'expected_output', 'context']
testset_short = testset_short[columns_to_keep].copy()
testset_full = testset_full[columns_to_keep].copy()

# %%
import opik
from opik import Opik

api_key = os.getenv("COMET_API_KEY")

opik.configure(api_key=api_key, 
                use_local=False,
                workspace="llm-testing")
opik_client = Opik()

dataset = opik_client.get_or_create_dataset(
    name="polsci-advisor-testset-short",
    description="Test set for the Pol Sci Advisor RAG chatbot. All measures are >= 4 individually.",
)

dataset.insert(testset_short.to_dict(orient="records"))

#%%
dataset_long = opik_client.get_or_create_dataset(
    name="polsci-advisor-testset-long",
    description="Test set for the Pol Sci Advisor RAG chatbot. Mean of measures is >= 4.",
)
dataset_long.insert(testset_full.to_dict(orient="records"))


opik_client.flush()
#%%
