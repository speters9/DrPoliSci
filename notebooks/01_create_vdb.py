
#%%
import os
import re
import uuid

from pyprojroot.here import here
from dotenv import load_dotenv
from tqdm import tqdm

from flair.nn import Classifier
from flair.splitter import SegtokSentenceSplitter

from langchain_text_splitters import MarkdownHeaderTextSplitter

import chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

from IPython.display import Markdown, display

load_dotenv()
vdb_path = here() / "data/vdb"
output_path = here() / "data/processed"

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
                    # Replace the entity text with '[NAME]' but exclude 'Pol Sci' or 'political science' from being replaced
                    text = re.sub(pattern, '[NAME]', text) if not re.search(r'\b(?:Pol Sci|Political Science|For Ar Stu)\b', text, re.IGNORECASE) else text
        sentence_list.append(text)
    if return_as_is:
        return original_text
    else:
        return " ".join(sentence_list)

def extract_course_name(text):
    # Remove leading hashes and spaces first
    text = re.sub(r'^[#\s]+', '', text.strip())
    
    # Then apply the regex pattern
    pattern = r'^([A-Za-z ]+\d{3}[A-Za-z]?)\.'
    match = re.match(pattern, text)
    return match.group(1) if match else None

def combine_subsection(doc_chunk):
    if 'Subsection' in doc_chunk.metadata.keys():
        course = extract_course_name(doc_chunk.metadata.get("Subsection"))
        if course:
            doc_chunk.metadata['Subsection'] = course
    return doc_chunk

splitter = SegtokSentenceSplitter()
tagger = Classifier.load('ner')


#%%
# load documents
reader = SimpleDirectoryReader(input_dir=str(output_path))
documents = reader.load_data()


#%%
# split markdown text into headers and add unique identifiers
headers_to_split_on = [
    ("#", "Document"),
    ("##", "Chapter"),
    ("###", "Section"),
    ("####", "Subsection"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)

split_docs = []
for doc in documents:
     split_docs.extend(markdown_splitter.split_text(doc.text))

clean_docs = []
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
    cleaned_doc = combine_subsection(split_doc)
    clean_docs.append(cleaned_doc)



llama_index_docs = [Document.from_langchain_format(doc) for doc in clean_docs]

# %%

# # create collection
client = chromadb.PersistentClient(path=str(vdb_path))
collection = client.get_or_create_collection(name="coi",
                                                metadata={"description": "usafa course of instruction"})

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents=llama_index_docs,
    embed_model=embed_model,
    storage_context=storage_context
)



#%%
if __name__ == "__main__":
    from src.utils.chatbot import get_model
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core import get_response_synthesizer
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core import Settings

    # retriever = vector_store.as_retriever(search_type="similarity", k=15)

    # query = "What should I major in if I'm interested in flying after I graduate?"

    # response = retriever.invoke(query)

    # print([doc.page_content for doc in response])

    
    Settings.llm = get_model(model_name="gpt-4o-mini")
    Settings.context_window = 128000

    retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=8
            )

    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
    )
    #query_engine = index.as_retriever(llm=llm)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )

    response = query_engine.query("What is Pol Sci 300 about?")
    display(Markdown(f"<b>{response}</b>"))

# %%
