
#%%
import os
import re
import uuid

from pyprojroot.here import here
from dotenv import load_dotenv


from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import chromadb
from chromadb.utils.embedding_functions import create_langchain_embedding

from llama_index.core import SimpleDirectoryReader

load_dotenv()
vdb_path = here() / "data/vdb"
output_path = here() / "data/processed"

#%%

# load documents
reader = SimpleDirectoryReader(input_dir=str(output_path))
documents = reader.load_data()


#%%
# split markdown text into headers and add unique identifiers
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

split_docs = []
for doc in documents:
     split_docs.extend(markdown_splitter.split_text(doc.text))

for split_doc in split_docs:
    # pull dict representation of the document
    if not split_doc.model_dump().get('id', None):
        split_doc.id = str(uuid.uuid4())


# %%

# create db and add documents
client = chromadb.PersistentClient(path=str(vdb_path))

# define embedding function
embeddings_fn_raw = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")  # "sentence-transformers/msmarco-distilbert-cos-v5"
embeddings_fn = create_langchain_embedding(embeddings_fn_raw)

# create collection
collection = client.get_or_create_collection(name="coi",
                                                embedding_function=embeddings_fn,
                                                metadata={"description": "usafa course of instruction"})

vector_store = Chroma(
    client=client,
    collection_name="coi",
    embedding_function=embeddings_fn
)

# add documents to collection
vector_store.add_documents(documents=split_docs)



#%%
if __name__ == "__main__":
    retriever = vector_store.as_retriever(search_type="similarity", k=15)

    query = "What should I major in if I'm interested in flying after I graduate?"

    response = retriever.invoke(query)

    print([doc.page_content for doc in response])

# %%
