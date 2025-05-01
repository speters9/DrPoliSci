# <img src="static/drps.png" alt="Logo" width="32" style="vertical-align: middle;"> Dr. PoliSci: Academic Advisor Chatbot

This repository contains a multi-component chatbot designed for academic advising tasks. It combines LangGraph (for routing and state handling), LlamaIndex (for retrieval), and ChromaDB (as a vector store), with session memory and summarization via SQLite.

The system routes student queries to one of several nodes based on context: a clarifier, a rephraser, a retriever, or an advisor model. It supports conversational memory, follow-up questions, and document-grounded responses.

---

## Functionality Overview

- **Query Routing**: Queries are first passed to a router node which determines if the question depends on memory, requires clarification, or should be answered using retrieval from external documents.
- **Clarification Loop**: If the question lacks sufficient context, a clarification question is asked and a revised version of the query is constructed.
- **Rephrasing/Expansion**: Vague or ambiguous queries are reformulated to improve downstream retrieval.
- **RAG Retrieval**: If the query pertains to course descriptions, policies, or other factual content, relevant context is retrieved from a vector database using a hybrid retriever (vector + BM25 with reranking).
- **Advisor Response**: A final advisor node generates the reply using a predefined persona, incorporating the retrieved documents and prior conversation history.
- **Session Memory**: Conversations are checkpointed with SQLite. At the end of a session (manual or idle), a structured summary is generated and saved.
- **Idle Watchdog**: An optional background process monitors chat activity and automatically finalizes and prunes idle sessions.
- **Dual Interfaces**:
  - Tkinter GUI via `chat_with_agent.py`
  - Streamlit web app via `chat_ui.py` with live session tracking

---

## Directory Layout

```
chatbot/
├── notebooks/
│   ├── 00_convert_and_chunk_coi.py       # Convert PDFs to Markdown, chunk text
│   ├── 01_create_vdb.py                  # Clean + index into vector DB
│   ├── 02_rag_chat.py                    # Run Tkinter chatbot
│   ├── 03_rag_chat_streamlit.py         # Launch Streamlit + watchdog
│   └── 04_create_test_set.py            # Generate QA pairs for testing
├── src/utils/
│   ├── app/
│   │   ├── chat_ui.py                   # Streamlit frontend
│   │   └── watch_idle_sessions.py      # Ends idle sessions
│   ├── chatbot.py                       # Defines LangGraph and nodes
│   ├── chat_with_agent.py              # Tkinter GUI logic
│   ├── config_types.py                 # Dataclass config schema
│   ├── global_helpers.py               # Config loader, logging
│   ├── prompts.py                      # Advisor, router, and helper prompts
│   ├── rag_memory.py                   # SQLite memory logic + summarization
│   └── rag_tools.py                    # Retriever classes, embedding logic
```

---

## How to Run

1. **Document Preparation**  
   Convert and preprocess course materials:
   ```bash
   python notebooks/00_convert_and_chunk_coi.py
   python notebooks/01_create_vdb.py
   ```

2. **Run the Bot**
   - Tkinter:
     ```bash
     python notebooks/02_rag_chat.py
     ```
   - Streamlit:
     ```bash
     python notebooks/03_rag_chat_streamlit.py
     ```

3. **Generate Evaluation Data (Optional)**  
   Creates a test set of QA pairs for RAG evaluation:
   ```bash
   python notebooks/04_create_test_set.py
   ```

---

## Configuration

System behavior is controlled by `config.yaml`, parsed via dataclasses in `config_types.py`. This includes model choice, retriever settings, file paths, and session timeout durations.

---

## Requirements

- Python 3.10+
- `.env` file with keys for OpenAI, Claude, or Gemini models
- GPU optional but recommended for document preprocessing
- External dependencies: LlamaIndex, LangGraph, Chroma, Streamlit

---
