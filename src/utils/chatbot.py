#%%
import os
import json

from src.utils.global_helpers import build_opik_tracer
from dotenv import load_dotenv
from datetime import datetime

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from chromadb.utils.embedding_functions import create_langchain_embedding

from IPython.display import Image, display

from pyprojroot.here import here

from src.utils.prompts import polisci_advisor, router
import sqlite3
import chromadb
import uuid

#%% 

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
    

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    session_id: Optional[str] = None
    thread_id: Optional[str] = None


class AdvisorNode:
    def __init__(
        self, 
        model_name: str, 
        thread_id: str,
        system_prompt: str, 
        temperature: float = 0,
        **kwargs
    ):
        self.system_message = SystemMessage(content=system_prompt)
        self.model = get_model(model_name, temperature)
        self.tracer = build_opik_tracer(workspace="llm-testing", 
                                        project_name="polsci-advisor", 
                                        thread_id=thread_id,
                                        tags = 'advisor')

    def run(self, state: State):
        messages = state["messages"]

        # Ensure system message is always 
        if not isinstance(messages[0], SystemMessage):
            messages.insert(0, self.system_message)

        response = self.model.invoke(messages, config = {"callbacks": [self.tracer]})
        state["messages"].append(response)
        self.tracer.flush()
        
        return state


class RouterNode:
    def __init__(self, 
                 model_name: str, 
                 thread_id: str,
                 system_prompt: str = router.personality,
                 temperature: float = 0.5, 
                 **kwargs):
        self.system_message = SystemMessage(content=system_prompt)
        self.model = get_model(model_name, temperature)
        self.tracer = build_opik_tracer(workspace="llm-testing", 
                                        project_name="polsci-advisor", 
                                        thread_id=thread_id,
                                        tags = 'router')

    def run(self, state: State):
        # Get the original query from the last human message.
        original_query = state["messages"][-1].content

        # Build a message list with the system prompt and the human query.
        messages = json.dumps([{st.type: st.content} for st in state['messages']])
        prompt = self.system_message.content.format(context=messages, user_query=original_query)

        # Invoke the model to obtain an augmented query.
        augmented_query = self.model.invoke(prompt, config = {"callbacks": [self.tracer]})
        # Append the augmented query to the conversation state.
        state["messages"].append(augmented_query)
        # print(f"augmented_query: {augmented_query}")

        self.tracer.flush()

        return state

class RagNode:
    def __init__(self, 
                 model_name: str, 
                 thread_id: str,
                 temperature: float = 0, 
                 vdb_path: str = here() / "data/vdb",
                 **kwargs):
        self.model = get_model(model_name, temperature)
        self.client = chromadb.PersistentClient(path=str(vdb_path))
        self.embeddings_fn = create_langchain_embedding(
                        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                        )
        self.collection = self.client.get_or_create_collection(name="coi",
                                                embedding_function=self.embeddings_fn,
                                                metadata={"description": "usafa course of instruction"})
        self.vector_store = Chroma(
            client=self.client,
            collection_name="coi",
            embedding_function=self.embeddings_fn
        )
        self.search_type = kwargs.get('search_type', "similarity")
        self.k = kwargs.get('k', 15)
        self.retriever = self.vector_store.as_retriever(search_type=self.search_type, k=self.k)
        self.tracer = build_opik_tracer(workspace="llm-testing", 
                                        project_name="polsci-advisor", 
                                        thread_id=thread_id,
                                        tags = 'rag')
    
    def format_docs(self, docs):
        formatted_docs = []
        for doc in docs:
            meta = "\n".join([f"{header}: {meta}" for header, meta in doc.metadata.items()])
            formatted_doc = "\n\n".join([f"Metadata: {meta}", f"Content: {doc.page_content}"])
            formatted_docs.append(formatted_doc)

        return "\n## Reference Document:\n".join(formatted_docs)

    def run(self, state: State):
        query = state["messages"][-1].content
        docs = self.retriever.invoke(query, config = {"callbacks": [self.tracer]})
        formatted_docs = self.format_docs(docs)

        state["messages"].append(AIMessage(content=f"Here are some documents that may help you: {formatted_docs}"))

        self.tracer.flush()
        # print(f"formatted_docs: {formatted_docs}")

        return state
    

class ChatbotGraph:
    def __init__(self, model_name: str, temperature: float = 0, thread_id: str = uuid.uuid4(), **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.dbDir = kwargs.get('dbDir', here() / "data/chat_histories/conversation_database.db")
        self.thread_id = thread_id

        # Nodes
        self.system_prompt = kwargs.get('system_prompt', polisci_advisor.personality)
        self.chatbot_node = AdvisorNode(self.model_name, self.thread_id, system_prompt=self.system_prompt, temperature=self.temperature)
        self.rag_node = RagNode(self.model_name, self.thread_id, temperature=self.temperature)
        self.router_node = RouterNode(self.model_name, self.thread_id, temperature=self.temperature)

        # Memory saver       
        self.conn = sqlite3.connect(self.dbDir, check_same_thread=False)
        self.memory_saver = SqliteSaver(self.conn)
        self.memory_saver.setup()

    @property
    def graph(self):
        if hasattr(self, "_graph"):
            return self._graph
        self._graph = self.graph_builder()
        return self._graph

    def graph_builder(self):
        graph_builder = StateGraph(State)

        # Node
        graph_builder.add_node("router", self.router_node.run)
        graph_builder.add_node("chatbot_agent", self.chatbot_node.run)
        graph_builder.add_node("rag", self.rag_node.run)

        # Edge
        graph_builder.add_edge(START, "router")
        graph_builder.add_edge("router", "rag")
        graph_builder.add_edge("rag", "chatbot_agent")

        return graph_builder.compile(checkpointer=self.memory_saver)
    
    def display(self):
        display(Image(self.graph.get_graph(xray=True).draw_mermaid_png()))

    def invoke(self, input: str):
        state: State = {"messages": [], 
                        "session_id": datetime.now().isoformat(),
                        "thread_id": self.thread_id}  

        config = {"configurable": {"thread_id": state.get('thread_id'), "user_id": "USER"}}
        
        # Append the user input as a message and then process the graph.
        state["messages"].append(HumanMessage(content=input))
        result_state = self.graph.invoke(state, config)

        return result_state
    
#%%

if __name__ == "__main__":
    # Create the chatbot graph
    chatbot_graph = ChatbotGraph(model_name="gpt-4o-mini", temperature=0.3, thread_id="test_thread")

    # Display the graph
    chatbot_graph.display()

    # Test the chatbot
    response = chatbot_graph.invoke("What should I major in if I'm interested in flying after I graduate?")

    print(response["messages"][-1].content)

    response = chatbot_graph.invoke("What about political science?")

    print(response["messages"][-1].content)
# %%
