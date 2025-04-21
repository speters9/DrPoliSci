#%%
import time
start_time = time.time()
import os
import json
import opik
import re
import sys
import sqlite3
import uuid

from datetime import datetime

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from opik.integrations.llama_index import LlamaIndexCallbackHandler
from opik import Opik

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager

from IPython.display import Image, display

from pyprojroot.here import here
from dotenv import load_dotenv


sys.path.append(str(here()))
from src.utils.prompts import polisci_advisor, rephraser, router, clarifier
from src.utils.rag_tools import get_model, RetrieverFactory, build_opik_tracer_langchain, deduplicate_nodes
from src.utils.rag_memory import finalize_and_prune, get_all_messages_from_thread


print(f"Imports loaded in {time.time() - start_time:.2f} seconds.")

#%% 
load_dotenv()
api_key = os.getenv("COMET_API_KEY")

opik.configure(api_key=api_key, 
                use_local=False,
                workspace="llm-testing")

opik_callback_handler = LlamaIndexCallbackHandler(project_name="polsci-advisor",)
Settings.callback_manager = CallbackManager([opik_callback_handler])
    
class State(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    clarification_context: Optional[dict] = None
    route: Optional[str] = None
    rag_context: Optional[list] = None
    rephrased_queries: Optional[dict] = None

class RouterNode:
    def __init__(self, model_name: str, 
                 thread_id: str, 
                 temperature: float = 0, 
                 system_prompt: str = router.personality,
                 **kwargs):
        self.model = get_model(model_name, temperature)
        self.tracer = build_opik_tracer_langchain(
            workspace="llm-testing",
            project_name="polsci-advisor",
            thread_id=thread_id,
            tags=['router', thread_id]
        )
        self.system_message = SystemMessage(content=system_prompt)

    def run(self, state: State):
        user_query = state["messages"][-1].content
        history = json.dumps([{m.type: m.content} for m in state["messages"][:-1]])
        prompt = self.system_message.content.format(conversation_history=history, user_query=user_query)
        route = self.model.invoke(prompt, config={"callbacks": [self.tracer]}).content.lower().strip()

        self.tracer.flush()

        valid_routes = {"memory", "clarify", "rag"}

        # if state.get("clarification_context", {}):
        #     route = "rephrase"

        # if route not in valid_routes:
        #     route = "rephrase"


        # state["route"] = (
        #     "history" if route == "memory"
        #     else "clarify" if route == "clarify"
        #     else "rephrase"
        # )
        #         # if state.get("clarification_context", {}):
        # #     route = "rephrase"

        # # if route not in valid_routes:
        # #     route = "rephrase"

        # next step is rephrase or clarify (or chat history if memory)
        #print(f"before router: {state.get("clarification_context")}")

        if state.get("clarification_context", {}) or route == "rag":
            final_route = "rephrase"  # rephrase/augment clarified query
        elif route == "memory":
            final_route = "history"
        elif route == "clarify":
            final_route = "clarify"
        else:  # route == "rag" or unrecognized
            final_route = "rephrase"

        state["route"] = final_route if final_route in valid_routes else "rephrase"
        #print(final_route)

        return state


class ClarificationNode:
    def __init__(self, model_name: str, 
                 thread_id: str, 
                 system_prompt: str = clarifier.personality,
                 temperature: float = 0.3, 
                 **kwargs):
        self.model = get_model(model_name, temperature)
        self.tracer = build_opik_tracer_langchain(
            workspace="llm-testing",
            project_name="polsci-advisor",
            thread_id=thread_id,
            tags=['clarifier', thread_id]
        )
        self.system_message = SystemMessage(content=system_prompt)

    def run(self, state: State):
        user_query = state["messages"][-1].content
        conversation_history = json.dumps([{m.type: m.content} for m in state["messages"][:-1]])

        prompt = self.system_message.content.format(conversation_history=conversation_history, user_query=user_query)
        response = self.model.invoke(prompt, config={"callbacks": [self.tracer]})

        state["clarification_context"] = {
                "original_query": user_query,
                "clarifying_question": response.content
            }

        state["messages"].append(AIMessage(content=response.content))

        self.tracer.flush()
        return state



class RephrasingNode:
    def __init__(self, 
                 model_name: str, 
                 thread_id: str,
                 system_prompt: str = rephraser.personality,
                 temperature: float = 0.5, 
                 **kwargs):
        self.system_message = SystemMessage(content=system_prompt)
        self.model = get_model(model_name, temperature, streaming=False)
        self.tracer = build_opik_tracer_langchain(workspace="llm-testing", 
                                        project_name="polsci-advisor", 
                                        thread_id=thread_id,
                                        tags = ['rephraser', thread_id])
        self.expand = True if rephraser.preface == "expand" else False

    def run(self, state: State):
        # Get the original query from the last human message.
        clarification = state.get("clarification_context")
        if clarification:
            clarifier_q = clarification["clarifying_question"]
            student_followup = state["messages"][-1].content
            original_query = clarification["original_query"]
            
            composed_query = (
                "The student asked an ambiguous question and then clarified. Here is that record:\n"
                f"Clarification question: {clarifier_q}\n"
                f"Student answer to clarification: {student_followup}\n"
                f"Original question: {original_query}"
            )
        else:
            composed_query = state["messages"][-1].content

        # Build a message list with the system prompt and the human query.
        messages = json.dumps([{st.type: st.content} for st in state['messages'][:-1]])
        prompt = self.system_message.content.format(user_query=composed_query, conversation_history=messages)

        # Invoke the model to obtain an augmented query.
        augmented_query = self.model.invoke(prompt, config = {"callbacks": [self.tracer]})

        if self.expand:
            import yaml
            content = re.sub(r'^```yaml|```$', '', augmented_query.content).strip()
            augmented_query = yaml.safe_load(content).get("expanded_queries", [])
            expanded_query_list = [AIMessage(content=q) for q in augmented_query if q.strip()]
            state["rephrased_queries"] = {'user_original': composed_query, 
                                         'expanded_queries': expanded_query_list}

        else:
            # Append the augmented query to the conversation state.
            state["rephrased_queries"] = {'user_original': composed_query, 
                                         'expanded_queries': augmented_query}

        state["clarification_context"]= None
        #print(f"after pop: {state.get('clarification_context')=}")
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
        self.thread_id = thread_id
        self.model = get_model(model_name, temperature, streaming=False)

        # set retriever params
        self.k = kwargs.get('k', 15)

        # hybrid retriever with weights of [vector=0.7, bm25=0.3] performed the best on experimentation
        self.retriever_type = kwargs.get('retriever_type', "hybrid_rerank")
        self.retriever_weights = kwargs.get('retriever_weights', [0.7, 0.3]) 

        # build retriever
        self.factory = RetrieverFactory(db_path=vdb_path, db_name="coi", top_k=self.k)
        self.retriever = self.factory.get_retriever(type=self.retriever_type, retriever_weights=self.retriever_weights)
    
        self.client = Opik()


    def format_docs(self, docs):
        # formatted_docs = []
        #  for i, doc in enumerate(docs):
            #meta = "\n".join([f"{header}: {meta}" for header, meta in doc.metadata.items()])
            #formatted_doc = "\n\n".join([f"Metadata: {meta}", f"Content: {doc.text}"])
            # formatted_doc = f"Next Document {i}:\n{doc.text}"
            # formatted_docs.append(formatted_doc)

        return "\n\n# Next Reference Document:\n".join([doc.text for doc in docs])

    def run(self, state: State):
        #query = state["messages"][-1].content
        original_query = state["rephrased_queries"].get("user_original")
        if isinstance(state.get("rephrased_queries").get("expanded_queries"), str):
            query = state["rephrased_queries"].get("expanded_queries")
            nodes = self.retriever.retrieve(query)
        else:
            nodelist = []
            querylist = state["rephrased_queries"].get("expanded_queries")
            for query in querylist:
                nodelist.extend(self.retriever.retrieve(query.content))
            nodes = deduplicate_nodes(nodelist)

        formatted_docs = self.format_docs(nodes)

        # rerank if applicable
        if self.retriever.ranker:
            reranked_nodes = self.retriever.ranker.postprocess_nodes(nodes, query_str=original_query)
            formatted_docs = self.format_docs(reranked_nodes)
        
        self.client.trace(name="rag",
            input=original_query,
            output=formatted_docs,
            project_name="polsci-advisor",
            thread_id=self.thread_id,
            tags=['rag', self.thread_id])
        self.client.flush()

        state["rag_context"] = [AIMessage(content=f"Here are some documents that may help you: {formatted_docs}")]
        state["rephrased_queries"] = None
        # print(f"formatted_docs: {formatted_docs}")

        return state
    
class AdvisorNode:
    def __init__(
        self, 
        model_name: str, 
        thread_id: str,
        system_prompt: str, 
        temperature: float = 0,
        **kwargs
    ):
        self.prompt = ChatPromptTemplate.from_messages(
                    [
                        SystemMessage(
                            content=(system_prompt)
                        ),
                        HumanMessagePromptTemplate.from_template(polisci_advisor.preface)
                    ]
                )
        self.model = get_model(model_name, temperature, streaming=False)
        self.chain = self.prompt | self.model

        self.tracer = build_opik_tracer_langchain(workspace="llm-testing", 
                                        project_name="polsci-advisor", 
                                        thread_id=thread_id,
                                        tags = ['advisor', thread_id])

    def run(self, state: State):
        user_query = state["messages"][-1].content
        message_history = json.dumps([{m.type: m.content} for m in state["messages"][:-1]])

        rag_context_list = state.get("rag_context") or []
        rag_context = rag_context_list[-1] if rag_context_list else "No additional context available."
        rag_context = rag_context.content if isinstance(rag_context, AIMessage) else rag_context

        response = self.chain.invoke({"user_query": user_query,
                                      "message_history": message_history,
                                      "rag_context": rag_context}, 
                                      config = {"callbacks": [self.tracer]})
        
        state["messages"].append(response)
        state["rag_context"]= None
        self.tracer.flush()
        return state


class ChatbotGraph:
    def __init__(self, model_name: str, temperature: float = 0, thread_id: str = uuid.uuid4(), **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.dbDir = kwargs.get('dbDir', here() / "data/chat_histories/conversation_database.db")
        self.thread_id = thread_id
        self.has_greeted_user = False
        self.k = kwargs.get('k', 10)
        self.retriever_type = kwargs.get('retriever_type', "hybrid_rerank")
        self.retriever_weights = kwargs.get('retriever_weights', [0.7, 0.3])
    
        # Memory saver       
        self.conn = sqlite3.connect(self.dbDir, check_same_thread=False)
        self.memory_saver = SqliteSaver(self.conn)
        self.memory_saver.setup()
        self.last_thread_questions = self.get_last_summary()
        self.state = {
                "messages": [],
                "session_id": datetime.now().isoformat(),
                "thread_id": self.thread_id,
                "rag_context": [],
                "clarification_context": None,
                "rephrased_queries": None,
                "route": None
            }

        # Nodes
        self.system_prompt = kwargs.get('system_prompt', polisci_advisor.personality)
        self.chatbot_node = AdvisorNode(self.model_name, self.thread_id, system_prompt=self.system_prompt, temperature=self.temperature)
        self.rag_node = RagNode(self.model_name, self.thread_id, temperature=self.temperature, 
                                k=self.k, retriever_type=self.retriever_type, retriever_weights=self.retriever_weights)
        self.rephrasing_node = RephrasingNode(self.model_name, self.thread_id, temperature=self.temperature)
        self.router_node = RouterNode(self.model_name, self.thread_id, temperature=self.temperature)
        self.clarifier_node = ClarificationNode(self.model_name, self.thread_id, temperature=self.temperature)
    
    @property
    def graph(self):
        if hasattr(self, "_graph"):
            return self._graph
        self._graph = self.graph_builder()
        return self._graph

    def graph_builder(self):
        graph_builder = StateGraph(State)

        # Nodes
        graph_builder.add_node("router", self.router_node.run)
        graph_builder.add_node("rephrase", self.rephrasing_node.run)
        graph_builder.add_node("rag", self.rag_node.run)
        graph_builder.add_node("chatbot_agent", self.chatbot_node.run)
        graph_builder.add_node("clarify", self.clarifier_node.run)
 
        # Edges
        graph_builder.add_edge(START, "router")
        graph_builder.add_conditional_edges("router", 
                lambda state: state["route"], {
                    "rephrase": "rephrase",
                    "clarify": "clarify",
                    "history": "chatbot_agent"
                })

        graph_builder.add_edge("rephrase", "rag")
        graph_builder.add_edge("rag", "chatbot_agent")

        return graph_builder.compile(checkpointer=self.memory_saver)

    def display(self):
        display(Image(self.graph.get_graph(xray=True).draw_mermaid_png()))

    def get_last_summary(self):
        messages = get_all_messages_from_thread(self.memory_saver, self.thread_id)

        try:
            last_summary = messages.content[-1].content 
        except IndexError:
            last_summary = "No summary available."

        def extract_student_questions(summary: str) -> str:
            match = re.search(r"##\s*\*{2}Student Questions\*{2}\s*:\s*(.*?)(?:\n\s*##|\Z)", summary, re.DOTALL)
    
            if not match:
                # In case it's the last section with no following "##"
                match = re.search(r"##\s*\*\*Student Questions\*\*\s*:\s*(.*)", summary, re.DOTALL)
            
            if match:
                questions_text = match.group(1).strip()
                questions = re.sub(r"^\s*-\s*", "", questions_text, flags=re.MULTILINE)
                return questions.strip()
            return ""
        
        questions_last_time = extract_student_questions(last_summary)
        if questions_last_time.strip():
            return questions_last_time
        else:
            return "No questions found in the summary."
    
    def get_startup_message(self) -> str:
        if self.last_thread_questions and "No questions" not in self.last_thread_questions:
            return (
                f"""Welcome back **{self.thread_id}**! I created a summary of our last conversation, shown here:\n\n"*{self.last_thread_questions}*"\n\nWould you like to continue with this or start something new?"""
                )
        return ""
        
    def invoke(self, input: str):
        config = {"configurable": {"thread_id": self.state.get('thread_id'), "user_id": "USER"}}
        
        if not self.has_greeted_user:
            startup_msg = self.get_startup_message()
            if startup_msg.strip():
                self.state["messages"].append(AIMessage(content=startup_msg))
            self.has_greeted_user = True  
            # print(state["messages"])

        # Append the user input as a message and then process the graph.
        self.state["messages"].append(HumanMessage(content=input))
        result_state = self.graph.invoke(self.state, config)
        self.state = result_state

        return self.state
    
    def end_session(self):
        # Close the database connection and clean up checkpoints
        summary = finalize_and_prune(thread_id=self.thread_id, memory_saver=self.memory_saver)
        self.conn.close()

        #return f"Conversation summary: \n\n{summary}"
        print("Conversation summarized and checkpoints pruned.")

            
#%%

if __name__ == "__main__":
    from src.utils.global_helpers import load_config
    import time

    config = load_config()
    model_args = config.model.to_dict()
    graph_args = config.graph.to_dict()
    # Create the chatbot graph
    start_time = time.time()
    graph = ChatbotGraph(**model_args,**graph_args, thread_id="speters0")
    print(f"Graph created in {time.time() - start_time:.2f} seconds.")

#%%
    # Display the graph
    graph.display()
#%%
    # Test the chatbot
    response = graph.invoke("What should I major in if I'm interested in flying after I graduate?")

    print(response["messages"][-1].content)

#%%

    response = graph.invoke("I'd like to be a pilot")

    print(response["messages"][-1].content)

#%%
    #print(response["messages"][-1].content)

    response = graph.invoke("What about political science?")

    print(response["messages"][-1].content)

#%%
    #print(response["messages"][-1].content)

    response = graph.invoke("Where did we leave off last time?")

    print(response["messages"][-1].content)
# %%
