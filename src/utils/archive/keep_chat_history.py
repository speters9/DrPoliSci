"""This module contains a class that implements a chat history that stores messages in a file."""
# %%
import os
from typing import Sequence
import sqlite3
from pathlib import Path
from pyprojroot.here import here
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, MessagesState, START


class SQLiteConversationMemory:
    def __init__(self, table_name: str, builder: StateGraph,  db_path: Path = here()/"data/conversation_db"):
        self.table_name = table_name
        self.db_path = db_path
        self.config = {"configurable": {
            "thread_id": "primary", "user_id": "USER"}}

    def load_history(self):
        with SqliteSaver.from_conn_string(self.db_path) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            query = [h for h in graph.get_state_history(self.config)]
            records = query[0].values['messages']

            return records


# %%
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from pyprojroot.here import here
    from src.utils.personas import project_manager

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    import tkinter as tk
    from tkinter import scrolledtext

    dbDir = here() / "data/conversation_database"
    dbDir_pm = dbDir / project_manager.db_name

    load_dotenv()
    OPENAI_KEY = os.getenv("OPENAI_KEY")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_KEY)

    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")


    # create new saver instance instead of context manager -- need to close connection manually
    conn = sqlite3.connect(dbDir_pm, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    # %%
    def run_graph(prompt, query_type="user", thread_id:str = "primary"):
        config = {"configurable": {"thread_id": thread_id, "user_id": "USER"}}
        if not isinstance(prompt, list):
            query = [(query_type, prompt)]
        else:
            query = prompt
        return graph.invoke({"messages": query}, config=config)
        

    # create the persona
    persona_inputs = [("system", f"{project_manager.personality}")]
    run_graph(persona_inputs, thread_id="project_manager")



    def chat_with_agent(agent="project_manager"):
        """
        Open a popup window for an interactive chat with a given persona.
        'agent' is used to route all messages to/from that persona's memory.
        """
        root = tk.Tk()
        root.title(f"Chat with Persona: {agent}")

        # A comfortable dark gray background (like Spyder's dark theme)
        dark_bg = "#2d2d2d"
        light_fg = "#dcdcdc"
        slightly_lighter_bg = "#3b3b3b"
        color_args = {"fg": light_fg, "insertbackground": light_fg, "relief": tk.FLAT}

        root.configure(bg=dark_bg)

        # A ScrolledText widget to display the conversation
        chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20,
                                                bg=dark_bg,
                                             **color_args)
        chat_display.pack(expand=True, fill='both')

        # An Entry widget for user to type new messages
        user_input = tk.Text(root, 
                     wrap=tk.WORD,      # enable word-wrap
                     bg=slightly_lighter_bg, 
                     fg=light_fg,
                     insertbackground=light_fg,
                     height=3)  # display ~3 text lines
        user_input.pack(fill='x', padx=5, pady=5)

        def format_lm_response(raw_text: str) -> str:
            """
            Convert escaped newline chars, etc., to real newlines.
            Optionally handle bullet points or other styling needs.
            """
            # Simple approach: replace literal "\n" with an actual newline
            text = raw_text.replace("\\n", "\n")
            # If you see literal "\\t", you might do text = text.replace("\\t", "    ")
            return text


        def send_message(event=None):
            """Handle sending user input and displaying model response."""
            prompt = user_input.get("1.0", "end-1c").strip()  
            user_input.delete("1.0", "end") # clear the input

            if not prompt:
                return  # ignore empty

            # Display user's message in the chat
            chat_display.insert(tk.END, f"You: {prompt}\n")

            # Call your run_graph function to get the persona's response
            response = run_graph(prompt, query_type="user", thread_id=agent)['messages']
            # 'response' might be a string or list of messages. Adjust as needed.
            # Suppose it's just a string for simplicity:
            if isinstance(response, list) and len(response) > 0:
                # assume last element is the AI's reply
                ai_reply = response[-1].content
            else:
                # or if run_graph returns something else
                ai_reply = response.content

            ai_reply = format_lm_response(ai_reply)
            # Display the agent's reply
            chat_display.insert(tk.END, f"Agent ({agent}): {ai_reply}\n")

            # Optionally scroll to bottom
            chat_display.see(tk.END)

        # Pressing Enter in the Entry triggers send_message
        def on_return(event):
            """
            If SHIFT is pressed, insert a newline; otherwise send the message.
            """
            # event.state is a bitmask of modifier keys.
            # SHIFT typically sets bit 0x0001 (1), but can vary by system.
            # We'll check if SHIFT is pressed:
            
            SHIFT_HEX = 0x0001
            if event.state & SHIFT_HEX:
                # Insert a literal newline
                user_input.insert(tk.INSERT, "\n")
            else:
                # Normal Enter -> send message
                send_message()

            # "break" stops the default Text behavior
            return "break"

        # Bind the key press event for Return
        user_input.bind("<Return>", on_return)

        # A Send button if user doesn't want to press Enter
        send_button = tk.Button(root, text="Send", command=send_message,
                            bg=slightly_lighter_bg, fg=light_fg, 
                            activebackground="#444444", activeforeground=light_fg,
                            relief = tk.FLAT
                            )
        send_button.pack(padx=5, pady=5)

        def quit_chat():
            # close out db connection
            if saver is not None:
                saver.conn.close()
            root.destroy()

        def on_closing():
            quit_chat()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        root.mainloop()

    # when done:
    #saver.conn.close()


# %%












# %%
    from typing import List, Optional, Literal, TypedDict, Annotated

    from langchain.agents import initialize_agent, Tool
    from langchain.agents.agent_types import AgentType
    from langchain_core.tools import tool
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    from langchain_core.runnables.config import RunnableConfig

    from langgraph.prebuilt import create_react_agent, InjectedStore
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.types import Command
    from langgraph.store.base import BaseStore
    from langgraph.store.memory import InMemoryStore
    from langgraph.checkpoint.memory import MemorySaver

    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver

    class AgentState(TypedDict):
        messages: List[BaseMessage]
        foo: str

    def format_for_model(state: AgentState):
        # You can do more complex modifications here
        return prompt.invoke({"messages": state["messages"]})

    def save_memory(memory: str, *, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]) -> str:
        '''Save the given memory for the current user.'''
        # This is a **tool** the model can use to save memories to storage
        user_id = config.get("configurable", {}).get("user_id")
        namespace = ("memories", user_id)
        store.put(namespace, f"memory_{len(store.search(namespace))}", {
                  "data": memory})
        return f"Saved memory: {memory}"

    def prepare_model_inputs(state: AgentState, config: RunnableConfig, store: BaseStore):
        # Retrieve user memories and add them to the system message
        # This function is called **every time** the model is prompted. It converts the state to a prompt
        user_id = config.get("configurable", {}).get("user_id")
        namespace = ("memories", user_id)
        memories = [m.value["data"] for m in store.search(namespace)]
        system_msg = f"User memories: {', '.join(memories)}"
        return [{"role": "system", "content": system_msg}] + state["messages"]

    store = InMemoryStore()

    agent = create_react_agent(
        model=llm,
        tools=[],
        # state_modifier=prepare_model_inputs,
        # store=store,
        checkpointer=memory
    )

# %%
    from langgraph.graph import StateGraph, MessagesState, START

    def call_model(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")

# %%

    with SqliteSaver.from_conn_string(dbDir / "checkpoints.sqlite") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        inputs = [{"type": "system",
                   "content": f"{project_manager.personality}"},
                  {"type": "user",
                   "content": "What questions do you have about the research you will be starting?"}
                  ]
        graph.invoke(
            {"messages": ["How many questions did you just ask me back?"]}, config=config)
