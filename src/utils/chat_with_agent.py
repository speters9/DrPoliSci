"""This module creates a framework to align agents with their desired roles."""
# %%
import os
import sqlite3
from pathlib import Path
from pyprojroot.here import here

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, START

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

import tkinter as tk
from tkinter import scrolledtext


# %%
# def run_graph(prompt, graph, query_type="user", thread_id: str = "speters9"):
#     config = {"configurable": {"thread_id": thread_id, "user_id": "USER"}}
#     if not isinstance(prompt, list):
#         query = [(query_type, prompt)]
#     else:
#         query = prompt
#     return graph.invoke({"messages": query}, config=config)


def chat_with_agent(graph: StateGraph):
    """
    Open a popup window for an interactive chat with a given persona.
    'agent' is used to route all messages to/from that persona's memory.
    """
    thread_id = input("Please enter your username. If you do not have one, please create one and remember it. \n\nThis is how the system will remember you: ")

    root = tk.Tk()
    root.title("Chat with Dr. PoliSci")
    saver = graph.memory_saver

    # A comfortable dark gray background (like Spyder's dark theme)
    dark_bg = "#2d2d2d"
    light_fg = "#dcdcdc"
    slightly_lighter_bg = "#3b3b3b"
    color_args = {"fg": light_fg,
                  "insertbackground": light_fg, "relief": tk.FLAT}

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
        user_input.delete("1.0", "end")  # clear the input

        if not prompt:
            return  # ignore empty

        # Display user's message in the chat
        chat_display.insert(tk.END, f"\nYou: {prompt}\n")

        # Call your run_graph function to get the persona's response
        response = graph.invoke(prompt)

        ai_reply = format_lm_response(response['messages'][-1].content)

        # Display the agent's reply
        chat_display.insert(tk.END, f"\nDr. PoliSci: {ai_reply}\n")

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
                            relief=tk.FLAT
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


def chat_with_docs(graph: StateGraph):
    """
    Open a popup window for an interactive chat with a given persona.
    'agent' is used to route all messages to/from that persona's memory.
    """
    root = tk.Tk()
    root.title(f"Chat with Persona: Rag Retriever")
    saver = graph.checkpointer

    # A comfortable dark gray background (like Spyder's dark theme)
    dark_bg = "#2d2d2d"
    light_fg = "#dcdcdc"
    slightly_lighter_bg = "#3b3b3b"
    color_args = {"fg": light_fg,
                  "insertbackground": light_fg, "relief": tk.FLAT}

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
        user_input.delete("1.0", "end")  # clear the input

        if not prompt:
            return  # ignore empty

        # Display user's message in the chat
        chat_display.insert(tk.END, f"\nYou: {prompt}\n")

        # Call your run_graph function to get the persona's response
        response = graph.invoke({"question": prompt})['answer']
        # 'response' might be a string or list of messages. Adjust as needed.
        # Suppose it's just a string for simplicity:
        if isinstance(response, list) and len(response) > 0:
            # assume last element is the AI's reply
            ai_reply = response[-1]
        else:
            # or if run_graph returns something else
            ai_reply = response

        ai_reply = format_lm_response(ai_reply)
        # Display the agent's reply
        chat_display.insert(tk.END, f"\nAgent (Rag Retriever): {ai_reply}\n")

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
                            relief=tk.FLAT
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


# %%
if __name__ == "__main__":
    from contextlib import closing
    from src.utils.prompts import polisci_advisor

    dbDir = here() / "data/vdb/chat_histories.sqlite"

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
    conn = sqlite3.connect(dbDir, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    graph = builder.compile(checkpointer=saver)

    # create the persona
    agent = "project_manager"
    with closing(conn.cursor()) as cursor:
        cursor.execute("""SELECT thread_id 
                        FROM checkpoints 
                        WHERE thread_id = ?""", (agent,))
        threads = cursor.fetchall()

    if not threads:
        persona_inputs = [("system", f"{polisci_advisor.personality}")]
        run_graph(persona_inputs, thread_id=agent)

    chat_with_agent(agent, graph)

    # when done:
    # saver.conn.close()


# %%
