"""This module creates a framework to align agents with their desired roles."""
# %%
import os
import sqlite3
from pathlib import Path
from pyprojroot.here import here

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, MessagesState, START

from langchain_openai import ChatOpenAI

from src.utils.chatbot import ChatbotGraph
import time
from dotenv import load_dotenv

import tkinter as tk
from tkinter import scrolledtext, Scrollbar
from tkhtmlview import HTMLLabel
import markdown

from src.utils.global_helpers import load_config

config = load_config()
model_args = config.model.to_dict()

# %%


def chat_with_agent():
    """
    Open a popup window for an interactive chat with a given persona.
    'agent' is used to route all messages to/from that persona's memory.
    """

    root = tk.Tk()
    root.title("Chat with Dr. PoliSci")

    # A comfortable dark gray background (like Spyder's dark theme)
    dark_bg = "#2d2d2d"
    light_fg = "#dcdcdc"
    slightly_lighter_bg = "#3b3b3b"
    color_args = {"fg": light_fg,
                  "insertbackground": light_fg, "relief": tk.FLAT}

    #root.configure(bg=light_fg)

    # A ScrolledText widget to display the conversation
    # chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=20,
    #                                          bg=dark_bg,
    #                                          **color_args)
    # read markdown add
    chat_display = HTMLLabel(root, html="", width=60, height=20,
                                             bg=light_fg,
                                             **color_args)
    chat_display.pack(expand=True, fill='both')

    # An Entry widget for user to type new messages
    user_input = tk.Text(root,
                         wrap=tk.WORD,      # enable word-wrap
                         bg=light_fg,
                         fg=slightly_lighter_bg,
                         insertbackground=slightly_lighter_bg,
                         height=4)  # display ~3 text lines
    user_input.pack(fill='x', padx=5, pady=5)

    # Initial setup: Prompt for username
    username_entered = False
    graph = None 
    html_log = ""

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
        nonlocal username_entered, graph, html_log  

        prompt = user_input.get("1.0", "end-1c").strip()
        user_input.delete("1.0", "end")

        if not prompt:
            return
        
        # read markdown add    
        html_response = markdown.markdown(f"**You:**<br>{prompt}")

        #chat_display.insert(tk.END, f"\nYou: {prompt}\n")

        # if not username_entered:
        #     # First message sets username and initializes graph
        #     thread_id = prompt

        #     graph = ChatbotGraph(model_name="gpt-4o-mini", temperature=0.3, thread_id=thread_id)
            
        #     if not graph.memory_saver.get(config = {"configurable": {"thread_id": thread_id, "user_id": "USER"}}):
        #         chat_display.insert(tk.END, f"\nDr. PoliSci: Hello, {thread_id}! Looks like I don't have a record of our prior conversations. "
        #                                     "How can I help you today?\n")
        #     else:
        #         chat_display.insert(tk.END, f"\nDr. PoliSci: Hello, {thread_id}! How can I help you today?\n")

        #     username_entered = True
        # else:
        #     # Normal message flow after username setup
        #     response = graph.invoke(prompt)
        #     ai_reply = format_lm_response(response['messages'][-1].content)
        #     chat_display.insert(tk.END, f"\nDr. PoliSci: {ai_reply}\n")


        if not username_entered:
            # First input sets up the thread ID
            thread_id = prompt
            graph = ChatbotGraph(**model_args, thread_id=thread_id)

            # Personalized greeting
            if not graph.memory_saver.get(config={"configurable": {"thread_id": thread_id, "user_id": "USER"}}):
                reply = f"Hello, {thread_id}! Looks like I don't have a record of our prior conversations. How can I help you today?"
            else:
                reply = f"Hello, {thread_id}! How can I help you today?"

            username_entered = True
        else:
            # NORMAL CHAT FLOW
            response = graph.invoke(prompt)
            reply = response["messages"][-1].content

        # read markdown add
        html_response += markdown.markdown(f"**Dr. PoliSci:**<br>{reply}")
        html_log += html_response
        html_response += "<hr>"
        chat_display.set_html(html_log)
        #chat_display.see(tk.END)

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
                            # bg=light_fg, fg=light_fg,
                            # activebackground="#444444", activeforeground=light_fg,
                            fg="black",  # optional for more contrast
                            activebackground=light_fg,
                            activeforeground="black",
                            background=light_fg,
                            borderwidth=1,
                            relief=tk.FLAT
                            )
    send_button.pack(padx=5, pady=5)

    def quit_chat():
        # close out the chat
        if graph and graph.memory_saver:
            graph.end_session()
        root.destroy()

    def on_closing():
        quit_chat()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Initial prompt for username in GUI
    # chat_display.insert(tk.END, "Dr. PoliSci: Please enter your username.\n\n"
    #                             "If you do not have one, please create one and remember it. "
    #                             "This is how the system will remember you. "
    #                             "\n\nAfter you enter your username, it might take a few seconds to load our conversation records. "
    #     )

    # read markdown add
    intro = """
    <p><b>Dr. PoliSci:</b> Please enter your username.</p>
    <p>If you do not have one, please create one and remember it. This is how the system will remember you.</p>
    <p>It might take a few seconds to load the memory...</p>
    """
    chat_display.set_html(intro)
 
    root.mainloop()


# %%
if __name__ == "__main__":
    from contextlib import closing
    from src.utils.prompts import polisci_advisor


    chat_with_agent()

    # when done:
    # saver.conn.close()


# %%
