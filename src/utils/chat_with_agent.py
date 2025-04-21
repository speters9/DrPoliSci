"""This module creates a framework to align agents with their desired roles."""
# %%
import sys
import markdown
import tkinter as tk
from tkinter import Canvas, Scrollbar, Frame
from tkhtmlview import HTMLText
import markdown
from pyprojroot.here import here

sys.path.append(str(here()))
from src.utils.global_helpers import load_config
from src.utils.chatbot import ChatbotGraph

avatar_path = here("static/drps_avatar.png")
avatar_uri = avatar_path.as_uri()

#%%
config = load_config()
model_args = config.model.to_dict()
graph_args = config.graph.to_dict()

FONT = "Tahoma" # "Arial" # "Segoe UI" 
SIZE = 11
text_font = (FONT, SIZE)

# %%
def render_markdown(md_text: str, font=FONT, size=SIZE, color="black") -> str:
    """
    Convert Markdown to HTML for use in tkhtmlview's HTMLLabel.
    
    - Handles newline conversion
    - Adjusts font size and color
    - Preferred fonts: "Segoe UI", "Arial"
    """
    
    md_text = md_text.replace("\\n", "\n").replace("\n", "  \n")
    html_body = markdown.markdown(md_text, extensions=["extra"])
    html_body = html_body.replace("<em>", "<i>").replace("</em>", "</i>")
    html_body = html_body.replace("<strong>", "<b>").replace("</strong>", "</b>")

    formatted_html = f"""<font face="{font}" size="{size}" color="{color}">
        {html_body}
        </font>"""
    return formatted_html

def chat_with_agent():
    """
    Open a popup window for an interactive chat with a given persona.
    'agent' is used to route all messages to/from that persona's memory.
    """

    root = tk.Tk()
    root.title("Chat with Dr. PoliSci")
    root.iconbitmap(default=here("static/drps_icon.ico"))

    light_fg = "#dcdcdc"
    slightly_lighter_bg = "#3b3b3b"
    color_args = {"fg": light_fg,
                  "insertbackground": light_fg, "relief": tk.FLAT}
    
    chat_frame = Frame(root, bg=light_fg)
    chat_frame.pack(fill="both", expand=True)

    # Create canvas inside the frame
    chat_canvas = Canvas(chat_frame, bg="white", highlightthickness=0)
    chat_canvas.pack(side="left", fill="both", expand=True)

    # Scrollbar next to the canvas
    scrollbar = Scrollbar(chat_frame, command=chat_canvas.yview)
    scrollbar.pack(side="right", fill="y")
    chat_canvas.configure(yscrollcommand=scrollbar.set)

    def resize_label(event):
        chat_display.configure(width=event.width)

    chat_canvas.bind("<Configure>", resize_label)   

    # Frame *inside* canvas that holds the actual HTMLLabel
    chat_inner_frame = Frame(chat_canvas, bg=light_fg)
    chat_window_id = chat_canvas.create_window((0, 0), window=chat_inner_frame, anchor="nw")

    # Make the canvas scroll when the inner frame grows
    def on_configure(event):
        chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))

    chat_inner_frame.bind("<Configure>", on_configure)

    def on_canvas_resize(event):
        chat_canvas.itemconfig(chat_window_id, width=event.width, height=event.height)

    chat_canvas.bind("<Configure>", on_canvas_resize)

    chat_display = HTMLText(chat_inner_frame, 
                             html="", 
                             width=60, height=20,
                             bg=light_fg,
                             **color_args)
    chat_display.pack(anchor="nw", fill="both", expand=True)

    # An Entry widget for user to type new messages
    user_input = tk.Text(root,
                         wrap=tk.WORD,      # enable word-wrap
                         bg=light_fg,
                         fg="black",
                         insertbackground=slightly_lighter_bg,
                         font=text_font,
                         height=4)  # display ~3 text lines
    user_input.pack(fill='x', padx=5, pady=5)

    # Initial setup: Prompt for username
    username_entered = False
    graph = None 
    html_log = ""


    def send_message(event=None):
        nonlocal username_entered, graph, html_log  

        prompt = user_input.get("1.0", "end-1c").strip()
        user_input.delete("1.0", "end")

        if not prompt:
            return
        
        # read markdown add    
        html_response = render_markdown(f"**You:**<br>{prompt}")

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
            graph = ChatbotGraph(**model_args, **graph_args, thread_id=thread_id)

            # Personalized greeting
            startup = graph.get_startup_message()
            if startup:
                reply = startup
            else:
                reply = f"Hello, **{thread_id}**! Looks like I have no prior conversation history.\n\nHow can I help you today?"
            username_entered = True
        else:
            # NORMAL CHAT FLOW
            response = graph.invoke(prompt)
            reply = response["messages"][-1].content

        def scroll_to_bottom():
            chat_canvas.update_idletasks()
            chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
            chat_canvas.yview_moveto(1.0)

        # read markdown add
        html_response += render_markdown(f"**Dr. PoliSci:**\n{reply}")
        html_log += html_response
        html_response += "<hr>"
        chat_display.set_html(html_log)
        root.after(100, scroll_to_bottom)
        


        root.after(50, lambda: root.after(50, scroll_to_bottom))
        chat_display.see(tk.END)

        # A Send button if user doesn't want to press Enter
    send_button = tk.Button(root, text="Send", command=send_message,
                            fg="black",  # optional for more contrast
                            activebackground=light_fg,
                            activeforeground="black",
                            background=light_fg,
                            borderwidth=1,
                            relief=tk.FLAT
                            )
    send_button.pack(padx=5, pady=5)

    # Pressing Enter in the Entry triggers send_message
    def on_return(event):
        """
        If SHIFT is pressed, insert a newline; otherwise send the message.
        """
        # Newline if Shift + Enter is pressed. Otherwise, send the message on Enter
        SHIFT_HEX = 0x0001
        if event.state & SHIFT_HEX:
            user_input.insert(tk.INSERT, "\n")
        else:
            # Visually trigger the button press (depress + callback)
            # Simulate visual feedback
            send_button.config(relief=tk.SUNKEN)
            root.update_idletasks()

            # Wait a short moment to show the press
            root.after(100, lambda: send_button.config(relief=tk.FLAT))

            send_button.invoke()

        return "break"

    # Bind the key press event for Return
    user_input.bind("<Return>", on_return)

    def quit_chat():
        # close out the chat
        if graph and graph.memory_saver:
            graph.end_session()
        root.destroy()

    def on_closing():
        quit_chat()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # read markdown add
    intro = render_markdown("""
    **Dr. PoliSci:**\nPlease enter your username.\n\nIf you do not have one, please create one and remember it. This is how the system will remember you.\n\n*It might take a few seconds to load the memory...*
    """)
    chat_display.set_html(intro)
 
    root.mainloop()


# %%
if __name__ == "__main__":


    chat_with_agent()

# %%
