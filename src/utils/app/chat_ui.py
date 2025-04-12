#%%
import streamlit as st
import sys
import time
from pyprojroot.here import here
from streamlit_autorefresh import st_autorefresh
import json
from pathlib import Path

sys.path.append(str(here()))
from src.utils.chatbot import ChatbotGraph
from src.utils.global_helpers import load_config

config = load_config()
MAX_IDLE = config.timeouts.idle_seconds
model_args = config.model.to_dict()
graph_args = config.graph.to_dict()
#%%

st.set_page_config(page_title="Dr. PoliSci", page_icon="🎓", layout="centered")
st.title("🎓 Chat with Dr. PoliSci")

SESSION_TRACK_DIR = here(config.paths.session_tracking)
SESSION_TRACK_DIR.mkdir(exist_ok=True)

# @st.cache_resource
# def load_graph(username: str):
#     return ChatbotGraph(**model_args, **graph_args, thread_id=username)

def save_session_status():
    if 'thread_id' in st.session_state and 'last_ping' in st.session_state:
        session_data = {
            "thread_id": st.session_state.thread_id,
            "last_ping": st.session_state.last_ping,
            "timestamp": time.time()
        }
        with open(SESSION_TRACK_DIR / f"{st.session_state.thread_id}.json", "w") as f:
            json.dump(session_data, f)
    else:
        st.warning("Session state variables 'thread_id' or 'last_ping' are not initialized.")

def clear_session():
    if 'thread_id' in st.session_state:
        session_file = SESSION_TRACK_DIR / f"{st.session_state.thread_id}.json"
        if session_file.exists():
            session_file.unlink()
    else:
        st.warning("Session state variable 'thread_id' is not initialized.")

# set heartbeat
 # 5 minutes
st_autorefresh(interval=config.timeouts.autorefresh_seconds * 1000, key="heartbeat", limit=None)
if "last_ping" not in st.session_state:
    st.session_state["last_ping"] = time.time()


# Initialize session state
if "graph" not in st.session_state:
    st.session_state.graph = None
    st.session_state.username_entered = False
    st.session_state.thread_id = ""
    st.session_state.chat_log = []

if "session_active" not in st.session_state:
    st.session_state.session_active = True

if st.session_state.get("clear_prompt"):
    st.session_state.prompt = ""
    st.session_state.clear_prompt = False


# Username prompt (first run)
if not st.session_state.username_entered:
    username = st.text_input("Enter your username to begin:  \n*(may take a few seconds to load chat history)*")
    if username:
        st.session_state.graph = ChatbotGraph(**model_args, **graph_args, thread_id=username) #load_graph(username=username)
        st.session_state.username_entered = True
        st.session_state.thread_id = username
        if st.session_state.graph.get_startup_message():
            st.session_state.chat_log.append(("Dr. PoliSci", st.session_state.graph.get_startup_message()))
        else:
            st.session_state.chat_log.append(("Dr. PoliSci", f"Hello, **{username}**! How can I help you today?"))
        st.session_state["last_ping"] = time.time()
    
        save_session_status()
        st.rerun()



# Chat UI
if st.session_state.username_entered:
    for speaker, message in st.session_state.chat_log:
        st.markdown(f"**{speaker}:** {message}", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=False):
        prompt = st.text_area(
            "Your message:",
            key="prompt",
            height=100,
            placeholder="Type your question...",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Send")

        if submitted and st.session_state.prompt.strip():
            st.session_state.chat_log.append(("You", st.session_state.prompt))

            result = st.session_state.graph.invoke(st.session_state.prompt)
            reply = result["messages"][-1].content.strip()
            st.session_state.chat_log.append(("Dr. PoliSci", reply))

            st.session_state["last_ping"] = time.time()
            save_session_status()

            st.session_state.clear_prompt = True
            st.rerun()


    # prompt = st.text_area("Your message:", 
    #                       height=100, 
    #                       key="prompt",
    #                       placeholder="Type your question...")

    # if st.button("Send", type="primary") and prompt.strip():
    #     st.session_state.chat_log.append(("You", prompt))

    #     # Get AI response
    #     result = st.session_state.graph.invoke(prompt)
    #     reply = result["messages"][-1].content.strip()
    #     st.session_state.chat_log.append(("Dr. PoliSci", reply))

    #     # # Add a placeholder for streaming output
    #     # response_placeholder = st.empty()

    #     # # Stream response from advisor
    #     # streamed_text = ""
    #     # response_stream = st.session_state.graph.advisor_stream(prompt)  # We'll add this method next

    #     # for chunk in response_stream:
    #     #     if hasattr(chunk, "content"):
    #     #         streamed_text += chunk.content
    #     #         response_placeholder.markdown(f"**Dr. PoliSci:** {streamed_text}", unsafe_allow_html=True)

    #     # # Save the full response after stream ends
    #     # st.session_state.chat_log.append(("Dr. PoliSci", streamed_text.strip()))
    #     st.session_state["last_ping"] = time.time()

    #     save_session_status()

    #     st.session_state.clear_prompt = True
    #     st.rerun()
    
    if st.button("End Chat"):
        if st.session_state.graph and not st.session_state.get("session_ended", False):
            st.session_state.graph.end_session()
            clear_session()
        st.session_state.session_ended = True
        st.success("Session ended and summarized.")


    if "graph" in st.session_state and not st.session_state.get("session_ended", False):
        last_ping = st.session_state.get("last_ping", time.time())
        idle_time = time.time() - last_ping
        st.caption(f"⏱ Last heartbeat: {round(idle_time, 0)} sec ago")

        if idle_time > MAX_IDLE and not st.session_state.get("session_ended", False):
            st.session_state.graph.end_session()
            clear_session()
            st.session_state.session_ended = True
            st.warning("Session ended due to inactivity. Please close and restart the chat.")





# %%
