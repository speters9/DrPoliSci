#%%
import uuid

from datetime import datetime

from langchain_core.messages import BaseMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from dataclasses import dataclass
from src.utils.rag_tools import get_model
from src.utils.global_helpers import load_config

config = load_config()
model_args = config.model.to_dict()

@dataclass
class SavedMessages:
    thread_id: str
    content: list[BaseMessage]

SUMMARY_PROMPT = """
You are an academic advisor. Summarize the key points of the following conversation between you and a student. 
Focus on their interests, questions, goals, and any advice or clarifications you gave.

The purpose of this summary is to serve as a reminder for the next time you meet with the student.

A helpful way to summarize is to use the following format. Each section should be no longer than 1-2 sentences:
# Summary of our last conversation:
    ## **Academics**
        - Student Interests, Background, or Goals
    ## **Career**:
        - Student's Desired Career Path or Interests
    ## **Student Questions**: 
        - Student's Questions or Concerns
    ## **Your Advice**: 
        - Your Recommendations or Clarifications (if any)
    ## **Next Steps**: 
        - Decisions Made or Agreed-Upon Actions (if any)
    
## Guidelines:
    - **Do NOT** include information in your summary that was not discussed in the conversation.
        - For example, if a student's career plans were not discussed, do not include that section in the summary. 
    - **Do NOT** include any new information or assumptions that were not part of the conversation.

Here is the conversation history:
    <conversation>
        {messages}
    </conversation>
"""

def get_all_messages_from_thread(memory_saver, thread_id: str) -> list[BaseMessage]:
    config = {"configurable": {"thread_id": thread_id,
                               "user_id": "USER"}}
    all_checkpoints = list(memory_saver.list(config))

    # Accumulate messages from all checkpoints (may have some duplicates)
    messages = SavedMessages(thread_id=thread_id, content=[])
    seen_ids = set()

    for checkpoint in all_checkpoints:
        state = checkpoint.checkpoint  # this is a `State`, which should include `messages`
        for msg in state.get('channel_values', {}).get("messages", []):
            if not hasattr(msg, "type") or not hasattr(msg, "content"):
                continue  # or log and skip
            # Optionally deduplicate by message content or ID
            msg_id = (msg.type, msg.content)
            if msg_id not in seen_ids:
                messages.content.append(msg)
                seen_ids.add(msg_id)

    return messages

def format_messages_for_summary(messages: SavedMessages) -> str:
    lines = []
    thread_id = messages.thread_id
    for msg in messages.content:
        role = thread_id if msg.type == "human" else "Advisor"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def summarize_conversation(memory_saver, thread_id: str, model_name: str, temperature: float, streaming: bool) -> str:
    model = get_model(model_name, temperature=temperature, streaming=streaming)
    messages = get_all_messages_from_thread(memory_saver, thread_id)
    conversation_text = format_messages_for_summary(messages)

    prompt = SUMMARY_PROMPT.format(messages=conversation_text)
    summary = model.invoke(prompt)
    return summary.content

def prune_checkpoints(thread_id: str, memory_saver: SqliteSaver):
    with memory_saver.cursor() as cur:
        cur.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        cur.execute(
            "DELETE FROM writes WHERE thread_id = ?",
            (thread_id,)
        )


def build_summary_metadata(thread_id: str, 
                   summary: str, 
                   ) -> dict:
    summary_message = AIMessage(content=f"Summary of our last conversation:\n\n{summary}")

    metadata = {
                'source': 'loop',
                'writes': {'summary': {'messages': [summary_message],
                                        'session_id': datetime.now().isoformat(),
                                        'thread_id': thread_id,
                                        'summary_model': model_args['model_name']}},
                'thread_id': thread_id,
                'user_id': 'USER',
                'step': 999, # arbitrary high number to indicate summary
                'parents': {},
                }
    return metadata


def finalize_and_prune(thread_id: str, memory_saver):
    # Summarize
    summary = summarize_conversation(memory_saver, thread_id, **model_args)

    # Write summary as new base state
    checkpoint_id = str(uuid.uuid4())
    summary_state = {
        "messages": [AIMessage(content=f"Summary of our last conversation:\n\n{summary}")],
        "session_id": datetime.now().isoformat(),
        "thread_id": thread_id,
    }

    checkpoint = {
        "id": checkpoint_id,
        "ts": datetime.now().isoformat(),
        "channel_values": summary_state,
        "channel_versions": {},         # required for versioning
        "pending_sends": [],            # required for task scheduling
        "pending_receives": {},         # safe to leave empty
        "channel_writes": {},           # can be empty
        "versions_seen": {},            # can be empty
    }

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": "", "checkpoint_id": checkpoint_id}}
    metadata = build_summary_metadata(thread_id, summary)

    # Prune all older checkpoints
    prune_checkpoints(thread_id, memory_saver)
    memory_saver.put(config, checkpoint, metadata, new_versions={})

    return summary
#%%