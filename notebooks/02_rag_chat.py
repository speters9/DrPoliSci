#%%
from src.utils.chatbot import ChatbotGraph

from src.utils.chat_with_agent import chat_with_agent

#%%

chatbot_graph = ChatbotGraph(model_name="gpt-4o-mini", temperature=0.3, thread_id="test_thread")

# Display the graph
chatbot_graph.display()

# %%
chat_with_agent(chatbot_graph)

#%%