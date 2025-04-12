import time
import json
import sys
from pathlib import Path
from pyprojroot.here import here
import logging 
sys.path.append(str(here()))
from src.utils.chatbot import ChatbotGraph  # Make sure this can reload your graph
from src.utils.global_helpers import load_config

config = load_config()
tracker_path = here(config.paths.session_tracking)
model_args = config.model.to_dict()
MAX_IDLE = config.timeouts.idle_seconds
POLL_TIME = config.timeouts.watchdog_poll_seconds

log_path = tracker_path / "dog.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(log_path),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logging.info("Watchdog script started.")

while True:
    time.sleep(POLL_TIME)  # Check every min
    for file in tracker_path.glob("*.json"):
        with open(file) as f:
            data = json.load(f)

        last_ping = data.get("last_ping", 0)
        thread_id = data["thread_id"]

        if time.time() - last_ping > MAX_IDLE:
            logging.info(f"[watchdog] Session '{thread_id}' is idle. Summarizing and cleaning up...")

            try:
                graph = ChatbotGraph(**model_args, thread_id=thread_id)
                graph.end_session()
                logging.info(f"✅ Session {thread_id} summarized.")
                file.unlink()
            except Exception as e:
                logging.info(f"❌ Failed to close session {thread_id}: {e}")
