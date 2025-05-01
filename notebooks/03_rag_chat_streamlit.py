#%%

import sys
import subprocess
import threading
from pyprojroot.here import here

def launch_streamlit_chat(app_path=here("src/utils/app/chat_ui.py"), port=8501, browser=True):
    """
    Launch the Streamlit chat app in a new thread or subprocess.
    """
    def _run_streamlit():
        # Use Popen for async, redirect output to a log
        with open(str(here("data/chat_histories/session_tracking/streamlit_log.txt")), "w") as log:
            subprocess.Popen(
                ["streamlit", "run", app_path, "--server.port", str(port)],
                stdout=log,
                stderr=log
            )

    def _run_watchdog():
        watchdog_path = here("src/utils/app/watch_idle_sessions.py")
        if watchdog_path.exists():
            with open(str(here("data/chat_histories/session_tracking/dog.log")), "w") as log:
                subprocess.Popen([sys.executable, str(watchdog_path)], #sys.executable to make sure it uses the same python env
                                 stdout=log, stderr=log)
        else:
            print("⚠️ No watchdog found at", watchdog_path)

    # Launch both in background threads
    threading.Thread(target=_run_streamlit, daemon=True).start()
    threading.Thread(target=_run_watchdog, daemon=True).start()

    # if browser:
    #     webbrowser.open(f"http://localhost:{port}")

    print(f"✅ Streamlit app launching on http://localhost:{port}")
    print("Check streamlit_log.txt if it fails to open.")

# %%

launch_streamlit_chat()
# %%
