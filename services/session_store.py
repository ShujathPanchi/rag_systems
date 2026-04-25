import json
import os


# ---------------------------------
# SYSTEM FOLDER
# ---------------------------------
os.makedirs(
    "system",
    exist_ok=True
)

CHAT_FILE = "system/chat_history.json"


# ---------------------------------
# LOAD CHAT
# ---------------------------------
def load_chat():

    if os.path.exists(CHAT_FILE):

        try:
            with open(
                CHAT_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                return json.load(f)

        except:
            return []

    return []


# ---------------------------------
# SAVE CHAT
# ---------------------------------
def save_chat(messages):

    with open(
        CHAT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            messages,
            f,
            ensure_ascii=False,
            indent=2
        )


# ---------------------------------
# CLEAR CHAT
# ---------------------------------
def clear_chat():

    save_chat([])