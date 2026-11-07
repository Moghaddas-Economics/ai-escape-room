"""
AI Escape Room - Version 1 (Interactive AI Escape Room)
The LLM (Gemini) acts as the Game Master. Run with:  streamlit run app_v1.py
"""
import os
import re

import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-3.8-flash"  # check the current free model name in Google AI Studio


# ---------------------------------------------------------------
# API key: read from Streamlit secrets or an environment variable
# (never hard-code it in this file)
# ---------------------------------------------------------------
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


API_KEY = get_api_key()

st.set_page_config(page_title="AI Escape Room - Version 1", page_icon="🗝️")
st.title("🗝️ AI Escape Room - Version 1")

if not API_KEY:
    st.error(
        "No API key found. Put GEMINI_API_KEY in .streamlit/secrets.toml "
        "or set it as an environment variable, then restart the app."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)

# ---------------------------------------------------------------
# Game design (given to the LLM as its rules)
# ---------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are the Game Master of a text-based escape room set in a locked, dimly lit library.
Stay in character, describe things vividly but briefly (3-6 sentences), and never break the fiction.

THE ROOM (these are the ONLY interactive objects):
1. Bookshelf: four books are arranged differently from the rest. Each has a handwritten
   number inside: 4, 1, 7, 3 (in the order they stand on the shelf, left to right).
2. Desk: sturdy wooden desk with a locked drawer and a few dusty papers (no clues).
3. Desk drawer: locked with a 4-digit code. The correct code is 4173 (the numbers in the books).
   Inside is a small BRASS KEY.
4. Oil painting: hangs slightly crooked. Behind it is a small wall box with a keyhole.
   The BRASS KEY opens it. Inside is a NOTE reading:
   "Time stopped at the hour of the great clock. Hour first, then minutes, opens the way."
5. Grandfather clock: stopped. Its hands show 9:47.
6. Exit door: secured by a 4-digit keypad. The correct code is 0947.

PUZZLE CHAIN:
- Puzzle 1: Find the four numbers in the books -> use 4173 on the drawer -> get the brass key.
- Puzzle 2: Use the brass key on the wall box behind the painting -> get the note.
- Puzzle 3: Combine the note with the clock (9:47) -> enter 0947 on the door keypad.

RULES:
- Only describe the objects listed above. If the player asks about something else, say it is
  not there or not important.
- Reveal clues only when the player examines the right object. Do not give the codes away.
  If the player is stuck, give a gentle hint, not the answer.
- The drawer needs code 4173. The wall box needs the brass key. The door needs 0947.
  Wrong codes do not work.
- Whenever the player picks up or receives an item, add the tag [ITEM: item name] at the end
  of your reply (for example [ITEM: Brass Key] or [ITEM: Mysterious Note]).
- When the player enters the correct door code 0947, describe their escape and end your reply
  with the tag [ESCAPED].
"""

INTRO = (
    "You wake up on the cold floor of a dimly lit library. The air smells of old paper. "
    "You can make out a **desk**, a tall **bookshelf**, an **oil painting**, a stopped "
    "**grandfather clock**, and a heavy **exit door** with a 4-digit electronic keypad. "
    "What do you do?"
)

HOW_TO_PLAY = """
**How to play**
- Type what you want to do in plain English, e.g. *"I examine the bookshelf"* or *"I try 1234 on the door"*.
- Look at objects, pick things up, combine clues and try codes.
- Goal: escape through the exit door.
- Use **Restart Game** in the sidebar to start over.
"""


# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
def reset_game():
    st.session_state.messages = [{"role": "model", "text": INTRO}]
    st.session_state.inventory = []
    st.session_state.escaped = False


if "messages" not in st.session_state:
    reset_game()

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Game Status")
    st.markdown(HOW_TO_PLAY)
    st.subheader("Inventory")
    if st.session_state.inventory:
        for item in st.session_state.inventory:
            st.write(f"- {item}")
    else:
        st.write("Your pockets are empty.")
    st.divider()
    if st.button("Restart Game"):
        reset_game()
        st.rerun()

# ---------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message("assistant" if msg["role"] == "model" else "user"):
        st.markdown(msg["text"])

if st.session_state.escaped:
    st.success("🎉 You escaped the library! Click 'Restart Game' to play again.")
    st.stop()

# ---------------------------------------------------------------
# Handle player input
# ---------------------------------------------------------------
if prompt := st.chat_input("What do you want to do?"):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send the whole conversation so the Game Master remembers earlier actions
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
        for m in st.session_state.messages
    ]

    with st.spinner("The Game Master is thinking..."):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                ),
            )
            reply = response.text or "The room is silent. (No response, try again.)"
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.session_state.messages.pop()  # remove the unanswered message
            st.stop()

    # Read the tags the Game Master added, then hide them from the player
    for item in re.findall(r"\[ITEM:\s*([^\]]+)\]", reply):
        item = item.strip()
        if item and item not in st.session_state.inventory:
            st.session_state.inventory.append(item)
    if "[ESCAPED]" in reply:
        st.session_state.escaped = True

    clean_reply = re.sub(r"\[ITEM:[^\]]*\]|\[ESCAPED\]", "", reply).strip()
    st.session_state.messages.append({"role": "model", "text": clean_reply})
    st.rerun()