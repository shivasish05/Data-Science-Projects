import streamlit as st
import os
import openai
from streamlit_chat import message
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def api_calling(prompt):
    messages = [
        {"role": "system", "content":"Translate the following English text into Hinglish, a combination of English and Hindi (Devanagari script). Maintain the style of having the Hindi part in Devanagari script. Only translate specific words into Hindi while leaving the rest in English. Take the below given example:Statement: I had about a 30-minute demo just using this new headset.Desired result: मझे थी एक ३०-मिनट की डेमो, using इस नए headset."},
        {"role": "user", "content": prompt},
    ]

    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.5,
    )

    response = completions.choices[0].message['content']
    return response

st.title("Language Translater Listed Assignment")

if 'user_input' not in st.session_state:
    st.session_state.user_input = []

if 'openai_response' not in st.session_state:
    st.session_state.openai_response = []

user_input = st.text_area("Write your text in English", key="input")

if st.button("Translate"):
    if user_input:
        # Check if the user input has changed to avoid recomputation
        if not st.session_state.user_input or user_input != st.session_state.user_input[-1]:
            output = api_calling(user_input).strip()
            st.session_state.openai_response.append(output)
            st.session_state.user_input.append(user_input)

message_history = st.empty()

if st.session_state.user_input:
    for i in range(len(st.session_state.user_input) - 1, -1, -1):
        message(st.session_state.user_input[i], key=str(i), avatar_style="icons")
        message(st.session_state.openai_response[i], avatar_style="miniavs", is_user=True, key=str(i) + 'data_by_user')
