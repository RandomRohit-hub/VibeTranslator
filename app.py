import streamlit as st
from models import (
    sentiment_pipeline,
    emotion_pipeline,
    translator,
    intent_pipeline,
    sarcasm_pipeline
)

from utils import (
    detect_language,
    translate_to_english,
    detect_intent,
    detect_sarcasm
)

from memory import add_to_history, get_context
from llm import analyze_with_llm

st.set_page_config(page_title="ConvoSense AI", layout="wide")

st.title("💬 ConvoSense AI — Chat Intelligence System")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_input = st.chat_input("Type a message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Add memory
    add_to_history(user_input)
    context = get_context()
    full_text = context + " " + user_input

    # NLP Pipeline
    lang = detect_language(user_input)
    translated = translate_to_english(full_text, translator, lang)

    sentiment = sentiment_pipeline(translated)[0]['label']
    emotion = emotion_pipeline(translated)[0][0]['label']
    intent = detect_intent(translated, intent_pipeline)
    sarcasm = detect_sarcasm(translated, sarcasm_pipeline)

    # LLM Analysis (Ollama)
    analysis = analyze_with_llm(
        user_input,
        emotion,
        sentiment,
        intent,
        sarcasm
    )

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(analysis)

    st.session_state.messages.append(
        {"role": "assistant", "content": analysis}
    )
