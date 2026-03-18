
import nbformat as nbf
import os

os.makedirs("d:/tomato/ml_model_notebook", exist_ok=True)

nb = nbf.v4.new_notebook()

cell1 = """# importing stuff I found on stackoverflow
from transformers import pipeline
from langdetect import detect
import requests
import json
"""

cell2 = """print("loading my models... this takes forever lol")

# sentiment model
my_sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# emotion
my_emotion = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# translation model so I can understand hindi
my_translator = pipeline("translation", model="facebook/m2m100_418M")

# intent stuff
my_intent = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# sarcasm
my_sarcasm = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

print("done loading!")
"""

cell3 = """# list to keep previous chats
my_chat_history = []

def add_chat(txt):
    my_chat_history.append(txt)
    # keep only 5
    if len(my_chat_history) > 5:
        my_chat_history.pop(0)

def detect_lang_simple(t):
    try:
        ans = detect(t)
        return ans
    except:
        return "idk"

def do_translation(t, lang_is):
    # if I don't know the language I just assume hindi
    if lang_is == "idk":
        lang_is = "hi"
    
    if lang_is == "en":
        return t
        
    res = my_translator(t, src_lang=lang_is, tgt_lang="en")
    return res[0]['translation_text']
"""

cell4 = """# using ollama locally for free!
my_url = "http://localhost:11434/api/generate"

def ask_ollama(text, emo, sent, intnt, sarc):
    my_prompt = f"Analyze this:\\nMessage: {text}\\nEmotion: {emo}\\nSentiment: {sent}\\nIntent: {intnt}\\nSarcasm: {sarc}\\nTell me what they mean and suggest a reply."
    
    # sending request
    req_data = {
        "model": "llama3",
        "prompt": my_prompt,
        "stream": False
    }
    
    resp = requests.post(my_url, json=req_data)
    
    try:
        jsn = resp.json()
        if "error" in jsn:
            return "error: " + jsn["error"]
        return jsn["response"]
    except:
        return "broke"
"""

cell5 = """# let's test it out!
test_msg = "Tumko time hi nahi hai mere liye"
print("Input:", test_msg)

# 1. detect lang
l = detect_lang_simple(test_msg)
print("Language is:", l)

# 2. translate
translated_msg = do_translation(test_msg, l)
print("English translation:", translated_msg)

# 3. get emotion
e = my_emotion(translated_msg)[0][0]['label']
s = my_sentiment(translated_msg)[0]['label']
i = my_intent(translated_msg, ["complaint", "request", "casual talk"])['labels'][0]
sarc = my_sarcasm(translated_msg)[0]['label']

print("Emotion:", e)
print("Sentiment:", s)
print("Intent:", i)
print("Sarcasm:", sarc)

# 4. finally ask llm
final_ans = ask_ollama(test_msg, e, s, i, sarc)
print("\\n--- AI RESPONSE ---\\n")
print(final_ans)
"""

nb['cells'] = [
    nbf.v4.new_code_cell(cell1),
    nbf.v4.new_code_cell(cell2),
    nbf.v4.new_code_cell(cell3),
    nbf.v4.new_code_cell(cell4),
    nbf.v4.new_code_cell(cell5)
]

nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata.language_info = {"name": "python"}

with open('d:/tomato/ml_model_notebook/rookie_model.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook rookie_model.ipynb created successfully.")
