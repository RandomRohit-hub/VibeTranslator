from transformers import pipeline

# Sentiment (multilingual)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

# Emotion detection
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# Translation (multilingual)
# Translation (multilingual) - manual loading to avoid pipeline errors
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
_model_name = "facebook/m2m100_418M"
tokenizer_m2m = M2M100Tokenizer.from_pretrained(_model_name)
model_m2m = M2M100ForConditionalGeneration.from_pretrained(_model_name)

def get_safe_pipeline(task, model, **kwargs):
    """
    Attempts to load a pipeline normally. If any error occurs (like connection issues),
    it falls back to loading from local files only.
    """
    try:
        # First attempt: normal load (allows checking for updates)
        return pipeline(task, model=model, **kwargs)
    except Exception as e:
        # Fallback to local files only for ANY error (connection, etc.)
        # This is robust for offline/unstable environments
        print(f"⚠️  Normal load failed for {model}. Error: {str(e)[:100]}... Falling back to local cache.")
        try:
            return pipeline(task, model=model, local_files_only=True, **kwargs)
        except Exception as e2:
            print(f"❌ Failed to load {model} even from cache: {e2}")
            raise e2

# We pack them into a simple object so utils.py can still treat it like a 'translator'
class Translator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    def __call__(self, text, src_lang, tgt_lang="en"):
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang))
        return [{"translation_text": self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]}]

translator = Translator(model_m2m, tokenizer_m2m)

# Intent detection (zero-shot)
intent_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

import streamlit as st
import os as _os

@st.cache_resource
def load_sarcasm_model(path_or_name, is_custom=False):
    if is_custom:
        return pipeline("text-classification", model=path_or_name, tokenizer=path_or_name)
    return pipeline("text-classification", model=path_or_name)

_sarcasm_model_path = _os.path.join(_os.path.dirname(__file__), "sarcasm_model")

if _os.path.exists(_sarcasm_model_path):
    sarcasm_pipeline = load_sarcasm_model(_sarcasm_model_path, is_custom=True)
    print("✅ Loaded custom trained sarcasm model (cached)")
else:
    sarcasm_pipeline = load_sarcasm_model("cardiffnlp/twitter-roberta-base-irony")
    print("ℹ️  Using default sarcasm model (cached)")
