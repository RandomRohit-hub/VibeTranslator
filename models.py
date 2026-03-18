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
translator = pipeline(
    "translation",
    model="facebook/m2m100_418M"
)

# Intent detection (zero-shot)
intent_pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Sarcasm detection
sarcasm_pipeline = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-irony"
)
