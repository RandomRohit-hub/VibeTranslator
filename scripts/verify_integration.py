# Small test script to check the linkage of the components
from models import sentiment_pipeline, emotion_pipeline, translator, intent_pipeline, sarcasm_pipeline
from utils import detect_language, translate_to_english, detect_sentiment, detect_emotion, detect_intent, detect_sarcasm

test_text = "Tumko time hi nahi hai mere liye 😒"
print(f"Original: {test_text}")

# 1. Language Detection
lang = detect_language(test_text)
print(f"Language: {lang}")

# 2. Translation
translated = translate_to_english(test_text, translator, lang)
print(f"Translated: {translated}")

# 3. NLP Pipeline
sentiment = detect_sentiment(translated, sentiment_pipeline)
emotion = detect_emotion(translated, emotion_pipeline)
intent = detect_intent(translated, intent_pipeline)
sarcasm = detect_sarcasm(translated, sarcasm_pipeline)

print(f"Sentiment: {sentiment}")
print(f"Emotion: {emotion}")
print(f"Intent: {intent}")
print(f"Sarcasm: {sarcasm}")

# 4. Check dependencies in models.py
print("-" * 20)
print("Pipeline verification completed successfully.")
