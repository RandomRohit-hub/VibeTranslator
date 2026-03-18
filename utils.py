from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


def detect_sentiment(text, sentiment_pipeline):
    result = sentiment_pipeline(text)
    if isinstance(result[0], list):
        return result[0][0]['label']
    return result[0]['label']

def detect_emotion(text, emotion_pipeline):
    result = emotion_pipeline(text)
    if isinstance(result[0], list):
        return result[0][0]['label']
    return result[0]['label']

def translate_to_english(text, translator, src_lang="hi"):
    if src_lang == "unknown" or src_lang not in translator.tokenizer.lang_code_to_id:
        src_lang = "hi"
    
    if src_lang == "en":
        return text
        
    result = translator(text, src_lang=src_lang, tgt_lang="en")
    return result[0]['translation_text']


def detect_intent(text, intent_pipeline):
    labels = [
        "complaint",
        "request",
        "emotional expression",
        "casual talk",
        "attention seeking"
    ]
    result = intent_pipeline(text, labels)
    return result['labels'][0]


# sarcastic signal words common in Hinglish / English
_SARC_KEYWORDS = [
    "obviously", "of course", "wow", "great", "perfect", "sure",
    "haan haan", "haan obviously", "bilkul", "acha thik hai",
    "kay mast", "rehne do", "ho na", "wah", "nice",
]

def detect_sarcasm(text, sarcasm_pipeline):
    text_lower = text.lower()

    # rule-based: check for sarcasm trigger words
    rule_detected = any(kw in text_lower for kw in _SARC_KEYWORDS)

    # model prediction
    result = sarcasm_pipeline(text)[0]
    score = result["score"]
    label = result["label"]

    # custom model uses LABEL_1 for sarcasm, irony model uses "irony"
    is_sarc_label = (label == "LABEL_1") or (label == "irony")

    # ensemble: rule OR confident model prediction
    if rule_detected:
        return "sarcasm"
    elif is_sarc_label and score > 0.6:  # Slightly lowered threshold as per user request
        return "sarcasm"
    else:
        return "no sarcasm"


def generate_basic_meaning(sentiment, emotion, intent, sarcasm):
    return f"""
Emotion: {emotion}
Sentiment: {sentiment}
Intent: {intent}
Sarcasm: {sarcasm}
"""
