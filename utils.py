from langdetect import detect

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"


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


def detect_sarcasm(text, sarcasm_pipeline):
    result = sarcasm_pipeline(text)[0]
    return result['label']


def generate_basic_meaning(sentiment, emotion, intent, sarcasm):
    return f"""
Emotion: {emotion}
Sentiment: {sentiment}
Intent: {intent}
Sarcasm: {sarcasm}
"""
