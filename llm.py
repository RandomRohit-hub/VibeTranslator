import requests
import random

OLLAMA_URL = "http://localhost:11434/api/generate"

def detect_mode(emotion, sentiment, sarcasm, text):
    """
    Auto-detects the response tone based on analysis results and text content.
    """
    text = text.lower()

    # 🔥 Priority logic
    if sarcasm == "High":
        return "Savage 😈"

    if "fine" in text or "thik" in text or "rehne do" in text:
        return "Cooked 💀"

    if sentiment == "Negative" and emotion in ["sadness", "anger", "fear"]:
        return "Therapist 🧠"

    if sentiment == "Positive":
        return "Cool Friend"

    # Default to Funny Roast if it's neutral or ambiguous
    return "Funny Roast 😂"

def build_prompt(text, emotion, sentiment, intent, sarcasm):
    """
    Builds a hybrid, dynamic prompt based on the auto-detected mode.
    """
    mode = detect_mode(emotion, sentiment, sarcasm, text)

    return f"""
You are a highly intelligent, human-like conversation expert.

MODE SELECTED: {mode}

STYLE RULES:
- Sound like a real person (not AI)
- Use simple English + a bit of Hinglish (mix naturally)
- Add a light cool tone (mate, bro, yeah, nah)
- Keep it short and relatable (no over-explaining)
- Match your personality to the situation (Savage, Therapist, Cool Friend, or GenZ/Cooked)
- If situation is serious → be calm (therapist style)
- If sarcasm detected → be sharp/savage
- If emotional → be supportive
- If obvious drama → add light humor

Message: "{text}"

Detected:
- Emotion: {emotion}
- Sentiment: {sentiment}
- Intent: {intent}
- Sarcasm: {sarcasm}

Now respond in this specific format:

1. What they ACTUALLY mean (simple + real meaning, no sugarcoat)
2. How they feel (simple, 1 line)
3. What they WANT from you (straight to the point)
4. Best reply (short, very natural, WhatsApp style)

IMPORTANT:
- Make it feel like a human friend explaining to another friend.
- No robotic language or patterns.
- Keep it under 6-8 lines total.
"""

def analyze_with_llm(text, emotion, sentiment, intent, sarcasm):
    prompt = build_prompt(text, emotion, sentiment, intent, sarcasm)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120  # wait up to 2 minutes for LLM response
        )
        data = response.json()
        if "error" in data:
            return f"⚠️ Ollama Error: {data['error']}"
        return data.get("response", "No response text from Ollama.")
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot connect to Ollama. Make sure it is running: `ollama serve`"
    except requests.exceptions.Timeout:
        return "⚠️ Ollama took too long to respond. Try a shorter message."
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"
