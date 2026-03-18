import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def analyze_with_llm(text, emotion, sentiment, intent, sarcasm):
    prompt = f"""
You are an expert in human conversation analysis.

Message: "{text}"

Detected:
- Emotion: {emotion}
- Sentiment: {sentiment}
- Intent: {intent}
- Sarcasm: {sarcasm}

Explain clearly:
1. What the person actually means
2. Emotional state in simple words
3. What they expect from the other person
4. Suggest a perfect reply (short, natural)
"""

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
