# 💬 ConvoSense AI — Decode the Vibes Mate

## 🧠 What’s this all about?

Ever been stuck thinking:

> “Did she mean that… or did she *mean* that?” 🤔

Yeah mate, we’ve all been there.

**ConvoSense AI** is your digital wingman that helps you figure out:

* What someone *actually* meant
* Whether that “okay 👍” is chill… or a red flag 🚩
* If you're in trouble (spoiler: you probably are 😅)

---

## 🎯 The Big Idea

Most AI tools just say:

> “Positive / Negative”

Useless.

This project goes deeper:

* Detects **sarcasm (the real killer)**
* Understands **emotion + intent**
* Handles **Hinglish + Marathi + English chats**
* Gives you **actual meaning + reply suggestions**

👉 Basically:
**We’re decoding human drama using AI**

---

## 🔥 Features

* 🧠 **Custom-trained sarcasm model (trained from scratch)**
* 💬 **Emotion detection**
* 🎯 **Intent analysis**
* 🌍 **Multilingual support**
* 🧠 **Context-aware memory**
* 🤖 **Local LLM via Ollama**
* 💡 **Smart reply suggestions**

---

## 🧠 Workflow (How this beast actually works)

```
User Input (chat message)
        ↓
Context Injection (previous messages)
        ↓
Custom Sarcasm Model 
        ↓
Emotion Detection + Sentiment Analysis
        ↓
Intent Classification
        ↓
All signals combined
        ↓
LLM (Ollama) for reasoning
        ↓
Final Output:
   - Meaning
   - Emotion
   - Intent
   - Suggested Reply
```

👉 Translation layer removed for speed
👉 Fully optimized pipeline for real-time feel

---

## 🧠 Custom Model + Dataset

This is where things get serious 👇

### 🔥 Custom Sarcasm Model

* Built using **DistilBERT / lightweight transformer**
* Fine-tuned specifically for **conversations IYKYK**
* Optimized for:

  * Hinglish
  * Marathi tone
  * Passive-aggressive messages

---

### 📊 Dataset

The model is trained on a **custom multilingual dataset (~10,000+ samples)** including:

* English
* Hindi
* Hinglish
* Marathi

👉 Data includes:

* Real conversational patterns
* Emotion-heavy sentences
* Passive-aggressive and sarcastic tones

---

### ⚠️ Important Note (Ethical)

The dataset is:

* **Anonymized**
* Based on **generic conversational patterns**
* Used strictly for **educational and research purposes**

👉 No personal or sensitive data is exposed.

---

## 😂 Why this exists

Because:

> “I’m fine” is never actually fine
> “Do whatever you want” is a trap
> “Okay 👍” could mean *anything*

And honestly…
someone had to solve this problem 😭

---

## ⚙️ Tech Stack

* Python
* HuggingFace Transformers
* Custom BERT Model
* Streamlit
* Ollama (Local LLM)
* Scikit-learn

---

## 🚀 How to Run

### 1. Install stuff

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama run llama3
```

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Open in browser

```
http://localhost:8501
```

---

## 💬 Try this

```
Acha thik hai tum busy ho toh rehne do
```

👉 Watch the AI go:
**“Yeah nah… she’s not fine mate” 😬**

---

## 🔐 Privacy

Runs locally.
No data leaves your system.
No spying. No nonsense.

---

## 🎯 Future Improvements

* Real-time chat analysis (WhatsApp style 👀)
* Voice input + tone detection
* Chrome extension
* Higher sarcasm accuracy

---

## 🧠 Final Thought

> “Helping you not mess things up” 💀

---

## 🤝 Contributing

If you’ve ever misunderstood a message…
you’re already qualified 😭

---

## ⭐ Give it a star

If this saves you from even **one argument**,
that’s a win mate.

---

## 📌 Note

This project is a foundational step towards an upcoming, more advanced system called **DORA** — aiming to take conversational intelligence and human-AI interaction to the next level.

Stay tuned… it’s gonna be big 👀

---

Made with Predator + confusion + emotional damage.
