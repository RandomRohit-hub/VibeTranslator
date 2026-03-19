# ConvoSense AI – Project Workflow & Architecture

![Project Workflow](file:///D:/tomato/docs/workflow_diagram.png)

## 1. Overview
ConvoSense AI is a modular, AI-powered conversational intelligence system designed to analyze human communication in real time. It integrates multiple NLP models with a Streamlit-based interface to generate human-like interpretations of conversations. The system follows a pipeline architecture where each module has a clearly defined responsibility, ensuring scalability, maintainability, and clarity.

## 2. Project Architecture
The application is divided into the following core modules:

### 2.1 app.py (Main Controller / UI Layer)
Acts as the entry point of the Streamlit application.
**Responsible for:**
- Handling user input
- Displaying chat interface and message history
- Coordinating the entire workflow

**Execution Flow:**
1. Accept user input
2. Update conversation history using `memory.py`
3. Detect language and translate (if required) via `utils.py`
4. Execute NLP pipelines using `models.py`
5. Send processed results to `llm.py`
6. Display final response in UI

### 2.2 models.py (Machine Learning Hub)
Central module for loading and managing all ML models. Uses Hugging Face Transformers.
**Models Integrated:**
- **Sentiment Analysis** → Twitter-XLM-RoBERTa
- **Emotion Detection** → DistilRoBERTa
- **Translation** → M2M-100
- **Intent Classification** → BART
- **Sarcasm Detection** → Custom/Default RoBERTa

**Key Feature:** *Offline Resilience*. Automatically switches to locally cached models if internet is unavailable.

### 2.3 utils.py (Processing Layer)
Acts as an intermediate layer between raw ML outputs and application logic.
**Core Functions:**
- `detect_language()`
- `translate_to_english()`
- `detect_intent()`
- `detect_sarcasm()`

### 2.4 llm.py (LLM Interaction Layer)
Handles communication with the local Ollama server (LLaMA 3.2).
**Core Responsibility:** Converts raw ML outputs into human-like interpretations.
**Key Feature:** *Dynamic Personality Engine*. Automatically selects response tone based on analysis: Savage Mode 😈, Friendly Mode 😄, Therapist Mode 🧠, Casual Bro Mode 🤝.

### 2.5 memory.py (Context Management)
Maintains a sliding window of conversation history.
**Purpose:** Provides context to the LLM. Ensures responses are not isolated but conversation-aware.

## 3. Role of Machine Learning in the System
The ML models act as the intelligence layer of the application.

- **Sentiment & Emotion Analysis**: Determines user feelings (Sad, Angry, Happy, etc.).
- **Sarcasm Detection**: Identifies hidden tone or passive-aggressive intent (e.g., “Acha thik hai”).
- **Intent Classification**: Determines user purpose (Asking for help, Expressing frustration, making a request).
- **Combined Intelligence**: All outputs are merged into structured metadata then used by the LLM to generate meaningful responses.

## 4. System Workflow (End-to-End Pipeline)
1. **[USER INPUT]**
2. **[app.py]**
3. **[memory.py]** → Store context
4. **[utils.py]** → Language detection & translation
5. **[models.py]** → NLP Analysis (Sentiment, Emotion, Intent, Sarcasm)
6. **[llm.py]** → Mode Detection + Prompt Generation + LLM Response
7. **[app.py]** → Display Final Output

## 5. Key Highlights
- **Modular architecture**: Easy to scale and maintain.
- **Multi-model NLP pipeline**: Rich analysis beyond basic chatbots.
- **Context-aware responses**: Uses conversation memory.
- **Dynamic personality system**: Human-like interaction.
- **Offline capability**: Reliable in low connectivity environments.

## 6. Conclusion
ConvoSense AI goes beyond traditional chat systems by combining multi-layered NLP analysis, context awareness, and dynamic personality generation. This results in a system that not only understands what users say, but also how they feel and what they mean.
