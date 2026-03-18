chat_history = []

def add_to_history(text):
    chat_history.append(text)
    if len(chat_history) > 5:
        chat_history.pop(0)

def get_context():
    return " ".join(chat_history)
