import json
import torch
import random
from datetime import datetime
from tkinter import *
from tkinter import scrolledtext
from tokenizer import bag_of_words
from model import NeuralNet

# ---- Load Model ----
data  = torch.load("neural_bot.pth")
model = NeuralNet(data["input_size"], data["hidden1_size"],
                  data["hidden2_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

vocabulary = data["vocabulary"]
tags       = data["tags"]

with open("intents.json") as f:
    intents_data = json.load(f)

# ---- Memory System ----
conversation_history = []
user_name            = None
last_topic           = None

def get_response(user_input):
    global user_name, last_topic

    user_input_lower = user_input.lower().strip()

    # Remember user name
    if "my name is" in user_input_lower:
        user_name = user_input_lower.split("my name is")[-1].strip().capitalize()
        conversation_history.append(f"User: {user_input}")
        return f"Nice to meet you {user_name}! I will remember your name!"

    if "i am" in user_input_lower or "i'm" in user_input_lower:
        name = user_input_lower.replace("i am", "").replace("i'm", "").strip().capitalize()
        if len(name) > 1 and len(name) < 20:
            user_name = name
            conversation_history.append(f"User: {user_input}")
            return f"Hello {user_name}! I will remember that!"

    # Remember what was talked about
    if "what did i say" in user_input_lower or "what did we talk" in user_input_lower:
        if len(conversation_history) == 0:
            return "We haven't talked about anything yet!"
        last = conversation_history[-1]
        return f"Last you said: {last}"

    if "do you remember me" in user_input_lower or "what is my name" in user_input_lower:
        if user_name:
            return f"Of course! Your name is {user_name}!"
        return "I don't know your name yet. Tell me — what is your name?"

    if "how many messages" in user_input_lower or "how long have we talked" in user_input_lower:
        return f"We have exchanged {len(conversation_history)} messages so far!"

    if "what did we talk about" in user_input_lower:
        if len(conversation_history) == 0:
            return "We haven't talked about anything yet!"
        topics = ", ".join(conversation_history[-5:])
        return f"Recently we talked about: {topics}"

    # Neural network response
    bow        = bag_of_words(user_input, vocabulary)
    X          = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
    output     = model(X)
    probs      = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)

    if confidence.item() < 0.75:
        conversation_history.append(f"User: {user_input}")
        if user_name:
            return f"I'm not sure I understand that {user_name}. Can you rephrase?"
        return "I'm not sure I understand. Can you rephrase?"

    tag = tags[predicted.item()]
    last_topic = tag
    conversation_history.append(f"User: {user_input} (topic: {tag})")

    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            if response == "__TIME__":
                return f"Current time: {datetime.now().strftime('%H:%M:%S')}"
            if response == "__DATE__":
                return f"Today is: {datetime.now().strftime('%B %d, %Y')}"
            # Add name to greeting if known
            if tag == "greeting" and user_name:
                return f"Hey {user_name}! How can I help you?"
            return response

def update_memory_bar():
    status = f"Memory: {user_name}" if user_name else "Memory: No name saved yet"
    memory_bar.config(text=status)

def send_message():
    user_text = input_field.get()
    if not user_text.strip():
        return
    
    chat_display.config(state=NORMAL)
    chat_display.insert(END, f"You: {user_text}\n", "user")
    
    response = get_response(user_text)
    chat_display.insert(END, f"Bot: {response}\n\n", "bot")
    chat_display.config(state=DISABLED)
    chat_display.see(END)
    
    input_field.delete(0, END)
    update_memory_bar()

# ---- GUI ----
window = Tk()
window.title("NeuralBot AI Assistant")
window.geometry("500x650")
window.resizable(False, False)
window.configure(bg="#1e1e2e")

# Title
title = Label(window, text="NeuralBot", font=("Arial", 20, "bold"),
              bg="#1e1e2e", fg="#cdd6f4")
title.pack(pady=10)

# Memory status bar
memory_bar = Label(window, text="Memory: No name saved yet",
                   font=("Arial", 10), bg="#313244", fg="#a6e3a1", pady=5)
memory_bar.pack(fill=X, padx=10)

# Chat display
chat_display = scrolledtext.ScrolledText(window, wrap=WORD, state=DISABLED,
                                         bg="#313244", fg="#cdd6f4", font=("Arial", 10))
chat_display.pack(fill=BOTH, expand=True, padx=10, pady=10)
chat_display.tag_config("user", foreground="#89b4fa")
chat_display.tag_config("bot", foreground="#a6e3a1")

# Input field
input_field = Entry(window, font=("Arial", 10), bg="#45475a", fg="#cdd6f4", insertbackground="#cdd6f4")
input_field.pack(fill=X, padx=10, pady=5)
input_field.bind("<Return>", lambda e: send_message())

# Send button
send_btn = Button(window, text="Send", command=send_message, bg="#89b4fa", fg="#1e1e2e", font=("Arial", 10))
send_btn.pack(pady=5)

window.mainloop()