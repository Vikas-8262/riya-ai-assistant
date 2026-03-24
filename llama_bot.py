import json
import random
import threading
from datetime import datetime
from tkinter import *
from tkinter import scrolledtext
import ollama
from tokenizer import bag_of_words
from model import NeuralNet
import torch

# ---- Load Neural Network ----
data     = torch.load("neural_bot.pth")
nn_model = NeuralNet(data["input_size"], data["hidden1_size"],
                     data["hidden2_size"], data["output_size"])
nn_model.load_state_dict(data["model_state"])
nn_model.eval()
vocabulary = data["vocabulary"]
tags       = data["tags"]

with open("intents.json") as f:
    intents_data = json.load(f)

# ---- Memory ----
conversation_history = []
user_name            = None
last_topic           = None

# ---- Riya (TinyLLaMA) Response ----
def riya_response(user_input):
    try:
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {"role": "system", "content": "You are Riya, a helpful friendly AI assistant. Your name is Riya. Give short clear answers in 2-3 sentences."},
                {"role": "user",   "content": user_input}
            ]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return "I had trouble thinking about that. Try again!"

# ---- Smart Response ----
KNOWN_INTENTS = ["greeting", "goodbye", "thanks"]

def get_response(user_input):
    global user_name, last_topic

    user_lower = user_input.lower().strip()
    conversation_history.append(user_input)

    # Save name
    for phrase in ["my name is", "i am", "i'm"]:
        if phrase in user_lower:
            name = user_lower.split(phrase)[-1].strip().capitalize()
            if 1 < len(name) < 20 and name.isalpha():
                user_name = name
                return f"Nice to meet you {user_name}! I am Riya and I will remember that!"

    # Recall name
    if any(p in user_lower for p in ["what is my name", "who am i", "do you remember me"]):
        return f"Your name is {user_name}!" if user_name else "I don't know your name yet! Tell me!"

    # Message count
    if "how many messages" in user_lower:
        return f"We have exchanged {len(conversation_history)} messages!"

    # Recall conversation
    if "what did we talk about" in user_lower:
        if len(conversation_history) < 2:
            return "We just started talking!"
        recent = ", ".join(conversation_history[-5:-1])
        return f"Recently you said: {recent}"

    # Who are you
    if any(p in user_lower for p in ["what is your name", "who are you", "your name"]):
        return "I am Riya, your personal AI Assistant! How can I help you?"

    # Neural Network for known intents only
    bow        = bag_of_words(user_input, vocabulary)
    X          = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
    output     = nn_model(X)
    probs      = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    tag        = tags[predicted.item()]

    if confidence.item() >= 0.99 and tag in KNOWN_INTENTS:
        last_topic = tag
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                if tag == "greeting" and user_name:
                    return f"Hey {user_name}! I am Riya. How can I help you?"
                return response

    # Riya (TinyLLaMA) handles everything else
    last_topic = "open conversation"
    return riya_response(user_input)

# ---- GUI ----
window = Tk()
window.title("Riya - AI Assistant")
window.geometry("520x680")
window.resizable(False, False)
window.configure(bg="#1e1e2e")

Label(window, text="Riya AI", font=("Arial", 20, "bold"),
      bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)

memory_bar = Label(window, text="Status: Ready | Powered by TinyLLaMA",
                   font=("Arial", 9), bg="#313244",
                   fg="#a6adc8", anchor="w", padx=10)
memory_bar.pack(fill=X, padx=15)

chat_box = scrolledtext.ScrolledText(window, state=DISABLED, wrap=WORD,
                                      font=("Arial", 12), bg="#313244",
                                      fg="#cdd6f4", relief=FLAT,
                                      padx=10, pady=10)
chat_box.pack(padx=15, pady=10, fill=BOTH, expand=True)

def update_status():
    name_str  = user_name or "unknown"
    topic_str = last_topic or "none"
    memory_bar.config(text=f"User: {name_str} | Messages: {len(conversation_history)} | Topic: {topic_str}")

def display_response(user_input):
    response = get_response(user_input)
    chat_box.config(state=NORMAL)
    chat_box.delete("end-2l", "end-1l")
    chat_box.insert(END, f"Riya: {response}\n\n")
    chat_box.config(state=DISABLED)
    chat_box.yview(END)
    update_status()

def send_message():
    user_input = entry.get().strip()
    if not user_input:
        return
    entry.delete(0, END)
    chat_box.config(state=NORMAL)
    chat_box.insert(END, f"You: {user_input}\n")
    chat_box.insert(END, f"Riya: thinking...\n")
    chat_box.config(state=DISABLED)
    chat_box.yview(END)
    window.update()
    threading.Thread(target=display_response, args=(user_input,), daemon=True).start()

frame = Frame(window, bg="#1e1e2e")
frame.pack(pady=5, padx=15, fill=X)

entry = Entry(frame, font=("Arial", 13), bg="#45475a", fg="#cdd6f4",
              relief=FLAT, insertbackground="white")
entry.pack(side=LEFT, fill=X, expand=True, ipady=8, padx=(0, 10))
entry.bind("<Return>", lambda e: send_message())

Button(frame, text="Send", font=("Arial", 12, "bold"),
       bg="#89b4fa", fg="#1e1e2e", relief=FLAT,
       padx=15, pady=8, cursor="hand2",
       command=send_message).pack(side=RIGHT)

chat_box.config(state=NORMAL)
chat_box.insert(END, "Riya: Hello! I am Riya, your personal AI Assistant! Ask me anything!\n\n")
chat_box.config(state=DISABLED)

window.mainloop()