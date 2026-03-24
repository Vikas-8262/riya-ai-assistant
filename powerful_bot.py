import torch
import json
import random
from datetime import datetime
from tkinter import *
from tkinter import scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tokenizer import bag_of_words
from model import NeuralNet

print("Loading models... please wait...")

# ---- Load your Neural Network ----
data  = torch.load("neural_bot.pth")
nn_model = NeuralNet(data["input_size"], data["hidden1_size"],
                     data["hidden2_size"], data["output_size"])
nn_model.load_state_dict(data["model_state"])
nn_model.eval()
vocabulary = data["vocabulary"]
tags       = data["tags"]

with open("intents.json") as f:
    intents_data = json.load(f)

# ---- Load GPT-2 ----
print("Loading GPT-2 language model...")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model     = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()
print("All models loaded!")

# ---- Memory ----
conversation_history = []
user_name            = None
last_topic           = None

# ---- GPT-2 Response ----
def gpt2_response(user_input):
    prompt = (
        f"The following is a conversation with a helpful, knowledgeable AI assistant.\n"
        f"The AI gives clear, detailed, and accurate answers.\n\n"
        f"Human: {user_input}\n"
        f"AI:"
    )
    
    inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = gpt2_model.generate(
            inputs,
            max_length=inputs.shape[1] + 150,
            temperature=0.6,
            top_p=0.85,
            top_k=50,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    
    full_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "AI:" in full_text:
        response = full_text.split("AI:")[-1].strip()
        response = response.split("Human:")[0].strip()
        response = response.split("\n")[0].strip()
        if len(response) > 15:
            return response
    
    return "That is a great question! I am still learning about that topic."

# ---- Smart Response ----
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
                return f"Nice to meet you {user_name}! I will remember that!"

    # Recall name
    if any(p in user_lower for p in ["what is my name", "do you remember me", "who am i"]):
        return f"Your name is {user_name}!" if user_name else "I don't know your name yet!"

    # Message count
    if "how many messages" in user_lower:
        return f"We have exchanged {len(conversation_history)} messages!"

    # Recall conversation
    if "what did we talk about" in user_lower:
        if len(conversation_history) < 2:
            return "We just started talking!"
        recent = ", ".join(conversation_history[-5:-1])
        return f"Recently you said: {recent}"

    # Only use Neural Network for exact intent matches
    KNOWN_INTENTS = ["greeting", "goodbye", "thanks"]
    
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
                    return f"Hey {user_name}! How can I help you?"
                return response

    # Everything else goes to GPT-2
    last_topic = "open conversation"
    return gpt2_response(user_input)

# ---- GUI ----
window = Tk()
window.title("NeuralBot AI - Powered Up!")
window.geometry("520x680")
window.resizable(False, False)
window.configure(bg="#1e1e2e")

Label(window, text="NeuralBot AI", font=("Arial", 20, "bold"),
      bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)

memory_bar = Label(window, text="Status: Ready",
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
    memory_bar.config(text=f"User: {name_str} | Messages: {len(conversation_history)} | Last topic: {topic_str}")

def send_message():
    user_input = entry.get().strip()
    if not user_input:
        return
    entry.delete(0, END)
    chat_box.config(state=NORMAL)
    chat_box.insert(END, f"You: {user_input}\n")
    chat_box.insert(END, "NeuralBot: thinking...\n")
    chat_box.config(state=DISABLED)
    chat_box.yview(END)
    window.update()

    response = get_response(user_input)

    chat_box.config(state=NORMAL)
    chat_box.delete("end-2l", "end-1l")
    chat_box.insert(END, f"NeuralBot: {response}\n\n")
    chat_box.config(state=DISABLED)
    chat_box.yview(END)
    update_status()

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
chat_box.insert(END, "NeuralBot: Hello! I am now powered by GPT-2! Ask me anything!\n\n")
chat_box.config(state=DISABLED)

window.mainloop()