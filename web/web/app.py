from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import random
import torch
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import bag_of_words
from model import NeuralNet

try:
    from weather import get_weather
    WEATHER_ENABLED = True
except:
    WEATHER_ENABLED = False

try:
    from web_search import is_web_search, web_search_response
    WEBSEARCH_ENABLED = True
except:
    WEBSEARCH_ENABLED = False

app  = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data     = torch.load(os.path.join(BASE_DIR, "neural_bot.pth"),
                      map_location="cpu")
nn_model = NeuralNet(data["input_size"], data["hidden1_size"],
                     data["hidden2_size"], data["output_size"])
nn_model.load_state_dict(data["model_state"])
nn_model.eval()
vocabulary = data["vocabulary"]
tags       = data["tags"]

with open(os.path.join(BASE_DIR, "intents.json")) as f:
    intents_data = json.load(f)

KNOWN_INTENTS = ["greeting", "goodbye", "thanks"]

def get_response(user_input, user_name=None):
    user_lower = user_input.lower().strip()

    for phrase in ["my name is", "i am", "i'm"]:
        if phrase in user_lower:
            name = user_lower.split(phrase)[-1].strip().capitalize()
            if 1 < len(name) < 20 and name.isalpha():
                return {"response": f"Nice to meet you {name}! I am Riya! 😊",
                        "name": name}

    if any(p in user_lower for p in ["who are you", "your name"]):
        return {"response": "I am Riya, your personal AI Assistant! 🌟"}

    if any(p in user_lower for p in ["what time", "current time"]):
        return {"response": f"It is {datetime.now().strftime('%I:%M %p')} ⏰"}

    if any(p in user_lower for p in ["what date", "today"]):
        return {"response": f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"}

    if WEATHER_ENABLED and any(w in user_lower for w in
            ["weather", "temperature", "forecast"]):
        words = user_input.split()
        city  = None
        for i, word in enumerate(words):
            if word.lower() in ["in", "for"] and i + 1 < len(words):
                city = words[i + 1]
                break
        return {"response": get_weather(city or "Pune")}

    if WEBSEARCH_ENABLED and is_web_search(user_input):
        return {"response": web_search_response(user_input)}

    bow        = bag_of_words(user_input, vocabulary)
    X          = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
    output     = nn_model(X)
    probs      = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    tag        = tags[predicted.item()]

    if confidence.item() >= 0.99 and tag in KNOWN_INTENTS:
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                return {"response": random.choice(intent["responses"])}

    try:
        import ollama
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {"role": "system", "content": (
                    "You are Riya, a friendly personal AI assistant. "
                    "Give clear concise answers in 2-3 sentences."
                )},
                {"role": "user", "content": user_input}
            ]
        )
        return {"response": response["message"]["content"].strip()}
    except:
        return {"response": "I am Riya! How can I help you today? 😊"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data      = request.json
    message   = data.get("message", "")
    user_name = data.get("user_name", None)
    if not message:
        return jsonify({"error": "No message"}), 400
    result = get_response(message, user_name)
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "online",
        "name":    "Riya AI",
        "version": "3.0",
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)