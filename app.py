from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import random
import torch
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import bag_of_words
from model import NeuralNet

app  = Flask(__name__,
             template_folder="web/templates",
             static_folder="web/static")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

def get_response(user_input):
    user_lower = user_input.lower().strip()

    if any(p in user_lower for p in ["who are you", "your name"]):
        return "I am Riya, your personal AI Assistant! 🌟"

    if any(p in user_lower for p in ["what time", "current time"]):
        return f"It is {datetime.now().strftime('%I:%M %p')} ⏰"

    if any(p in user_lower for p in ["what date", "today"]):
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"

    bow        = bag_of_words(user_input, vocabulary)
    X          = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
    output     = nn_model(X)
    probs      = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    tag        = tags[predicted.item()]

    if confidence.item() >= 0.99 and tag in KNOWN_INTENTS:
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

    return "I am Riya! How can I help you today? 😊"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data    = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message"}), 400
    response = get_response(message)
    return jsonify({"response": response})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "online", "name": "Riya AI"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)