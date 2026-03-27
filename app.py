from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import random
import torch
import os
import sys
from datetime import datetime

# 1. Safely set up base directory and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 2. Import custom modules safely
try:
    from tokenizer import bag_of_words
    from model import NeuralNet
except ImportError as e:
    print(f"CRITICAL ERROR: Missing Python file! {e}")

template_dir = os.path.join(BASE_DIR, 'web', 'templates')
static_dir = os.path.join(BASE_DIR, 'web', 'static')

app = Flask(__name__, 
            template_folder=template_dir, 
            static_folder=static_dir)
CORS(app)

# 3. Safely load intents.json
intents_path = os.path.join(BASE_DIR, "intents.json")
if not os.path.exists(intents_path):
    print(f"CRITICAL ERROR: Could not find {intents_path}")
    intents_data = {"intents": []}
else:
    with open(intents_path, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)

# 4. Safely load the PyTorch model
model_path = os.path.join(BASE_DIR, "neural_bot.pth")
model_loaded = False

if not os.path.exists(model_path):
    print(f"CRITICAL ERROR: Could not find {model_path}")
else:
    try:
        data = torch.load(model_path, map_location="cpu")
        nn_model = NeuralNet(data["input_size"], data["hidden1_size"],
                             data["hidden2_size"], data["output_size"])
        nn_model.load_state_dict(data["model_state"])
        nn_model.eval()
        vocabulary = data["vocabulary"]
        tags = data["tags"]
        model_loaded = True
    except Exception as e:
        print(f"CRITICAL ERROR LOADING MODEL: {e}")

KNOWN_INTENTS = ["greeting", "goodbye", "thanks"]

def get_response(user_input):
    user_lower = user_input.lower().strip()

    if any(p in user_lower for p in ["who are you", "your name"]):
        return "I am Riya, your personal AI Assistant! 🌟"

    if any(p in user_lower for p in ["what time", "current time"]):
        return f"It is {datetime.now().strftime('%I:%M %p')} ⏰"

    if any(p in user_lower for p in ["what date", "today"]):
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"

    if not model_loaded:
        return "My brain (model) is currently offline! Check the server logs."

    bow = bag_of_words(user_input, vocabulary)
    X = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
    output = nn_model(X)
    probs = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    tag = tags[predicted.item()]

    if confidence.item() >= 0.99 and tag in KNOWN_INTENTS:
        for intent in intents_data.get("intents", []):
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

    return "I am Riya! How can I help you today? 😊"

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        # If the template is missing, this will print the EXACT path Render is looking for directly on the webpage!
        return f"<h1>Template Error!</h1><p>{e}</p><p>Flask is looking in this folder: <b>{template_dir}</b></p>", 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "No message"}), 400
    response = get_response(message)
    return jsonify({"response": response})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online", 
        "name": "Riya AI", 
        "model_loaded": model_loaded,
        "template_path": template_dir
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    