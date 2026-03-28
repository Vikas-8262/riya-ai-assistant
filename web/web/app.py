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

# ---- Groq AI ----
try:
    from groq import Groq
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    if GROQ_API_KEY:
        groq_client  = Groq(api_key=GROQ_API_KEY)
        GROQ_ENABLED = True
        print("Groq AI connected!")
    else:
        GROQ_ENABLED = False
        print("No Groq API key found")
except Exception as e:
    GROQ_ENABLED = False
    print(f"Groq not available: {e}")

# ---- Weather ----
try:
    from weather import get_weather
    WEATHER_ENABLED = True
except:
    WEATHER_ENABLED = False

# ---- Web Search ----
try:
    from web_search import is_web_search, web_search_response
    WEBSEARCH_ENABLED = True
except:
    WEBSEARCH_ENABLED = False

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "templates"),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "static"))
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data     = torch.load(os.path.join(BASE_DIR, "neural_bot.pth"), map_location="cpu")
nn_model = NeuralNet(data["input_size"], data["hidden1_size"],
                     data["hidden2_size"], data["output_size"])
nn_model.load_state_dict(data["model_state"])
nn_model.eval()
vocabulary = data["vocabulary"]
tags       = data["tags"]

with open(os.path.join(BASE_DIR, "intents.json")) as f:
    intents_data = json.load(f)

KNOWN_INTENTS = ["greeting", "goodbye", "thanks"]

def groq_response(user_input, user_name=None):
    try:
        memory = f"The user's name is {user_name}." if user_name else ""
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": (
                    "You are Riya, a friendly warm intelligent personal AI assistant. "
                    "You were built from scratch by Vikas, a Python Developer from Pune India. "
                    "Your name is Riya. You have a cheerful helpful personality. "
                    "Give clear concise answers in 2-3 sentences. "
                    "Use emojis naturally to be expressive. "
                    f"{memory}"
                )},
                {"role": "user", "content": user_input}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return None

def get_response(user_input, user_name=None):
    user_lower = user_input.lower().strip()

    # Name detection
    for phrase in ["my name is", "i am", "i'm", "call me"]:
        if phrase in user_lower:
            name = user_lower.split(phrase)[-1].strip().capitalize()
            if 1 < len(name) < 20 and name.replace(" ", "").isalpha():
                return {"response": f"Nice to meet you {name}! 😊 I am Riya!", "name": name}

    # Recall name
    if any(p in user_lower for p in ["what is my name", "who am i"]):
        if user_name:
            return {"response": f"Your name is {user_name}! 😊"}
        return {"response": "I don't know your name yet! Tell me! 😊"}

    # Time
    if any(p in user_lower for p in ["what time", "current time"]):
        return {"response": f"It is {datetime.now().strftime('%I:%M %p')} ⏰"}

    # Date
    if any(p in user_lower for p in ["what date", "today's date", "what day", "what is today"]):
        return {"response": f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"}

    # Weather
    if WEATHER_ENABLED and any(w in user_lower for w in ["weather", "temperature", "forecast"]):
        words = user_input.split()
        city  = None
        for i, word in enumerate(words):
            if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                city = words[i + 1]
                break
        return {"response": get_weather(city or "Pune")}

    # Web search
    if WEBSEARCH_ENABLED and is_web_search(user_input):
        return {"response": web_search_response(user_input)}

    # Neural Network for known intents
    bow        = bag_of_words(user_input, vocabulary)
    X          = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
    output     = nn_model(X)
    probs      = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)
    tag        = tags[predicted.item()]

    if confidence.item() >= 0.99 and tag in KNOWN_INTENTS:
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                resp = random.choice(intent["responses"])
                if tag == "greeting" and user_name:
                    return {"response": f"Hey {user_name}! 😊 How can I help you?"}
                return {"response": resp}

    # Groq AI for everything else
    if GROQ_ENABLED:
        response = groq_response(user_input, user_name)
        if response:
            return {"response": response}

    # Fallback
    return {"response": random.choice([
        "That's interesting! 🤔 Could you tell me more?",
        "I am Riya! 🌟 Ask me about weather, time or just chat!",
        "Great question! 💫 I am still learning!",
    ])}

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
        "status":    "online",
        "name":      "Riya AI",
        "version":   "3.0",
        "ai_engine": "Groq Llama3" if GROQ_ENABLED else "Neural Network",
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)