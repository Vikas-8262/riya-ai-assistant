import json
import torch
import random
from datetime import datetime
from tokenizer import bag_of_words
from model import NeuralNet

data  = torch.load("neural_bot.pth")
model = NeuralNet(data["input_size"], data["hidden1_size"],
                  data["hidden2_size"], data["output_size"])
model.load_state_dict(data["model_state"])
model.eval()

vocabulary = data["vocabulary"]
tags       = data["tags"]

with open("intents.json") as f:
    intents_data = json.load(f)

CONFIDENCE_THRESHOLD = 0.75

def get_response(user_input):
    bow        = bag_of_words(user_input, vocabulary)
    X          = torch.tensor(bow, dtype=torch.float32).unsqueeze(0)
    output     = model(X)
    probs      = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probs, dim=1)

    if confidence.item() < CONFIDENCE_THRESHOLD:
        return "I'm not sure I understand. Can you rephrase?"

    tag = tags[predicted.item()]

    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            response = random.choice(intent["responses"])
            if response == "__TIME__":
                return f"Current time: {datetime.now().strftime('%H:%M:%S')}"
            if response == "__DATE__":
                return f"Today is: {datetime.now().strftime('%B %d, %Y')}"
            return response

print("NeuralBot ready! Type 'quit' to exit.\n")
while True:
    user = input("You: ").strip()
    if not user:
        continue
    if user.lower() in ["quit", "exit"]:
        print("NeuralBot: Goodbye!")
        break
    print(f"NeuralBot: {get_response(user)}\n")