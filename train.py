import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizer import build_vocabulary, bag_of_words
from model import NeuralNet

with open("intents.json") as f:
    intents_data = json.load(f)

vocabulary = build_vocabulary(intents_data)
tags       = [intent["tag"] for intent in intents_data["intents"]]

X_data, y_data = [], []

for idx, intent in enumerate(intents_data["intents"]):
    for pattern in intent["patterns"]:
        X_data.append(bag_of_words(pattern, vocabulary))
        y_data.append(idx)

X_data = np.array(X_data)
y_data = np.array(y_data)

class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = torch.tensor(X_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

INPUT_SIZE    = len(vocabulary)
HIDDEN1_SIZE  = 128
HIDDEN2_SIZE  = 64
OUTPUT_SIZE   = len(tags)
LEARNING_RATE = 0.001
BATCH_SIZE    = 8
EPOCHS        = 1000

dataset    = ChatDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model     = NeuralNet(INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training started...")
for epoch in range(EPOCHS):
    for X_batch, y_batch in dataloader:
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {loss.item():.4f}")

torch.save({
    "model_state":  model.state_dict(),
    "input_size":   INPUT_SIZE,
    "hidden1_size": HIDDEN1_SIZE,
    "hidden2_size": HIDDEN2_SIZE,
    "output_size":  OUTPUT_SIZE,
    "vocabulary":   vocabulary,
    "tags":         tags
}, "neural_bot.pth")

print("\nModel trained and saved to neural_bot.pth ✓")