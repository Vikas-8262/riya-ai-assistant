import numpy as np
import re

def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z\s]', '', sentence)
    return sentence.split()

def build_vocabulary(intents_data):
    vocabulary = set()
    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            words = tokenize(pattern)
            vocabulary.update(words)
    return sorted(list(vocabulary))

def bag_of_words(sentence, vocabulary):
    words = tokenize(sentence)
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for word in words:
        if word in vocabulary:
            idx = vocabulary.index(word)
            bag[idx] = 1.0
    return bag