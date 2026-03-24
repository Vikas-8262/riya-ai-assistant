from textblob import TextBlob
import random

EMOTIONS = {
    "happy":   ["happy", "great", "awesome", "excellent", "wonderful", "fantastic",
                "amazing", "love", "good", "joy", "excited", "glad", "pleased"],
    "sad":     ["sad", "unhappy", "depressed", "miserable", "heartbroken", "crying",
                "upset", "down", "lonely", "hopeless", "grief", "sorrow"],
    "angry":   ["angry", "furious", "mad", "annoyed", "frustrated", "hate",
                "rage", "irritated", "outraged", "bitter"],
    "anxious": ["anxious", "worried", "nervous", "scared", "afraid", "fear",
                "stress", "panic", "tense", "uneasy", "overwhelmed"],
    "tired":   ["tired", "exhausted", "sleepy", "bored", "drained",
                "fatigue", "worn out", "lazy"],
    "confused":["confused", "lost", "unsure", "puzzled", "stuck",
                "dont understand", "help me", "no idea"],
    "neutral": []
}

RESPONSES = {
    "happy":    [
        "I love your positive energy! 😊 Keep it up!",
        "That's wonderful to hear! 🌟 You deserve all the happiness!",
        "Your happiness makes me happy too! 😄"
    ],
    "sad":      [
        "I am really sorry to hear that 💙 I am here for you.",
        "It's okay to feel sad sometimes. I am listening 💙",
        "I care about you. Would you like to talk about it? 💙"
    ],
    "angry":    [
        "I understand your frustration 😤 Take a deep breath.",
        "It's okay to feel angry. Want to talk about what happened?",
        "I hear you. Let's work through this together 💪"
    ],
    "anxious":  [
        "Everything will be okay 🌸 Take it one step at a time.",
        "I am here with you. You are not alone 🌸",
        "Take a deep breath. You've got this! 💫"
    ],
    "tired":    [
        "Make sure you get some rest! 😴 You deserve it.",
        "Take a break — you have been working hard! ☕",
        "Rest is important! Take care of yourself 💤"
    ],
    "confused": [
        "Don't worry! I will help you figure it out 🤔",
        "Let me help you understand! That's what I am here for 😊",
        "No worries! Let's work through it together step by step 💡"
    ],
    "neutral":  [
        "I am here to help! What can I do for you? 😊",
        "Tell me more! I am listening 👂",
        "Got it! How can I assist you? 🌟"
    ]
}

def detect_emotion(text):
    text_lower = text.lower()

    for emotion, keywords in EMOTIONS.items():
        if emotion == "neutral":
            continue
        for keyword in keywords:
            if keyword in text_lower:
                return emotion

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.3:
        return "happy"
    elif polarity < -0.3:
        return "sad"
    else:
        return "neutral"

def get_emotion_response(text):
    emotion = detect_emotion(text)
    response = RESPONSES.get(emotion, RESPONSES["neutral"])
    return emotion, random.choice(response)

if __name__ == "__main__":
    print("Emotion Detection Test")
    print("=" * 30)
    while True:
        text = input("You: ")
        if text.lower() == "quit":
            break
        emotion, response = get_emotion_response(text)
        print(f"Detected emotion: {emotion}")
        print(f"Riya: {response}\n")