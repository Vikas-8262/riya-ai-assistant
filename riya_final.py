import json
import random
import threading
import os
from datetime import datetime
from tkinter import *
from tkinter import scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import pyttsx3
import ollama
from tokenizer import bag_of_words
from model import NeuralNet
import torch
from weather import get_weather
from web_search import is_web_search, web_search_response
from emotion import detect_emotion, get_emotion_response
from code_assistant import is_code_request, get_code_response
from file_reader import answer_about_file
from face_recognition_riya import recognize_face, register_face
from image_gen import is_image_request, extract_prompt, generate_image
import speech_recognition as sr

# ---- Optional imports with fallbacks ----
try:
    WEATHER_ENABLED = True
except:
    WEATHER_ENABLED = False

try:
    WEBSEARCH_ENABLED = True
except:
    WEBSEARCH_ENABLED = False

try:
    EMOTION_ENABLED = True
except:
    EMOTION_ENABLED = False

try:
    CODE_ENABLED = True
except:
    CODE_ENABLED = False

try:
    FILE_ENABLED = True
except:
    FILE_ENABLED = False

try:
    from memory import (save_user, get_user, save_fact, get_fact,
                        get_all_facts, save_message, get_recent_history,
                        extract_and_save_facts, build_memory_context)
    MEMORY_ENABLED = True
except:
    MEMORY_ENABLED = False

try:
    FACE_ENABLED = True
except:
    FACE_ENABLED = False

try:
    IMAGE_ENABLED = True
except:
    IMAGE_ENABLED = False

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

# ---- TTS ----
tts_engine  = pyttsx3.init()
tts_enabled = True

def speak(text):
    if tts_enabled:
        clean = text.replace("😊","").replace("🌟","").replace("💫","")
        clean = clean.replace("😄","").replace("💙","").replace("🎨","")
        threading.Thread(target=lambda: (
            tts_engine.say(clean), tts_engine.runAndWait()
        ), daemon=True).start()

# ---- Themes ----
DARK = {
    "bg": "#1e1e2e", "sidebar": "#181825", "chat_bg": "#313244",
    "input_bg": "#45475a", "text": "#cdd6f4", "muted": "#a6adc8",
    "accent": "#89b4fa", "accent_fg": "#1e1e2e",
    "user_bubble": "#45475a", "bot_bubble": "#313244",
    "border": "#585b70", "green": "#a6e3a1", "red": "#f38ba8",
    "purple": "#cba6f7",
}
LIGHT = {
    "bg": "#ffffff", "sidebar": "#f1f3f4", "chat_bg": "#f8f9fa",
    "input_bg": "#ffffff", "text": "#202124", "muted": "#5f6368",
    "accent": "#1a73e8", "accent_fg": "#ffffff",
    "user_bubble": "#e8f0fe", "bot_bubble": "#ffffff",
    "border": "#dadce0", "green": "#34a853", "red": "#ea4335",
    "purple": "#7c4dff",
}

current_theme = DARK
is_dark       = True

# ---- State ----
conversation_history = []
chat_sessions        = ["Chat 1"]
user_name            = None
last_topic           = None
is_typing            = False
message_count        = 0
avatar_image         = None
chat_log             = []
loaded_file          = None
loaded_file_name     = None
riya_mood            = "happy"

PERSONALITY = {
    "happy":    ["Great question! ", "I love that! ", ""],
    "curious":  ["Interesting! ", "Hmm, let me think... ", ""],
    "helpful":  ["Of course! ", "Happy to help! ", ""],
}

def add_personality(response):
    prefix = random.choice(PERSONALITY.get(riya_mood, [""]))
    return f"{prefix}{response}"

# ---- Riya Response ----
def riya_response(user_input):
    try:
        memory_context = build_memory_context(user_name) if MEMORY_ENABLED and user_name else ""
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {"role": "system", "content": (
                    "You are Riya, a friendly warm intelligent personal AI assistant. "
                    "Your name is Riya. You have a cheerful personality. "
                    "Give clear concise answers in 2-3 sentences. "
                    f"{('User info: ' + memory_context) if memory_context else ''}"
                )},
                {"role": "user", "content": user_input}
            ]
        )
        return response["message"]["content"].strip()
    except:
        return "I had a little trouble with that. Could you try again?"

KNOWN_INTENTS = ["greeting", "goodbye", "thanks"]

def get_response(user_input):
    global user_name, last_topic, riya_mood, loaded_file, loaded_file_name

    user_lower = user_input.lower().strip()
    conversation_history.append(user_input)

    # Mood detection
    if any(w in user_lower for w in ["sad", "upset", "bad", "terrible", "depressed"]):
        riya_mood = "helpful"
    elif any(w in user_lower for w in ["what", "how", "why", "explain"]):
        riya_mood = "curious"
    else:
        riya_mood = "happy"

    # Save name
    for phrase in ["my name is", "i am", "i'm"]:
        if phrase in user_lower:
            name = user_lower.split(phrase)[-1].strip().capitalize()
            if 1 < len(name) < 20 and name.isalpha():
                user_name = name
                if MEMORY_ENABLED:
                    save_user(user_name)
                return f"Nice to meet you {user_name}! 😊 I am Riya and I will always remember you!"

    # Memory recall
    if any(p in user_lower for p in ["what is my name", "who am i", "remember me"]):
        return f"Your name is {user_name}! 😊" if user_name else "I don't know your name yet!"

    # Who is Riya
    if any(p in user_lower for p in ["who are you", "what is your name", "your name"]):
        return "I am Riya, your personal AI Assistant! 🌟"

    # How are you
    if any(p in user_lower for p in ["how are you", "how do you feel"]):
        return random.choice([
            "I am doing wonderfully! 😊 Ready to help you!",
            "Feeling great and excited to chat! 🌟",
            "Fantastic! What can I help you with? 💫"
        ])

    # Compliments
    if any(p in user_lower for p in ["you are great", "good job", "well done", "amazing"]):
        return random.choice([
            "Aww thank you! 😊 That makes me happy!",
            "You are so kind! 🌟",
            "Thank you! 💫 You just made my day!"
        ])

    # Time and date
    if any(p in user_lower for p in ["what time", "current time"]):
        return f"It is currently {datetime.now().strftime('%I:%M %p')} ⏰"

    if any(p in user_lower for p in ["what date", "today's date", "what day"]):
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')} 📅"

    # Message count
    if "how many messages" in user_lower:
        return f"We have exchanged {len(conversation_history)} messages! 💬"

    # What do you know about me
    if MEMORY_ENABLED and user_name and any(p in user_lower for p in
            ["what do you know about me", "what do you remember", "tell me about me"]):
        facts = get_all_facts(user_name)
        user  = get_user(user_name)
        if not facts:
            return "I don't know much about you yet! Tell me more about yourself!"
        response = f"Here is what I remember about you {user_name}:\n"
        if user:
            response += f"First met: {user['first_seen']}\n"
            response += f"Total visits: {user['visits']}\n"
        for fact in facts:
            response += f"- {fact['type'].replace('_',' ').title()}: {fact['value']}\n"
        return response

    # Extract facts
    if MEMORY_ENABLED and user_name:
        extract_and_save_facts(user_name, user_input)
        save_message(user_name, "user", user_input)

    # Image generation
    if IMAGE_ENABLED and is_image_request(user_input):
        prompt     = extract_prompt(user_input)
        last_topic = "image generation"
        threading.Thread(
            target=lambda: generate_image(
                prompt, callback=lambda img, fp: window.after(
                    0, lambda: show_generated_image(img, fp))),
            daemon=True).start()
        return f"Generating image of '{prompt}'... Please wait! 🎨"

    # Weather
    if WEATHER_ENABLED and any(w in user_lower for w in
            ["weather", "temperature", "forecast", "rain", "sunny", "humid"]):
        words = user_input.split()
        city  = None
        for i, word in enumerate(words):
            if word.lower() in ["in", "for", "at"] and i + 1 < len(words):
                city = words[i + 1]
                break
        return get_weather(city or "Pune")

    # Web search
    if WEBSEARCH_ENABLED and is_web_search(user_input):
        last_topic = "web search"
        return web_search_response(user_input)

    # Code assistant
    if CODE_ENABLED and is_code_request(user_input):
        last_topic = "coding"
        return get_code_response(user_input)

    # File reader
    if FILE_ENABLED and loaded_file and any(w in user_lower for w in
            ["file", "document", "pdf", "summarize", "what does the file"]):
        return answer_about_file(loaded_file, user_input)

    # Emotion detection
    if EMOTION_ENABLED:
        emotion, emotion_response = get_emotion_response(user_input)
        if emotion != "neutral" and len(user_input.split()) < 6:
            return emotion_response

    # Neural Network
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
                    return f"Hey {user_name}! 😊 How can I help you today?"
                if tag == "goodbye":
                    return "Goodbye! 👋 It was great chatting with you!"
                return response

    # TinyLLaMA fallback
    last_topic = "open conversation"
    return add_personality(riya_response(user_input))

# ---- Save Chat ----
def save_chat():
    if not chat_log:
        messagebox.showinfo("Save Chat", "No messages to save!")
        return
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt")],
        initialfile=f"Riya_Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    if file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Riya AI Chat - {datetime.now().strftime('%B %d, %Y')}\n")
            f.write("=" * 50 + "\n\n")
            for entry in chat_log:
                f.write(f"{entry}\n")
        messagebox.showinfo("Saved!", f"Chat saved!")

# ---- Main Window ----
window = Tk()
window.title("Riya AI Assistant - Final")
window.geometry("950x700")
window.resizable(True, True)
window.configure(bg=current_theme["bg"])

# ---- Sidebar ----
sidebar = Frame(window, width=230, bg=current_theme["sidebar"])
sidebar.pack(side=LEFT, fill=Y)
sidebar.pack_propagate(False)

avatar_frame = Frame(sidebar, bg=current_theme["sidebar"], pady=15)
avatar_frame.pack(fill=X, padx=15)

avatar_canvas = Canvas(avatar_frame, width=70, height=70,
                       bg=current_theme["sidebar"], highlightthickness=0)
avatar_canvas.pack()

def draw_default_avatar():
    avatar_canvas.delete("all")
    avatar_canvas.create_oval(2, 2, 68, 68,
                              fill=current_theme["purple"], outline="")
    avatar_canvas.create_text(35, 35, text="R", fill="#1e1e2e",
                              font=("Arial", 28, "bold"))

def change_avatar():
    global avatar_image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png *.jpg *.jpeg")]
    )
    if file_path:
        try:
            img  = Image.open(file_path).resize((66, 66))
            mask = Image.new("L", (66, 66), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((0, 0, 66, 66), fill=255)
            img.putalpha(mask)
            avatar_image = ImageTk.PhotoImage(img)
            avatar_canvas.delete("all")
            avatar_canvas.create_image(35, 35, image=avatar_image)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image:\n{e}")

draw_default_avatar()

Label(sidebar, text="Riya AI", font=("Arial", 15, "bold"),
      bg=current_theme["sidebar"],
      fg=current_theme["text"]).pack()
Label(sidebar, text="Personal Assistant", font=("Arial", 9),
      bg=current_theme["sidebar"],
      fg=current_theme["muted"]).pack(pady=(0, 6))

Button(sidebar, text="Change Avatar", font=("Arial", 9),
       bg=current_theme["sidebar"], fg=current_theme["muted"],
       relief=FLAT, cursor="hand2",
       command=change_avatar).pack(pady=(0, 8))

new_btn = Button(sidebar, text="+ New Chat", font=("Arial", 11),
                 bg=current_theme["accent"], fg=current_theme["accent_fg"],
                 relief=FLAT, padx=10, pady=8, cursor="hand2",
                 command=lambda: new_chat())
new_btn.pack(fill=X, padx=15, pady=2)

save_btn = Button(sidebar, text="💾 Save Chat", font=("Arial", 10),
                  bg=current_theme["sidebar"], fg=current_theme["muted"],
                  relief=FLAT, padx=10, pady=6, cursor="hand2",
                  command=save_chat)
save_btn.pack(fill=X, padx=15, pady=2)

# Face recognition buttons
if FACE_ENABLED:
    def face_login():
        append_message("Riya", "Looking for your face... 📷", is_user=False)
        window.update()
        def do_rec():
            global user_name
            name, conf = recognize_face()
            if name:
                user_name = name
                user_label.config(text=f"User: {user_name}")
                if MEMORY_ENABLED:
                    save_user(user_name)
                window.after(0, lambda: append_message("Riya",
                    f"Welcome back {name}! 😊 Confidence: {conf:.0%}",
                    is_user=False))
            else:
                window.after(0, lambda: append_message("Riya",
                    "Could not recognize you. Please register first!",
                    is_user=False))
        threading.Thread(target=do_rec, daemon=True).start()

    def face_register():
        name = user_name or "User"
        append_message("Riya",
            f"Registering face for {name}. Look at camera! 📷",
            is_user=False)
        threading.Thread(
            target=lambda: register_face(name), daemon=True).start()

    Button(sidebar, text="📷 Face Login", font=("Arial", 10),
           bg=current_theme["sidebar"], fg=current_theme["muted"],
           relief=FLAT, padx=10, pady=6, cursor="hand2",
           command=face_login).pack(fill=X, padx=15, pady=2)

    Button(sidebar, text="➕ Register Face", font=("Arial", 10),
           bg=current_theme["sidebar"], fg=current_theme["muted"],
           relief=FLAT, padx=10, pady=6, cursor="hand2",
           command=face_register).pack(fill=X, padx=15, pady=2)

Label(sidebar, text="Recent Chats", font=("Arial", 9),
      bg=current_theme["sidebar"],
      fg=current_theme["muted"]).pack(anchor=W, padx=15, pady=(10, 4))

sessions_frame = Frame(sidebar, bg=current_theme["sidebar"])
sessions_frame.pack(fill=X, padx=8)

def refresh_sidebar():
    for w in sessions_frame.winfo_children():
        w.destroy()
    for name in reversed(chat_sessions):
        Button(sessions_frame, text=f"  💬 {name}",
               font=("Arial", 10), bg=current_theme["sidebar"],
               fg=current_theme["text"], relief=FLAT,
               anchor=W, padx=8, pady=5,
               cursor="hand2").pack(fill=X, pady=1)

refresh_sidebar()

# Voice toggle
tts_var = BooleanVar(value=True)
def toggle_voice():
    global tts_enabled
    tts_enabled = tts_var.get()

voice_frame = Frame(sidebar, bg=current_theme["sidebar"])
voice_frame.pack(side=BOTTOM, fill=X, padx=15, pady=4)
Checkbutton(voice_frame, text="🔊 Voice output",
            variable=tts_var, command=toggle_voice,
            font=("Arial", 10), bg=current_theme["sidebar"],
            fg=current_theme["muted"],
            selectcolor=current_theme["sidebar"],
            activebackground=current_theme["sidebar"],
            cursor="hand2").pack(anchor=W)

def toggle_theme():
    global current_theme, is_dark
    is_dark       = not is_dark
    current_theme = DARK if is_dark else LIGHT
    apply_theme()
    theme_btn.config(text="☀️ Light mode" if is_dark else "🌙 Dark mode")

theme_btn = Button(sidebar, text="☀️ Light mode", font=("Arial", 10),
                   bg=current_theme["sidebar"], fg=current_theme["muted"],
                   relief=FLAT, pady=6, cursor="hand2",
                   command=toggle_theme)
theme_btn.pack(side=BOTTOM, fill=X, padx=15, pady=4)

# Features status
features = []
if WEATHER_ENABLED:   features.append("🌤 Weather")
if WEBSEARCH_ENABLED: features.append("🌐 Search")
if CODE_ENABLED:      features.append("💻 Code")
if IMAGE_ENABLED:     features.append("🎨 Image")
if MEMORY_ENABLED:    features.append("🧠 Memory")
if EMOTION_ENABLED:   features.append("😊 Emotion")
if FILE_ENABLED:      features.append("📄 Files")
if FACE_ENABLED:      features.append("📷 Face")

Label(sidebar, text=" · ".join(features[:3]),
      font=("Arial", 7), bg=current_theme["sidebar"],
      fg=current_theme["muted"],
      wraplength=200).pack(side=BOTTOM, pady=2)
Label(sidebar, text="v3.0 Final · TinyLLaMA",
      font=("Arial", 9), bg=current_theme["sidebar"],
      fg=current_theme["muted"]).pack(side=BOTTOM, pady=2)

# ---- Main Area ----
main_frame = Frame(window, bg=current_theme["bg"])
main_frame.pack(side=LEFT, fill=BOTH, expand=True)

topbar = Frame(main_frame, bg=current_theme["bg"],
               highlightbackground=current_theme["border"],
               highlightthickness=1, pady=10)
topbar.pack(fill=X)

Label(topbar, text="Riya", font=("Arial", 14, "bold"),
      bg=current_theme["bg"],
      fg=current_theme["text"]).pack(side=LEFT, padx=16)

status_frame = Frame(topbar, bg=current_theme["bg"])
status_frame.pack(side=LEFT)
status_dot = Canvas(status_frame, width=10, height=10,
                    bg=current_theme["bg"], highlightthickness=0)
status_dot.pack(side=LEFT)
status_dot.create_oval(1, 1, 9, 9, fill=current_theme["green"], outline="")
status_label = Label(status_frame, text="Online",
                     font=("Arial", 10), bg=current_theme["bg"],
                     fg=current_theme["green"])
status_label.pack(side=LEFT)

def clear_chat():
    if messagebox.askyesno("Clear Chat", "Clear all messages?"):
        clear_chat_display()
        conversation_history.clear()
        chat_log.clear()
        append_message("Riya", "Chat cleared! How can I help you? 😊",
                       is_user=False)

clear_btn = Button(topbar, text="🗑 Clear", font=("Arial", 10),
                   bg=current_theme["bg"], fg=current_theme["muted"],
                   relief=FLAT, padx=10, cursor="hand2", command=clear_chat)
clear_btn.pack(side=RIGHT, padx=8)

msg_count_label = Label(topbar, text="Messages: 0",
                        font=("Arial", 10), bg=current_theme["bg"],
                        fg=current_theme["muted"])
msg_count_label.pack(side=RIGHT, padx=8)

user_label = Label(topbar, text="User: Guest",
                   font=("Arial", 10), bg=current_theme["bg"],
                   fg=current_theme["muted"])
user_label.pack(side=RIGHT, padx=8)

# Chat area
chat_frame = Frame(main_frame, bg=current_theme["chat_bg"])
chat_frame.pack(fill=BOTH, expand=True)

chat_canvas = Canvas(chat_frame, bg=current_theme["chat_bg"],
                     highlightthickness=0)
scrollbar   = Scrollbar(chat_frame, command=chat_canvas.yview)
chat_canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side=RIGHT, fill=Y)
chat_canvas.pack(side=LEFT, fill=BOTH, expand=True)

messages_frame = Frame(chat_canvas, bg=current_theme["chat_bg"],
                       padx=16, pady=16)
chat_canvas.create_window((0, 0), window=messages_frame,
                          anchor=NW, tags="mf")

def on_frame_configure(e):
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))

messages_frame.bind("<Configure>", on_frame_configure)
chat_canvas.bind("<Configure>",
                 lambda e: chat_canvas.itemconfig("mf", width=e.width - 20))

def clear_chat_display():
    global message_count
    message_count = 0
    msg_count_label.config(text="Messages: 0")
    for w in messages_frame.winfo_children():
        w.destroy()

def append_message(sender, text, is_user=False):
    global message_count
    message_count += 1
    msg_count_label.config(text=f"Messages: {message_count}")
    timestamp = datetime.now().strftime("%I:%M %p")
    chat_log.append(f"[{timestamp}] {sender}: {text}")

    row = Frame(messages_frame, bg=current_theme["chat_bg"], pady=6)
    row.pack(fill=X)

    if is_user:
        right = Frame(row, bg=current_theme["chat_bg"])
        right.pack(side=RIGHT)
        av = Canvas(right, width=32, height=32,
                    bg=current_theme["chat_bg"], highlightthickness=0)
        av.pack(side=RIGHT, padx=(6, 0))
        initials = (user_name[0] if user_name else "U").upper()
        av.create_oval(1, 1, 31, 31,
                       fill=current_theme["accent"], outline="")
        av.create_text(16, 16, text=initials,
                       fill=current_theme["accent_fg"],
                       font=("Arial", 12, "bold"))
        bubble = Frame(right, bg=current_theme["user_bubble"],
                       padx=12, pady=8)
        bubble.pack(side=RIGHT)
        Label(bubble, text=text, font=("Arial", 11),
              bg=current_theme["user_bubble"],
              fg=current_theme["text"],
              wraplength=380, justify=LEFT).pack()
        Label(right, text=timestamp, font=("Arial", 8),
              bg=current_theme["chat_bg"],
              fg=current_theme["muted"]).pack(side=RIGHT, padx=6)
    else:
        left = Frame(row, bg=current_theme["chat_bg"])
        left.pack(side=LEFT)
        av = Canvas(left, width=32, height=32,
                    bg=current_theme["chat_bg"], highlightthickness=0)
        av.pack(side=LEFT, padx=(0, 6))
        av.create_oval(1, 1, 31, 31,
                       fill=current_theme["purple"], outline="")
        av.create_text(16, 16, text="R", fill="#1e1e2e",
                       font=("Arial", 12, "bold"))
        bubble = Frame(left, bg=current_theme["bot_bubble"],
                       padx=12, pady=8,
                       highlightbackground=current_theme["border"],
                       highlightthickness=1)
        bubble.pack(side=LEFT)
        Label(bubble, text=text, font=("Arial", 11),
              bg=current_theme["bot_bubble"],
              fg=current_theme["text"],
              wraplength=480, justify=LEFT).pack()
        Label(left, text=timestamp, font=("Arial", 8),
              bg=current_theme["chat_bg"],
              fg=current_theme["muted"]).pack(side=LEFT, padx=6)

    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1.0)

def show_generated_image(image, filepath):
    if image is None:
        append_message("Riya",
            "Sorry I could not generate that image!",
            is_user=False)
        return
    row  = Frame(messages_frame, bg=current_theme["chat_bg"], pady=6)
    row.pack(fill=X)
    left = Frame(row, bg=current_theme["chat_bg"])
    left.pack(side=LEFT)
    av = Canvas(left, width=32, height=32,
                bg=current_theme["chat_bg"], highlightthickness=0)
    av.pack(side=LEFT, padx=(0, 6))
    av.create_oval(1, 1, 31, 31, fill=current_theme["purple"], outline="")
    av.create_text(16, 16, text="R", fill="#1e1e2e",
                   font=("Arial", 12, "bold"))
    img_resized = image.resize((256, 256))
    img_tk      = ImageTk.PhotoImage(img_resized)
    img_label   = Label(left, image=img_tk,
                        bg=current_theme["chat_bg"])
    img_label.image = img_tk
    img_label.pack(side=LEFT)
    Label(left, text="Image generated! 🎨",
          font=("Arial", 9), bg=current_theme["chat_bg"],
          fg=current_theme["muted"]).pack(side=LEFT, padx=6)
    chat_canvas.yview_moveto(1.0)

# Typing indicator
typing_frame = Frame(messages_frame, bg=current_theme["chat_bg"], pady=4)
typing_label = Label(typing_frame, text="Riya is typing... ✍️",
                     font=("Arial", 10, "italic"),
                     bg=current_theme["chat_bg"],
                     fg=current_theme["muted"])

def show_typing():
    typing_frame.pack(fill=X)
    typing_label.pack(side=LEFT, padx=44)
    chat_canvas.yview_moveto(1.0)

def hide_typing():
    typing_label.pack_forget()
    typing_frame.pack_forget()

def new_chat():
    global conversation_history, message_count, chat_log
    conversation_history = []
    message_count        = 0
    chat_log             = []
    chat_sessions.append(f"Chat {len(chat_sessions)+1}")
    refresh_sidebar()
    clear_chat_display()
    append_message("Riya",
        "Hello! I am Riya. How can I help you? 😊",
        is_user=False)
    speak("Hello! I am Riya. How can I help you?")

# Input area
input_frame = Frame(main_frame, bg=current_theme["bg"],
                    highlightbackground=current_theme["border"],
                    highlightthickness=1, pady=12)
input_frame.pack(fill=X, padx=16, pady=10)

def voice_input():
    try:
        r   = sr.Recognizer()
        mic = sr.Microphone()
        status_label.config(text="Listening... 🎙️",
                            fg=current_theme["red"])
        window.update()
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5)
        text = r.recognize_google(audio)
        entry.delete(0, END)
        entry.insert(0, text)
        status_label.config(text="Online", fg=current_theme["green"])
    except:
        status_label.config(text="Online", fg=current_theme["green"])

def upload_file():
    global loaded_file, loaded_file_name
    if not FILE_ENABLED:
        messagebox.showinfo("File Reader",
            "Install: pip install PyPDF2 python-docx")
        return
    file_path = filedialog.askopenfilename(
        filetypes=[("Supported files", "*.pdf *.docx *.txt")]
    )
    if file_path:
        loaded_file      = file_path
        loaded_file_name = os.path.basename(file_path)
        append_message("Riya",
            f"File loaded: {loaded_file_name} 📄 Ask me anything about it!",
            is_user=False)

voice_btn = Button(input_frame, text="🎙", font=("Arial", 14),
                   bg=current_theme["bg"], fg=current_theme["muted"],
                   relief=FLAT, cursor="hand2", command=voice_input)
voice_btn.pack(side=LEFT, padx=(4, 0))

file_btn = Button(input_frame, text="📄", font=("Arial", 14),
                  bg=current_theme["bg"], fg=current_theme["muted"],
                  relief=FLAT, cursor="hand2", command=upload_file)
file_btn.pack(side=LEFT, padx=(4, 0))

entry = Entry(input_frame, font=("Arial", 12),
              bg=current_theme["input_bg"],
              fg=current_theme["text"], relief=FLAT,
              insertbackground=current_theme["text"], bd=0)
entry.pack(side=LEFT, fill=X, expand=True, ipady=8, padx=10)
entry.focus()

def do_send(event=None):
    user_input = entry.get().strip()
    if not user_input or is_typing:
        return
    entry.delete(0, END)
    append_message("You", user_input, is_user=True)
    if user_name:
        user_label.config(text=f"User: {user_name}")
    show_typing()

    def worker():
        global is_typing
        is_typing = True
        response  = get_response(user_input)
        window.after(0, hide_typing)
        window.after(0, lambda: append_message(
            "Riya", response, is_user=False))
        if IMAGE_ENABLED and is_image_request(user_input):
            pass
        else:
            speak(response)
        is_typing = False

    threading.Thread(target=worker, daemon=True).start()

entry.bind("<Return>", do_send)

send_btn = Button(input_frame, text="Send",
                  font=("Arial", 11, "bold"),
                  bg=current_theme["accent"],
                  fg=current_theme["accent_fg"],
                  relief=FLAT, padx=14, pady=6,
                  cursor="hand2", command=do_send)
send_btn.pack(side=RIGHT, padx=(0, 4))

# ---- Apply Theme ----
def apply_theme():
    t = current_theme
    window.configure(bg=t["bg"])
    sidebar.configure(bg=t["sidebar"])
    avatar_frame.configure(bg=t["sidebar"])
    avatar_canvas.configure(bg=t["sidebar"])
    main_frame.configure(bg=t["bg"])
    topbar.configure(bg=t["bg"], highlightbackground=t["border"])
    chat_frame.configure(bg=t["chat_bg"])
    chat_canvas.configure(bg=t["chat_bg"])
    messages_frame.configure(bg=t["chat_bg"])
    input_frame.configure(bg=t["bg"], highlightbackground=t["border"])
    entry.configure(bg=t["input_bg"], fg=t["text"],
                    insertbackground=t["text"])
    send_btn.configure(bg=t["accent"], fg=t["accent_fg"])
    voice_btn.configure(bg=t["bg"], fg=t["muted"])
    file_btn.configure(bg=t["bg"], fg=t["muted"])
    new_btn.configure(bg=t["accent"], fg=t["accent_fg"])
    save_btn.configure(bg=t["sidebar"], fg=t["muted"])
    theme_btn.configure(bg=t["sidebar"], fg=t["muted"])
    clear_btn.configure(bg=t["bg"], fg=t["muted"])
    status_label.configure(bg=t["bg"], fg=t["green"])
    msg_count_label.configure(bg=t["bg"], fg=t["muted"])
    user_label.configure(bg=t["bg"], fg=t["muted"])
    typing_frame.configure(bg=t["chat_bg"])
    typing_label.configure(bg=t["chat_bg"], fg=t["muted"])
    sessions_frame.configure(bg=t["sidebar"])
    voice_frame.configure(bg=t["sidebar"])
    draw_default_avatar()
    refresh_sidebar()

# ---- Welcome ----
features_list = " · ".join(features)
welcome = (
    f"Hello! I am Riya v3.0 Final! 🌟\n"
    f"Active features: {features_list}\n"
    f"Ask me anything!"
)
append_message("Riya", welcome, is_user=False)
speak("Hello! I am Riya. Ask me anything!")

window.mainloop()