import json
import random
import threading
from datetime import datetime
from tkinter import *
from tkinter import scrolledtext, messagebox
import ollama
from tokenizer import bag_of_words
from model import NeuralNet
import torch
import speech_recognition as sr

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

# ---- Themes ----
DARK = {
    "bg":          "#1e1e2e",
    "sidebar":     "#181825",
    "chat_bg":     "#313244",
    "input_bg":    "#45475a",
    "text":        "#cdd6f4",
    "muted":       "#a6adc8",
    "accent":      "#89b4fa",
    "accent_fg":   "#1e1e2e",
    "user_bubble": "#45475a",
    "bot_bubble":  "#313244",
    "border":      "#585b70",
    "btn_hover":   "#585b70",
    "green":       "#a6e3a1",
    "red":         "#f38ba8",
}
LIGHT = {
    "bg":          "#ffffff",
    "sidebar":     "#f1f3f4",
    "chat_bg":     "#f8f9fa",
    "input_bg":    "#ffffff",
    "text":        "#202124",
    "muted":       "#5f6368",
    "accent":      "#1a73e8",
    "accent_fg":   "#ffffff",
    "user_bubble": "#e8f0fe",
    "bot_bubble":  "#ffffff",
    "border":      "#dadce0",
    "btn_hover":   "#e8eaed",
    "green":       "#34a853",
    "red":         "#ea4335",
}

current_theme = DARK
is_dark       = True

# ---- State ----
conversation_history = []
chat_sessions        = ["Chat 1"]
current_session      = 0
user_name            = None
last_topic           = None
is_typing            = False
typing_dots          = 0
message_count        = 0

# ---- Riya Response ----
def riya_response(user_input):
    try:
        messages = [{"role": "system", "content": "You are Riya, a helpful friendly AI assistant. Your name is Riya. Give short clear answers in 2-3 sentences."}]
        for i, msg in enumerate(conversation_history[-6:]):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": msg})
        messages.append({"role": "user", "content": user_input})
        response = ollama.chat(model="tinyllama", messages=messages)
        return response["message"]["content"].strip()
    except:
        return "I had trouble thinking about that. Please try again!"

KNOWN_INTENTS = ["greeting", "goodbye", "thanks"]

def get_response(user_input):
    global user_name, last_topic
    user_lower = user_input.lower().strip()
    conversation_history.append(user_input)

    for phrase in ["my name is", "i am", "i'm"]:
        if phrase in user_lower:
            name = user_lower.split(phrase)[-1].strip().capitalize()
            if 1 < len(name) < 20 and name.isalpha():
                user_name = name
                return f"Nice to meet you {user_name}! I am Riya and I will remember that!"

    if any(p in user_lower for p in ["what is my name", "who am i", "do you remember me"]):
        return f"Your name is {user_name}!" if user_name else "I don't know your name yet!"

    if any(p in user_lower for p in ["what is your name", "who are you", "your name"]):
        return "I am Riya, your personal AI Assistant!"

    if "how many messages" in user_lower:
        return f"We have exchanged {len(conversation_history)} messages!"

    if "what time" in user_lower or "current time" in user_lower:
        return f"Current time is {datetime.now().strftime('%I:%M %p')}"

    if "what date" in user_lower or "today's date" in user_lower:
        return f"Today is {datetime.now().strftime('%B %d, %Y')}"

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
                    return f"Hey {user_name}! How can I help you today?"
                return response

    last_topic = "open conversation"
    return riya_response(user_input)

# ---- Main Window ----
window = Tk()
window.title("Riya AI Assistant")
window.geometry("900x650")
window.resizable(True, True)
window.configure(bg=current_theme["bg"])

# ---- Frames ----
sidebar = Frame(window, width=220, bg=current_theme["sidebar"])
sidebar.pack(side=LEFT, fill=Y)
sidebar.pack_propagate(False)

main_frame = Frame(window, bg=current_theme["bg"])
main_frame.pack(side=LEFT, fill=BOTH, expand=True)

# ---- Sidebar Content ----
# Logo
logo_frame = Frame(sidebar, bg=current_theme["sidebar"], pady=15)
logo_frame.pack(fill=X, padx=15)

logo_circle = Canvas(logo_frame, width=36, height=36,
                     bg=current_theme["sidebar"], highlightthickness=0)
logo_circle.pack(side=LEFT)
logo_circle.create_oval(2, 2, 34, 34, fill="#89b4fa", outline="")
logo_circle.create_text(18, 18, text="R", fill="#1e1e2e",
                        font=("Arial", 16, "bold"))

Label(logo_frame, text="  Riya AI", font=("Arial", 16, "bold"),
      bg=current_theme["sidebar"],
      fg=current_theme["text"]).pack(side=LEFT, pady=5)

# New Chat Button
def new_chat():
    global conversation_history, message_count
    conversation_history = []
    message_count        = 0
    session_name = f"Chat {len(chat_sessions)+1}"
    chat_sessions.append(session_name)
    refresh_sidebar()
    clear_chat_display()
    append_message("Riya", "Hello! I am Riya. How can I help you?", is_user=False)

new_btn = Button(sidebar, text="+ New Chat",
                 font=("Arial", 11), bg=current_theme["accent"],
                 fg=current_theme["accent_fg"], relief=FLAT,
                 padx=10, pady=8, cursor="hand2", command=new_chat)
new_btn.pack(fill=X, padx=15, pady=8)

Label(sidebar, text="Recent Chats", font=("Arial", 9),
      bg=current_theme["sidebar"],
      fg=current_theme["muted"]).pack(anchor=W, padx=15, pady=(8, 4))

sessions_frame = Frame(sidebar, bg=current_theme["sidebar"])
sessions_frame.pack(fill=X, padx=8)

def refresh_sidebar():
    for w in sessions_frame.winfo_children():
        w.destroy()
    for i, name in enumerate(reversed(chat_sessions)):
        btn = Button(sessions_frame, text=f"  {name}",
                     font=("Arial", 10), bg=current_theme["sidebar"],
                     fg=current_theme["text"], relief=FLAT,
                     anchor=W, padx=8, pady=6, cursor="hand2")
        btn.pack(fill=X, pady=1)

refresh_sidebar()

# Theme toggle at bottom of sidebar
def toggle_theme():
    global current_theme, is_dark
    is_dark       = not is_dark
    current_theme = DARK if is_dark else LIGHT
    apply_theme()
    theme_btn.config(text="Light mode" if is_dark else "Dark mode")

theme_btn = Button(sidebar, text="Light mode",
                   font=("Arial", 10), bg=current_theme["sidebar"],
                   fg=current_theme["muted"], relief=FLAT,
                   pady=6, cursor="hand2", command=toggle_theme)
theme_btn.pack(side=BOTTOM, fill=X, padx=15, pady=10)

Label(sidebar, text="v1.0 · TinyLLaMA", font=("Arial", 9),
      bg=current_theme["sidebar"],
      fg=current_theme["muted"]).pack(side=BOTTOM, pady=4)

# ---- Top Bar ----
topbar = Frame(main_frame, bg=current_theme["bg"],
               highlightbackground=current_theme["border"],
               highlightthickness=1, pady=10)
topbar.pack(fill=X)

Label(topbar, text="Riya", font=("Arial", 14, "bold"),
      bg=current_theme["bg"],
      fg=current_theme["text"]).pack(side=LEFT, padx=16)

# Status indicator
status_frame = Frame(topbar, bg=current_theme["bg"])
status_frame.pack(side=LEFT, padx=8)

status_dot = Canvas(status_frame, width=10, height=10,
                    bg=current_theme["bg"], highlightthickness=0)
status_dot.pack(side=LEFT)
status_dot.create_oval(1, 1, 9, 9, fill=current_theme["green"], outline="")

status_label = Label(status_frame, text="Online",
                     font=("Arial", 10), bg=current_theme["bg"],
                     fg=current_theme["green"])
status_label.pack(side=LEFT)

# Clear button
def clear_chat():
    if messagebox.askyesno("Clear Chat", "Clear all messages?"):
        clear_chat_display()
        conversation_history.clear()
        append_message("Riya", "Chat cleared! How can I help you?", is_user=False)

clear_btn = Button(topbar, text="Clear",
                   font=("Arial", 10), bg=current_theme["bg"],
                   fg=current_theme["muted"], relief=FLAT,
                   padx=10, cursor="hand2", command=clear_chat)
clear_btn.pack(side=RIGHT, padx=8)

# Message count
msg_count_label = Label(topbar, text="Messages: 0",
                        font=("Arial", 10), bg=current_theme["bg"],
                        fg=current_theme["muted"])
msg_count_label.pack(side=RIGHT, padx=8)

# User label
user_label = Label(topbar, text="User: Guest",
                   font=("Arial", 10), bg=current_theme["bg"],
                   fg=current_theme["muted"])
user_label.pack(side=RIGHT, padx=8)

# ---- Chat Area ----
chat_frame = Frame(main_frame, bg=current_theme["chat_bg"])
chat_frame.pack(fill=BOTH, expand=True, padx=0, pady=0)

chat_canvas = Canvas(chat_frame, bg=current_theme["chat_bg"],
                     highlightthickness=0)
scrollbar = Scrollbar(chat_frame, command=chat_canvas.yview)
chat_canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side=RIGHT, fill=Y)
chat_canvas.pack(side=LEFT, fill=BOTH, expand=True)

messages_frame = Frame(chat_canvas, bg=current_theme["chat_bg"], padx=16, pady=16)
chat_canvas.create_window((0, 0), window=messages_frame, anchor=NW, tags="mf")

def on_frame_configure(e):
    chat_canvas.configure(scrollregion=chat_canvas.bbox("all"))
    chat_canvas.itemconfig("mf", width=chat_canvas.winfo_width() - 20)

messages_frame.bind("<Configure>", on_frame_configure)
chat_canvas.bind("<Configure>", lambda e: chat_canvas.itemconfig(
    "mf", width=e.width - 20))

def clear_chat_display():
    for w in messages_frame.winfo_children():
        w.destroy()

def append_message(sender, text, is_user=False):
    global message_count
    message_count += 1
    msg_count_label.config(text=f"Messages: {message_count}")

    row = Frame(messages_frame, bg=current_theme["chat_bg"], pady=6)
    row.pack(fill=X)

    if is_user:
        # User message - right aligned
        right = Frame(row, bg=current_theme["chat_bg"])
        right.pack(side=RIGHT)

        # Avatar
        av = Canvas(right, width=32, height=32,
                    bg=current_theme["chat_bg"], highlightthickness=0)
        av.pack(side=RIGHT, padx=(6, 0))
        initials = (user_name[0] if user_name else "U").upper()
        av.create_oval(1, 1, 31, 31, fill="#89b4fa", outline="")
        av.create_text(16, 16, text=initials, fill="#1e1e2e",
                       font=("Arial", 12, "bold"))

        bubble = Frame(right, bg=current_theme["user_bubble"],
                       padx=12, pady=8)
        bubble.pack(side=RIGHT)
        Label(bubble, text=text, font=("Arial", 11),
              bg=current_theme["user_bubble"],
              fg=current_theme["text"],
              wraplength=400, justify=LEFT).pack()

        ts = Label(right, text=datetime.now().strftime("%I:%M %p"),
                   font=("Arial", 8), bg=current_theme["chat_bg"],
                   fg=current_theme["muted"])
        ts.pack(side=RIGHT, padx=6)

    else:
        # Riya message - left aligned
        left = Frame(row, bg=current_theme["chat_bg"])
        left.pack(side=LEFT)

        av = Canvas(left, width=32, height=32,
                    bg=current_theme["chat_bg"], highlightthickness=0)
        av.pack(side=LEFT, padx=(0, 6))
        av.create_oval(1, 1, 31, 31, fill="#cba6f7", outline="")
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

        ts = Label(left, text=datetime.now().strftime("%I:%M %p"),
                   font=("Arial", 8), bg=current_theme["chat_bg"],
                   fg=current_theme["muted"])
        ts.pack(side=LEFT, padx=6)

    chat_canvas.update_idletasks()
    chat_canvas.yview_moveto(1.0)

# Typing indicator
typing_frame = Frame(messages_frame, bg=current_theme["chat_bg"], pady=4)
typing_label = Label(typing_frame, text="Riya is typing...",
                     font=("Arial", 10, "italic"),
                     bg=current_theme["chat_bg"],
                     fg=current_theme["muted"])

def show_typing():
    typing_frame.pack(fill=X)
    typing_label.pack(side=LEFT, padx=38)
    chat_canvas.yview_moveto(1.0)

def hide_typing():
    typing_label.pack_forget()
    typing_frame.pack_forget()

# ---- Input Area ----
input_frame = Frame(main_frame, bg=current_theme["bg"],
                    highlightbackground=current_theme["border"],
                    highlightthickness=1, pady=12)
input_frame.pack(fill=X, padx=16, pady=10)

# Voice button
def voice_input():
    try:
        r   = sr.Recognizer()
        mic = sr.Microphone()
        status_label.config(text="Listening...", fg=current_theme["red"])
        window.update()
        with mic as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5)
        text = r.recognize_google(audio)
        entry.delete(0, END)
        entry.insert(0, text)
        status_label.config(text="Online", fg=current_theme["green"])
    except ImportError:
        messagebox.showinfo("Voice", "Install SpeechRecognition:\npip install SpeechRecognition pyaudio")
        status_label.config(text="Online", fg=current_theme["green"])
    except Exception as e:
        status_label.config(text="Online", fg=current_theme["green"])

voice_btn = Button(input_frame, text="🎙",
                   font=("Arial", 14), bg=current_theme["bg"],
                   fg=current_theme["muted"], relief=FLAT,
                   cursor="hand2", command=voice_input)
voice_btn.pack(side=LEFT, padx=(4, 0))

entry = Entry(input_frame, font=("Arial", 12),
              bg=current_theme["input_bg"],
              fg=current_theme["text"],
              relief=FLAT, insertbackground=current_theme["text"],
              bd=0)
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
        window.after(0, lambda: hide_typing())
        window.after(0, lambda: append_message("Riya", response, is_user=False))
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
    new_btn.configure(bg=t["accent"], fg=t["accent_fg"])
    theme_btn.configure(bg=t["sidebar"], fg=t["muted"])
    clear_btn.configure(bg=t["bg"], fg=t["muted"])
    status_label.configure(bg=t["bg"], fg=t["green"])
    msg_count_label.configure(bg=t["bg"], fg=t["muted"])
    user_label.configure(bg=t["bg"], fg=t["muted"])
    typing_frame.configure(bg=t["chat_bg"])
    typing_label.configure(bg=t["chat_bg"], fg=t["muted"])
    sessions_frame.configure(bg=t["sidebar"])
    refresh_sidebar()

# ---- Welcome Message ----
append_message("Riya", "Hello! I am Riya, your personal AI Assistant! Ask me anything!", is_user=False)

window.mainloop()