from tinydb import TinyDB, Query
from datetime import datetime
import os
import re

db_path = os.path.join(os.path.dirname(__file__), "riya_memory.json")
db      = TinyDB(db_path)

users_table   = db.table("users")
facts_table   = db.table("facts")
history_table = db.table("history")

User = Query()

def save_user(name):
    existing = users_table.search(User.name == name)
    if not existing:
        users_table.insert({
            "name":       name,
            "first_seen": datetime.now().strftime("%B %d, %Y"),
            "visits":     1
        })
    else:
        users_table.update(
            {"visits": existing[0]["visits"] + 1},
            User.name == name
        )

def get_user(name):
    result = users_table.search(User.name == name)
    return result[0] if result else None

def save_fact(user_name, fact_type, value):
    existing = facts_table.search(
        (User.user == user_name) & (User.type == fact_type)
    )
    if existing:
        facts_table.update(
            {"value": value, "updated": datetime.now().strftime("%B %d, %Y %H:%M")},
            (User.user == user_name) & (User.type == fact_type)
        )
    else:
        facts_table.insert({
            "user":    user_name,
            "type":    fact_type,
            "value":   value,
            "updated": datetime.now().strftime("%B %d, %Y %H:%M")
        })

def get_fact(user_name, fact_type):
    result = facts_table.search(
        (User.user == user_name) & (User.type == fact_type)
    )
    return result[0]["value"] if result else None

def get_all_facts(user_name):
    return facts_table.search(User.user == user_name)

def save_message(user_name, role, message):
    history_table.insert({
        "user":      user_name,
        "role":      role,
        "message":   message,
        "timestamp": datetime.now().strftime("%B %d, %Y %H:%M")
    })

def get_recent_history(user_name, limit=10):
    history = history_table.search(User.user == user_name)
    return history[-limit:] if len(history) > limit else history

def extract_and_save_facts(user_name, text):
    text_lower = text.lower()
    saved      = []

    # Age
    age_match = re.search(r"i am (\d+) years old|my age is (\d+)", text_lower)
    if age_match:
        age = age_match.group(1) or age_match.group(2)
        save_fact(user_name, "age", age)
        saved.append(f"age: {age}")

    # Job
    job_match = re.search(r"i (?:am|work as) (?:a |an )?(.+?)(?:\.|$)", text_lower)
    if job_match:
        job = job_match.group(1).strip()
        if len(job) < 30:
            save_fact(user_name, "job", job)
            saved.append(f"job: {job}")

    # Location
    location_match = re.search(r"i (?:live|am from|am in) (.+?)(?:\.|$)", text_lower)
    if location_match:
        location = location_match.group(1).strip()
        if len(location) < 30:
            save_fact(user_name, "location", location)
            saved.append(f"location: {location}")

    # Hobby
    hobby_match = re.search(r"i (?:love|like|enjoy) (.+?)(?:\.|$)", text_lower)
    if hobby_match:
        hobby = hobby_match.group(1).strip()
        if len(hobby) < 30:
            save_fact(user_name, "hobby", hobby)
            saved.append(f"hobby: {hobby}")

    # Favorite
    fav_match = re.search(r"my favorite (.+?) is (.+?)(?:\.|$)", text_lower)
    if fav_match:
        fav_type  = fav_match.group(1).strip()
        fav_value = fav_match.group(2).strip()
        save_fact(user_name, f"favorite_{fav_type}", fav_value)
        saved.append(f"favorite {fav_type}: {fav_value}")

    return saved

def build_memory_context(user_name):
    if not user_name:
        return ""

    context = f"User's name: {user_name}\n"
    user    = get_user(user_name)

    if user:
        context += f"First met: {user['first_seen']}\n"
        context += f"Total visits: {user['visits']}\n"

    facts = get_all_facts(user_name)
    if facts:
        context += "Known facts:\n"
        for fact in facts:
            context += f"- {fact['type']}: {fact['value']}\n"

    return context

if __name__ == "__main__":
    print("Memory System Test")
    print("=" * 40)
    name = input("Your name: ")
    save_user(name)

    while True:
        text = input("You: ").strip()
        if text.lower() == "quit":
            break

        saved = extract_and_save_facts(name, text)
        if saved:
            print(f"Riya remembered: {', '.join(saved)}")

        save_message(name, "user", text)

    print("\nAll facts I know about you:")
    for fact in get_all_facts(name):
        print(f"- {fact['type']}: {fact['value']}")