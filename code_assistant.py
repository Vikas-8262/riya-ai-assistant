import ollama

CODE_KEYWORDS = [
    "write code", "code for", "program for", "script for",
    "how to code", "debug", "fix this code", "explain code",
    "what does this code", "function for", "create a function",
    "write a function", "write a program", "build a program",
    "python code", "javascript code", "html code", "css code",
    "java code", "c++ code", "sql query", "regex for"
]

def is_code_request(text):
    """Check if user input is a code-related request."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CODE_KEYWORDS)

def get_code_response(user_input):
    """Generate code response using Ollama."""
    try:
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {"role": "system", "content": (
                    "You are Riya, an expert coding assistant. "
                    "When asked to write code, always provide clean, "
                    "well-commented code with a brief explanation. "
                    "Format code blocks properly. "
                    "Support all programming languages."
                )},
                {"role": "user", "content": user_input}
            ]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"I had trouble generating code. Error: {str(e)}"

if __name__ == "__main__":
    print("Riya Code Assistant Ready!")
    print("=" * 40)
    while True:
        user = input("You: ").strip()
        if user.lower() == "quit":
            break
        if is_code_request(user):
            print(f"\nRiya: {get_code_response(user)}\n")
        else:
            print("Riya: Please ask me to write or explain some code!\n")