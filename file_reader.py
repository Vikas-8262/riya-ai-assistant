import os
import ollama

try:
    import PyPDF2
    PDF_ENABLED = True
except:
    PDF_ENABLED = False

try:
    import docx
    DOCX_ENABLED = True
except:
    DOCX_ENABLED = False

def read_pdf(file_path):
    """Extract text from PDF files"""
    if not PDF_ENABLED:
        return "Install PyPDF2: pip install PyPDF2"
    try:
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        return f"Could not read PDF: {e}"

def read_docx(file_path):
    """Extract text from Word documents"""
    if not DOCX_ENABLED:
        return "Install python-docx: pip install python-docx"
    try:
        doc  = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        return f"Could not read Word file: {e}"

def read_txt(file_path):
    """Read text files"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        return f"Could not read text file: {e}"

def read_file(file_path):
    """Read file based on extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext == ".txt":
        return read_txt(file_path)
    else:
        return "Sorry I can only read PDF, DOCX and TXT files!"

def answer_about_file(file_path, question):
    """Answer questions about file content using AI"""
    content = read_file(file_path)
    
    # Check for errors
    if content.startswith("Could not") or content.startswith("Sorry") or content.startswith("Install"):
        return content
    
    # Limit content to prevent token overflow
    content = content[:3000]
    
    try:
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {"role": "system", "content": (
                    "You are Riya, a helpful friendly AI assistant. "
                    "Answer questions based on the file content provided. "
                    "Be clear, concise and accurate. "
                    "If the answer is not in the file, say so."
                )},
                {"role": "user", "content": (
                    f"File content:\n\n{content}\n\n"
                    f"Question: {question}"
                )}
            ]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return "I had trouble reading that file. Please try again!"

if __name__ == "__main__":
    print("Riya File Reader Module")
    print("=" * 50)
    file_path = input("Enter file path: ").strip()
    
    if not os.path.exists(file_path):
        print("❌ File not found!")
    else:
        print(f"✅ File loaded: {os.path.basename(file_path)}")
        while True:
            question = input("\n❓ Ask about the file (or 'quit'): ").strip()
            if question.lower() == "quit":
                break
            if question:
                print(f"\n🤖 Riya: {answer_about_file(file_path, question)}\n")