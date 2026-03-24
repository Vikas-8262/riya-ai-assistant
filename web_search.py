from duckduckgo_search import DDGS
import ollama
import time

WEB_KEYWORDS = [
    "search for", "look up", "find information",
    "what is the latest", "current news", "today's news",
    "who is", "what happened", "when did", "where is",
    "how much does", "price of", "news about",
    "latest", "recent", "right now", "currently",
    "search", "google", "look up"
]

# Cache for search results
search_cache = {}
CACHE_EXPIRY = 3600

def is_web_search(text):
    """Check if user input is asking for a web search"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in WEB_KEYWORDS)

def search_web(query, max_results=3):
    """Search the web using DuckDuckGo"""
    # Check cache first
    if query in search_cache:
        cached_time, cached_results = search_cache[query]
        if time.time() - cached_time < CACHE_EXPIRY:
            return cached_results
    
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body":  r.get("body", ""),
                    "url":   r.get("href", "")
                })
        
        # Cache the results
        search_cache[query] = (time.time(), results)
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def web_search_response(query):
    """Perform web search and return AI-summarized response"""
    try:
        results = search_web(query)
        if not results:
            return "I could not find any results. Try rephrasing your question!"

        # Build context from search results
        context = ""
        for i, r in enumerate(results, 1):
            context += f"Result {i}: {r['title']}\n{r['body']}\n\n"

        # Use TinyLLaMA to summarize results
        response = ollama.chat(
            model="tinyllama",
            messages=[
                {"role": "system", "content": (
                    "You are Riya, a helpful friendly AI assistant. "
                    "Summarize the web search results clearly in 2-3 sentences. "
                    "Be informative and accurate."
                )},
                {"role": "user", "content": (
                    f"Search query: {query}\n\n"
                    f"Search results:\n{context}\n\n"
                    f"Please summarize the key information from these results."
                )}
            ]
        )

        answer  = response["message"]["content"].strip()
        sources = "\n\n📚 Sources:\n"
        for i, r in enumerate(results, 1):
            sources += f"{i}. {r['title']}\n   🔗 {r['url']}\n"

        return answer + sources

    except Exception as e:
        return "I could not search the web right now. Please try again!"

if __name__ == "__main__":
    print("Riya Web Search Module")
    print("=" * 50)
    while True:
        query = input("Search (or 'quit'): ").strip()
        if query.lower() == "quit":
            break
        if query:
            print(f"\nRiya: {web_search_response(query)}\n")