from typing import Optional
try:
    import requests
    from googlesearch import search
    from bs4 import BeautifulSoup
    WEB_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Bot: Warning: Web search dependencies not installed ({e}). Web search disabled.")
    WEB_SEARCH_AVAILABLE = False

class WebSearcher:
    """Handles web searches for answers, including general and Stack Overflow searches."""
   
    def fetch_web_answer(self, query: str, stack_overflow: bool = False) -> Optional[str]:
        """Fetch an answer from the web or Stack Overflow."""
        if not WEB_SEARCH_AVAILABLE:
            return None
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            search_query = f"{query} site:stackoverflow.com" if stack_overflow else query
            for url in search(search_query, num_results=3):
                try:
                    response = requests.get(url, headers=headers, timeout=5)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    text = ' '.join(p.get_text() for p in paragraphs[:3] if p.get_text())
                    if text:
                        return text[:500] + "..." if len(text) > 500 else text
                except requests.RequestException:
                    continue
            return None
        except Exception as e:
            print(f"Bot: Error fetching web results: {e}")
            return None