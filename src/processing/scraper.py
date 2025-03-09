import requests
from bs4 import BeautifulSoup
import re
import concurrent.futures

def scrape_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
        
        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)
        
        return f"Source: {url}\n\n{text}"
    except Exception:
        return ""

def process_urls(urls, progress_callback=None):
    texts = []
    successful_urls = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_url, url): url for url in urls}
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_url)):
            url = future_to_url[future]
            try:
                text = future.result()
                if text:
                    texts.append(text)
                    successful_urls.append(url)
            except Exception:
                pass
            
            if progress_callback:
                progress_callback((i + 1) / len(urls))
    
    return texts, successful_urls