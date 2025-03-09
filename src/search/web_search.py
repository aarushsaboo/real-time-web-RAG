from duckduckgo_search import DDGS

def search_web(query, num_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append(r['href'])
    return results