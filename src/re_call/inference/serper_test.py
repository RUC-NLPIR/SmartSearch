import requests
import json
from urllib.parse import urlencode
import langid


def fetch_search_results(query):
    serper_api_key = "your_serper_api_key_here"
    
    if langid.classify(query)[0] == "zh":
        gl = "cn"
        hl = "zh-cn"
    else:
        gl = "us"
        hl = "en"
    
    serper_url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    data = {
        "q": query,
        "gl": gl,
        "hl": hl,
        "num": 10,
        "autocorrect": True
    }
    
    try:
        response = requests.post(serper_url, headers=headers, json=data)
        response.raise_for_status()
        serper_data = response.json()
    except Exception as e:
        return f"Error fetching search results: {str(e)}"
        
    formatted_results = []
    
    if 'answerBox' in serper_data:
        ab = serper_data['answerBox']
        title = ab.get('title', '')
        snippet = ab.get('snippet', '')
        formatted_results.append(f'"{title}"\n{snippet}')
    
    if 'organic' in serper_data:
        for result in serper_data['organic']:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            formatted_results.append(f'"{title}"\n{snippet}')
    
    return "\n\n".join(formatted_results)