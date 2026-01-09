import requests

tool_call_str = 'Crydamoure'
url = 'http://0.0.0.0:9090/execute'
wikipedia_search_env = """import requests

def wikipedia_search(query: str, top_n: int = 5):
    url = "http://0.0.0.0:8080/search"
    
    if query == '':
        return 'invalid query'
    
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)
    retrieval_text = ''
    for line in response.json():
        retrieval_text += f"{line['contents']}\\n\\n"
    retrieval_text = retrieval_text.strip()
    
    return retrieval_text"""
func_name = "wikipedia_search"
arguments = {"query": tool_call_str, "top_n": 5}
args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
call_str = f"{func_name}({args_str})"
data = {
    'env': wikipedia_search_env,
    'call': call_str
}
response = requests.post(url, json=data, timeout=10)
print(response.json())