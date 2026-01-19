import os
import json
import datasets
import jsonlines
import argparse
import random
random.seed(42)

wikipedia_search_env = """import requests

def wikipedia_search(query: str, top_n: int = 5):
    url = "<search-url-placeholder>/search"
    
    if query == '':
        return 'invalid query'
    
    data = {'query': query, 'top_n': top_n}
    response = requests.post(url, json=data)
    retrieval_text = ''
    for line in response.json():
        retrieval_text += f"{line['contents']}\\n\\n"
    retrieval_text = retrieval_text.strip()
    
    return retrieval_text"""

wikipedia_search_schemas = [{
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": "Search Wikipedia for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query to search for."
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of results to return. The default value is 5.",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]
wikipedia_search_schemas = json.dumps(wikipedia_search_schemas, indent=4)

if __name__ == '__main__':
    train_data_path = 'data/asearcher/train.jsonl'
    lines = []
    with jsonlines.open(train_data_path) as reader:
        for line in reader:
            lines.append(line)
    train_data = []
    for line in lines:
        train_data.append({
            "data_source": "re_call",
            "question": line['question'],
            "ability": "re_call",
            "reward_model": {
                    "style": "rule",
                    "ground_truth": line['golden_answers']
                },
            "extra_info": {
                "env": wikipedia_search_env,
                "func_schemas": wikipedia_search_schemas
            }
        })
        
    train_data = random.sample(train_data, 2000)

    dev_data_1 = []
    dev_data_2 = []
    dev_data_3 = []
    dev_data_4 = []
    
    dev_data_path = 'data/bamboogle/test.jsonl'
    lines = []
    with jsonlines.open(dev_data_path) as reader:
        for line in reader:
            lines.append(line)
    for line in lines:
        dev_data_1.append({
            "data_source": "re_call",
            "question": line['question'],
            "ability": "re_call",
            "reward_model": {
                "style": "rule",
                "ground_truth": line['golden_answers']
            },
            "extra_info": {
                # "id": line['id'],
                "env": wikipedia_search_env,
                "func_schemas": wikipedia_search_schemas
            }
        })
        
    dev_data_path = 'data/2wikimultihopqa/test_all.jsonl'
    lines = []
    with jsonlines.open(dev_data_path) as reader:
        for line in reader:
            lines.append(line)
    for line in lines:
        dev_data_2.append({
            "data_source": "re_call",
            "question": line['question'],
            "ability": "re_call",
            "reward_model": {
                "style": "rule",
                "ground_truth": line['golden_answers']
            },
            "extra_info": {
                # "id": line['id'],
                "env": wikipedia_search_env,
                "func_schemas": wikipedia_search_schemas
            }
        })
        
    dev_data_path = 'data/hotpotqa/test_all.jsonl'
    lines = []
    with jsonlines.open(dev_data_path) as reader:
        for line in reader:
            lines.append(line)
    for line in lines:
        dev_data_3.append({
            "data_source": "re_call",
            "question": line['question'],
            "ability": "re_call",
            "reward_model": {
                "style": "rule",
                "ground_truth": line['golden_answers']
            },
            "extra_info": {
                # "id": line['id'],
                "env": wikipedia_search_env,
                "func_schemas": wikipedia_search_schemas
            }
        })
        
    dev_data_path = 'data/musique/test_all.jsonl'
    lines = []
    with jsonlines.open(dev_data_path) as reader:
        for line in reader:
            lines.append(line)
    for line in lines:
        dev_data_4.append({
            "data_source": "re_call",
            "question": line['question'],
            "ability": "re_call",
            "reward_model": {
                "style": "rule",
                "ground_truth": line['golden_answers']
            },
            "extra_info": {
                "env": wikipedia_search_env,
                "func_schemas": wikipedia_search_schemas
            }
        })
    
    dev_data_1 = random.sample(dev_data_1, 100)
    dev_data_2 = random.sample(dev_data_2, 100)
    dev_data_3 = random.sample(dev_data_3, 100)
    dev_data_4 = random.sample(dev_data_4, 100)
    
    dev_data = dev_data_1 + dev_data_2 + dev_data_3 + dev_data_4
    
    print(len(train_data))
    print(len(dev_data))

    train_dataset = datasets.Dataset.from_list(train_data)
    test_dataset = datasets.Dataset.from_list(dev_data)

    train_dataset.to_parquet(os.path.join('data/grpo', 'train.parquet'))
    test_dataset.to_parquet(os.path.join('data/grpo', 'test.parquet'))