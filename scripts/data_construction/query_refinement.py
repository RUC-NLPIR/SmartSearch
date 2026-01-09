import json
import requests
import re
import argparse
from tqdm import tqdm

class Rewrite:
    def __init__(self, model_url):
        self.model_url = model_url

    def generate_response(self, curr_prompt):
        try:
            response = requests.post(
                f'{self.model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": 0.6,
                        "max_new_tokens": 2048,
                        "stop": ['</search>']
                    }
                }).json()
        except Exception as e:
            return ""
        return response.get('text', '') + "</search>"

    def rewrite_search(self, data):
        assistant_prefix = f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        for entry in tqdm(data):
            question = entry['question']

            for i, process_score in enumerate(entry['process_scores']):
                context = process_score['context']
                score = process_score['score']
                
                redundancy_value = entry.get('redundancy', [])[i] if i < len(entry.get('redundancy', [])) else 0
                explanation = ""
                if score == "0":
                    explanation += process_score['explanation']
                if redundancy_value > 1:
                    explanation += ' The agent\'s query of the current search round is redundant, meaning that the query result duplicates information from previous search rounds.'
                
                if score == "0" or redundancy_value > 1:
                    user_prompt = f'''
You are a query-refine assistant. Your task is to refine a search agent's query of the current search round within <search> </search> according to the user's question, the agent's search process up to the current search round and the issues of the query.

The details of the refinement are in the Refine Guideline, please read it carefully.

### User's question
{question}

### Agent's search process up to the current search round
{context}

### Issues of the query
{explanation}

### Refine Guideline
1. The refined query is meant to replace the query of the current round, so **don't rely on any query result within <result> </result> from the current round** when refining the query.
2. If the issues of the query indicate that the query intent is unreasonable, the refined query should **serve for a more necessary and actionable query intent**.
3. The refined query can be expressed as **a complete semantic question or a keyphrase-based query**, and you may **add or remove information from the original query**. All depends on which option best serves the agent's query intent, ensuring that the query result contains the answer to the agent's query intent (not the user's question).

### Output format:
<search> refined query </search>
<explanation> explanation for the refined query </explanation>
'''
                    user_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>"
                    input_prompt = user_prompt + "\n" + assistant_prefix
                    process_score['refined'] = self.get_modified_query(input_prompt)
        
        return data

    def get_modified_query(self, curr_prompt):
        response = self.generate_response(curr_prompt)
        search = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
        search = search.group(1).strip() if search else ''
        
        result = {
            "search": search
        }
        
        return result

    def save_to_json(self, evaluated_data, output_file):
        with open(output_file, 'w') as f:
            json.dump(evaluated_data, f, indent=4)
    
    def load_json(self, input_file):
        with open(input_file, 'r') as f:
            return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search results")
    parser.add_argument('--model_url', type=str, required=True, help="The URL of the model")
    parser.add_argument('--input_file', type=str, required=True, help="Input JSON file with data to evaluate")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to save the evaluation results")
    args = parser.parse_args()
    
    rewriter = Rewrite(args.model_url)
    input_data = rewriter.load_json(args.input_file)
    modified_data = rewriter.rewrite_search(input_data)
    rewriter.save_to_json(modified_data, args.output_file)
