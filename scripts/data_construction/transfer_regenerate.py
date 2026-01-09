import json
import requests
import re
import argparse
from tqdm import tqdm

class Transfer:
    def transfer(self, data):
        results = []
        for entry in tqdm(data):
            for process_score in entry["process_scores"]:
                prefix = process_score['context']
                if "refined" in process_score:
                    original_content = prefix
                    search = process_score['refined']['search']

                    if search == "":
                        continue
                    
                    idx = original_content.rfind("<result>")
                    original_content = original_content[:idx]
                    
                    matches = list(re.finditer(r'(<search>)(.*?)(</search>)', original_content, flags=re.DOTALL))
                    if not matches:
                        continue
                    else:
                        last = matches[-1]
                        start, end = last.span(2)
                        original_content = original_content[:start] + search + original_content[end:]
                    
                    result = {
                        "question": entry['question'],
                        "golden_answers": [entry['golden_answer']],
                        "prefix": original_content
                    }
                    results.append(result)
                    
        return results

    def save_to_jsonl(self, evaluated_data, output_file):
        with open(output_file, 'w') as f:
            for entry in evaluated_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def load_json(self, input_file):
        with open(input_file, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search results")
    parser.add_argument('--input_file', type=str, required=True, help="Input JSON file with data to evaluate")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to save the evaluation results")
    args = parser.parse_args()
    
    transfer = Transfer()
    input_data = transfer.load_json(args.input_file)
    transfered_data = transfer.transfer(input_data)
    transfer.save_to_jsonl(transfered_data, args.output_file)