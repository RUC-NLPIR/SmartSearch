import json
import requests
import re
import argparse
from tqdm import tqdm

class Transfer:
    def transfer(self, data):
        system_prompt = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> tags and \\boxed{{}} with latex format respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> The final answer is \\[ \\boxed{fianl exact answer here} \\]. In the last part of the response, the final exact answer is enclosed within \\boxed{} with latex format.
        """
        
        filtered_data = [
            {
                "instruction": system_prompt,
                "input": item["question"],
                "output": item["response"]
            }
            for item in data
            if item.get("format", 0) == 1 and
            all(num <= 1 for num in item.get("redundancy", [])) and
            all(process["score"] == "1" for process in item.get("process_scores", [])) and
            item.get("outcome_score", {}).get("f1") >= 0.8
        ]
                    
        return filtered_data
                
    def save_to_json(self, evaluated_data, output_file):
        with open(output_file, 'w') as f:
            json.dump(evaluated_data, f, indent=4)
    
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
    transfer.save_to_json(transfered_data, args.output_file)