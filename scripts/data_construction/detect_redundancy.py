import json
import requests
import re
import argparse
from tqdm import tqdm

class Redundancy:
    def validate_template_format(self, text: str) -> tuple[bool, str]:
        if text.count('<think>') != text.count('</think>'):
            return False, "<think> </think> unpair"
        
        if text.count('<think>') == 0 or text.count('</think>') == 0:
            return False, "less <think> or </think> label"
        
        if text.count('<answer>') != 1 or text.count('</answer>') != 1:
            return False, "the appearance time for <answer> or </answer> is not 1"        
        
        current_pos = 0
        while True:
            search_pos = text.find('<search>', current_pos)
            if search_pos == -1:
                break
                
            result_pos = text.find('<result>', search_pos)
            search_end_pos = text.find('</search>', search_pos)
            result_end_pos = text.find('</result>', result_pos)
            
            if -1 in (result_pos, search_end_pos, result_end_pos):
                return False, "search/result is uncomplete"
                
            if not (search_pos < search_end_pos < result_pos < result_end_pos):
                return False, "search/result order error"
                
            current_pos = result_end_pos
        
        answer_start = text.find('<answer>')
        answer_end = text.find('</answer>')
        if answer_start > answer_end:
            return False, "<answer> must exist before </answer>"
        answer_content = text[answer_start:answer_end]
        if '\\boxed{' not in answer_content or '}' not in answer_content:
            return False, "answer needs \\boxed{}"
        
        return True, "correct format"
    def detect_redundancy(self, data):
        for entry in tqdm(data):
            text = entry['response']
            if self.validate_template_format(text)[0] is False:
                entry['format'] = 0
            else:
                entry['format'] = 1
            
            result_pattern = r'<result>(.*?)</result>'
            result_matches = re.findall(result_pattern, text, re.DOTALL)

            processed_results = []
            for result in result_matches:
                cleaned_result = result.replace("result: ", "").strip()
                document_fragments = cleaned_result.split('\n\n')
                processed_results.append(document_fragments)

            entry['redundancy'] = []
            cur = set()
            for i in range(len(processed_results)):
                entry['redundancy'].append(len(cur.intersection(set(processed_results[i]))))
                cur.update(set(processed_results[i]))
        return data

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
    
    redundancy = Redundancy()
    input_data = redundancy.load_json(args.input_file)
    output_data = redundancy.detect_redundancy(input_data)
    redundancy.save_to_json(output_data, args.output_file)