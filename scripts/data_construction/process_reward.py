import json
import requests
import re
import argparse
from tqdm import tqdm

class SearchEvaluation:
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
                        # "stop": ['</answer>']
                    }
                }).json()
        except Exception as e:
            return ""
        return response.get('text', '')
        # return response.get('text', '') + "</answer>"

    def evaluate_search(self, data):
        assistant_prefix = f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        result = []
        
        for entry in tqdm(data):
            question = entry['question']
            golden_answer = entry['golden_answers'][0]
            response = entry['output']['final_response']
            pred = entry['output']['pred']
            outcome_score = entry['output']['metric_score']
            process_scores = []
            
            think_matches = list(re.finditer(r'<think>', response))
            if len(think_matches) < 4:
                continue
            fourth_think_pos = think_matches[3].start()
            response = response[fourth_think_pos:]
            
            result_matches = list(re.finditer(r'</result>', response))
            last_end = 0
            substrings = []
            for match in result_matches:
                substrings.append(response[last_end: match.end()])
            
            for prefix in substrings:
                user_prompt = f'''
You are a query-evaluation assistant. Your task is to assess the quality of a search agent's query of the current search round according to the user's question, the golden answer and the agent's search process up to the current search round.

If the agent's query intent of the current search round is necessary and actionable, and the corresponding query result includes the answer for the query, the score for query should be 1. Otherwise, the score for the query should be 0. The details of the assessment are in the Evaluation Guideline, please read it carefully.

### User's question
{question}

### Golden answer
{golden_answer}

### Agent's search process up to the current search round
{prefix}

### Evaluation Guideline
1. Identify the agent's query intent of the current search round accurately (**last round** in the agent's search process up to the current search round).
2. The query result **doesn't need to solve the user's question directly**; but it must include the information that address the agent's query intent completely (check/seek for information), related entities alone is not enough.
3. The intended entity and the one found in the query result **must be exactly the same (don't assume typos or other excuses)**, otherwise, the score should be 0.

### Output Format:
<answer> score for the query </answer>
<explanation> explanation for the score </explanation>'''
                user_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>"
                input_prompt = user_prompt + "\n" + assistant_prefix
                score = self.get_model_score(input_prompt)
                score['context'] = prefix
                score['prompt'] = input_prompt
                process_scores.append(score)
            
            result.append({
                "question": question,
                "golden_answer": golden_answer,
                "response": response,
                "pred": pred,
                "outcome_score": outcome_score,
                "process_scores": process_scores
            })
        
        return result

    def get_model_score(self, curr_prompt):
        response = self.generate_response(curr_prompt)
        answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        explanation = re.search(r'<explanation>(.*?)</explanation>', response, re.DOTALL)

        original_content = response
        extracted_answer = answer.group(1).strip() if answer else ''
        extracted_explanation = explanation.group(1).strip() if explanation else ''

        result = {
            "original_content": original_content,
            "score": extracted_answer,
            "explanation": extracted_explanation
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
    
    evaluator = SearchEvaluation(args.model_url)
    input_data = evaluator.load_json(args.input_file)
    evaluated_data = evaluator.evaluate_search(input_data)
    evaluator.save_to_json(evaluated_data, args.output_file)
