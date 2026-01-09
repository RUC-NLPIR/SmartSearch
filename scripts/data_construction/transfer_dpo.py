import json
from collections import defaultdict
import argparse
from tqdm import tqdm

class Transfer:
    def count_correct_process(self, entry):
        correct_count = 0
        process_scores = entry.get("process_scores", [])
        redundancy = entry.get("redundancy", [])
        for i, ps in enumerate(process_scores):
            score = ps.get("score")
            redundancy_value = redundancy[i] if i < len(redundancy) else 0
            if score == "1" and redundancy_value <= 1:
                correct_count += 1
        return correct_count

    def count_wrong_process(self, entry):
        wrong_count = 0
        process_scores = entry.get("process_scores", [])
        redundancy = entry.get("redundancy", [])
        for i, ps in enumerate(process_scores):
            score = ps.get("score")
            redundancy_value = redundancy[i] if i < len(redundancy) else 0
            if score == "0" or redundancy_value > 1:
                wrong_count += 1
        return wrong_count
    
    def transfer(self, data1, data2):
        system_prompt = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. The reasoning process and answer are enclosed within <think> </think> tags and \\boxed{{}} with latex format respectively, and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process. </think> The final answer is \\[ \\boxed{fianl exact answer here} \\]. In the last part of the response, the final exact answer is enclosed within \\boxed{} with latex format.
        """
        result = []
        cnt1 = 0
        cnt2 = 0

        q2group = defaultdict(list)
        for entry in data2:
            q2group[entry["question"]].append(entry)

        for entry1 in data1:
            q = entry1["question"]
            f1_1 = entry1["outcome_score"]["f1"]
            group2 = q2group.get(q, [])

            if not group2:
                continue

            if f1_1 >= 0.8:
                wrong_entries = [e for e in group2 if e["outcome_score"]["f1"] < 0.8]
                if wrong_entries:
                    chosen_wrong = min(wrong_entries, key=self.count_correct_process)
                    result.append({
                        "instruction": system_prompt,
                        "input": q,
                        "chosen": entry1.get("response", ""),
                        "rejected": chosen_wrong.get("response", "")
                    })
                    cnt1 += 1

            elif f1_1 < 0.8:
                correct_entries = [e for e in group2 if e["outcome_score"]["f1"] >= 0.8]
                if correct_entries:
                    chosen_correct = min(correct_entries, key=self.count_wrong_process)
                    result.append({
                        "instruction": system_prompt,
                        "input": q,
                        "chosen": chosen_correct.get("response", ""),
                        "rejected": entry1.get("response", "")
                    })
                    cnt2 += 1
                    
        print(f"cnt1: {cnt1}, cnt2: {cnt2}")
        return result
                
    def save_to_json(self, evaluated_data, output_file):
        with open(output_file, 'w') as f:
            json.dump(evaluated_data, f, indent=4)
    
    def load_json(self, input_file):
        with open(input_file, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search results")
    parser.add_argument('--input_file1', type=str, required=True, help="Input JSON file with data to evaluate")
    parser.add_argument('--input_file2', type=str, required=True, help="Input JSON file with data to evaluate")
    parser.add_argument('--output_file', type=str, required=True, help="Output file to save the evaluation results")
    args = parser.parse_args()
    
    transfer = Transfer()
    input_data1 = transfer.load_json(args.input_file1)
    input_data2 = transfer.load_json(args.input_file2)
    transfered_data = transfer.transfer(input_data1, input_data2)
    transfer.save_to_json(transfered_data, args.output_file)