import re
import string
from typing import Union, List
from collections import Counter
import requests

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    
    return final_metric['f1']

def validate_template_format(text: str) -> tuple[bool, str]:
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

def generate_response(curr_prompt):
    try:
        response = requests.post(
            'http://0.0.0.0:9001/generate',
            json={
                "text": curr_prompt,
                "sampling_params": {
                    "temperature": 0.6,
                    "max_new_tokens": 2048,
                    "stop": ['</answer>']
                }
            }).json()
    except Exception as e:
        return ""
    return response.get('text', '') + '</answer>'

def evaluate_search(question, response, golden_answer):
    assistant_prefix = f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    result = []
    
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
        result.append(get_model_score(input_prompt))
    print('val process reward: ', result)
    return result

def get_model_score(curr_prompt):
    response = generate_response(curr_prompt)
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    extracted_answer = answer.group(1).strip() if answer else ''
    return extracted_answer

def detect_redundancy(response):
    result_pattern = r'<result>(.*?)</result>'
    result_matches = re.findall(result_pattern, response, re.DOTALL)

    processed_results = []
    for result in result_matches:
        cleaned_result = result.replace("result: ", "").strip()
        document_fragments = cleaned_result.split('\n\n')
        processed_results.append(document_fragments)

    result = []
    cur = set()
    for i in range(len(processed_results)):
        result.append(len(cur.intersection(set(processed_results[i]))))
        cur.update(set(processed_results[i]))
    return result

def compute_score_with_format(tokenizer, solution_str, ground_truth) -> tuple[float, str]:
    score = 0.0
    text = ""
    
    matches = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
    if matches:
        question = matches.group(1).strip()
    else:
        return 0, 'no user prompt found'
    
    think_matches = list(re.finditer(r'<think>', solution_str))
    if len(think_matches) >= 4: 
        fourth_think_pos = think_matches[3].start()
        response = solution_str[fourth_think_pos:]
    else:
        return 0, 'not response found'

    # outcome reward 1
    try:
        answer = remove_boxed(last_boxed_only_string(response))
    except Exception as e:
        answer = ""
        text += f'find box error: {e}\n'
    f1_score = get_f1_score(answer, ground_truth)
    if f1_score >= 0.8:
        text += f'correct answer, get f1 score: {f1_score}\n'
        score += 1.0
        
    # process reward 0.1 * times
    process_scores = evaluate_search(question, response, ground_truth)
    redundancy = detect_redundancy(response)
    corret = 0
    wrong = 0
    for i, entry in enumerate(process_scores):
        redundancy_value = redundancy[i] if len(process_scores) == len(redundancy) else 0
        if entry == '0' or redundancy_value > 1:
            wrong += 1
        else:
            corret += 1

    if f1_score >= 0.8:
        score -= 0.1 * wrong
        score = max(score, 0.7)
    else:
        score += 0.1 * corret
        score = min(score, 0.3)
        
    # format 0.1
    valid_template, reason = validate_template_format(response)
    if not valid_template:
        text += f'bad format: {reason}\n'
    else:
        score += 0.1
        
    if '</search<' in response:
        text += 'invalid search tag found\n'
        score = 0.0
        
    return score