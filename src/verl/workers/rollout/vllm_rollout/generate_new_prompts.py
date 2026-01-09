from verl import DataProto
from transformers import AutoTokenizer
import re
import requests
import torch
import random
import torch.nn.functional as F

def generate_response(curr_prompt):
    try:
        response = requests.post(
            'http://0.0.0.0:9001/generate',
            json={
                "text": curr_prompt,
                "sampling_params": {
                    "temperature": 0.6,
                    "max_new_tokens": 2048
                }
            }).json()
    except Exception as e:
        return ""
    return response.get('text', '')

def evaluate_search(question, response, ground_truth):
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
{ground_truth}

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
        score = get_model_score(input_prompt)
        score['context'] = prefix
        result.append(score)
    
    return result

def rewrite_search(question, score, redundancy):
    assistant_prefix = f"<|im_start|>assistant\n<think>\n\n</think>\n\n"

    for i, entry in enumerate(score):
        context = entry['context']
        redundancy_value = redundancy[i] if len(score) == len(redundancy) else 0
        explanation = ""
        if entry['score'] == "0":
            explanation += entry['explanation']
        if redundancy_value > 1:
            explanation += ' The agent\'s query of the current search round is redundant, meaning that the query result duplicates information from previous search rounds.'
        if entry['score'] == "0" or redundancy_value > 1:
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
'''
            user_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>"
            input_prompt = user_prompt + "\n" + assistant_prefix
            entry['refined'] = get_modified_query(input_prompt)
    
    return score

def get_modified_query(curr_prompt):
    response = generate_response(curr_prompt)
    search = re.search(r'<search>(.*?)</search>', response, re.DOTALL)
    search = search.group(1).strip() if search else ''
    return search

def get_model_score(curr_prompt):
    response = generate_response(curr_prompt)
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    explanation = re.search(r'<explanation>(.*?)</explanation>', response, re.DOTALL)

    extracted_answer = answer.group(1).strip() if answer else ''
    extracted_explanation = explanation.group(1).strip() if explanation else ''

    result = {
        "score": extracted_answer,
        "explanation": extracted_explanation
    }

    return result

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

def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)


def count_consecutive(tensor):
    result = []
    current_value = tensor[0].item()
    count = 1
    for i in range(1, len(tensor)):
        if tensor[i].item() == current_value:
            count += 1
        else:
            result.append((count, current_value))
            current_value = tensor[i].item()
            count = 1
    result.append((count, current_value))
    return result

def generate_new_prompts(prompts: DataProto, tokenizer: AutoTokenizer, ground_truth):
    ori_input_ids = prompts.batch['prompts'][0]
    input_ids = prompts.batch['input_ids'][0]
    response = prompts.batch['responses'][0]
    loss_mask = prompts.batch['loss_mask'][0]
    max_length = ori_input_ids.shape[-1]
    if type(ground_truth) == list:
        ground_truth = ground_truth[0]
    
    trajectory = tokenizer.decode(input_ids)
    matches = re.search(r'<\|im_start\|>user\n(.*?)<\|im_end\|>', trajectory, re.DOTALL)
    if matches:
        question_text = matches.group(1).strip()
    else:
        return [],[]
    response_text = tokenizer.decode(response, skip_special_tokens=True)
    
    score = evaluate_search(question_text, response_text, ground_truth)
    redundancy = detect_redundancy(response_text)
    result = rewrite_search(question_text, score, redundancy)
    
    input_ids_list = []
    loss_mask_list = []
    loss_mask_all = []
    
    for i in range(1, loss_mask.size(0)):
        if loss_mask[i - 1] == 1 and loss_mask[i] == 0:
            loss_mask_all.append(loss_mask[:i])
    loss_mask_all = loss_mask_all[:-1]
    
    if len(result) != len(loss_mask_all):
        return [],[]
    
    for i, entry in enumerate(result):
        if 'refined' in entry and entry['refined'].strip() != "":
            original_content = entry['context']
            search = entry['refined']
            idx = original_content.rfind("<result>")
            original_content = original_content[:idx]
            matches = list(re.finditer(r'(<search>)(.*?)(</search>)', original_content, flags=re.DOTALL))
            if not matches:
                continue
            else:
                last = matches[-1]
                start, end = last.span(2)
                original_content = original_content[:start] + search + original_content[end:]
                
                input_ids_prefix = tokenizer.encode(original_content)
                
                if abs(len(input_ids_prefix) - loss_mask_all[i].size(0)) >= 50:
                    continue
                if len(input_ids_prefix) < loss_mask_all[i].size(0):
                    loss_mask_prefix = loss_mask_all[i][:len(input_ids_prefix)]
                else:
                    loss_mask_prefix = torch.cat((loss_mask_all[i], torch.ones(len(input_ids_prefix) - loss_mask_all[i].size(0), dtype=loss_mask_all[i].dtype, device=loss_mask_all[i].device)), dim=-1)
                
                non_pad_index = torch.nonzero(ori_input_ids != tokenizer.pad_token_id, as_tuple=False)[0][0]
                ori_input_ids = ori_input_ids[non_pad_index:]
                input_ids_prefix = torch.cat((ori_input_ids, torch.tensor(input_ids_prefix, device=ori_input_ids.device)), dim=-1)
                
                sequence_length = input_ids_prefix.shape[-1]
                if sequence_length < max_length:
                    input_ids_prefix = pad_sequence_to_length(input_ids_prefix,
                                                    max_seq_len=max_length,
                                                    pad_token_id=tokenizer.pad_token_id,
                                                    left_pad=True)
                    loss_mask_prefix = pad_sequence_to_length(loss_mask_prefix,
                                                    max_seq_len=max_length,
                                                    pad_token_id=0,
                                                    left_pad=True)
                elif sequence_length > max_length:
                    continue
                
                input_ids_list.append(input_ids_prefix)
                loss_mask_list.append(loss_mask_prefix)
                
    if len(input_ids_list) > 0 and len(loss_mask_list) > 0:
        input_ids_list = [input_ids_list[0]]
        loss_mask_list = [loss_mask_list[0]]

    return input_ids_list, loss_mask_list