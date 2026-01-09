import re
import json
import requests
import time
from typing import List
from functools import wraps

def retry(max: int=10, sleep: int=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[retry] try {i} times")
                    if i == max - 1:
                        # raise Exception("Retry {} failed after {} times".format(func.__name__, max))
                        return ""
                    elif sleep:
                        time.sleep(sleep)
        return wrapper
    return decorator

class ReCall():
    system_prompt = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. \
Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""

    def __init__(self, model_url, executor_url):
        self.model_url = model_url
        self.executor_url = executor_url
        
    def init_prompt(self, func_schemas, question):
        system_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>"
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix
    
    def init_prompt_prefix(self, func_schemas, question, prefix, env):
        system_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>"
        user_prompt = f"<|im_start|>user\n{question}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n{prefix}"
        curr_prompt = system_prompt + "\n" + user_prompt + "\n" + assistant_prefix
        
        return curr_prompt
    
    def init_prompt_hint(self, func_schemas, question, hint):
        system_prompt = f"<|im_start|>system\n{self.system_prompt_hint}<|im_end|>"
        user_prompt = f"<|im_start|>user\nQuestion: {question}\nReasoning path: {hint}<|im_end|>"
        assistant_prefix = f"<|im_start|>assistant\n<think>"
        return system_prompt + "\n" + user_prompt + "\n" + assistant_prefix

    def cat_assistant_response(self, curr_prompt, assistant_response):
        return curr_prompt + assistant_response
    
    def cat_tool_results(self, curr_prompt, tool_calls, results):
        tool_response_str = ""
        for tool_call, result in zip(tool_calls, results):
            tool_response_str += f"<result>{result}\n</result>\n"
        return curr_prompt + tool_response_str

    def format_tool_call(self, tool_call_str: str):
        """Convert JSON function call description to Python executable code string."""
        try:
            func_name = "wikipedia_search"
            arguments = {"query": tool_call_str, "top_n": 5}
            
            args_str = ', '.join(f"{k}={repr(v)}" for k, v in arguments.items())
            return f"{func_name}({args_str})"
        except Exception as e:
            return f"Parse tool call failed: {e}"
    
    def execute_tool_calls(self, env: str, tool_calls: List[str]) -> List[str]:
        def exe_tool_call_web(env, call):
            from ..inference.serper_test import fetch_search_results

            try:
                response = fetch_search_results(call)
                ret_str = f'result: \n{response}\n'
                return ret_str.strip()
            except Exception as e:
                return str(e)
    
        def exe_tool_call(env, call):
            url = self.executor_url + '/execute'

            call_str = self.format_tool_call(call)
            if call_str.startswith("error: parse tool call failed"):
                return call_str

            try:
                data = {
                    'env': env,
                    'call': call_str
                }
                response = requests.post(url, json=data, timeout=10)
                if response.status_code != 200:
                    return f"error: {response.status_code}"
                response = response.json()
                ret_str = ''
                if response['result']:
                    ret_str += f'result: \n{response["result"]}\n'
                if response['output']:
                    ret_str += f'output: \n{response["output"]}\n'
                if response['error']:
                    ret_str += f'error: \n{response["error"]}\n'
                return ret_str.strip()
            except requests.exceptions.Timeout:
                return "error: execution timed out"
            except Exception as e:
                return str(e)
        
        results = []
        for tool_call in tool_calls:
            result = exe_tool_call(env, tool_call)
            results.append(result)
        return results
    
    def validate_tool_calls(self, output_str):
        start_tags = re.findall(r'<search>', output_str)
        end_tags = re.findall(r'</search>', output_str)
        
        if len(start_tags) != len(end_tags):
            return False
            
        start_positions = [m.start() for m in re.finditer(r'<search>', output_str)]
        end_positions = [m.start() for m in re.finditer(r'</search>', output_str)]
        
        for start, end in zip(start_positions, end_positions):
            if start >= end:
                return False
                
        return True

    def extract_tool_calls(self, output_str):
        if not self.validate_tool_calls(output_str):
            return []

        try:
            pattern = r'<search>((?:(?!</search>).)*)</search>'
            matches = re.finditer(pattern, output_str, re.DOTALL)
            
            return [match.group(1).strip() for match in matches]
        except Exception as e:
            return []
        
    @retry(max=5, sleep=1)
    def run(self, env, func_schemas, question):
        curr_prompt = self.init_prompt(func_schemas, question)
        for _ in range(10):
            response = requests.post(
                f'{self.model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 8192,
                        "stop": ['</search>', '</answer>']
                    }
                }
            ).json()
            stop_reason = response['meta_info']['finish_reason'].get('type', '')
            stop_matched = response['meta_info']['finish_reason'].get('matched', '')
            if stop_reason == 'stop' and isinstance(stop_matched, str) and '</search>' in stop_matched:
                output_str = response['text'] + '</search>'
            elif stop_reason == 'stop' and isinstance(stop_matched, str) and '</answer>' in stop_matched:
                output_str = response['text'] + '</answer>'
            else:
                output_str = response['text']
            
            curr_prompt = self.cat_assistant_response(curr_prompt, output_str)

            tool_calls: List[str] = self.extract_tool_calls(output_str)
            if len(tool_calls) == 0:
                break

            results: List[str] = self.execute_tool_calls(env, tool_calls)
            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

        return curr_prompt
    
    @retry(max=5, sleep=1)
    def run_prefix(self, env, func_schemas, question, prefix):
        curr_prompt = self.init_prompt_prefix(func_schemas, question, prefix, env)
        for _ in range(10):
            response = requests.post(
                f'{self.model_url}/generate', 
                json={
                    "text": curr_prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 8192,
                        "stop": ['</search>']
                    }
                }
            ).json()
            stop_reason = response['meta_info']['finish_reason'].get('type', '')
            stop_matched = response['meta_info']['finish_reason'].get('matched', '')
            if stop_reason == 'stop' and isinstance(stop_matched, str) and '</search>' in stop_matched:
                output_str = response['text'] + '</search>'
            else:
                output_str = response['text']
            
            curr_prompt = self.cat_assistant_response(curr_prompt, output_str)

            tool_calls: List[str] = self.extract_tool_calls(output_str)
            if len(tool_calls) == 0:
                break

            results: List[str] = self.execute_tool_calls(env, tool_calls)
            curr_prompt = self.cat_tool_results(curr_prompt, tool_calls, results)

        return curr_prompt