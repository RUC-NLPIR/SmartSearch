from openai import OpenAI
import logging
import argparse
import json
import jsonlines
import os

from utils import retry, execute, init_logger

LOG_NAME = "llm_judge"

llm_judge_prompt = """You will be given a question and its ground truth answer list where each item can be a ground truth answer. \
Provided a pred_answer, you need to judge if the pred_answer correctly answers the question based on the ground truth answer list. \
You should first give your rationale for the judgement, and then give your judgement result (i.e., correct or incorrect).

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

The output should in the following json format:
```json 
{{
    "rationale": "your rationale for the judgement, as a text",
    "judgement": "your judgement result, can only be 'correct' or 'incorrect'"
}}
```

Your output:"""

def read_lines(args):
    lines = json.load(open(os.path.join(args.input_dir, 'intermediate_data.json')))

    return lines

def cal_llm_judge_metric(args):
    num_correct = 0
    num_total = 0
    with jsonlines.open(args.output_path) as reader:
        for line in reader:
            if line['llm_judge']['judgement'] == 'correct':
                num_correct += 1
            num_total += 1
    
    score = num_correct / num_total
    
    with open(args.metric_path, 'w') as f:
        f.write(f"llm_judge_metric: {score}\n")

def run(args):
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )
    logger = logging.getLogger(LOG_NAME)
    
    lines = read_lines(args)
    logger.info(f"Read {len(lines)} lines from {args.input_dir}")

    @retry(max=5, sleep=1, logger=logger)
    def run_judge(line):
        prompt = llm_judge_prompt.format(
            question=line['question'], 
            gt_answer=str(line['golden_answers']), 
            pred_answer=line['output']['pred']
        )

        response = client.chat.completions.create(
            model=args.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        try:
            res = json.loads(response.choices[0].message.content)
            assert 'judgement' in res and 'rationale' in res
        except Exception as e:
            logger.info(f"Error parsing response: {e}")
            logger.info(f"Response: {response.choices[0].message.content}")
            raise e

        line['llm_judge'] = res
        return line

    execute(run_judge, lines, args.output_path, args.max_workers, logger)

    cal_llm_judge_metric(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", 
        type=str,
        help="The directory of the evaluation output of FlashRAG"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="The name of the judge LLM"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="base-url-of-judge-llm",
        help="The base URL of the judge LLM"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="api-key-of-judge-llm",
        help="The API key of the judge LLM"
    )
    parser.add_argument(
        "--log_path", 
        type=str, 
        default="llm_judge.log",
        help="The path of the log file"
    )
    parser.add_argument(
        "--max_workers", 
        type=int, 
        default=10,
        help="The maximum number of workers for the parallel execution"
    )
    args = parser.parse_args()
    args.output_path = os.path.join(args.input_dir, 'llm_judge.jsonl')
    args.metric_path = os.path.join(args.input_dir, 'llm_judge_metric.txt')
    logger = init_logger(args.log_path, LOG_NAME)
    for k, v in args.__dict__.items():
        logger.info(f"{k}: {v}")
    run(args)