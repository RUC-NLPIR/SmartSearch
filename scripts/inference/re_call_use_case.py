import pandas as pd
from re_call import ReCall

model_url = "<the-hosted-re-call-model-url>"
sandbox_url = "<the-sandbox-url>"
data_path = "<the-parquet-data-path>"

# load some data
test_lines = []
test_data = pd.read_parquet(data_path)
for row in test_data.iterrows():
    curr_line = {}
    curr_line['question'] = row[1]['question']
    curr_line['answer'] = row[1]['reward_model']['ground_truth']
    curr_line['env'] = row[1]['extra_info']['env']
    curr_line['func_schemas'] = row[1]['extra_info']['func_schemas']
    test_lines.append(curr_line)

# initialize the re_call model
re_call = ReCall(model_url, sandbox_url)

# run the re_call model
response = re_call.run(test_lines[1]['env'], test_lines[1]['func_schemas'], test_lines[1]['question'])
print(response)