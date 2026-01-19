mkdir -p hotpotqa
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/hotpotqa/train.jsonl -O hotpotqa/train.jsonl
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/hotpotqa/dev.jsonl -O hotpotqa/dev.jsonl

mkdir -p 2wikimultihopqa
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/train.jsonl -O 2wikimultihopqa/train.jsonl
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/2wikimultihopqa/dev.jsonl -O 2wikimultihopqa/dev.jsonl

mkdir -p musique
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/train.jsonl -O musique/train.jsonl
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/musique/dev.jsonl -O musique/dev.jsonl

mkdir -p bamboogle
wget https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/resolve/main/bamboogle/test.jsonl -O bamboogle/test.jsonl



