from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse

def naive(args, config_dict):
    from flashrag.pipeline import SequentialPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    pipeline = SequentialPipeline(config)

    result = pipeline.run(test_data)

def zero_shot(args, config_dict):
    from flashrag.pipeline import SequentialPipeline
    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate

    templete = PromptTemplate(
        config=config,
        system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
        user_prompt="Question: {question}",
    )
    pipeline = SequentialPipeline(config, templete)
    result = pipeline.naive_run(test_data)

def iterretgen(args, config_dict):
    """
    Reference:
        Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative
                            Retrieval-Generation Synergy"
        in EMNLP Findings 2023.

        Zhangyin Feng et al. "Retrieval-Generation Synergy Augmented Large Language Models"
        in EMNLP Findings 2023.
    """
    iter_num = 3

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import IterativePipeline

    pipeline = IterativePipeline(config, iter_num=iter_num)
    result = pipeline.run(test_data)

def ircot(args, config_dict):
    """
    Reference:
        Harsh Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
        in ACL 2023
    """
    from flashrag.pipeline import IRCOTPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    print(config["generator_model_path"])
    pipeline = IRCOTPipeline(config, max_iter=5)

    result = pipeline.run(test_data)

def re_call(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    
    from flashrag.pipeline import ReCallPipeline
    pipeline = ReCallPipeline(config)
    result = pipeline.run(test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--config_path", type=str, default="./eval_config.yaml")
    parser.add_argument("--method_name", type=str, default="re-call")
    parser.add_argument("--data_dir", type=str, default="your-data-dir")
    parser.add_argument("--dataset_name", type=str, default="bamboogle")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default="your-save-dir")
    parser.add_argument("--save_note", type=str, default='your-save-note-for-identification')
    parser.add_argument("--sgl_remote_url", type=str, default="your-sgl-remote-url")
    parser.add_argument("--sandbox_url", type=str, default="your-sandbox-url")
    parser.add_argument("--remote_retriever_url", type=str, default="your-remote-retriever-url")
    parser.add_argument("--generator_model", type=str, default="your-local-model-path")

    func_dict = {
        "naive": naive,
        "zero-shot": zero_shot,
        "iterretgen": iterretgen,
        "ircot": ircot,
        "re-call": re_call,
    }

    args = parser.parse_args()

    config_dict = {
        "data_dir": args.data_dir,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "save_dir": args.save_dir,
        "save_note": args.save_note if args.save_note else args.method_name,
        "sgl_remote_url": args.sgl_remote_url,
        "remote_retriever_url": args.remote_retriever_url,
        "generator_model": args.generator_model,
        "sandbox_url": args.sandbox_url,
    }

    func = func_dict[args.method_name]
    func(args, config_dict)
