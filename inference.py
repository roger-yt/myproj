from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import multiprocessing
import json, os, re
from datasets import Dataset
from task_configs import task_config_check, task_data_set, get_stoppings
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-GPU inference on questions and answers.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path or identifier of the model to be used for inference.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="The path of output dataset.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="prompts/math_prompt.txt",
        help= "path to get the cot prompt",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="math_metamath",
        help= "math or code",
    )
    parser.add_argument(
        "--dataset_fraction",
        type=str
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512
    )
    parser.add_argument(
        "--upload_token",
        type=str
    )
    parser.add_argument("--for_sft", action='store_true')
    return parser.parse_args()



NUM_GPUS = torch.cuda.device_count()
print("NUM_GPUS:", NUM_GPUS)
GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
GPUS = [int(gpu_id) for gpu_id in GPUS.split(",")]
print("GPUS:", GPUS)
split_list = lambda l, n: [l[i * len(l) // n: (i + 1) * len(l) // n] for i in range(n)]
def run_inference_one_gpu(gpu_id, question_list, answer_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    #os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[gpu_id]
    return generate_rational(question_list, answer_list)

def run_inference_multi_gpu(questions, answers):
    split_questions = split_list(questions, NUM_GPUS)
    split_answers = split_list(answers, NUM_GPUS)
    inputs = [(i+GPUS[0], p, split_answers[i]) for i, p in enumerate(split_questions)]
    with multiprocessing.Pool(processes=NUM_GPUS) as pool:
        results = pool.starmap(run_inference_one_gpu, inputs)
    outputs = []
    for result in results:
        outputs.extend(result)
    return outputs


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_path
    # print("model_name:", model_name)
    task_config = task_config_check(args.task_type)
    dataset_fraction = args.dataset_fraction
    task_config = task_config_check(args.task_type)
    train_path, dataset_ = task_data_set(args.task_type)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stop_strings, stop_tokens = get_stoppings(model_name, tokenizer)
    print(stop_tokens)
    _, dataset_ = task_data_set(args.task_type+dataset_fraction)
    # dataset_ = load_dataset(train_path, split="train"+ dataset_fraction)
    dataset_ = dataset_.map(task_config.inference_tokenize(tokenizer), num_proc=16)
    # dataset_ = dataset_.select(range(10))
    questions = dataset_["template_question"]
    answers = dataset_["answer_text"]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        seed=42,
        max_tokens=args.max_length,
        min_tokens=50,
        n=1,
        # frequency_penalty=1.0,
        stop_token_ids=[tokenizer.eos_token_id],
        stop=task_config.stop_str_gen_z + stop_strings,
    )

    def generate_rational(few_shot_questions, answers):
        llm = LLM(model=model_name, tokenizer=model_name, dtype="bfloat16", seed=42, gpu_memory_utilization=0.9)
        rational = llm.generate(few_shot_questions, sampling_params, use_tqdm=True)
        print("rational:", rational)
        rational_answer = [rational[i].outputs[0].text + answer_text for i, answer_text in enumerate(answers)]
        return rational_answer

    num_train_data = len(dataset_)
    gathered_data = []
    if not args.for_sft:
        print("Start inference")
        rational_answer = run_inference_multi_gpu(questions, answers)
        print("Inference done")
        for i in range(num_train_data):
            tmp_data = {"question": dataset_[i][task_config.x_colname], "answer": dataset_[i][task_config.y_colname],
                        "rational_answer": rational_answer[i]}
            gathered_data.append(tmp_data)
    else:
        for i in range(num_train_data):
            tmp_data = {"question": dataset_[i][task_config.x_colname],
                        "rational_answer": dataset_[i][task_config.y_colname]}
            gathered_data.append(tmp_data)
    print("gathered_data:", gathered_data)
    dataset = Dataset.from_list(gathered_data)
    dataset.push_to_hub(args.dataset_path, private=False, token=args.upload_token)
