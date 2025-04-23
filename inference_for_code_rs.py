from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import multiprocessing
import json, os, re
from datasets import Dataset
from task_configs import task_config_check, task_data_set, get_stoppings
import argparse
import torch
from rm_code import rm_code
from tqdm import tqdm
import pickle as pkl

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
        "--gen_nums",
        type=int
    )
    parser.add_argument(
        "--temp",
        type=float
    )
    parser.add_argument("--for_sft", action='store_true')
    return parser.parse_args()



NUM_GPUS = torch.cuda.device_count()
print("NUM_GPUS:", NUM_GPUS)
GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
GPUS = [int(gpu_id) for gpu_id in GPUS.split(",")]
print("GPUS:", GPUS)
split_list = lambda l, n: [l[i * len(l) // n: (i + 1) * len(l) // n] for i in range(n)]
def run_inference_one_gpu(gpu_id, question_list, answer_list, seq_id_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    #os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[gpu_id]
    return generate_rational(question_list, answer_list, seq_id_list)

def run_inference_multi_gpu(questions, answers, seq_ids):
    split_questions = split_list(questions, NUM_GPUS)
    split_answers = split_list(answers, NUM_GPUS)
    split_seq_ids = split_list(seq_ids, NUM_GPUS)
    inputs = [(i+GPUS[0], p, split_answers[i], split_seq_ids[i]) for i, p in enumerate(split_questions)]
    with multiprocessing.Pool(processes=NUM_GPUS) as pool:
        results = pool.starmap(run_inference_one_gpu, inputs)
    outputs = []
    for result in results:
        outputs.extend(result)
    return outputs


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_path
    print("model_name:", model_name)
    task_config = task_config_check(args.task_type)
    dataset_fraction = args.dataset_fraction
    task_config = task_config_check(args.task_type)
    train_path, dataset_ = task_data_set(args.task_type)
    print("generating rm")
    RM = rm_code()
    print("end rm")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stop_strings, stop_tokens = get_stoppings(model_name, tokenizer)
    print(stop_tokens)
    _, dataset_ = task_data_set(args.task_type+dataset_fraction)
    # dataset_ = load_dataset(train_path, split="train"+ dataset_fraction)
    dataset_ = dataset_.map(task_config.inference_tokenize(tokenizer), num_proc=16)
    # dataset_ = dataset_.select(range(10))
    questions = dataset_["template_question"]
    answers = dataset_["answer_text"]
    seq_ids = dataset_["seq_id"]
    sampling_params = SamplingParams(
        temperature=args.temp,
        top_p=1.0,
        top_k=-1,
        seed=42,
        max_tokens=512,
        min_tokens=50,
        n=1,
        # frequency_penalty=1.0,
        stop_token_ids=[tokenizer.eos_token_id],
        stop=task_config.stop_str_gen_z + stop_strings,
    )

    def generate_rational(few_shot_questions, answers, seq_ids):
        llm = LLM(model=model_name, tokenizer=model_name, dtype="bfloat16", seed=42, gpu_memory_utilization=0.9)
        rationals = []
        for i in range(args.gen_nums):
            rational = llm.generate(few_shot_questions, sampling_params, use_tqdm=True)
            rationals.append(rational)
        all_rationals = []
        for i, answer_text in enumerate(answers):
            item = {"answer": answer_text, "seq_id": seq_ids[i], "rationals":[]}
            for j in range(args.gen_nums):
                item["rationals"].append(rationals[j][i].outputs[0].text)
            all_rationals.append(item)
        # rational_answer = [modify_rational(rational[i].outputs[0].text, answer_text, seq_ids[i])  for i, answer_text in enumerate(answers)]
        return all_rationals
    def filter_rational_answer(all_rationals):
        def modify_rational(rational, answer_text):
            #rational = rational.outputs[0].text
            rational = rational.split("""[Implementation]""")[0]
            rational = rational.split("""[Reasoning]\n""")[-1]
            begin_prompt = """
            We will organize our response by two parts: [Reasoning] and [Implementation]."""
            return rational, begin_prompt + """\n[Reasoning]\n""" + rational + """\n[Implementation]\n""" + answer_text
        res = []
        for j, item in tqdm(enumerate(all_rationals)):
            seq_id = item["seq_id"]
            item_new = {"seq_id": seq_id, "answer": item["answer"]}
            for i in range(args.gen_nums):
                reasoning, rational_answer = modify_rational(item["rationals"][i], item["answer"])
                # if j == 1977 or j==1978:
                #     acc=0
                #     break
                acc = RM.get_binary_reward(seq_id, reasoning)
                # print("reasoning:", reasoning)
                # print("rational_answer:", rational_answer)
                # print("acc:", acc)
                if acc == 1:
                    break
            item_new["rational_answer"] = rational_answer
            item_new["acc"] = acc
            res.append(item_new)
        return res

    num_train_data = len(dataset_)
    gathered_data = []
    print("Start inference")
    data_nm = f"data_{args.gen_nums}_temp{args.temp}.pkl"
    if os.path.exists(data_nm):
        with open(data_nm, "rb") as f:
            all_rationals = pkl.load(f)
    else:
        all_rationals = run_inference_multi_gpu(questions, answers, seq_ids)
        with open(data_nm, "wb") as f:
            pkl.dump(all_rationals, f)
    print("Inference done")



    print("Start filtering")
    rational_answers = filter_rational_answer(all_rationals)
    print("Filter done")
    for i in range(num_train_data):
        if rational_answers[i]["acc"] == 1:
            tmp_data = {"question": dataset_[i][task_config.x_colname], "answer": dataset_[i][task_config.y_colname],
                        "rational_answer": rational_answers[i]["rational_answer"]}
            gathered_data.append(tmp_data)
    print("len(gathered_data):", len(gathered_data))
    for index in [0,1,-1]:
        print(f"gathered_data[-1]:\n{gathered_data[-1]}")
    dataset = Dataset.from_list(gathered_data)
    dataset.push_to_hub(args.dataset_path, private=False)
