from datasets import load_dataset

class Config_Math():
    def __init__(self):
        self.stop_str_gen_z = ["Question:"]
        self.prompt_path = "prompts/math_prompt.txt"
        self.x_colname = "query"
        self.y_colname = "response"
    def tokenize_E(self):
        pass
    def M_sft_cot_prefix(self):
        def cot_prefix(sample):
            sample["text"] = 'Question: ' + sample["question"] + ' Answer: ' + sample["rational_answer"]
        #    sample["prompt"] = few_shot_cot_prompt + sample["question"]
        #    sample["completion"] = sample["rational_answer"]
            return sample
        return cot_prefix
    def baseline_sft_cot_prefix(self):
        def cot_prefix(sample):
            sample["text"] = 'Question: ' + sample["query"] + ' Answer: ' + sample["response"]
            return sample
        return cot_prefix
    def inference_tokenize(self):
        def tokenize(sample):
            answer_text = sample['response'].split("The answer is")[-1].strip()
            sample["few_shot_cot_question"] = self.few_shot_cot_prompt + sample['query']
            sample["answer_text"] = f"The answer is {answer_text}."
            return sample
        return tokenize
    
class Config_Math_GSM(Config_Math):
    def __init__(self):
        super(Config_Math_GSM, self).__init__()
        with open(self.prompt_path, "r") as file:
            # Read the content of the file
            self.few_shot_cot_prompt = file.read()
    def tokenize_E(self,tokenizer):
        def tokenize(sample):
            tokenized_q = tokenizer(self.few_shot_cot_prompt + sample['question'], truncation=True)
            answer_text = sample['answer'].split('####')[-1].strip()
            answer = f"The answer is {answer_text}."
            tokenized_a = tokenizer(answer, truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"]
            sample["attention_mask_a"] = tokenized_a["attention_mask"]
            return sample
        return tokenize



class Config_Math_MetaMath(Config_Math):
    def __init__(self):
        super(Config_Math_MetaMath, self).__init__()
        with open(self.prompt_path, "r") as file:
            # Read the content of the file
            self.few_shot_cot_prompt = file.read()


    def tokenize_E(self,tokenizer):
        def tokenize(sample):
            tokenized_q = tokenizer(self.few_shot_cot_prompt + sample['query'], truncation=True)
            answer_text = sample['response'].split('The answer is: ')[-1].strip()
            answer = f"The answer is {answer_text}."
            tokenized_a = tokenizer(answer, truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"]
            sample["attention_mask_a"] = tokenized_a["attention_mask"]
            return sample
        return tokenize

class Config_Code(Config_Math):
    def __init__(self):
        super(Config_Code, self).__init__()
        self.stop_str_gen_z = ["""[Implementation]"""]
        self.prompt_path = "prompts/code_prompt.txt"
        with open(self.prompt_path, "r") as file:
            # Read the content of the file
            self.few_shot_cot_prompt = file.read()
        self.x_colname = "instruction"
        self.y_colname = "output"

class Config_Code_Opencoder_edu(Config_Code):
    def __init__(self):
        super(Config_Code_Opencoder_edu, self).__init__()
        with open(self.prompt_path, "r") as file:
            # Read the content of the file
            self.few_shot_cot_prompt = file.read()
    def tokenize_E(self,tokenizer):
        def tokenize(sample):
            tokenized_q = tokenizer(self.few_shot_cot_prompt + sample[self.x_colname], truncation=True)
            answer_text = sample[self.y_colname].strip()
            answer = f"[Implementation]\n{answer_text}."
            tokenized_a = tokenizer(answer, truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"]
            sample["attention_mask_a"] = tokenized_a["attention_mask"]
            return sample
        return tokenize

    def inference_tokenize(self):
        def tokenize(sample):
            answer_text = sample[self.y_colname].strip()
            sample["few_shot_cot_question"] = self.few_shot_cot_prompt + sample[self.x_colname]
            sample["answer_text"] = f"[Implementation]\n{answer_text}."
            return sample
        return tokenize
    def M_sft_cot_prefix(self):
        """
        recall that we save the inference dataset as follows:
        tmp_data = {"question": dataset_[i][task_config.x_colname], "answer": dataset_[i][task_config.y_colname],
            "rational_answer": rational_answer[i]}
        Here:  rational_answer == cat(z,y)
        """
        def cot_prefix(sample):
            sample["text"] = '### Instruction\n' + sample["question"] + '### Response\n[Reasoning]\n' + sample["rational_answer"] ##+ '[Implementation]\n' + sample["answer"]
            return sample
        return cot_prefix
    def baseline_sft_cot_prefix(self):
        def cot_prefix(sample):
            sample["text"] = '### Instruction\n' + sample[self.x_colname] + '### Response\n[Implementation]\n' + sample[self.y_colname]
            return sample
        return cot_prefix
def task_config_check(task_name):
    if task_name == "math_gsm":
        return Config_Math_GSM()
    elif  task_name == "math_metamath":
        return Config_Math_MetaMath() 
    elif task_name == "code_opencoder_edu":
        return Config_Code_Opencoder_edu()
    else:
        raise(NotImplementedError)
    
def task_data_set(task_name):
    if task_name == "math_gsm":
        train_set_path = "openai/gsm8k"
        return train_set_path, load_dataset(train_set_path, 'main')["train"]
    elif  task_name == "math_metamath":
        train_set_path = "meta-math/MetaMathQA"
        return train_set_path, load_dataset(train_set_path)["train"]
    elif task_name == "code_opencoder_edu":
        train_set_path = "OpenCoder-LLM/opc-sft-stage2"
        return train_set_path,  load_dataset(train_set_path, "educational_instruct")["train"]
    else:
        raise(NotImplementedError)