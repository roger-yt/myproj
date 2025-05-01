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



class Config_Math_GSM(Config_Math):
    def __init__(self):
        super(Config_Math_GSM, self).__init__()
        self.x_colname = "question"
        self.y_colname = "answer"

    def tokenize_E(self, tokenizer):
        def tokenize(sample):
            #tokenized_q = tokenizer(self.few_shot_cot_prompt + sample['query'], truncation=True)
            input = [{"role": "user", "content": sample['question']}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            tokenized_q = tokenizer.apply_chat_template(input, tokenize=True, add_generation_prompt=True, truncation=True)
            answer_text = sample['answer'].split('####')[-1].strip()
            answer = f"The answer is {answer_text}."
            input_answer = [{"role": "user", "content": sample['question']}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            tokenized_a = tokenizer(answer, truncation=True)
            sample["input_ids_q"] = tokenized_q
            sample["attention_mask_q"] = [1 for _ in range(len(tokenized_q))]
            sample["input_ids_a"] = tokenized_a["input_ids"]
            sample["attention_mask_a"] = tokenized_a["attention_mask"]
            return sample
        return tokenize

    def inference_tokenize(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample['question']}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            answer_text = sample['answer'].split('####')[-1].strip()
            answer = f"The answer is {answer_text}."
            input_answer = [{"role": "user", "content": sample['question']}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            sample["template_question"] = q
            sample["answer_text"] = answer
            sample["answer_num"] = answer_text
            hint_input = [{"role": "user", "content": sample['question'] + " "+ sample["answer"]+ " " +"Answer:"}]
            hint_q = tokenizer.apply_chat_template(hint_input, tokenize=False, add_generation_prompt=True)
            sample["hint_template_question"] = hint_q
            return sample
        return tokenize
    def inference_box_tokenize(self, tokenizer):
        def tokenize(sample):
            instruction = "Please reason step by step, and put your final answer within \\boxed{{}}."
            sample_x = instruction + " " + sample['question']
            input = [{"role": "user", "content": sample_x}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            answer_text = sample['answer'].split('####')[-1].strip()
            answer = f"The answer is {answer_text}."
            input_answer = [{"role": "user", "content": sample_x}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            sample["template_question"] = q
            sample["answer_text"] = answer
            sample["answer_num"] = answer_text
            hint_input = [{"role": "user", "content": sample_x + " "+ sample["answer"]+ " " +"Answer:"}]
            hint_q = tokenizer.apply_chat_template(hint_input, tokenize=False, add_generation_prompt=True)
            sample["hint_template_question"] = hint_q
            return sample
        return tokenize
    def sft_tokenize(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample['question']}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            return sample
        return tokenize
    def sft_box_tokenize(self, tokenizer):
        def tokenize(sample):
            instruction = "Please reason step by step, and put your final answer within \\boxed{{}}."
            sample_x = instruction + " " + sample['question']
            input = [{"role": "user", "content": sample_x}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            return sample
        return tokenize
    def m_sft_tokenize(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample["question"]}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            sample["text"]= q+sample["rational_answer"]
            return sample
        return tokenize

class Config_Math_Math(Config_Math):
    def __init__(self):
        super(Config_Math_Math, self).__init__()
        self.x_colname = "problem"
        self.y_colname = "solution"

    def extract_boxed_content(self, s):
        start_idx = s.rfind('\\boxed{')
        if start_idx == -1:
            # return None
            start_ids = s.rfind('\\boxed')
            idx = start_ids + len('\\boxed')
            content = ""
            while s[idx] == " ":
                idx += 1
            while idx < len(s) and s[idx]!= "$":
                content += s[idx]
                idx += 1
            print("hi")
            print("s=", s)
            print("content=", content)
            return content
        idx = start_idx + len('\\boxed{')
        depth = 1
        content = ''
        while idx < len(s) and depth > 0:
            char = s[idx]
            if char == '{':
                depth += 1
                content += char
            elif char == '}':
                depth -= 1
                if depth > 0:
                    content += char
            else:
                content += char
            idx += 1
        if depth == 0:
            return content
        else:
            return None
    def tokenize_E(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample['problem']}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            tokenized_q = tokenizer.apply_chat_template(input, tokenize=True, add_generation_prompt=True, truncation=True)
            # print("begin")
            answer_text = self.extract_boxed_content(sample['solution'])
            # print("end")
            # print("solution=", sample['solution'])
            # print("answer_text=", answer_text)
            try:
                answer = f"The answer is {answer_text}."
            except Exception as e:
                print("error=", e)
                print("sample=", sample)
                print("answer_text=", answer_text)
            input_answer = [{"role": "user", "content": sample['problem']}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            tokenized_a = tokenizer(answer, truncation=True)
            sample["input_ids_q"] = tokenized_q
            sample["attention_mask_q"] = [1 for _ in range(len(tokenized_q))]
            sample["input_ids_a"] = tokenized_a["input_ids"]
            sample["attention_mask_a"] = tokenized_a["attention_mask"]
            return sample
        return tokenize

    def inference_tokenize(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample['problem']}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            answer_text = self.extract_boxed_content(sample['solution'])
            answer = f"The answer is {answer_text}."
            input_answer = [{"role": "user", "content": sample['problem']}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            sample["template_question"] = q
            sample["answer_text"] = answer
            sample["answer_num"] = answer_text
            return sample
        return tokenize

    def inference_box_tokenize(self, tokenizer):
        def tokenize(sample):
            instruction = "Please reason step by step, and put your final answer within \\boxed{{}}."
            sample_x = instruction + " " + sample['problem']
            input = [{"role": "user", "content": sample_x}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            answer_text = self.extract_boxed_content(sample['solution'])
            answer = f"The answer is {answer_text}."
            input_answer = [{"role": "user", "content": sample_x}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            sample["template_question"] = q
            sample["answer_text"] = answer
            sample["answer_num"] = answer_text
            return sample
        return tokenize
    
    def inference_ans_tokenize(self, tokenizer):
        def tokenize(sample):
            instruction = "Please reason step by step, and put your final answer after \"The answer is \"."
            sample_x = instruction + " " + sample['problem']
            input = [{"role": "user", "content": sample_x}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            answer_text = self.extract_boxed_content(sample['solution'])
            answer = f"The answer is {answer_text}."
            input_answer = [{"role": "user", "content": sample_x}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            sample["template_question"] = q
            sample["answer_text"] = answer
            sample["answer_num"] = answer_text
            return sample
        return tokenize
    
    def sft_tokenize(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample['question']}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            return sample
        return tokenize
    
    def sft_box_tokenize(self, tokenizer):
        def tokenize(sample):
            instruction = "Please reason step by step, and put your final answer within \\boxed{{}}."
            sample_x = instruction + " " + sample['question']
            input = [{"role": "user", "content": sample_x}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            return sample
        return tokenize
    
    def sft_ans_tokenize(self, tokenizer):
        def tokenize(sample):
            instruction =  "Please reason step by step, and put your final answer after \"The answer is \"."
            sample_x = instruction + " " + sample['question']
            input = [{"role": "user", "content": sample_x}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            return sample
        return tokenize
    def sft_myans_tokenize(self, tokenizer):
        def tokenize(sample):
            instruction =  "Please show your reasoning step by step. In the end, clearly state your final answer."
            sample_x = instruction + " " + sample['question']
            input = [{"role": "user", "content": sample_x}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            return sample
        return tokenize

    def m_sft_tokenize(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample['question']}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            sample["text"]= q+sample["rational_answer"]
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

    def tokenize_E(self, tokenizer):
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
            sample["text"] = '### Instruction\n' + sample["question"] + '### Response\n[Reasoning]\n' + sample[
                "rational_answer"]  ##+ '[Implementation]\n' + sample["answer"]
            return sample

        return cot_prefix

    def baseline_sft_cot_prefix(self):
        def cot_prefix(sample):
            sample["text"] = '### Instruction\n' + sample[self.x_colname] + '### Response\n[Implementation]\n' + sample[
                self.y_colname]
            return sample

        return cot_prefix


def task_config_check(task_name):
    if task_name.startswith("math_gsm"):
        return Config_Math_GSM()
    elif task_name.startswith("math_math"):
        return Config_Math_Math()
    elif task_name == "code_opencoder_edu":
        return Config_Code_Opencoder_edu()


def task_data_set(task_name):
    if "math_gsm" in task_name:
        train_set_path = "openai/gsm8k"
        split = task_name.split("math_gsm")[-1]
        if split == "":
            data = load_dataset(train_set_path, "main")["train"]
        else:
            sp = [0 if i=='' else int(i) for i in split.strip("[]").split(":")]
            data = load_dataset(train_set_path, "main")["train"]
            data_length = len(data)
            assert sp[0] < data_length
            data = data.select(range(sp[0], min(sp[1], data_length)))
        return train_set_path, data
    if "math_math" in task_name:
        train_set_path = "YYT-t/MATH"
        split = task_name.split("math_math")[-1]
        if split == "":
            data = load_dataset(train_set_path)["train"]
            print("data_length=", len(data))
        else:
            sp = [0 if i=='' else int(i) for i in split.strip("[]").split(":")]
            data = load_dataset(train_set_path)["train"]
            data_length = len(data)
            assert sp[0] < data_length
            data = data.select(range(sp[0], min(sp[1], data_length)))
        return train_set_path, data
    elif task_name == "code_opencoder_edu":
        train_set_path = "OpenCoder-LLM/opc-sft-stage2"
        return train_set_path, load_dataset(train_set_path, "educational_instruct")["train"]

def get_stoppings(model_name, tokenizer):
    stop_strings = []
    if "llama" in model_name.lower():
        stop_strings = ["<|eot_id|>"]
    elif "mistral" in model_name.lower():
        stop_strings = ["</s>"]
    elif "gemma-1.1" in model_name.lower():
        stop_strings = ["<eos>"]
    elif "gemma-2" in model_name.lower():
        stop_strings = ["<end_of_turn>"]
    elif "qwen" in model_name.lower():
        stop_strings = ["<|im_end|>"]
    stop_tokens = []
    for stop_string in stop_strings:
        stop_tokens += tokenizer(stop_string)["input_ids"]
    return stop_strings, stop_tokens

def get_stoppings_new(model_name, tokenizer):
    stop_strings = []
    if "llama" in model_name.lower():
        stop_strings = ["<|eot_id|>"]
    elif "mistral" in model_name.lower():
        stop_strings = ["</s>"]
    elif "gemma-1.1" in model_name.lower():
        stop_strings = ["<eos>", "<bos>"]
    elif "gemma-2" in model_name.lower():
        stop_strings = ["<end_of_turn>"]
    elif "qwen" in model_name.lower():
        stop_strings = ["<|im_end|>"]
    stop_tokens = []
    for stop_string in stop_strings:
        stop_tokens += tokenizer(stop_string)["input_ids"]
    return stop_strings, stop_tokens
