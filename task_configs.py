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
            tokenized_q = tokenizer(q, truncation=True)
            answer_text = sample['answer'].split('####')[-1].strip()
            answer = f"The answer is {answer_text}."
            input_answer = [{"role": "user", "content": sample['question']}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            tokenized_a = tokenizer(answer, truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
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
            return sample
        return tokenize
    def sft_tokenize(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample['question']}]
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
            tokenized_q = tokenizer(q, truncation=True)
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
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
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
            return sample
        return tokenize
    
    def sft_tokenize(self, tokenizer):
        def tokenize(sample):
            input = [{"role": "user", "content": sample['question']}]
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
        self.id_colname = "seq_id"


class Config_Code_Opencoder_edu(Config_Code):
    def __init__(self):
        super(Config_Code_Opencoder_edu, self).__init__()
        self.x_colname = "instruction"
        self.y_colname = "output"
        with open(self.prompt_path, "r") as file:
            # Read the content of the file
            self.few_shot_cot_prompt = file.read()
    def tokenize_E(self, tokenizer):
        def tokenize(sample):
            input_str = self.few_shot_cot_prompt + sample[self.x_colname]
            input = [{"role": "user", "content": input_str}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            tokenized_q = tokenizer(q, truncation=True)
            answer_text = sample[self.y_colname].strip()
            answer = f"[Implementation]\n{answer_text}."
            input_answer = [{"role": "user", "content": input_str}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            tokenized_a = tokenizer(answer, truncation=True)
            sample["input_ids_q"] = tokenized_q["input_ids"]
            sample["attention_mask_q"] = tokenized_q["attention_mask"]
            sample["input_ids_a"] = tokenized_a["input_ids"]
            sample["attention_mask_a"] = tokenized_a["attention_mask"]
            return sample

        return tokenize

    def inference_tokenize(self, tokenizer):
        def tokenize(sample):
            input_str = self.few_shot_cot_prompt + sample[self.x_colname]
            input = [{"role": "user", "content": input_str}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            
            answer_text = sample[self.y_colname].strip()
            answer = f"[Implementation]\n{answer_text}."
            input_answer = [{"role": "user", "content": input_str}, {"role": "assistant", "content": answer}]
            answer = tokenizer.apply_chat_template(input_answer, tokenize=False).replace(q, '')
            sample["template_question"] = q
            sample["answer_text"] = answer
            return sample

        return tokenize
    def sft_tokenize(self, tokenizer):
        """

    the colnames of x, y, z are changed after inference.py
        """
        def tokenize(sample):
            input_str = sample['question']
            input = [{"role": "user", "content": input_str}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            return sample
        return tokenize
    def m_sft_tokenize(self, tokenizer):
        """
        def tokenize(sample):
            input_str = sample['question']
            input = [{"role": "user", "content": input_str}]
            q = tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
            sample["question"] = q
            sample["text"]= q+sample["rational_answer"]
            return sample
        """
        x_label="question"
        y_label="rational_answer"
        def tokenize(batch):
        # 将 prompt 和 response 拼接
            inputs = [p + r for p, r in zip(batch[x_label], batch[y_label])]
            # 使用分词器处理输入
            tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

            # 创建 labels
            labels = tokenized_inputs["input_ids"].clone()  # 复制 input_ids 作为 labels
            for i, (prompt, response) in enumerate(zip(batch[x_label], batch[y_label])):
                # 找到 prompt 的 token 长度
                prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
                prompt_length = len(prompt_tokens)
                # 将 prompt 部分的 labels 设置为 -100（忽略）
                labels[i, :prompt_length] = -100

            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        return tokenize



def task_config_check(task_name):
    if task_name.startswith("math_gsm"):
        return Config_Math_GSM()
    elif task_name.startswith("math_metamath"):
        return Config_Math_MetaMath()
    elif task_name.startswith("math_math"):
        return Config_Math_Math()
    elif "code_opencoder_edu" in task_name:
        return Config_Code_Opencoder_edu()
    else:
        raise(NotImplementedError)

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
        train_set_path = "lighteval/MATH"
        split = task_name.split("math_math")[-1]
        if split == "":
            data = load_dataset(train_set_path, "all")["train"]
        else:
            sp = [0 if i=='' else int(i) for i in split.strip("[]").split(":")]
            data = load_dataset(train_set_path, "all")["train"]
            data_length = len(data)
            assert sp[0] < data_length
            data = data.select(range(sp[0], min(sp[1], data_length)))
        return train_set_path, data
    elif "code_opencoder_edu" in task_name:

        train_set_path = "OpenCoder-LLM/opc-sft-stage2"
        split = task_name.split("code_opencoder_edu")[-1]
        if split == "":
            data = load_dataset(train_set_path, "educational_instruct")["train"]
        else:
            sp = [0 if i=='' else int(i) for i in split.strip("[]").split(":")]
            data = load_dataset(train_set_path, "educational_instruct")["train"]
            data_length = len(data)
            assert sp[0] < data_length
            data = data.select(range(sp[0], min(sp[1], data_length)))
        return train_set_path, data
        

def get_stoppings(model_name, tokenizer):
    stop_strings = []
    if "llama" in model_name.lower():
        stop_strings = ["<|eot_id|>"]
    elif "mistral" in model_name.lower():
        stop_strings = ["</s>"]
    elif "gemma" in model_name.lower():
        stop_strings = ["<eos>"]
    elif "deepseek-coder" in model_name.lower():
        stop_string = ["<|EOT|>", "[Implementation]", "```python"]
    else:
        print(model_name.lower())
        raise(ValueError)
    stop_tokens = []
    for stop_string in stop_strings:
        stop_tokens += tokenizer(stop_string)["input_ids"]
    return stop_strings, stop_tokens

