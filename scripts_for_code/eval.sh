conda activate evalplus || conda create --name evalplus python=3.11 && pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"
conda activate evalplus

path="./${model_name}-${task_suf}_sample_${num_samples}_tp"
export HF_TOKEN=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

# Or `pip install "evalplus[vllm]" --upgrade` for the latest stable release
#MODEL="ZhangShenao/code_opencoder_edu-deepseek-coder-1.3b-instruct-m-iter-1_sample_4000_tp"
#MODEL="ZhangShenao/code_opencoder_edu-deepseek-coder-1.3b-instruct-sft-sample_4000_tp"
MODEL="ZhangShenao/code_opencoder_edu-deepseek-coder-6.7b-instruct-m-iter-1_sample_4000_tp"
#MODEL="ZhangShenao/code_opencoder_edu-Llama-3.1-8B-Instruct-sft-sample_4000_tp"
#MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
#MODEL="ZhangShenao/code_opencoder_edu-deepseek-coder-6.7b-instruct-sft-sample_4000_tp"
DATA=humaneval
CUDA_VISIBLE_DEVICES=5 evalplus.evaluate --model $MODEL \
                  --dataset $DATA          \
                  --backend vllm                         \
                  --greedy


MODEL_SUFF="code_opencoder_edu-deepseek-coder-6.7b-instruct-m-iter-1_sample_4000_tp"
MODEL="ZhangShenao/"${MODEL_SUFF}
DATA=BigCodeBench
mkdir results/$DATA
mkdir results/$DATA/$MODEL_SUFF
CUDA_VISIBLE_DEVICES=6 python src/main.py \
    --model_name $MODEL \
    --save_path results/$DATA/$MODEL_SUFF \
    --num_gpus 1 \
    --batch_size 1 \
    --task $DATA \
    --model_type "Chat" \
    --prompt_type "Instruction" \
    --prompt_prefix "" \
    --prompt_suffix "" \
    --trust_remote_code

MODEL_SUFF="code_opencoder_edu-deepseek-coder-6.7b-instruct-m-iter-1_sample_4000_tp"
MODEL="ZhangShenao/"${MODEL_SUFF}
MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
DATA=BigCodeBench
mkdir results/$DATA
mkdir results/$DATA/$MODEL_SUFF
CUDA_VISIBLE_DEVICES=6,7,8 python src/main.py \
    --model_name $MODEL \
    --save_path results/$DATA/$MODEL_SUFF \
    --num_gpus 3 \
    --batch_size 10 \
    --task $DATA \
    --model_type "Chat" \
    --prompt_type "Instruction" \
    --prompt_prefix "" \
    --prompt_suffix "" \
    --trust_remote_code
#MODEL_SUFF="code_opencoder_edu-deepseek-coder-6.7b-instruct-m-iter-1_sample_4000_tp"
#MODEL="ZhangShenao/"${MODEL_SUFF}
MODEL="ZhangShenao/code_opencoder_edu-deepseek-coder-6.7b-instruct-e-iter-1_sample_4000_tp"
#MODEL="ZhangShenao/code_opencoder_edu-deepseek-coder-6.7b-instruct-sft-sample_4000_tp"
#MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
export E2B_API_KEY="e2b_0a231fa3b0a2b01690ab6c66a23b55c0979ce4ee"

CUDA_VISIBLE_DEVICES=8,9 bigcodebench.evaluate \
  --model ${MODEL} \
  --execution e2b \
  --split instruct \
  --subset full \
  --backend vllm

CUDA_VISIBLE_DEVICES=8 bigcodebench.evaluate \
  --samples generation-m1.jsonl \
  --execution e2b \
  --split instruct \
  --subset hard \
  --backend vllm