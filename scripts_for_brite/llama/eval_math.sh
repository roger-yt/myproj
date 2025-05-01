export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES="1"


PROMPT_TYPE=llama-3-8b-it
MODEL_NAME_OR_PATH=YYT-t/math_math-Meta-Llama-3-8B-Instruct-m-iter-1_sample_7500_nsk_ml256_mlr5e-5_ent0.0
DATA_NAME="math"
bash evaluation/sh/my_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME