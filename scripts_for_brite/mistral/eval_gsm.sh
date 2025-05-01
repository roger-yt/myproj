export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES="1"


PROMPT_TYPE=mistral
MODEL_NAME_OR_PATH=YYT-t/math_gsm-Mistral-7B-Instruct-v0.2-m-iter-1_sample_7500_nsksm_ml256_mlr5e-5_ent0.0
DATA_NAME="gsm8k"
cd evaluation
bash sh/my_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME