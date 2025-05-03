export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES="0"


PROMPT_TYPE=mistral
MODEL_NAME_OR_PATH=YYT-t/math_gsm-Mistral-7B-Instruct-v0.2-rs-sample_7500_temp_1.0_gen_30_mlr5e-5
DATA_NAME="gsm8k"
cd evaluation
bash sh/my_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME