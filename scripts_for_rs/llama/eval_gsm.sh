export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES="1"


PROMPT_TYPE=llama-3-8b-it_box
MODEL_NAME_OR_PATH=YYT-t/math_gsm-Meta-Llama-3-8B-Instruct-rs-sample_7500_temp_1.0_gen_30_mlr5e-5
DATA_NAME="gsm8k"
cd evaluation
bash sh/my_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME