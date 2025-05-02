export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES="1"


PROMPT_TYPE=gemma-it
MODEL_NAME_OR_PATH=YYT-t/math_math-gemma-1.1-7b-it-rs-sample_7500_temp_1.0_gen_30_mlr5e-5
DATA_NAME="math"
cd evaluation
bash sh/my_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME