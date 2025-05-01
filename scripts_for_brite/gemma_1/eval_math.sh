export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES="1"


PROMPT_TYPE=gemma-it
MODEL_NAME_OR_PATH=YYT-t/math_math-gemma-1.1-7b-it-m-iter-1_sample_7500_nsksm_ml256_mlr5e-5_ent0.0
DATA_NAME="math"
cd evaluation
bash sh/my_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME