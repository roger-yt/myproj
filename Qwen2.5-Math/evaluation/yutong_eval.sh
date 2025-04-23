# conda env create -f environment.yaml
# conda activate math_eval

# cd latex2sympy
# pip install -e .
# cd ..
# pip install -r requirements.txt 
# pip install vllm==0.5.1 --no-build-isolation
# pip install transformers==4.42.3
export HF_TOKEN=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
export CUDA_VISIBLE_DEVICES="1,3"
PROMPT_TYPE="gemma-1.1-7b-it"
MODEL_NAME_OR_PATH="ZhangShenao/math_math-gemma-1.1-7b-it-e-iter-1_sample_7500_nsk_ml512" #"ZhangShenao/math_math-gemma-2-9b-it-m-iter-1_sample_7500_tp"
DATA_NAME="math"
bash sh/my_eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $DATA_NAME