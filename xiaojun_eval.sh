task_pre="math"
task_suf="gsm"

export HF_TOKEN=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
CUDA_VISIBLE_DEVICES="0" python xiaojun_eval.py\
    --pmodel_name "meta-llama/Meta-Llama-3-8B-Instruct"\
    --qmodel_name "ZhangShenao/math_gsm-Meta-Llama-3-8B-Instruct-e-iter-1_sample_7000_tp" \
    --task_type "${task_pre}_${task_suf}" \