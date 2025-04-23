task_pre="math"
task_suf="gsm"
split="[:7473]"

export HF_TOKEN=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
CUDA_VISIBLE_DEVICES="2" python xiaojun_eval_noskip.py\
    --pmodel_name "meta-llama/Meta-Llama-3-8B-Instruct"\
    --qmodel_name "ZhangShenao/math_gsm-Meta-Llama-3-8B-Instruct-e-iter-1_sample_7473_tp_noskip" \
    --task_type "${task_pre}_${task_suf}${split}" \