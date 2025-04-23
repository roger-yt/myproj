task_pre="math"
task_suf="math"
split="[:7500]"

export HF_TOKEN=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
CUDA_VISIBLE_DEVICES="0" python xiaojun_eval_noskip.py\
    --pmodel_name "meta-llama/Meta-Llama-3-8B-Instruct"\
    --qmodel_name "ZhangShenao/math_math-Meta-Llama-3-8B-Instruct-e-iter-1_sample_7500_tp_noskip" \
    --task_type "${task_pre}_${task_suf}${split}" \