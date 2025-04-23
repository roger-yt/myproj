task_pre="math"
task_suf="gsm"
split="[:7473]"

export HF_TOKEN=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
CUDA_VISIBLE_DEVICES="3" python xiaojun_eval_noskip.py\
    --pmodel_name "meta-llama/Meta-Llama-3-8B-Instruct"\
    --qmodel_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --task_type "${task_pre}_${task_suf}${split}" \