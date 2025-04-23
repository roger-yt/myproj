task_pre="math"
task_suf="math"

export HF_TOKEN=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
CUDA_VISIBLE_DEVICES="1" python xiaojun_eval.py\
    --pmodel_name "google/gemma-2-9b-it"\
    --qmodel_name "google/gemma-2-9b-it" \
    --task_type "${task_pre}_${task_suf}" \