task_pre="math"
task_suf="math"

export HF_TOKEN=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
CUDA_VISIBLE_DEVICES="0" python xiaojun_eval.py\
    --pmodel_name "google/gemma-2-9b-it"\
    --qmodel_name "ZhangShenao/math_math-gemma-2-9b-it-e-iter-1_sample_7500_tp" \
    --task_type "${task_pre}_${task_suf}" \