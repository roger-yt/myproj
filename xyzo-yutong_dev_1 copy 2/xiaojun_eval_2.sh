task_pre="math"
task_suf="gsm"

export HF_TOKEN=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
CUDA_VISIBLE_DEVICES="2" python xiaojun_eval.py\
    --pmodel_name "google/gemma-2-9b-it"\
    --qmodel_name "ZhangShenao/math_gsm-gemma-2-9b-it-e-iter-1_sample_7473_tp" \
    --task_type "${task_pre}_${task_suf}" \