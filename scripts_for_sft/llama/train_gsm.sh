conda env create -f environment.yaml
eval "$(conda shell.bash hook)"
conda activate yy

export CUDA_VISIBLE_DEVICES="0,1,2,3"

export HF_TOKEN=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
upload_token=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

company="meta-llama"
model_name="Meta-Llama-3-8B-Instruct"
critic_model_name="${company}/${model_name}"
task_pre="math"
task_suf="gsm"
num_samples=7500
sft_learning_rate="5e-5"
path="./${model_name}-${task_suf}_sample_${num_samples}_tp"
mkdir $path

sft_model_dir="${path}/sft_model"
sft_hub_id="${task_pre}_${task_suf}-${model_name}-sft-sample_${num_samples}_tp_mlr_${sft_learning_rate}"
dataset_path="YYT-t/sft-${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_tp"
split="[:${num_samples}]"

python inference.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split --for_sft --upload_token $upload_token || exit 1
python sft.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $sft_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --learning_rate $sft_learning_rate|| exit 1
huggingface-cli upload "YYT-t/$sft_hub_id" "${sft_model_dir}_zq_raw" --token $upload_token
