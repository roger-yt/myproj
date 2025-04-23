# source ~/.bashrc
export CUDA_VISIBLE_DEVICES="0"
#export WANDB_API_KEY=ee43df2d6680a9ce636f698eba4b5534c4336452
#export HF_TOKEN=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
#huggingface-cli login  --token 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
#git clone https://github.com/YYT-t/xyzo.git
#cd xyzo
#conda env create -f environment.yaml
#pip install flash-attn==2.6.3
#conda env create -f environment_sft.yaml

#model_name="Mistral-7B-Instruct-v0.2"
#company="mistralai"

company="mistralai"
model_name="Mistral-7B-Instruct-v0.3"
critic_model_name="${company}/${model_name}"
task_pre="math"
task_suf="gsm"
# conda init bash
num_samples=7500
sft_learning_rate="5e-5"
path="./${model_name}-${task_suf}_sample_${num_samples}_tp"
export HF_TOKEN=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

conda activate yy
mkdir $path

star_model_dir="${path}/star_model"
star_hub_id="${task_pre}_${task_suf}-${model_name}-star-sample_${num_samples}_tp"
dataset_path="ZhangShenao/star-${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_tp"
split="[:${num_samples}]"

python inference_star.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split || exit 1

python inference_xiaojun.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $star_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --learning_rate $sft_learning_rate|| exit 1
huggingface-cli upload "ZhangShenao/$star_hub_id" "${star_model_dir}_zq_raw" --token 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
