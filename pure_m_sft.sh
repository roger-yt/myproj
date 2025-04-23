# source ~/.bashrc
export CUDA_VISIBLE_DEVICES="2,3"
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

company="meta-llama"
model_name="Meta-Llama-3-8B-Instruct"
critic_model_name="${company}/${model_name}"
task_pre="math"
task_suf="gsm"
# conda init bash
num_samples=7473
path="./${model_name}-${task_suf}_sample_${num_samples}_tp"
export HF_TOKEN=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
train_epochs=7

conda activate yy
mkdir $path

sft_model_dir="${path}/msft_model"
sft_hub_id="${task_pre}_${task_suf}-${model_name}-msft-sample_${num_samples}_tp"
dataset_path="ZhangShenao/msft-${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_tp"
split="[:${num_samples}]"

python inference.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split --for_sft || exit 1
accelerate launch  --main_process_port $PORT1 m_sft.py --deepspeed deepspeed_configs/deepspeed_3.json --num_train_epochs $train_epochs --model_name "${company}/${model_name}"  --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --train_set_path $dataset_path --output_dir $sft_model_dir --hub_model_id $sft_hub_id || exit 1
huggingface-cli upload "ZhangShenao/${sft_hub_id}" "${sft_model_dir}/final_checkpoint" --token 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
# python inference_xiaojun.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $sft_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" || exit 1
# huggingface-cli upload "ZhangShenao/$sft_hub_id" "${sft_model_dir}_zq_raw" --token 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
