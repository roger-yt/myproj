# source ~/.bashrc
#export WANDB_API_KEY=ee43df2d6680a9ce636f698eba4b5534c4336452
#export HF_TOKEN=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
#huggingface-cli login  --token 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
#git clone https://github.com/YYT-t/xyzo.git
#cd xyzo
#conda env create -f environment.yaml
#pip install flash-attn==2.6.3
#conda env create -f environment_sft.yaml

iter_num=3

#model_name="Mistral-7B-Instruct-v0.2"
#company="mistralai"

company="mistralai"
model_name="Mistral-7B-Instruct-v0.3"
critic_model_name="${company}/${model_name}"
if [ "$model_name" == "gemma-1.1-7b-it" ]; then
    critic_model_name="google/gemma-1.1-2b-it"
fi
if [ "$model_name" == "gemma-2-9b-it" ]; then
    critic_model_name="google/gemma-2-2b-it"
fi
task_pre="math"
task_suf="gsm"
# conda init bash
num_samples=7500
max_length=512
model_max_length=512
export HF_TOKEN=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
visible_devices="0,1"

conda activate yy

path="./${model_name}-${task_suf}_sample_${num_samples}_nsk_ml${max_length}_self"
mkdir $path

m_model_dir="${path}/m-iter-$i"
m_hub_id="${task_pre}_${task_suf}-${model_name}-m-iter-${i}_sample_${num_samples}_nsk_ml${max_length}_self"
dataset_path="ZhangShenao/${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_nsk_ml${max_length}_self"


split="[:${num_samples}]"

CUDA_VISIBLE_DEVICES="${visible_devices}" python inference.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split  --max_length $max_length|| exit 1
# accelerate launch  m_sft.py --deepspeed deepspeed_configs/deepspeed_3.json --num_train_epochs 3 --model_name $e_input_model  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --train_set_path $dataset_path --output_dir $m_model_dir --hub_model_id $m_hub_id || exit 1
# huggingface-cli upload "ZhangShenao/${m_hub_id}_msft" "${m_model_dir}/final_checkpoint" --token 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
CUDA_VISIBLE_DEVICES="${visible_devices}" python inference_xiaojun.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $m_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --model_max_length $model_max_length|| exit 1
huggingface-cli upload "ZhangShenao/$m_hub_id" "${m_model_dir}_zq_raw" --token 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
