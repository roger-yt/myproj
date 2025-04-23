# source ~/.bashrc
export CUDA_VISIBLE_DEVICES="0,1,2,3"
git clone -b yutong_dev --single-branch https://github.com/YYT-t/xyzo.git

cd xyzo

pip install pyyaml
python pip_install_from_yaml.py
num_samples=4000
sft_learning_rate="5e-5"
company="deepseek-ai"
model_name="deepseek-coder-6.7b-instruct"
critic_model_name="${company}/${model_name}"
task_pre="code"
task_suf="opencoder_edu"
# conda init bash

path="./${model_name}-${task_suf}_sample_${num_samples}_tp"
export HF_TOKEN=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


mkdir $path

sft_model_dir="${path}/sft_model"
sft_hub_id="${task_pre}_${task_suf}-${model_name}-sft-sample_${num_samples}_tp"
dataset_path="ZhangShenao/pure-sft-${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_tp"
split="[:${num_samples}]"

python inference_for_code.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split --for_sft 
# accelerate launch  m_sft.py --deepspeed deepspeed_configs/deepspeed_3.json --num_train_epochs 3 --model_name $e_input_model  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --train_set_path $dataset_path --output_dir $m_model_dir --hub_model_id $m_hub_id || exit 1
# huggingface-cli upload "ZhangShenao/${m_hub_id}_msft" "${m_model_dir}/final_checkpoint" --token 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
python inference_xiaojun.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $sft_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --learning_rate $sft_learning_rate
huggingface-cli upload "ZhangShenao/$sft_hub_id" "${sft_model_dir}_zq_raw" --token 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
