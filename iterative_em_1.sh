# source ~/.bashrc
# git clone -b math --single-branch https://github.com/roger-yt/myproj.git
# cd myproj
conda env create -f environment.yaml
# conda init bash
eval "$(conda shell.bash hook)"
conda activate yy


iter_num=1

export HF_TOKEN=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
upload_token=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

company=meta-llama
model_name=Meta-Llama-3-8B-Instruct
critic_model_name="${company}/${model_name}"
if [ "$model_name" == "gemma-1.1-7b-it" ]; then
    critic_model_name="google/gemma-1.1-2b-it"
fi
if [ "$model_name" == "gemma-2-9b-it" ]; then
    critic_model_name="google/gemma-2-2b-it"
fi
task_pre="math"
task_suf="math"
num_samples=7500
max_length=512
model_max_length=512
visible_devices="0,1,2,3"
ent_coeff=0.0

for i in $(seq 1 $iter_num); do
    if [ "$i" -eq 1 ]; then
        split="[:${num_samples}]"
        m_lr="5e-5"
    elif [ "$i" -eq 2 ]; then
        split="[$((num_samples)):$((num_samples*2))]"
        m_lr="5e-6"
    else
        split="[$((num_samples*2)):$((num_samples*3))]"
        m_lr="1e-6"
    fi
    
    path="./${model_name}-${task_suf}_sample_${num_samples}_ml${max_length}"
    mkdir $path
    e_input_model="${path}/m-iter-$((i-1))_zq_raw"
    e_model_dir="${path}/e-iter-$i"
    m_model_dir="${path}/m-iter-$i"
    e_hub_id="${task_pre}_${task_suf}-${model_name}-e-iter-${i}_sample_${num_samples}_ml${max_length}_mlr${m_lr}_ent${ent_coeff}"
    m_hub_id="${task_pre}_${task_suf}-${model_name}-m-iter-${i}_sample_${num_samples}_ml${max_length}_mlr${m_lr}_ent${ent_coeff}"
    dataset_path="YYT-t/${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_ml${max_length}_mlr${m_lr}_ent${ent_coeff}"

    if [ "$i" -eq 1 ]; then
        e_input_model="${company}/${model_name}"
    else
        echo "iteration $i"
    fi

    CUDA_VISIBLE_DEVICES="${visible_devices}" python xiaojun_E_step_ent_PPO.py --model_name $e_input_model --critic_model_name $critic_model_name --task_type "${task_pre}_${task_suf}${split}" --model_path $e_model_dir  --max_length $max_length --model_max_length $model_max_length --ent_coeff $ent_coeff
    

    huggingface-cli upload "YYT-t/$e_hub_id" "${e_model_dir}/final_checkpoint" --token $upload_token
    huggingface-cli upload "YYT-t/$e_hub_id" "${e_model_dir}/tb_log" --token $upload_token

    
    CUDA_VISIBLE_DEVICES="${visible_devices}" python inference.py --model_path "${e_model_dir}/final_checkpoint" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split  --max_length $max_length --upload_token $upload_token
    # accelerate launch  m_sft.py --deepspeed deepspeed_configs/deepspeed_3.json --num_train_epochs 3 --model_name $e_input_model  --per_device_train_batch_size 8 --gradient_accumulation_steps 4 --train_set_path $dataset_path --output_dir $m_model_dir --hub_model_id $m_hub_id || exit 1
    # huggingface-cli upload "YYT-t/${m_hub_id}_msft" "${m_model_dir}/final_checkpoint" --token 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
    CUDA_VISIBLE_DEVICES="${visible_devices}" python inference_xiaojun.py --model_path $e_input_model --dataset_path $dataset_path --save_prefix $m_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --model_max_length $model_max_length --learning_rate $m_lr
    huggingface-cli upload "YYT-t/$m_hub_id" "${m_model_dir}_zq_raw" --token $upload_token

done    