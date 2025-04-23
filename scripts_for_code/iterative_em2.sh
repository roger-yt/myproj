# source ~/.bashrc
git clone -b yutong_dev --single-branch https://github.com/YYT-t/xyzo.git

cd xyzo
pip install pyyaml
python pip_install_from_yaml.py

iter_num=1
#3
num_samples=120000
#40000
#model_name="Mistral-7B-Instruct-v0.2"
#company="mistralai"
task_pre="code"
task_suf="opencoder_edu"
company="deepseek-ai"
model_name="deepseek-coder-6.7b-instruct"
critic_model_name="${company}/${model_name}"
if [ "$model_name" == "gemma-1.1-7b-it" ]; then
    critic_model_name="google/gemma-1.1-2b-it"
fi
if [ "$model_name" == "gemma-2-9b-it" ]; then
    critic_model_name="google/gemma-2-2b-it"
fi
if [ "$model_name" == "deepseek-coder-6.7b-instruct" ]; then
    critic_model_name="deepseek-ai/deepseek-coder-6.7b-instruct"
fi

# conda init bash

# for debug
#5000
sft_learning_rate="5e-5"
path="./${model_name}-${task_suf}_sample_${num_samples}_tp"
export HF_TOKEN=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
#visible_devices="0,1,2,3"

for i in $(seq 1 $iter_num); do
    mkdir $path
    e_input_model="${path}/m-iter-$((i-1))_zq_raw"
    e_model_dir="${path}/e-iter-$i"
    m_model_dir="${path}/m-iter-$i-m-after-e"
    e_hub_id="${task_pre}_${task_suf}-${model_name}-e-iter-${i}_sample_${num_samples}_tp"
    m_hub_id="${task_pre}_${task_suf}-${model_name}-m-iter-${i}_sample_${num_samples}_tp-m-after-e"
    dataset_path="ZhangShenao/${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_tp"
    if [ "$i" -eq 1 ]; then
        e_input_model="${company}/${model_name}"
    else
        echo "iteration $i"
    fi
    if [ "$i" -eq 1 ]; then
        split="[:${num_samples}]"
    elif [ "$i" -eq 2 ]; then
        split="[$((num_samples)):$((num_samples*2))]"
    else
        split="[$((num_samples*2)):$((num_samples*3))]"
    fi
     python xiaojun_E_step_ent_PPO.py --model_name $e_input_model --critic_model_name $critic_model_name --task_type "${task_pre}_${task_suf}${split}" --model_path $e_model_dir 
    
    huggingface-cli upload "ZhangShenao/$e_hub_id" "${e_model_dir}/final_checkpoint" --token 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
    
    python inference_for_code.py --model_path "${e_model_dir}/final_checkpoint" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split 
    python inference_xiaojun.py --model_path "${e_model_dir}/final_checkpoint" --dataset_path $dataset_path --save_prefix $m_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --learning_rate $sft_learning_rate
    huggingface-cli upload "ZhangShenao/$m_hub_id" "${m_model_dir}_zq_raw" --token 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

done    