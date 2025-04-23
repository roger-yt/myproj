# source ~/.bashrc
#export WANDB_API_KEY=ee43df2d6680a9ce636f698eba4b5534c4336452
#export HF_TOKEN=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
#huggingface-cli login  --token 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
#git clone https://github.com/YYT-t/xyzo.git
#cd xyzo
#conda env create -f environment.yaml
#pip install flash-attn==2.6.3
#conda env create -f environment_sft.yaml

iter_num=1

#model_name="Mistral-7B-Instruct-v0.2"
#company="mistralai"
export HF_TOKEN=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
company="meta-llama"
model_name="Meta-Llama-3-8B-Instruct"
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
visible_devices="4,5,6,7"
ent_coeff=0.0

for i in $(seq 1 $iter_num); do
    conda activate yy
    if [ "$i" -eq 1 ]; then
        split="[:${num_samples}]"
        m_lr="5e-6"
    fi
    
    path="./${model_name}-${task_suf}_sample_${num_samples}_nsk_ml${max_length}"
    mkdir $path
    e_input_model="${path}/m-iter-$((i-1))_zq_raw"
    m_model_dir="${path}/m-iter-$i"
    m_hub_id="math_gsm-Meta-Llama-3-8B-Instruct-m-iter-1_sample_2500_nsk_ml512_mlr5e-6"
    dataset_path="ZhangShenao/${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_nsk_ml${max_length}_mlr${m_lr}_ent${ent_coeff}"

    if [ "$i" -eq 1 ]; then
        e_input_model="${company}/${model_name}"
    else
        echo "iteration $i"
    fi
    
    # CUDA_VISIBLE_DEVICES="${visible_devices}" python inference.py --model_path "ZhangShenao/math_gsm-Meta-Llama-3-8B-Instruct-e-iter-1_sample_2500_nsk_ml512_mlr5e-5" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split  --max_length $max_length|| exit 1
    # CUDA_VISIBLE_DEVICES="${visible_devices}" python inference_xiaojun.py --model_path $e_input_model --dataset_path $dataset_path --save_prefix $m_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --model_max_length $model_max_length --learning_rate $m_lr|| exit 1
    huggingface-cli upload "ZhangShenao/$m_hub_id" "${m_model_dir}_zq_raw" --token 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

done    