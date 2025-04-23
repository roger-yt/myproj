# source ~/.bashrc
export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd ..

pip install pyyaml
num_samples=4000
sft_learning_rate="5e-5"
company="deepseek-ai"
model_name="deepseek-coder-6.7b-instruct"
critic_model_name="${company}/${model_name}"
task_pre="code"
task_suf="opencoder_edu"
gen_nums=30
temp=1.0
# conda init bash

path="./${model_name}-${task_suf}_sample_${num_samples}_tp"
export HF_TOKEN=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


mkdir $path

rs_model_dir="${path}/rs_model_gen${gen_nums}_temp${temp}"
rs_hub_id="${task_pre}_${task_suf}-${model_name}-rs-sample_${num_samples}_tp_gen${gen_nums}_temp${temp}"
dataset_path="ZhangShenao/rs-${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_tp_gen${gen_nums}_temp${temp}"
split="[:${num_samples}]"

python inference_for_code_rs.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split  --gen_nums $gen_nums --temp $temp
python inference_xiaojun.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $rs_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --learning_rate $sft_learning_rate
huggingface-cli upload "ZhangShenao/$rs_hub_id" "${rs_model_dir}_zq_raw" --token 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
