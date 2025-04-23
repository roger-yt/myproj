# source ~/.bashrc
export CUDA_VISIBLE_DEVICES="0,1"
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

company="google"
model_name="gemma-1.1-7b-it"
critic_model_name="${company}/${model_name}"
task_pre="math"
task_suf="math"
# conda init bash
num_samples=7500
sft_learning_rate="5e-5"
temp=1.0
gen_nums=30
version=0


path="./${model_name}-${task_suf}_sample_${num_samples}_temp_${temp}_gen_${gen_nums}"
export HF_TOKEN=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

conda activate yy
mkdir $path

rs_model_dir="${path}/rs_newb_ver${version}_model_temp_${temp}_gen_${gen_nums}_mlr${sft_learning_rate}"
rs_hub_id="${task_pre}_${task_suf}-${model_name}-rs_newb_ver${version}-sample_${num_samples}_temp_${temp}_gen_${gen_nums}_mlr${sft_learning_rate}"
dataset_path="ZhangShenao/rs_newb_ver${version}-${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_temp_${temp}_gen_${gen_nums}_mlr${sft_learning_rate}"
split="[:${num_samples}]"

python inference_rs_newb.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split --temp $temp --gen_nums $gen_nums || exit 1

python inference_xiaojun_newb.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $rs_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --learning_rate $sft_learning_rate|| exit 1
huggingface-cli upload "ZhangShenao/$rs_hub_id" "${rs_model_dir}_zq_raw" --token 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

rs_model_dir="${path}/rs_newb_mix_ver${version}_model_temp_${temp}_gen_${gen_nums}_mlr${sft_learning_rate}"
rs_hub_id="${task_pre}_${task_suf}-${model_name}-rs_newb_mix_ver${version}-sample_${num_samples}_temp_${temp}_gen_${gen_nums}_mlr${sft_learning_rate}"

python inference_xiaojun_newb.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $rs_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --learning_rate $sft_learning_rate|| exit 1
huggingface-cli upload "ZhangShenao/$rs_hub_id" "${rs_model_dir}_zq_raw" --token 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