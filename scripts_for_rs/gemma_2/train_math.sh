conda env create -f environment.yaml
eval "$(conda shell.bash hook)"
conda activate yy

export CUDA_VISIBLE_DEVICES="0,1,2,3"

iter_num=1

export HF_TOKEN=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
upload_token=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

company="google"
model_name="gemma-2-9b-it"
task_pre="math"
task_suf="math"
num_samples=7500
sft_learning_rate="5e-5"
temp=1.0
gen_nums=30

path="./${model_name}-${task_suf}_sample_${num_samples}_temp_${temp}_gen_${gen_nums}"
mkdir $path

rs_model_dir="${path}/rs_model_temp_${temp}_gen_${gen_nums}_mlr${sft_learning_rate}"
rs_hub_id="${task_pre}_${task_suf}-${model_name}-rs-sample_${num_samples}_temp_${temp}_gen_${gen_nums}_mlr${sft_learning_rate}"
dataset_path="YYT-t/rs-${task_pre}_${task_suf}-${model_name}-iter${i}_sample_${num_samples}_temp_${temp}_gen_${gen_nums}_mlr${sft_learning_rate}"
split="[:${num_samples}]"

python inference_rs.py --model_path "${company}/${model_name}" --task_type "${task_pre}_${task_suf}" --dataset_path $dataset_path --dataset_fraction $split --temp $temp --gen_nums $gen_nums --upload_token $upload_token --mode ans || exit 1

python sft_rs.py --model_path "${company}/${model_name}" --dataset_path $dataset_path --save_prefix $rs_model_dir --sft_data_type zq_raw --train_step $num_samples  --task_type  "${task_pre}_${task_suf}" --learning_rate $sft_learning_rate --mode ans|| exit 1
huggingface-cli upload "YYT-t/$rs_hub_id" "${rs_model_dir}_zq_raw" --token $upload_token