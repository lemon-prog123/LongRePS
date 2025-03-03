# if model_name, model_path, eval_data_dir, file_name, sample_num, thresh, temperature, filtered_filename, are passed as arguments, then parse these arguments
if [[ $# -eq 8 ]]; then
    model_name=$1
    model_path=$2
    eval_data_dir=$3
    file_name=$4
    sample_num=$5
    thresh=$6
    temperature=$7
    filtered_filename=$8
    inference_mode=$(echo "$file_name" | grep -q "train" && echo "train" || echo "eval") # train (for sample data), eval (for evaluation)
else
    model_name=$1
    model_path=$2
    file_name=$3
    eval_data_dir="../dataset"
    mode="predicted_answer"
    inference_mode=$(echo "$file_name" | grep -q "train" && echo "train" || echo "eval") # train (for sample data), eval (for evaluation)
    if [[ $inference_mode == "train" ]]; then
        sample_num=30
        thresh=1.0
        temperature=0.7
        filtered_filename="${model_name}_sample${sample_num}temp${temperature}thresh${thresh}.jsonl"
    elif [[ $inference_mode == "eval" ]]; then
        sample_num=1
        temperature=0.0
    fi

fi



eval_dataset_name=$(echo "$file_name" | cut -d'_' -f1)
cot_mode=$(echo "$file_name" | grep -q "nocot" && echo "nocot" || echo "cot") # this is a trick in bash to implement in-line if-else using && and ||
result_dir="./pred_${inference_mode}"
output_dir="${result_dir}/${model_name}"
mkdir -p ${output_dir}
echo -e "\nScript executed with parameters: $@" >> ${output_dir}/new_launch_inference.sh

result_dir="${result_dir%/}" # remove the trailing slash if there is any
eval_data_dir="${eval_data_dir%/}"

echo "Launching inference for ${model_name}..."
echo "Model path: ${model_path}"
echo "Eval data dir: ${eval_data_dir}"
echo "File name: ${file_name}"
echo "Inference mode: ${inference_mode}"
echo "Sample num: ${sample_num}"
echo "Result dir: ${result_dir}"
echo "Temperature: ${temperature}"
echo "COT mode: ${cot_mode}"
echo "Dataset: ${eval_dataset_name}"
echo "Filtered filename: ${filtered_filename}"
echo "Thresh: ${thresh}"

#cp Qwen/Qwen2.5-7B-Instruct/tokenizer* ${model_path} #for Qwen Model
cp /mnt/xiyu/Model/meta-llama/Llama-3.1-8B-Instruct/tokenizer* ${model_path} #for Llama Model

for gpu_id in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${gpu_id} python -m vllm.entrypoints.openai.api_server \
    --served-model-name ${model_name} \
    --model ${model_path} \
    --tensor-parallel-size=1 \
    --trust-remote-code \
    --port 800${gpu_id} > ../log/vllm_${model_name}_gpu${gpu_id}.log 2>&1 &
done

sleep 30 # sleep 30s, wait for the servers to start

echo "Evaluating ${eval_dataset_name}..."
path_to_inference_output="${result_dir}/${model_name}/${eval_dataset_name}.temp${temperature}sample${sample_num}.${cot_mode}.jsonl"
path_to_extracted_result="${path_to_inference_output%.jsonl}_eval.jsonl" # remove the last ".jsonl" and add "_eval.jsonl"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python step1_eval_inference.py \
    --model ${model_name} \
    --model_path ${model_path} \
    --data_path ${eval_data_dir}/${file_name} \
    --output_path ${path_to_inference_output} \
    --sample_num ${sample_num} \
    --dataset_name ${eval_dataset_name} \
    --temperature ${temperature} \
    > ./inference.out

python step2_extract_preds_from_raw.py --path_to_src_file ${path_to_inference_output}
python step3_eval_f1.py --path_to_src_file ${path_to_extracted_result}

pkill -f vllm; pkill -f spawn_main
