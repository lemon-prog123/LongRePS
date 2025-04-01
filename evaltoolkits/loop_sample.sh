eval_model_list=(
    "Qwen-7B-Instruct-yarn-example /mnt/xiyu/Model/Qwen/Qwen2.5-7B-Instruct musique-Qwen-2.5-7B_prm_train.jsonl"
)
model_list=("${eval_model_list[@]}")

for model in "${model_list[@]}"; do
    model_name=$(echo $model | cut -d' ' -f1)
    model_path=$(echo $model | cut -d' ' -f2)
    file_name=$(echo $model | cut -d' ' -f3)
    echo "Launching inference for ${model_name}..."
    echo "Model path: ${model_path}"
    echo "File name: ${file_name}"
    bash launch_inference.sh ${model_name} ${model_path} ${file_name}
done