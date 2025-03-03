eval_model_list=(
    #"[Your Model Name] [Your Model Path] [Evaluate File Name]"
)
model_list=("${eval_model_list[@]}")

for model in "${model_list[@]}"; do
    model_name=$(echo $model | cut -d' ' -f1)
    model_path=$(echo $model | cut -d' ' -f2)
    file_name=$(echo $model | cut -d' ' -f3)
    echo "Launching inference for ${model_name}..."
    echo "Model path: ${model_path}"
    echo "File name: ${file_name}"
    bash new_launch_inference.sh ${model_name} ${model_path} ${file_name}
done