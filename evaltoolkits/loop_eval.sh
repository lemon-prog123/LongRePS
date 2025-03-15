eval_model_list=(
    "Llama-8B-warmup-lr1e-5-epoch1 ../saves/Llama3.1-8B/full/Llama-3.1-8B_warmup_train_lr1e-5_maxlen16k_2025-03-13-22-43-49/checkpoint-10 "
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