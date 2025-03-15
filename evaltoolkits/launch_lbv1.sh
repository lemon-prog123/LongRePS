
model_name="Llama-8B-sft-lr5e-6-sample30-temp0.7-thresh1.0-epoch1"
model_path="../saves/Llama3.1-8B/full/Llama-3.1-8B_sample30_thresh1.0_prm_train_lr5e-6_maxlen16k_2025-03-14-19-09-15/checkpoint-61"
mode="cot"

domain_list=("all")
eval_data_dir="../dataset/longbenchv1"
sample_num=1
temperature=0.0

cp /mnt/xiyu/Model/meta-llama/Llama-3.1-8B-Instruct/tokenizer* ${model_path} #for Llama Models
#cp Qwen/Qwen2.5-7B-Instruct/tokenizer* ${model_path}


for gpu_id in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${gpu_id} python -m vllm.entrypoints.openai.api_server \
    --served-model-name ${model_name} \
    --model ${model_path} \
    --tensor-parallel-size=1 \
    --trust-remote-code \
    --port 800${gpu_id} > ../log/vllm_${model_name}_gpu${gpu_id}.log 2>&1 &
done
sleep 30 # sleep 30s, wait for the servers to start


for domain in "${domain_list[@]}"; do
    file_name_list=("multifieldqa_en_${mode}.jsonl") #("musique_${mode}.jsonl" "hotpotqa_${mode}.jsonl" "multifieldqa_en_${mode}.jsonl" "qasper_${mode}.jsonl" "2wikimqa_${mode}.jsonl")
    for file_name in "${file_name_list[@]}"; do
        eval_dataset_name=$(echo "$file_name" | cut -d'_' -f1)
        cot_mode=$(echo "$file_name" | grep -q "nocot" && echo "nocot" || echo "cot") # this is a trick in bash to implement in-line if-else using && and ||
        result_dir="./pred_cot_vs_nocot"
        output_dir="${result_dir}/${model_name}"
        mkdir -p ${output_dir}
        #echo -e "\nScript executed with parameters: $@" >> ${output_dir}/dw_launch_cot_vs_nocot.sh

        result_dir="${result_dir%/}" # remove the trailing slash if there is any
        eval_data_dir="${eval_data_dir%/}"

        echo "Launching inference for ${model_name}..."
        echo "Model path: ${model_path}"
        echo "Eval data dir: ${eval_data_dir}"
        echo "File name: ${file_name}"
        echo "Sample num: ${sample_num}"
        echo "Result dir: ${result_dir}"
        echo "Temperature: ${temperature}"
        echo "COT mode: ${cot_mode}"
        echo "Dataset: ${eval_dataset_name}"

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
            > ./inference2.out

        python step2_extract_preds_from_raw.py --path_to_src_file ${path_to_inference_output}
        python step3_eval_f1.py --path_to_src_file ${path_to_extracted_result}

    done
done
pkill -f vllm; pkill -f spawn_main
