model_path="/mnt/xiyu/LongRePS/saves/Qwen2.5-7B/full/Qwen-2.5-7B_warmup_train_lr1e-5_maxlen16k_2025-03-26-17-19-47/checkpoint-18"
template="qwen"
learning_rate=5e-6
dataset="Qwen-2.5-7B_sample100_thresh1.0_yarn_checkstage3_prm"
echo "Dataname: ${dataset}"

output_path="saves/Qwen2.5-7B/full/${dataset}_train_lr${learning_rate}_maxlen16k_"$(date -d "+8 hours" +"%Y-%m-%d-%H-%M-%S")
mkdir -p ${output_path}

llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path ${model_path} \
        --preprocessing_num_workers 16 \
        --finetuning_type full \
        --template ${template} \
        --flash_attn auto \
        --dataset_dir data \
        --dataset ${dataset} \
        --cutoff_len 16384 \
        --learning_rate ${learning_rate} \
        --num_train_epochs 2 \
        --max_samples 100000 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type constant_with_warmup \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --save_strategy epoch \
        --warmup_steps 5 \
        --packing False \
        --save_only_model True \
        --report_to none \
        --output_dir ${output_path} \
        --bf16 True \
        --plot_loss True \
        --ddp_timeout 180000000 \
        --optim adamw_torch \
        --deepspeed cache/ds_z3_config.json > ${output_path}/output.log 2>&1
done