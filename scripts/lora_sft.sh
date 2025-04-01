model_path="/mnt/xiyu/LongRePS/saves/Qwen2.5-32B/lora/Qwen2.5-32B-warmup-epoch1"
template="qwen"
learning_rate=5e-5
dataset="Qwen-2.5-32B_sample30_temp0.7_thresh1.0_checkstage3_prm"
echo "Dataname: ${dataset}"

output_path="saves/Qwen2.5-32B/lora/${dataset}_train_lr${learning_rate}_maxlen16k_"$(date -d "+8 hours" +"%Y-%m-%d-%H-%M-%S")
mkdir -p ${output_path}
cp /mnt/xiyu/LongRePS/scripts/lora_sft.sh ${output_path}

llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path ${model_path} \
        --preprocessing_num_workers 16 \
        --finetuning_type lora \
        --lora_rank 128 \
        --lora_alpha 128 \
        --lora_dropout 0.05 \
        --lora_target all \
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
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --save_strategy epoch \
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