##please uncomment gllava/train/train.py lines 1261-1265
deepspeed --include=localhost:0,1,2,3,4,5,6,7 gllava/train/train.py \
                                            --mm_projector_lr 1e-5 \
                                            --deepspeed ./scripts/zero2.json \
                                            --model_name_or_path Qwen/Qwen2.5-Math-7B-Instruct \
                                            --version qwen_2 \
                                            --data_path ./playground/data/Geo170K/alignment.json \
                                            --image_folder ./playground/data/Geo170K/images \
                                            --vision_tower openai/clip-vit-large-patch14-336 \
                                            --mm_projector_type mlp2x_gelu \
                                            --mm_vision_select_layer -2 \
                                            --mm_use_im_start_end False \
                                            --mm_use_im_patch_token False \
                                            --image_aspect_ratio pad \
                                            --group_by_modality_length True \
                                            --bf16 True \
                                            --output_dir ./checkpoints/Qwen2.5-Math-7B-Instruct \
                                            --num_train_epochs 2 \
                                            --per_device_train_batch_size 6 \
                                            --per_device_eval_batch_size 4 \
                                            --gradient_accumulation_steps 1 \
                                            --evaluation_strategy "no" \
                                            --save_strategy "steps" \
                                            --save_steps 50000 \
                                            --save_total_limit 1 \
                                            --learning_rate 3e-5 \
                                            --weight_decay 0. \
                                            --warmup_ratio 0.03 \
                                            --lr_scheduler_type "cosine" \
                                            --logging_steps 1 \
                                            --tf32 True \
                                            --model_max_length 32768 \
                                            --gradient_checkpointing True \
                                            --dataloader_num_workers 4 \
                                            --lazy_preprocess True \
                                            --freeze_backbone