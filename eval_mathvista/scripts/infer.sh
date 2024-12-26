model_save_path="./checkpoints_sve/SVE-Math-Deepseek-7B"
output_file="pred_checkpoints_release.json"

python3 eval_mathvista/response.py \
    --rerun true \
    --conv_mode qwen_2 \
    --data_dir ./playground/MathVista/data \
    --input_file testmini.json \
    --output_dir ./eval_mathvista/outputs/ \
    --output_file  ${output_file} \
    --model_path ${model_save_path} &
