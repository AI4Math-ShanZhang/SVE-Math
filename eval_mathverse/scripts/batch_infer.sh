model_path="./checkpoints_sve/SVE-Math-Deepseek-7B"
answers_file="./eval_mathverse/outputs/pred_multimath-7b-llava-v1_5.json"

CHUNKS=8
for IDX in {0..7}; do
    CUDA_VISIBLE_DEVICES=$IDX python3 eval_mathverse/infer.py \
        --model_path ${model_path} \
        --prompt none \
        --conv_mode qwen_2 \
        --question_file ./playground/MathVerse/testmini.json \
        --image_folder ./playground/MathVerse/images \
        --answers_file ${answers_file}\
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done

wait

python3 eval_mathverse/merge_pred.py --answers_file ${answers_file}
