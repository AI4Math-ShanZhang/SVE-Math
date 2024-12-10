bash scripts/eval_multi.sh  ./checkpoints_sve/Qwen2.5_geo170k_qa_tuned ./playground/data/test_questions.jsonl results_try/Gllava-test ./playground/data/Geo170K/images 8 0 qwen_2

python scripts/geo_acc_calculate.py  --ground_truth_file ./playground/data/test_answers.jsonl --predictions_file results_try/Gllava-test_merged.jsonl