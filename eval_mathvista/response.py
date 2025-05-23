import os


import io
import time
import argparse

from tqdm import tqdm

import sys

sys.path.append('../')
from utilities import *

from build_query import create_query_data
from gllava.model.builder import load_pretrained_model
from gllava.mm_utils import get_model_name_from_path
from gllava.eval.run_llava import eval_model, evalmodel
from gllava.utils import disable_torch_init


multimath_prompt_zh = '''
请逐步推理并解答该数学问题，并将最终答案放置于\\boxed{}中。
每个步骤一行，使用如下形式：
Step X (所使用的数学定理/依据): 具体解答步骤。
Answer: \\boxed{}
'''

multimath_prompt_en = '''
Please reason step by step, and put your final answer within \\boxed{}.
Each step is placed on a new line, using the following format: 
Step X (Mathematical theorem/basis used): Detailed solution steps. 
Answer: \\boxed{}. 
'''

def extract_boxed_content(text):
    pattern = r'\\boxed\{(.*?)\}'
    match_list = re.findall(pattern, text)
    if match_list == []:
        return None
    else:
        return match_list[0]

def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response == None:
        return False
    if "Response Error" in response:
        return False
    return True


def evaluate_code(code_string):
    # execute_code_and_capture_output
    # Backup the original stdout
    old_stdout = sys.stdout

    # Redirect stdout to capture the output
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    # Try executing the code and capture any exception
    error = None
    try:
        exec(code_string)
    except Exception as e:
        error = e

    # Restore the original stdout
    sys.stdout = old_stdout

    # Get the captured output
    captured_output = new_stdout.getvalue()
    if isinstance(captured_output, str):
        captured_output = captured_output.strip()

    # Return the captured output or error
    return captured_output, error


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--data_dir', type=str, default='/mnt/bn/pengshuai-nas/data/MathDataset/MathVista/data')
    parser.add_argument('--input_file', type=str, default='testmini.json')
    # output
    parser.add_argument('--output_dir', type=str, default='/mnt/bn/pengshuai-nas/MathLLM/LLaVA/eval_mathvista/outputs/')
    parser.add_argument('--output_file', type=str, default='pred_deepseekmathrl_llavastage1_llavastage2_mathv360k_stage3.json')
    # model
    parser.add_argument('--model_path', type=str, default='liuhaotian/llava-v1.5-13b', help='path of lora or full model')
    parser.add_argument('--model_base', type=str, default=None, help='liuhaotian/llava-v1.5-13b for lora, =None for full model')
    parser.add_argument('--conv_mode', type=str, default='dpsk')
    # query
    parser.add_argument('--query_file', type=str, default='query.json')
    parser.add_argument('--caption_file', type=str, default=None)
    parser.add_argument('--ocr_file', type=str, default=None)
    parser.add_argument('--shot_type', type=str, default='solution', help='shot type',
                        choices=['solution', 'code'])
    parser.add_argument('--shot_num', type=int, default=0, help='number of shot examples')
    parser.add_argument('--use_caption', default=False, help='use caption data')
    parser.add_argument('--use_ocr', default=False, help='use ocr data')
    # other settings
    parser.add_argument('--use_multimath_prompt', action='store_true', default=False, help='append multimath prompt after query')
    parser.add_argument('--rerun', default=False, help='rerun answer extraction for all problems')
    parser.add_argument('--debug', default=False, help='debug mode')
    parser.add_argument("--num_of_kvs", type=int, default=4)
    parser.add_argument("--merge_version", type=str, default='combine')
    args = parser.parse_args()

    # load data
    input_file = os.path.join(args.data_dir, args.input_file)
    print(f"Reading {input_file}...")
    data = read_json(input_file)

    # load or create query data
    if args.query_file:
        query_file = os.path.join(args.data_dir, args.query_file)
        if os.path.exists(query_file):
            print(f"Loading existing {query_file}...")
            query_data = read_json(query_file)
    else:
        print("\nCreating new query...")
        # load caption
        caption_data = {}
        if args.use_caption:
            caption_file = args.caption_file
            if os.path.exists(caption_file):
                print(f"Reading {caption_file}...")
                try:
                    caption_data = read_json(caption_file)["texts"]
                    print("Caption data loaded.")
                except:
                    print("Caption data not found!! Please Check.")
                    # load ocr
        ocr_data = {}
        if args.use_ocr:
            ocr_file = args.ocr_file
            if os.path.exists(ocr_file):
                print(f"Reading {ocr_file}...")
                try:
                    ocr_data = read_json(ocr_file)["texts"]
                    print("OCR data loaded.")
                except:
                    print("OCR data not found!! Please Check.")
        # create query
        query_data = create_query_data(data, caption_data, ocr_data, args)

    if args.use_multimath_prompt:
        for pid in query_data.keys():
            query = query_data[pid]
            new_query = query + multimath_prompt_en
            query_data[pid] = new_query
    #print(query_data)

    # output file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)

    # load results
    if os.path.exists(output_file):
        print("\nResults already exist.")
        print(f"Reading {output_file}...")
        results = read_json(output_file)
    else:
        results = {}

    model_path = args.model_path
    model_base = args.model_base
    print(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=get_model_name_from_path(model_path),
        num_of_kvs=args.num_of_kvs,
        merge_version=args.merge_version
    )

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    ##

    # build final test pid list
    test_pids = list(data.keys())
    print("\nNumber of test problems in total:", len(test_pids))

    skip_pids = []
    if not args.rerun:
        print("\nRemoving problems with existing valid response...")
        for pid in test_pids:
            # print(f"Checking {pid}...")
            if pid in results and 'response' in results[pid]:
                response = results[pid]['response']
                if verify_response(response):
                    # print(f"Valid response found for {pid}.")
                    skip_pids.append(pid)
    else:
        print("\nRerun answer extraction for all problems...")

    test_pids = [pid for pid in test_pids if pid not in skip_pids]
    print("Number of test problems to run:", len(test_pids))
    # print(test_pids)

    # tqdm, enumerate results
    for _, pid in enumerate(tqdm(test_pids)):
        problem = data[pid]
        query = query_data[pid]
        image = problem['image']
        image_path = os.path.join(args.data_dir, image)

        if args.debug:
            print("--------------------------------------------------------------")
        print(f"\nGenerating response for {pid}...")
        
        try:

            args_llava = type('Args', (), {
                "model_path": model_path,
                "model_base": None,
                "model_name": get_model_name_from_path(model_path),
                "query": query,
                "conv_mode": args.conv_mode,
                "image_file": image_path,
                "sep": ",",
                "temperature": 0.2,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 32768
            })()
            response = evalmodel(args_llava, model_name, tokenizer, model, image_processor, context_len)
            results[pid] = problem
            results[pid]['query'] = query
            if args.shot_type == 'solution':
                results[pid]['response'] = response
            else:
                output, error = evaluate_code(response)
                results[pid]['response'] = response
                results[pid]['execution'] = output
                results[pid]['error'] = str(error)
            if args.use_multimath_prompt:
                results[pid]['extraction'] = extract_boxed_content(response)
            if args.debug:
                print(f"\n#Query: \n{query}")
                print(f"\n#Response: \n{response}")
        except Exception as e:
            print(e)
            print(f"Error in extracting answer for {pid}")
            results[pid]['error'] = e

        try:
            print(f"Saving results to {output_file}...")
            save_json(results, output_file)
            print(f"Results saved.")
        except Exception as e:
            print(e)
            print(f"Error in saving {output_file}")