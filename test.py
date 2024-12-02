from comet_ml import Experiment, ExistingExperiment

import torch
# import tensorflow as tf
import os, gc, argparse, pickle, numpy as np, pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from dataset import FigurativeDataset
from utils import set_all_seeds, create_prompt_flute, postprocess_output
from eval import Evaluator

HF_TOKEN = 'MY_HUGGINGFACE_TOKEN'                                                  

def test(model, tokenizer, test_data, evaluator, args=None, device=None, experiment=None):
    gc.collect()
    torch.cuda.empty_cache()
    full_texts = test_data.full_texts
    ref_labels, ref_explanations = test_data.labels, test_data.explanations
    pred_labels, pred_explanations = [], []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(full_texts), args.per_device_eval_batch_size)):
            full_texts_batch = full_texts[i:i+args.per_device_eval_batch_size]
            full_texts_batch = [text[:text.find('Label:')] + 'Label: ' for text in full_texts_batch]
            inputs = tokenizer(full_texts_batch, truncation=True, max_length=args.max_seq_length_src, padding='max_length', return_tensors='pt').to(device)
            if args.num_beams is not None and args.temperature is not None:
                model_outputs = model.generate(**inputs, max_new_tokens=args.max_seq_length_tgt, do_sample=True if args.temperature > 0 else False, temperature=args.temperature, num_beams=args.num_beams)
            elif args.num_beams is not None:
                model_outputs = model.generate(**inputs, max_new_tokens=args.max_seq_length_tgt, do_sample=False, temperature=0, num_beams=args.num_beams)
            elif args.temperature is not None:
                model_outputs = model.generate(**inputs, max_new_tokens=args.max_seq_length_tgt, do_sample=True if args.temperature > 0 else False, temperature=args.temperature, num_beams=1)
            else:
                model_outputs = model.generate(**inputs, max_new_tokens=args.max_seq_length_tgt, do_sample=False, temperature=0, num_beams=1)
            model_outputs[model_outputs == -100] = tokenizer.pad_token_id
            output_texts = tokenizer.batch_decode(model_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output_text in output_texts:
                pred_label, pred_explanation = postprocess_output(output_text, args.model_tag)
                pred_labels.append(pred_label)
                pred_explanations.append(pred_explanation)
    suffix = f'_t_{args.temperature}_b_{args.num_beams}'
    out_path = f'{args.output_folder}/outputs/test/'
    os.makedirs(out_path, exist_ok=True)
    df_pred_ref = pd.DataFrame({'text': full_texts, 'pred_label': pred_labels, 'ref_label': ref_labels, 'pred_explanation': pred_explanations, 'ref_explanation': ref_explanations})
    df_pred_ref.to_csv(f'{out_path}pred_ref_test{suffix}.csv', sep=',', header=True, index=False)
    rouge_scores = evaluator.compute_metric('rouge', pred_explanations, ref_explanations)
    bertscores = evaluator.compute_metric('bertscore', pred_explanations, ref_explanations, return_mean=False)
    evaluator.bertscore = None
    bleurt_scores = evaluator.compute_metric('bleurt', pred_explanations, ref_explanations, return_mean=False)
    evaluator.bleurt = None
    expl_scores = evaluator.compute_expl_scores(pred_labels, ref_labels, bertscores['bertscore_f1'], bleurt_scores['bleurt'])
    metrics = {**rouge_scores, 'bertscore_f1': np.mean(bertscores['bertscore_f1']), 'bleurt': np.mean(bleurt_scores['bleurt']), **expl_scores}
    pickle.dump(metrics, open(f'{out_path}metrics{suffix}.pkl', 'wb'))
    if experiment is not None:
        with experiment.test():
            experiment.log_metrics(metrics, step=0)
            experiment.log_table(f'./pred_ref_test{suffix}.csv', tabular_data=df_pred_ref, headers=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, dest="test_path", help='')
    parser.add_argument('--max_seq_length_src', type=int, default=256)
    parser.add_argument('--max_seq_length_tgt', type=int, default=None)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    
    parser.add_argument('--model_tag', type=str, dest="model_tag", help='')
    parser.add_argument('--no_apply_format', action='store_true', dest="no_apply_format", default=False, help='')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=None)

    parser.add_argument('--output_folder', type=str, dest="output_folder", default=None, help='')

    parser.add_argument('--use_gpu', action='store_true', dest="use_gpu", default=False, help='')
    parser.add_argument('--gpu_ids', type=str, dest="gpu_ids", default='0', help='')
    parser.add_argument('--num_workers', type=int, dest="num_workers", default=4, help='Number of workers used for dataloaders.')
    parser.add_argument('--pin_memory', action='store_true', dest="pin_memory", default=False, help='Whether to pin memory for data on GPU during data loading.')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--cache_dir', type=str, dest="cache_dir", default='/home/MY_CACHE_DIR/')
    parser.add_argument('--comet_logging', action='store_true', dest="comet_logging", default=False, help='Set flag to enable comet logging')
    parser.add_argument('--comet_key', type=str, dest="comet_key", default=None, help='Comet API key to log some metrics')
    parser.add_argument('--comet_workspace', type=str, dest="comet_workspace", default=None, help='Comet workspace name (usually username in Comet, used only if comet_key is not None)')
    parser.add_argument('--comet_project_name', type=str, dest="comet_project_name", default=None, help='Comet experiment name (used only if comet_key is not None)')

    args = parser.parse_args()

    if args.seed != -1: set_all_seeds(args.seed)

    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    device = torch.device("cuda") if torch.cuda.is_available() and args.use_gpu else torch.device("cpu")

    if args.max_seq_length_tgt is None: args.max_seq_length_tgt = args.max_seq_length_src

    if args.comet_logging:
        with open(f'{args.output_folder}args.txt', 'r') as f:
            args_lines = f.readlines()
            hyper_params = {line.split('=')[0]: line.split('=')[1].strip() for line in args_lines}
            args.model_tag = hyper_params['model_tag']
            hyper_params['max_seq_length_src'] = args.max_seq_length_src
            hyper_params['max_seq_length_tgt'] = args.max_seq_length_tgt
            hyper_params['temperature'] = args.temperature
            hyper_params['num_beams'] = args.num_beams
            if 'comet_experiment_id' in hyper_params:
                experiment = ExistingExperiment(api_key=args.comet_key, project_name=args.comet_project_name, workspace=args.comet_workspace, previous_experiment=hyper_params['comet_experiment_id'])
            else:
                experiment = Experiment(api_key=args.comet_key, project_name=args.comet_project_name, workspace=args.comet_workspace)
                experiment.log_parameters(hyper_params)
    else:
        experiment = None

    ckpt_path = f'{args.output_folder}/best_model/'
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, token=HF_TOKEN, cache_dir=args.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if 'gemma-2' in args.model_tag:
        model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, attn_implementation='eager', token=HF_TOKEN, cache_dir=args.cache_dir).to(device)
    else:
        model = AutoPeftModelForCausalLM.from_pretrained(ckpt_path, token=HF_TOKEN, cache_dir=args.cache_dir).to(device)
    
    evaluator = Evaluator(device)

    test_texts, test_labels, test_explanations = create_prompt_flute(args.test_path, prompt_id=0)
    test_dataset = FigurativeDataset(test_texts, test_labels, test_explanations, tokenizer, args.model_tag, apply_format=not args.no_apply_format, max_length=args.max_seq_length_src, device=device)
    print("Start testing...")
    test(model, tokenizer, test_dataset, evaluator, args, device, experiment)
    print("End testing...")
