from comet_ml import Experiment

import argparse, pathlib, os, random, numpy as np, pandas as pd, json, pickle, gc
from tqdm import tqdm
from datetime import datetime
import torch, tensorflow as tf
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import FigurativeDataset
from utils import set_all_seeds, create_prompt_flute, find_dream_scene, postprocess_output_few
from eval import Evaluator

HF_TOKEN = 'MY_HUGGINGFACE_TOKEN'

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

def extract_examples(dataset, k_shot, few_shot_type, args):
    train_dream_data = json.load(open(args.dream_train_path, 'r'))
    candidate_examples = []
    for dream_example in train_dream_data:
        if dream_example['premise_motivation'] != '' and dream_example['hypothesis_motivation'] != '' and dream_example['premise_emotion'] != '' and dream_example['hypothesis_emotion'] != '' and dream_example['premise_consequence'] != '' and dream_example['hypothesis_consequence'] != '' and dream_example['premise_rot'] != '' and dream_example['hypothesis_rot'] != '':
            for example in dataset:
                if dream_example['premise'] in example and dream_example['hypothesis'] in example:
                    candidate_examples.append((example, dream_example))
    if few_shot_type == 'random':
        indices = random.sample(range(len(candidate_examples)), k_shot)
        if k_shot > 1:
            _check = [candidate_examples[i][0] for i in indices]
            while int(any(['Entailment' in e for e in _check]))+int(any(['Contradiction' in e for e in _check])) < 2:
                indices = random.sample(range(len(candidate_examples)), k_shot)
                _check = [candidate_examples[i] for i in indices]
    elif few_shot_type == 'balanced':
        indices = []
        cnt_sar, cnt_met, cnt_sim, cnt_idi = 0, 0, 0, 0
        while len(indices) < k_shot:
            _indices = random.sample(range(len(candidate_examples)), k_shot)
            for ix in _indices:
                if ix in indices: continue
                if candidate_examples[ix][1]['data_type'] == 'sarcasm' and cnt_sar < k_shot//4:
                    indices.append(ix)
                    cnt_sar += 1
                elif candidate_examples[ix][1]['data_type'] == 'metaphor' and cnt_met < k_shot//4:
                    indices.append(ix)
                    cnt_met += 1
                elif candidate_examples[ix][1]['data_type'] == 'simile' and cnt_sim < k_shot//4:
                    indices.append(ix)
                    cnt_sim += 1
                elif candidate_examples[ix][1]['data_type'] == 'idiom' and cnt_idi < k_shot//4:
                    indices.append(ix)
                    cnt_idi += 1
    elif few_shot_type == 'figure':
        indices = []
        cnt_sar, cnt_met, cnt_sim, cnt_idi = 0, 0, 0, 0
        while len(indices) < k_shot:
            _indices = random.sample(range(len(candidate_examples)), k_shot)
            for ix in _indices:
                if ix in indices: continue
                if candidate_examples[ix][1]['data_type'] == 'sarcasm' and cnt_sar < k_shot//2:
                    indices.append(ix)
                    cnt_sar += 1
                elif candidate_examples[ix][1]['data_type'] == 'metaphor' and cnt_met < k_shot//6:
                    indices.append(ix)
                    cnt_met += 1
                elif candidate_examples[ix][1]['data_type'] == 'simile' and cnt_sim < k_shot//6:
                    indices.append(ix)
                    cnt_sim += 1
                elif candidate_examples[ix][1]['data_type'] == 'idiom' and cnt_idi < k_shot//6:
                    indices.append(ix)
                    cnt_idi += 1
    return [candidate_examples[i] for i in indices]

def find_dream_test(example, args):
    test_dream_data = json.load(open(args.dream_test_path, 'r'))
    premise = example.split('Premise:')[1].split('\n')[0].strip()
    hypothesis = example.split('Hypothesis:')[1].split('\n')[0].strip()
    if 'llama-3' in args.model_tag.lower():
        hypothesis = hypothesis.split('<|eot_id|>')[0].strip()
    if 'zephyr' in args.model_tag.lower():
        hypothesis = hypothesis.split('</s>')[0].strip()
    elif 'gemma' in args.model_tag.lower():
        hypothesis = hypothesis.split('<end_of_turn>')[0].strip()
    return (example, find_dream_scene(test_dream_data, premise, hypothesis))

def add_dream_scene(example, scene_type, args):
    example, dream = example
    header = example.split('Premise:')[0].strip()
    premise = example.split('Premise:')[1].split('\n')[0].strip()
    hypothesis = example.split('Hypothesis:')[1].split('\n')[0].strip()
    footer = '\n'.join(example.split('Hypothesis:')[1].split('\n')[1:])
    if 'llama-3' in args.model_tag.lower():
        hypothesis = hypothesis.split('<|eot_id|>')[0].strip()
    if 'zephyr' in args.model_tag.lower():
        hypothesis = hypothesis.split('</s>')[0].strip()
    elif 'gemma' in args.model_tag.lower():
        hypothesis = hypothesis.split('<end_of_turn>')[0].strip()
    if scene_type == 'motivation':
        premise = f'Premise: {premise}\nPremise (motivation): {dream["premise_motivation"]}'
        hypothesis = f'Hypothesis: {hypothesis}\nHypothesis (motivation): {dream["hypothesis_motivation"]}'
    elif scene_type == 'consequence':
        premise = f'Premise: {premise}\nPremise (consequence): {dream["premise_consequence"]}'
        hypothesis = f'Hypothesis: {hypothesis}\nHypothesis (consequence): {dream["hypothesis_consequence"]}'
    elif scene_type == 'emotion':
        premise = f'Premise: {premise}\nPremise (emotion): {dream["premise_emotion"]}'
        hypothesis = f'Hypothesis: {hypothesis}\nHypothesis (emotion): {dream["hypothesis_emotion"]}'
    elif scene_type == 'social norm':
        premise = f'Premise: {premise}\nPremise (social norm): {dream["premise_rot"]}'
        hypothesis = f'Hypothesis: {hypothesis}\nHypothesis (social norm): {dream["hypothesis_rot"]}'
    elif scene_type == 'all':
        premise = f'Premise: {premise}\nPremise (motivation): {dream["premise_motivation"]}\nPremise (consequence): {dream["premise_consequence"]}\nPremise (emotion): {dream["premise_emotion"]}\nPremise (social norm): {dream["premise_rot"]}'
        hypothesis = f'Hypothesis: {hypothesis}\nHypothesis (motivation): {dream["hypothesis_motivation"]}\nHypothesis (consequence): {dream["hypothesis_consequence"]}\nHypothesis (emotion): {dream["hypothesis_emotion"]}\nHypothesis (social norm): {dream["hypothesis_rot"]}'
    if 'llama-3' in args.model_tag.lower():
        hypothesis += '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    if 'zephyr' in args.model_tag.lower():
        hypothesis += '</s>'
    elif 'gemma' in args.model_tag.lower():
        hypothesis += '<end_of_turn>'
    return f'{header}\n{premise}\n{hypothesis}\n{footer}'


def test(model, tokenizer, test_data, evaluator, args=None, device=None, experiment=None):
    gc.collect()
    torch.cuda.empty_cache()
    full_texts = test_data[0]
    ref_labels, ref_explanations = test_data[1].labels, test_data[1].explanations
    pred_labels, pred_explanations = [], []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(full_texts):
            if any([args.model_tag == m for m in ['meta-llama/Meta-Llama-3.1-8B', 'google/gemma-2-9b', 'google/gemma-2-9b-it']]):
                sample_text = sample[:sample.rfind('Label:')]
            else:
                sample_text = sample[:sample.rfind('Label:')] + 'Label: '
            inputs = tokenizer(sample_text, return_tensors='pt').to(device)
            if args.num_beams is not None and args.temperature is not None:
                model_outputs = model.generate(**inputs, max_new_tokens=args.max_seq_length_tgt, do_sample=True if args.temperature > 0 else False, temperature=args.temperature, num_beams=args.num_beams)
            elif args.num_beams is not None:
                model_outputs = model.generate(**inputs, max_new_tokens=args.max_seq_length_tgt, do_sample=False, temperature=0, num_beams=args.num_beams)
            elif args.temperature is not None:
                model_outputs = model.generate(**inputs, max_new_tokens=args.max_seq_length_tgt, do_sample=True if args.temperature > 0 else False, temperature=args.temperature, num_beams=1)
            else:
                model_outputs = model.generate(**inputs, max_new_tokens=args.max_seq_length_tgt, do_sample=False, temperature=0, num_beams=1)
            if -100 in model_outputs:
                model_outputs[model_outputs == -100] = tokenizer.pad_token_id
            output_texts = tokenizer.batch_decode(model_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for output_text in output_texts:
                try:
                    pred_label, pred_explanation = postprocess_output_few(output_text, args.model_tag, args.k_shot)
                except:
                    pred_label, pred_explanation = '', ''
                    with open(f'{args.output_folder}/outputs/errors.txt', 'a') as f:
                        f.write(f'### Sample ###\n{sample}\n### Sample text ###\n{sample_text}\n### Output ###\n{output_text}\n### Pred label ###\n{pred_label}\n### Pred explanation ###\n{pred_explanation}\n@@@@@@@@@@@@@@@@@@@@@@\n')
                pred_labels.append(pred_label)
                pred_explanations.append(pred_explanation)
    suffix = f'_t_{args.temperature}_b_{args.num_beams}'
    out_path = f'{args.output_folder}/outputs/{args.phase}/'
    os.makedirs(out_path, exist_ok=True)
    df_pred_ref = pd.DataFrame({'text': full_texts, 'pred_label': pred_labels, 'ref_label': ref_labels, 'pred_explanation': pred_explanations, 'ref_explanation': ref_explanations})
    df_pred_ref.to_csv(f'{out_path}pred_ref_{args.phase}{suffix}.csv', sep=',', header=True, index=False)
    rouge_scores = evaluator.compute_metric('rouge', pred_explanations, ref_explanations)
    bertscores = evaluator.compute_metric('bertscore', pred_explanations, ref_explanations, return_mean=False)
    evaluator.bertscore = None
    bleurt_scores = evaluator.compute_metric('bleurt', pred_explanations, ref_explanations, return_mean=False)
    evaluator.bleurt = None
    expl_scores = evaluator.compute_expl_scores(pred_labels, ref_labels, bertscores['bertscore_f1'], bleurt_scores['bleurt'])
    metrics = {**rouge_scores, 'bertscore_f1': np.mean(bertscores['bertscore_f1']), 'bleurt': np.mean(bleurt_scores['bleurt']), **expl_scores}
    pickle.dump(metrics, open(f'{out_path}metrics{suffix}.pkl', 'wb'))
    if experiment is not None:
        context = experiment.validate if args.phase == 'val' else experiment.test
        with context():
            experiment.log_metrics(metrics, step=0)
            experiment.log_table(f'./pred_ref_{args.phase}{suffix}.csv', tabular_data=df_pred_ref, headers=True)


def zero_few_shot_figurative(args, device, experiment):
    model = AutoModelForCausalLM.from_pretrained(args.model_tag, torch_dtype=torch.float16, token=HF_TOKEN, cache_dir=args.cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_tag, padding_side='left', token=HF_TOKEN, cache_dir=args.cache_dir)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    train_texts, train_labels, train_explanations = create_prompt_flute(args.train_path, args.prompt_id, args=args)
    if '_TMP_EXP' in args.output_folder:
        test_texts, test_labels, test_explanations = create_prompt_flute(args.test_path, args.prompt_id, 15, args=args)
    else:
        test_texts, test_labels, test_explanations = create_prompt_flute(args.test_path, args.prompt_id, args=args)

    train_dataset = FigurativeDataset(train_texts, train_labels, train_explanations, tokenizer, args.model_tag, apply_format=not args.no_apply_format, max_length=None, device=device)
    test_dataset = FigurativeDataset(test_texts, test_labels, test_explanations, tokenizer, args.model_tag, apply_format=not args.no_apply_format, max_length=None, device=device)

    evaluator = Evaluator(device)
    
    few_shot_examples = ''
    if args.k_shot > 0:
        few_shot_examples = extract_examples(train_dataset.full_texts, args.k_shot, args.few_shot_type, args)
        if args.scene_type is not None:
            few_shot_examples = '\n'.join([add_dream_scene(example, args.scene_type, args) for example in few_shot_examples])
        else:
            few_shot_examples = '\n'.join([example[0] for example in few_shot_examples])
    test_samples = []
    for test_example in test_dataset.full_texts:
        if args.scene_type is not None: test_example = add_dream_scene(find_dream_test(test_example, args), args.scene_type, args)
        if args.k_shot > 0:
            test_samples.append(f'{few_shot_examples}\n{test_example}')
        else:
            test_samples.append(test_example)
    
    print(f'\nStart {args.phase}')
    test(model, tokenizer, (test_samples, test_dataset), evaluator, args, device, experiment)
    print(f'\n{args.phase} completed')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest="dataset", help='')
    parser.add_argument('--phase', type=str, choices=['val', 'test'], dest="phase", help='')
    parser.add_argument('--train_path', type=str, dest="train_path", help='')
    parser.add_argument('--test_path', type=str, dest="test_path", help='')
    parser.add_argument('--scene_type', type=str, dest="scene_type", default=None, help='')
    parser.add_argument('--dream_train_path', type=str, dest="dream_train_path", default=None, help='')
    parser.add_argument('--dream_test_path', type=str, dest="dream_test_path", default=None, help='')

    parser.add_argument('--model_tag', type=str, dest="model_tag", help='')
    parser.add_argument('--no_apply_format', action='store_true', dest="no_apply_format", default=False, help='')
    parser.add_argument('--max_seq_length_tgt', type=int, default=None)

    parser.add_argument('--prompt_id', type=int, dest="prompt_id", default=None, help='')
    parser.add_argument('--k_shot', type=int, default=0)
    parser.add_argument('--few_shot_type', type=str, default='random', help='')
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
    parser.add_argument('--exp_name', type=str, dest="exp_name", default=None, help='Experiment name to log on Comet')

    args = parser.parse_args()

    if args.seed != -1: set_all_seeds(args.seed)

    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    device = torch.device("cuda") if torch.cuda.is_available() and args.use_gpu else torch.device("cpu")

    if args.output_folder is None:
        args.output_folder = '../outputs/'
        if args.exp_name is not None:
            args.output_folder += args.exp_name.replace(' | ', '_').replace(' ', '_') + '/'
        else:
            args.output_folder += 'run_' + datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S") + '/'
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    os.makedirs(f'{args.output_folder}/outputs/', exist_ok=True)

    hyper_params = {}
    print("Arguments summary:\n")
    for key, value in vars(args).items():
        hyper_params[key] = value
        print(f"\t{key}:\t\t{value}")
    with open(f'{args.output_folder}/args.txt', 'w') as f:
        f.write('\n'.join([f'{k}={v}' for k, v in hyper_params.items()]))

    if args.comet_logging:
        experiment = Experiment(api_key=args.comet_key, project_name=args.comet_project_name, workspace=args.comet_workspace)
        if args.exp_name is not None:
            experiment.set_name(args.exp_name)
        experiment_id = experiment.id
        experiment.log_parameters(hyper_params)
        with open(f'{args.output_folder}/args.txt', 'a') as f:
            f.write(f'\ncomet_experiment_id={experiment_id}')
    else:
        experiment = None

    zero_few_shot_figurative(args, device, experiment)
