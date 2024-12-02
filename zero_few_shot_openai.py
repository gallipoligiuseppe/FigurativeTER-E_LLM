from comet_ml import Experiment
from openai import OpenAI

import argparse, pathlib, os, random, numpy as np, pandas as pd, json, pickle, gc
from tqdm import tqdm
from datetime import datetime
import torch, tensorflow as tf
from dataset import FigurativeDataset
from utils import set_all_seeds, create_prompt_flute, postprocess_output_few
from zero_few_shot import extract_examples, add_dream_scene, find_dream_test
from eval import Evaluator

API_KEY = 'MY_OPENAI_API_KEY'

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

def test(client, test_data, evaluator, args=None, device=None, experiment=None):
    gc.collect()
    torch.cuda.empty_cache()
    full_texts = test_data[0]
    ref_labels, ref_explanations = test_data[1].labels, test_data[1].explanations
    pred_labels, pred_explanations = [], []
    for ix, sample in tqdm(enumerate(full_texts), total=len(full_texts)):
        sample_text = sample[:sample.rfind('Label:')].strip()
        sample_text += '\nAnswer with\nLabel:\nExplanation:'
        messages = [{"role": "user", "content": sample_text}]
        try:
            completion = client.chat.completions.create(model=args.model_tag, messages=messages, temperature=args.temperature, max_tokens=args.max_seq_length_tgt, seed=args.seed)
            output_text = completion.choices[0].message.content
            pred_label, pred_explanation = postprocess_output_few(output_text, args.model_tag, args.k_shot)
        except:
            pred_label, pred_explanation = '', ''
            if not 'output_text' in locals(): output_text = 'API Error'
            with open(f'{args.output_folder}/outputs/errors.txt', 'a') as f:
                f.write(f'### Sample ###\n{sample}\n### Sample text ###\n{sample_text}\n### Output ###\n{output_text}\n### Pred label ###\n{pred_label}\n### Pred explanation ###\n{pred_explanation}\n@@@@@@@@@@@@@@@@@@@@@@\n')
        pred_labels.append(pred_label)
        pred_explanations.append(pred_explanation)
        if ix % 100 == 0:
            print(f'Sample text:\n{sample_text}')
            print(f'Output:\n{output_text}')
            print(f'Pred label: {pred_label}\nPred explanation: {pred_explanation}\n@@@@@@@@@@@')
    suffix = f'_t_{args.temperature}'
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
    client = OpenAI(api_key=API_KEY)

    train_texts, train_labels, train_explanations = create_prompt_flute(args.train_path, args.prompt_id, args=args)
    if '_TMP_EXP' in args.output_folder:
        test_texts, test_labels, test_explanations = create_prompt_flute(args.test_path, args.prompt_id, 5, args=args)
    else:
        test_texts, test_labels, test_explanations = create_prompt_flute(args.test_path, args.prompt_id, args=args)

    train_dataset = FigurativeDataset(train_texts, train_labels, train_explanations, None, args.model_tag, apply_format=not args.no_apply_format, max_length=None, device=device)
    test_dataset = FigurativeDataset(test_texts, test_labels, test_explanations, None, args.model_tag, apply_format=not args.no_apply_format, max_length=None, device=device)

    evaluator = Evaluator(device)
    
    few_shot_examples = ''
    if args.k_shot > 0:
        few_shot_examples = extract_examples(train_dataset.full_texts, args.k_shot, args.few_shot_type, args)
        if args.scene_type is not None:
            few_shot_examples = f"{'Example:\n' if args.k_shot == 1 else 'Examples:\n'}" + '\n'.join([add_dream_scene(example, args.scene_type, args) for example in few_shot_examples]) + '\n'
        else:
            few_shot_examples = f"{'Example:\n' if args.k_shot == 1 else 'Examples:\n'}" + '\n'.join([example[0] for example in few_shot_examples]) + '\n'
    test_samples = []
    for test_example in test_dataset.full_texts:
        if args.scene_type is not None: test_example = add_dream_scene(find_dream_test(test_example, args), args.scene_type, args)
        if args.k_shot > 0:
            test_samples.append(f'{few_shot_examples}\n{test_example}')
        else:
            test_samples.append(test_example)
    
    print(f'\nStart {args.phase}')
    test(client, (test_samples, test_dataset), evaluator, args, device, experiment)
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

    parser.add_argument('--output_folder', type=str, dest="output_folder", default=None, help='')

    parser.add_argument('--use_gpu', action='store_true', dest="use_gpu", default=False, help='')
    parser.add_argument('--gpu_ids', type=str, dest="gpu_ids", default='0', help='')
    parser.add_argument('--num_workers', type=int, dest="num_workers", default=4, help='Number of workers used for dataloaders.')
    parser.add_argument('--pin_memory', action='store_true', dest="pin_memory", default=False, help='Whether to pin memory for data on GPU during data loading.')
    parser.add_argument('--seed', type=int, default=-1)

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
