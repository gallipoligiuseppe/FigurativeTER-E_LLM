from comet_ml import Experiment

import argparse, pathlib, shutil, os, random, numpy as np, pandas as pd
from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
from utils import set_all_seeds, create_prompt_flute

API_KEY = 'MY_OPENAI_API_KEY'

def postprocess_output_ab(output_text):
    if '[[A]]' in output_text: return 'A'
    if '[[B]]' in output_text: return 'B'
    if '[[C]]' in output_text: return 'C'
    return 'See full output'

def ab_test(client, ab_data, args=None, experiment=None):
    system_instruct, ab_prompts = ab_data
    full_outputs, judgements = [], []
    for ix, sample in tqdm(enumerate(ab_prompts), total=len(ab_prompts)):
        messages = [{"role": "system", "content": system_instruct}, {"role": "user", "content": sample}]
        try:
            completion = client.chat.completions.create(model=args.evaluator_tag, messages=messages, temperature=args.temperature, max_tokens=args.max_seq_length_tgt)
            output_text = completion.choices[0].message.content
        except:
            output_text = 'API Error'
        judgement = postprocess_output_ab(output_text)
        if judgement == 'See full output':
            with open(f'{args.output_folder}errors.txt', 'a') as f:
                f.write(f'### Sample ###\n{sample}\n### Output ###\n{output_text}\n### Judgement ###\n{judgement}\n@@@@@@@@@@@@@@@@@@@@@@\n')
        full_outputs.append(output_text)
        judgements.append(judgement)
        if ix % 50 == 0:
            print(f'Sample text:\n{sample}')
            print(f'Output: {output_text}')
            print(f'Judgement: {judgement}\n@@@@@@@@@@@')
    suffix = f'_t_{args.temperature}'
    suffix += f'_{args.ix_start_stop.replace(",", "_")}' if args.ix_start_stop is not None else ''
    out_path = f'{args.output_folder}'
    df_pred_ref = pd.DataFrame({'prompts': ab_prompts, 'judgement': judgements, 'full_output': full_outputs})
    df_pred_ref.to_csv(f'{out_path}ab_test{suffix}.csv', sep=',', header=True, index=False)
    
    if experiment is not None:
        with experiment.test():
            experiment.log_table(f'./ab_test{suffix}.csv', tabular_data=df_pred_ref, headers=True)

def ab_test_prepare(args, experiment):
    client = OpenAI(api_key=API_KEY)

    if '_TMP_EXP' in args.output_folder:
        test_texts, test_labels, test_explanations = create_prompt_flute(args.test_path, prompt_id=0, max_samples=15)
    else:
        test_texts, test_labels, test_explanations = create_prompt_flute(args.test_path, prompt_id=0)

    modelA_df, modelB_df = pd.read_csv(args.modelA_out_file), pd.read_csv(args.modelB_out_file)
    assert len(modelA_df) == len(modelB_df) == len(test_texts)
    modelA_labels, modelA_explanations = modelA_df['pred_label'].tolist(), modelA_df['pred_explanation'].tolist()
    modelB_labels, modelB_explanations = modelB_df['pred_label'].tolist(), modelB_df['pred_explanation'].tolist()

    questions = [t[:t.find('Label:')].strip() for t in test_texts]
    ref_answers = [t[t.find('Label:'):].strip() for t in test_texts]
    modelA_answers = [f'Label: {l.strip()}\nExplanation: {e.strip()}' for l, e in zip(modelA_labels, modelA_explanations)]
    modelB_answers = [f'Label: {l.strip()}\nExplanation: {e.strip()}' for l, e in zip(modelB_labels, modelB_explanations)]

    if args.ix_start_stop is not None:
        ix_start, ix_stop = [int(ix) for ix in args.ix_start_stop.split(',')]
        print(f'Evaluating samples from {ix_start} to {ix_stop}')
        questions, ref_answers, modelA_answers, modelB_answers = questions[ix_start:ix_stop], ref_answers[ix_start:ix_stop], modelA_answers[ix_start:ix_stop], modelB_answers[ix_start:ix_stop]

    system_instruct = """Please act as an impartial judge and evaluate the quality of the responses provided by two models to the user question below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, model A’s answer, and model B’s answer. Begin your evaluation by comparing both assistants’ answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if model A is better, "[[B]]" if model B is better, and "[[C]]" for a tie."""
    prompt_template = "[User Question]\n@question@\n\n[Start of Reference Answer]\n@ref_answer@\n[End of Reference Answer]\n\n[Start of Model A's Answer]\n@modelA_answer@\n[End of Model A's Answer]\n\n[Start of Model B's Answer]\n@modelB_answer@\n[End of Model B's Answer]"
    ab_prompts = [prompt_template.replace('@question@', q).replace('@ref_answer@', ref_ans).replace('@modelA_answer@', modelA_ans).replace('@modelB_answer@', modelB_ans) for q, ref_ans, modelA_ans, modelB_ans in zip(questions, ref_answers, modelA_answers, modelB_answers)]
    
    print('\nStart AB test')
    ab_test(client, (system_instruct, ab_prompts), args, experiment)
    print('\nAB test completed')

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest="dataset", help='')
    parser.add_argument('--test_path', type=str, dest="test_path", help='')
    parser.add_argument('--no_apply_format', action='store_true', dest="no_apply_format", default=False, help='')
    parser.add_argument('--ix_start_stop', type=str, dest="ix_start_stop", default=None, help='')

    parser.add_argument('--evaluator_tag', type=str, dest="evaluator_tag", help='')
    parser.add_argument('--max_seq_length_tgt', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=0)

    parser.add_argument('--modelA_tag', type=str, dest="modelA_tag", help='')
    parser.add_argument('--modelB_tag', type=str, dest="modelB_tag", help='')
    parser.add_argument('--modelA_out_file', type=str, dest="modelA_out_file", help='')
    parser.add_argument('--modelB_out_file', type=str, dest="modelB_out_file", help='')

    parser.add_argument('--output_folder', type=str, dest="output_folder", default=None, help='')

    parser.add_argument('--use_gpu', action='store_true', dest="use_gpu", default=False, help='')
    parser.add_argument('--gpu_ids', type=str, dest="gpu_ids", default='0', help='')
    parser.add_argument('--num_workers', type=int, dest="num_workers", default=4, help='Number of workers used for dataloaders.')
    parser.add_argument('--seed', type=int, default=-1)

    parser.add_argument('--comet_logging', action='store_true', dest="comet_logging", default=False, help='Set flag to enable comet logging')
    parser.add_argument('--comet_key', type=str, dest="comet_key", default=None, help='Comet API key to log some metrics')
    parser.add_argument('--comet_workspace', type=str, dest="comet_workspace", default=None, help='Comet workspace name (usually username in Comet, used only if comet_key is not None)')
    parser.add_argument('--comet_project_name', type=str, dest="comet_project_name", default=None, help='Comet experiment name (used only if comet_key is not None)')
    parser.add_argument('--exp_name', type=str, dest="exp_name", default=None, help='Experiment name to log on Comet')

    args = parser.parse_args()

    if args.seed != -1: set_all_seeds(args.seed)

    args.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]

    if args.output_folder is None:
        args.output_folder = '../outputs/'
        if args.exp_name is not None:
            args.output_folder += args.exp_name.replace(' | ', '_').replace(' ', '_') + '/'
        else:
            args.output_folder += 'run_' + datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S") + '/'
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.modelA_out_file, f'{args.output_folder}{args.modelA_tag}_out.csv')
    shutil.copy(args.modelB_out_file, f'{args.output_folder}{args.modelB_tag}_out.csv')

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

    ab_test_prepare(args, experiment)
