from comet_ml import Experiment

import torch
# import tensorflow as tf
import os, shutil, gc, argparse, pathlib, pickle, numpy as np, pandas as pd
from datetime import datetime
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from dataset import FigurativeDataset
from utils import set_all_seeds, create_prompt_flute, postprocess_output, MyCallback, ext_metric_compute
from eval import Evaluator

HF_TOKEN = 'MY_HUGGINGFACE_TOKEN'

def compute_metrics_wrapper(model, tokenizer, val_data, evaluator, eval_save_steps, samples_ixs=None, args=None, device=None, experiment=None):
    def compute_metrics(preds):
        gc.collect()
        torch.cuda.empty_cache()
        full_texts = val_data.full_texts
        ref_labels, ref_explanations = val_data.labels, val_data.explanations
        if samples_ixs is not None:
            full_texts, ref_labels, ref_explanations = [full_texts[ix] for ix in samples_ixs], [ref_labels[ix] for ix in samples_ixs], [ref_explanations[ix] for ix in samples_ixs]
        pred_labels, pred_explanations = [], []
        with torch.no_grad():
            for i in range(0, len(full_texts), args.per_device_eval_batch_size):
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
        metrics = ext_metric_compute(args, pred_labels, ref_labels, pred_explanations, ref_explanations)
        df_pred_ref = pd.DataFrame({'text': full_texts, 'pred_label': pred_labels, 'ref_label': ref_labels, 'pred_explanation': pred_explanations, 'ref_explanation': ref_explanations})
        steps = sorted(filter(lambda f: 'step-' in f, os.listdir(f'{args.output_folder}/outputs/')), key=lambda f: int(f.split('-')[1]))
        curr_step = int(steps[-1].split('-')[1]) + eval_save_steps if len(steps) > 0 else eval_save_steps
        curr_step_path = f'{args.output_folder}/outputs/step-{curr_step}/'
        os.makedirs(curr_step_path, exist_ok=True)
        pickle.dump(metrics, open(f'{curr_step_path}metrics.pkl', 'wb'))
        df_pred_ref.to_csv(f'{curr_step_path}pred_ref.csv', sep=',', header=True, index=False)
        if experiment is not None:
            with experiment.validate():
                experiment.log_metrics(metrics)
                experiment.log_table(f'./pred_ref_step-{curr_step}.csv', tabular_data=df_pred_ref, headers=True)
        return metrics
    return compute_metrics


def finetune_figurative(args, device, experiment):
    model = AutoModelForCausalLM.from_pretrained(args.model_tag, token=HF_TOKEN, cache_dir=args.cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_tag, padding_side='left', token=HF_TOKEN, cache_dir=args.cache_dir)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.resize_token_embeddings(len(tokenizer))

    target_modules = ["query_key_value"]
    if True and ('llama' in args.model_tag or 'mistral' in args.model_tag or 'gemma' in args.model_tag or 'zephyr' in args.model_tag):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=target_modules, lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_texts, train_labels, train_explanations = create_prompt_flute(args.train_path, prompt_id=0, max_samples=args.max_samples_train)
    val_texts, val_labels, val_explanations = create_prompt_flute(args.val_path, prompt_id=0, max_samples=args.max_samples_val)

    train_dataset = FigurativeDataset(train_texts, train_labels, train_explanations, tokenizer, args.model_tag, apply_format=not args.no_apply_format, max_length=args.max_seq_length_src, device=device)
    val_dataset = FigurativeDataset(val_texts, val_labels, val_explanations, tokenizer, args.model_tag, apply_format=not args.no_apply_format, max_length=args.max_seq_length_src, device=device)

    epoch_steps = int(np.ceil(len(train_dataset)/args.per_device_train_batch_size))
    eval_save_steps = epoch_steps//4

    evaluator = Evaluator(device)
    samples_ixs = np.random.choice(len(val_dataset), args.n_samples_val, replace=False) if args.n_samples_val > 0 else None
    my_compute_metrics = compute_metrics_wrapper(model, tokenizer, val_dataset, evaluator, eval_save_steps, samples_ixs, args, device, experiment)
    my_callbacks = [MyCallback(args, evaluator)]

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_percentage,
        optim=args.optimizer,
        weight_decay=0.01,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.accumulation_steps,
        evaluation_strategy='steps',
        eval_steps=eval_save_steps,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        save_strategy='steps',
        save_steps=eval_save_steps,
        save_total_limit=3,
        logging_steps=50,
        logging_dir=f'{args.output_folder}/logs/',
        output_dir=f'{args.output_folder}',
        fp16=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=my_compute_metrics,
        callbacks=my_callbacks,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("Start training...")
    trainer.train()
    print("End training...")
    
    print("Saving best model...")
    for filename in os.listdir(args.output_folder):
        if filename.startswith('checkpoint-'):
            shutil.rmtree(os.path.join(args.output_folder, filename))
    trainer.save_model(args.output_folder + 'best_model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, dest="dataset", help='')
    parser.add_argument('--train_path', type=str, dest="train_path", help='')
    parser.add_argument('--max_samples_train', type=int, dest="max_samples_train", default=None, help='')
    parser.add_argument('--val_path', type=str, dest="val_path", help='')
    parser.add_argument('--max_samples_val', type=int, dest="max_samples_val", default=None, help='')
    parser.add_argument('--n_samples_val', type=int, dest="n_samples_val", default=-1, help='')
    parser.add_argument('--test_path', type=str, dest="test_path", help='')

    parser.add_argument('--model_tag', type=str, dest="model_tag", help='')
    parser.add_argument('--no_apply_format', action='store_true', dest="no_apply_format", default=False, help='')
    parser.add_argument('--max_seq_length_src', type=int, default=256)
    parser.add_argument('--max_seq_length_tgt', type=int, default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--optimizer', type=str, default='adamw_8bit')
    parser.add_argument('--warmup_percentage', type=float, default=0.1)
    parser.add_argument('--num_train_epochs', type=int, default=5)
    parser.add_argument('--accumulation_steps', type=int, default=1)

    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=None)

    parser.add_argument('--output_folder', type=str, dest="output_folder", default=None, help='')
    parser.add_argument('--metrics_sh', type=str, dest="metrics_sh", default=None, help='')

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
        args.output_folder += 'run_' + datetime.now().strftime(f"%d_%m_%Y_%H_%M_%S") + '/'
    pathlib.Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    os.makedirs(f'{args.output_folder}/outputs/', exist_ok=True)

    if args.max_seq_length_tgt is None: args.max_seq_length_tgt = args.max_seq_length_src

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

    finetune_figurative(args, device, experiment)
