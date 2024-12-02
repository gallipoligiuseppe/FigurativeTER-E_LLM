import os, shutil, subprocess, pickle, torch, transformers, gc, random, json, numpy as np
from transformers import TrainerCallback

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_prompt_flute(path, prompt_id, max_samples=None, args=None):
    raw_data = json.load(open(path, 'r'))
    if max_samples: raw_data = raw_data[:max_samples]
    full_data, labels, explanations = [], [], []
    for d in raw_data:
        scene_type = args.scene_type if args is not None and args.scene_type != 'all' else 'motivation, consequence, emotion, and social norm'
        sample = ''
        if prompt_id == 0:
            sample += "Is there a contradiction or entailment between the premise and hypothesis?\n"
            sample += f"Premise: {d['premise']}\nHypothesis: {d['hypothesis']}\n"
            sample += f"Label: {d['label']}\nExplanation: {d['explanation']}"
        elif prompt_id == 1:
            sample += "I will provide you with a pair of sentences consisting of a premise and a hypothesis containing figurative language. Is there a contradiction or entailment between the premise and hypothesis? Provide an explanation for your answer.\n"
            sample += f"Premise: {d['premise']}\nHypothesis: {d['hypothesis']}\n"
            sample += f"Label: {d['label']}\nExplanation: {d['explanation']}"
        elif prompt_id == 2:
            sample += "You are an expert in linguistics. I will provide you with a pair of sentences (a premise and a hypothesis) containing figurative language. Your task is to determine whether the premise entails or contradicts the hypothesis. Provide an explanation for your answer.\n"
            sample += f"Premise: {d['premise']}\nHypothesis: {d['hypothesis']}\n"
            sample += f"Label: {d['label']}\nExplanation: {d['explanation']}"
        elif prompt_id == 3:
            sample += "Is there a contradiction or entailment between the premise and hypothesis?\n"
            sample += f"Premise: {d['premise']}\nHypothesis: {d['hypothesis']}\n"
            sample += f"Label: {d['label']}\nExplanation: {d['explanation']}"
        elif prompt_id == 4:
            sample += f"I will provide you with a pair of sentences consisting of a premise and a hypothesis containing figurative language. For both the premise and the hypothesis, I will provide you with additional {scene_type} context. Is there a contradiction or entailment between the premise and hypothesis? Provide an explanation for your answer.\n"
            sample += f"Premise: {d['premise']}\nHypothesis: {d['hypothesis']}\n"
            sample += f"Label: {d['label']}\nExplanation: {d['explanation']}"
        elif prompt_id == 5:
            sample += f"You are an expert in linguistics. I will provide you with a pair of sentences (a premise and a hypothesis) containing figurative language. For both the premise and the hypothesis, I will provide you with additional {scene_type} context. Considering the pair of sentences and the additional context, your task is to determine whether the premise entails or contradicts the hypothesis. Provide an explanation for your answer.\n"
            sample += f"Premise: {d['premise']}\nHypothesis: {d['hypothesis']}\n"
            sample += f"Label: {d['label']}\nExplanation: {d['explanation']}"
        else:
            raise ValueError(f"Prompt id {prompt_id} is not valid.")
        full_data.append(sample)
        labels.append(d['label'])
        explanations.append(d['explanation'])
    return full_data, labels, explanations

def postprocess_output(output, model_tag):
    if 'llama' in model_tag.lower() or 'mistral' in model_tag.lower():
        label = output.split('Label:')[1].split('\n')[0].strip() if 'Label:' in output else ''
        explanation = output.split('Explanation:')[1].split('\n')[0].strip() if 'Explanation:' in output else ''
    elif 'gemma' in model_tag.lower():
        label = output.split('Label:')[1].split('\n')[0].strip() if 'Label:' in output else ''
        explanation = output.split('Explanation:')[1].split('\n')[0].strip() if 'Explanation:' in output else ''
    elif 'zephyr' in model_tag.lower():
        label = output.split('Label:')[1].split('\n')[0].strip() if 'Label:' in output else ''
        explanation = output.split('Explanation:')[1].split('\n')[0].strip() if 'Explanation:' in output else ''
    return label, explanation

def postprocess_output_few(output, model_tag, k_shot):
    label, explanation = '', ''

    if 'llama' in model_tag.lower() or 'mistral' in model_tag.lower():
        if 'Label:' in output:
            label = output.split('Label:')[k_shot+1].split('\n')[0].strip()
            label = 'Entailment' if 'entail' in label.lower() else 'Contradiction' if 'contradict' in label.lower() else ''
        if 'Explanation:' in output:
            if k_shot == 0:
                explanation = output.split('Explanation:')[1].split('\n')[0].strip()
            else:
                explanation = '\n'.join(output.split('Label:')[k_shot+1].split('\n')[1:]).strip()
                if 'Explanation:' in explanation:
                    explanation = explanation.split('Explanation:')[1].strip().split('\n')[0].strip()
                else:
                    explanation = explanation[explanation.find(':')+1:].strip()
    elif 'zephyr' in model_tag.lower():
        if 'Label:' in output:
            label = output.split('Label:')[k_shot+1].split('\n')[0].strip()
            label = 'Entailment' if 'entail' in label.lower() else 'Contradiction' if 'contradict' in label.lower() else ''
        if 'Explanation:' in output:
            if k_shot == 0:
                explanation = output.split('Explanation:')[1].split('\n')[0].strip()
            else:
                explanation = '\n'.join(output.split('Label:')[k_shot+1].split('\n')[1:]).strip()
                if 'Explanation:' in explanation:
                    explanation = explanation.split('Explanation:')[1].strip().split('\n')[0].strip()
                else:
                    explanation = explanation[explanation.find(':')+1:].strip()
    elif 'gemma' in model_tag.lower():
        if 'Label:' in output:
            label = output.split('Label:')[k_shot+1].split('\n')[0].strip()
            label = 'Entailment' if 'entail' in label.lower() else 'Contradiction' if 'contradict' in label.lower() else ''
        if 'Explanation:' in output:
            if k_shot == 0:
                explanation = output.split('Explanation:')[1].split('\n')[0].strip()
            else:
                explanation = '\n'.join(output.split('Label:')[k_shot+1].split('\n')[1:]).strip()
                if 'Explanation:' in explanation:
                    explanation = explanation.split('Explanation:')[1].strip().split('\n')[0].strip()
                else:
                    explanation = explanation[explanation.find(':')+1:].strip()
    elif 'gpt-35-turbo' in model_tag.lower() or 'gpt-4o' in model_tag.lower():
        if 'Label:' in output:
            label = output.split('Label:')[1].split('\n')[0].strip()
        else:
            label = output.split('\n')[0].strip()
        label = 'Entailment' if 'entail' in label.lower() else 'Contradiction' if 'contradict' in label.lower() else ''
        if 'Explanation:' in output:
            explanation = output.split('Explanation:')[1].split('\n')[0].strip()
        else:
            explanation = output[output.rfind(':')+1:].strip()
    return label, explanation

def find_dream_scene(dream_data, premise, hypothesis, scene_type=None):
    for dream_sample in dream_data:
        if dream_sample['premise'].strip() == premise and dream_sample['hypothesis'].strip() == hypothesis:
            if scene_type is not None:
                return dream_sample[f'premise_{scene_type}'], dream_sample[f'hypothesis_{scene_type}']
            else:
                return dream_sample
    return '', ''

class MyCallback(TrainerCallback):
    def __init__(self, args, evaluator):
        self.args = args
        self.evaluator = evaluator

    def on_evaluate(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        self.evaluator.bertscore = None
        self.evaluator.bleurt = None

def ext_metric_compute(args, pred_labels, ref_labels, pred_explanations, ref_explanations):
    metric_path = args.output_folder + '_metrics_tmp/'
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    data = {'pred_labels':pred_labels, 'ref_labels':ref_labels, 'pred_explanations':pred_explanations, 'ref_explanations':ref_explanations}
    pickle.dump(data, open(metric_path + 'data.pkl', 'wb'))
    subprocess.run([f'{args.metrics_sh} {metric_path}'], shell=True, check=True)
    if not os.path.exists(metric_path + 'metrics.pkl'):
        raise FileNotFoundError(f"File {metric_path + 'metrics.pkl'} does not exist.")
    metrics = pickle.load(open(metric_path + 'metrics.pkl', 'rb'))
    shutil.rmtree(metric_path)
    return metrics