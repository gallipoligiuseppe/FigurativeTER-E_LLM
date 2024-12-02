import torch
import tensorflow as tf
import os, sys, pickle, json, numpy as np, pandas as pd
from eval import Evaluator

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(device)
    
    metric_path = sys.argv[1]
    if metric_path.endswith('.csv'):
        data = pd.read_csv(metric_path).fillna('')
        pred_labels, ref_labels, pred_explanations, ref_explanations = data['pred_label'].tolist(), data['ref_label'].tolist(), data['pred_explanation'].tolist(), data['ref_explanation'].tolist()
    elif not os.path.exists(metric_path):
        raise FileNotFoundError(f"Directory {metric_path} does not exist.")
    else:
        data = pickle.load(open(metric_path + 'data.pkl', 'rb'))
        pred_labels, ref_labels, pred_explanations, ref_explanations = data['pred_labels'], data['ref_labels'], data['pred_explanations'], data['ref_explanations']
    
    if len(sys.argv) == 2:
        rouge_scores = evaluator.compute_metric('rouge', pred_explanations, ref_explanations)
        bertscores = evaluator.compute_metric('bertscore', pred_explanations, ref_explanations, return_mean=False)
        bleurt_scores = evaluator.compute_metric('bleurt', pred_explanations, ref_explanations, return_mean=False)
        expl_scores = evaluator.compute_expl_scores(pred_labels, ref_labels, bertscores['bertscore_f1'], bleurt_scores['bleurt'])
        metrics = {**rouge_scores, 'bertscore_f1': np.mean(bertscores['bertscore_f1']), 'bleurt': np.mean(bleurt_scores['bleurt']), **expl_scores}
        
        if metric_path.endswith('.csv'):
            pickle.dump(metrics, open(metric_path.replace('.csv', '_metrics.pkl'), 'wb'))
        else:
            pickle.dump(metrics, open(metric_path + 'metrics.pkl', 'wb'))

    elif len(sys.argv) == 3 and sys.argv[2] == 'types':
        test_data = json.load(open('./data/test.json'))
        fig_types = set([t['fig_type'] for t in test_data])
        assert len(pred_labels) == len(test_data)
        fig_types_outputs = {t: {'pred_labels':[], 'ref_labels':[], 'pred_explanations':[], 'ref_explanations':[]} for t in fig_types}
        for t, pred_label, ref_label, pred_explanation, ref_explanation in zip(test_data, pred_labels, ref_labels, pred_explanations, ref_explanations):
            fig_types_outputs[t['fig_type']]['pred_labels'].append(pred_label)
            fig_types_outputs[t['fig_type']]['ref_labels'].append(ref_label)
            fig_types_outputs[t['fig_type']]['pred_explanations'].append(pred_explanation)
            fig_types_outputs[t['fig_type']]['ref_explanations'].append(ref_explanation)
        assert sum([len(fig_types_outputs[t]['pred_labels']) for t in fig_types]) == len(pred_labels)
        for t in fig_types:
            rouge_scores = evaluator.compute_metric('rouge', fig_types_outputs[t]['pred_explanations'], fig_types_outputs[t]['ref_explanations'])
            bertscores = evaluator.compute_metric('bertscore', fig_types_outputs[t]['pred_explanations'], fig_types_outputs[t]['ref_explanations'], return_mean=False)
            bleurt_scores = evaluator.compute_metric('bleurt', fig_types_outputs[t]['pred_explanations'], fig_types_outputs[t]['ref_explanations'], return_mean=False)
            expl_scores = evaluator.compute_expl_scores(fig_types_outputs[t]['pred_labels'], fig_types_outputs[t]['ref_labels'], bertscores['bertscore_f1'], bleurt_scores['bleurt'])
            metrics = {**rouge_scores, 'bertscore_f1': np.mean(bertscores['bertscore_f1']), 'bleurt': np.mean(bleurt_scores['bleurt']), **expl_scores}
            pickle.dump(metrics, open(metric_path.replace('.csv', f'_{t}_metrics.pkl'), 'wb'))
