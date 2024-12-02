import numpy as np
import evaluate
from rouge_score import rouge_scorer, scoring
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator():
    def __init__(self, device):
        self.device = device
        
        self.rouge = evaluate.load('rouge')
        self.rouge_scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, tokenizer=None)

        self.bertscore = None
        self.bleurt = None
    
    def compute_metric(self, metric, predictions, references, return_mean=True):
        # predictions = list of strings | references = list of (list of) strings
        assert len(predictions) == len(references)
        if metric == 'rouge':
            all_scores = {'rouge1':[], 'rouge2':[], 'rougeL':[]}
        elif metric == 'rouge_all':
            all_scores = {f'{k}_{m}':[] for k in ['rouge1', 'rouge2', 'rougeL'] for m in ['p', 'r', 'f1']}
        elif metric == 'bertscore':
            all_scores = {'bertscore_p':[], 'bertscore_r':[], 'bertscore_f1':[]}
            if self.bertscore is None: self.bertscore = evaluate.load('bertscore')
        elif metric == 'bleurt':
            all_scores = {'bleurt':[]}
            if self.bleurt is None: self.bleurt = evaluate.load('bleurt', 'BLEURT-20', device=self.device)
        else:
            raise NotImplementedError('Metric not implemented yet')
        for p, r in zip(predictions, references):
            if metric == 'rouge':
                scores = self.rouge.compute(predictions=[p], references=[r], use_aggregator=True, rouge_types=['rouge1', 'rouge2', 'rougeL'])
                all_scores['rouge1'].append(scores['rouge1'])
                all_scores['rouge2'].append(scores['rouge2'])
                all_scores['rougeL'].append(scores['rougeL'])
            elif metric == 'rouge_all':
                rouge_aggregator = scoring.BootstrapAggregator()
                multi_ref = isinstance(references[0], list)
                for ref, pred in zip(references, predictions):
                    score = self.rouge_scorer.score_multi(ref, pred) if multi_ref else self.rouge_scorer.score(ref, pred)
                    rouge_aggregator.add_scores(score)
                if return_mean:
                    result = rouge_aggregator.aggregate()
                    for rouge_type in result:
                        all_scores[f'{rouge_type}_p'] = result[rouge_type].mid.precision
                        all_scores[f'{rouge_type}_r'] = result[rouge_type].mid.recall
                        all_scores[f'{rouge_type}_f1'] = result[rouge_type].mid.fmeasure
                else:  
                    result = rouge_aggregator._scores              
                    for rouge_type in result:
                        all_scores[f'{rouge_type}_p'] = [s.precision for s in result[rouge_type]]
                        all_scores[f'{rouge_type}_r'] = [s.recall for s in result[rouge_type]]
                        all_scores[f'{rouge_type}_f1'] = [s.fmeasure for s in result[rouge_type]]
                return all_scores
            elif metric == 'bertscore':
                res = self.bertscore.compute(predictions=[p], references=[r], lang='en', model_type='microsoft/deberta-large-mnli', device=self.device)
                all_scores['bertscore_p'].append(res['precision'][0])
                all_scores['bertscore_r'].append(res['recall'][0])
                all_scores['bertscore_f1'].append(res['f1'][0])
            elif metric == 'bleurt':
                res = self.bleurt.compute(predictions=[p], references=[r])['scores'][0]
                all_scores['bleurt'].append(res)
        return all_scores if not return_mean else {k: np.mean(v) for k, v in all_scores.items()}
    
    def compute_cls_metrics(self, predictions, references):
        all_scores = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            all_scores[metric] = accuracy_score(references, predictions)
            all_scores[metric] = precision_score(references, predictions)
            all_scores[metric] = recall_score(references, predictions)
            all_scores[metric] = f1_score(references, predictions)
        return all_scores
    
    def compute_expl_scores(self, pred_labels, ref_labels, bertscores, bleurt_scores):
        all_scores = {'expl_score':[]}
        count0, count50, count60 = 0, 0, 0
        for pred_label, ref_label, bscore, bleurt_score in zip(pred_labels, ref_labels, bertscores, bleurt_scores):
            if pred_label == ref_label:
                score = int((bscore + bleurt_score) * 50.0)
                all_scores['expl_score'].append(score)
                if score >= 0: count0 += 1
                if score >= 50: count50 += 1
                if score >= 60: count60 += 1
            else:
                all_scores['expl_score'].append(-1)
        all_scores['acc@0'] = count0 / len(pred_labels)
        all_scores['acc@50'] = count50 / len(pred_labels)
        all_scores['acc@60'] = count60 / len(pred_labels)
        return all_scores
    
    def dummy_bscore(self):
        predictions = ['i just left this car wash and was very satisfied !']
        references = [['i just left this car wash and was very satisfied !', 'i just left the car wash and i feel very satisfied', 'i did not leave this car wash and was very satisfied', 'i just left the car wash and i was very satisfied']]
        _ = self.bertscore.compute(predictions=predictions, references=references, lang='en', device=self.device)
        print('Dummy BERTScore computation end')

    def dummy_bleurt(self):
        predictions = ['i just left this car wash and was very satisfied !']
        references = ['i just left this car wash and was very satisfied !', 'i just left the car wash and i feel very satisfied', 'i did not leave this car wash and was very satisfied', 'i just left the car wash and i was very satisfied']
        _ = self.bleurt.compute(predictions=predictions, references=references)
        print('Dummy BLEURT computation end')