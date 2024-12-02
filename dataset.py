import json
from torch.utils.data import Dataset

class FigurativeDataset(Dataset):
    def __init__(self, full_texts, labels, explanations, tokenizer, model_tag, apply_format=True, max_length=256, device=None):
        self.full_texts = full_texts
        self.labels = labels
        self.explanations = explanations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        if apply_format:
            if 'llama-2' in model_tag.lower():
                self.full_texts = self._process_format_llama2(full_texts)
            if 'llama-3' in model_tag.lower():
                self.full_texts = self._process_format_llama3(full_texts)
            if 'mistral' in model_tag.lower():
                self.full_texts = self._process_format_mistral(full_texts)
            if 'gemma' in model_tag.lower():
                self.full_texts = self._process_format_gemma(full_texts)
            if 'zephyr' in model_tag.lower():
                self.full_texts = self._process_format_zephyr(full_texts)
            
    def _process_format_llama2(self, full_texts):
        formatted_texts = []
        for t in full_texts:
            text = ''
            text += '<s>[INST] ' + t.split('\n')[0] + ' [/INST]\n'
            text += '\n'.join(t.split('\n')[1:])
            text += '</s>'
            formatted_texts.append(text)
        return formatted_texts
    
    def _process_format_llama3(self, full_texts):
        formatted_texts = []
        for t in full_texts:
            text = ''
            text += '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n' + '\n'.join(t.split('\n')[:3]) + '<|eot_id|>'
            text += '<|start_header_id|>assistant<|end_header_id|>\n\n' + '\n'.join(t.split('\n')[3:]) + '<|eot_id|>'
            text += '<|end_of_text|>'
            formatted_texts.append(text)
        return formatted_texts
    
    def _process_format_mistral(self, full_texts):
        formatted_texts = []
        for t in full_texts:
            text = ''
            text += '<s>[INST] ' + t.split('\n')[0] + ' [/INST]\n'
            text += '\n'.join(t.split('\n')[1:])
            text += '</s>'
            formatted_texts.append(text)
        return formatted_texts
    
    def _process_format_gemma(self, full_texts):
        formatted_texts = []
        for t in full_texts:
            text = ''
            text += '<bos><start_of_turn>user\n' + '\n'.join(t.split('\n')[:3]) + '<end_of_turn>\n'
            text += '<start_of_turn>model\n' + '\n'.join(t.split('\n')[3:]) + '<end_of_turn>'
            formatted_texts.append(text)
        return formatted_texts
    
    def _process_format_zephyr(self, full_texts):
        formatted_texts = []
        for t in full_texts:
            text = ''
            text += '<|user|>\n' + '\n'.join(t.split('\n')[:3]) + '</s>\n'
            text += '<|assistant|>\n' + '\n'.join(t.split('\n')[3:]) + '</s>'
            formatted_texts.append(text)
        return formatted_texts

    def __len__(self):
        return len(self.full_texts)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.full_texts[idx],
                                truncation=True,
                                max_length=self.max_length,
                                padding='max_length',
                                return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()
        inputs = inputs.to(self.device)
        return inputs