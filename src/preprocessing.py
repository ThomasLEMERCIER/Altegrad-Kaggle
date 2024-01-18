import torch
import numpy as np

def process_graph(raw_path, gt):
    edge_index  = []
    x = []
    with open(raw_path, 'r') as f:
        next(f)
        for line in f: 
            if line != "\n":
                edge = *map(int, line.split()), 
                edge_index.append(edge)
            else:
                break
        next(f)
        for line in f: #get mol2vec features:
            substruct_id = line.strip().split()[-1]
            if substruct_id in gt.keys():
                x.append(gt[substruct_id])
            else:
                x.append(gt['UNK'])
    edge_index = np.array(edge_index).T
    x = np.array(x)
    return torch.LongTensor(edge_index), torch.FloatTensor(x)

def process_text(text, tokenizer):
    text_input = tokenizer([text],
                           return_tensors="pt", 
                           truncation=True, 
                           max_length=256,
                           padding="max_length",
                           add_special_tokens=True,)
    return text_input['input_ids'], text_input['attention_mask']
