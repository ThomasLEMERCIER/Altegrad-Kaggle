# Standard library imports
import os
from concurrent.futures import ThreadPoolExecutor
import os.path as osp

# Related third-party imports
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data

# Local application/library specific imports
from src.preprocessing import process_graph, process_text

class GraphTextDataset(Dataset):
    """
    Dataset for the graph and text data
    """
    def __init__(self, root, gt, split, tokenizer, nlp_model, in_memory=True):
        self.root = root
        self.gt = gt
        self.split = split
        self.nlp_model = nlp_model
        self.tokenizer = tokenizer
        self.in_memory = in_memory
        self.description = pd.read_csv(osp.join(self.root, split + '.tsv'), sep='\t', header=None).set_index(0)[1].to_dict()
        self.cids = list(self.description.keys())

        self.preprocessed_dir = osp.join(self.root, 'preprocessed', self.nlp_model, self.split)
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            self.preprocess()

        if self.in_memory:
            self.data = []
            for cid in tqdm(self.cids, desc='Loading data'):
                self.data.append(torch.load(osp.join(self.preprocessed_dir, 'data_{}.pt'.format(cid))))

        super(GraphTextDataset, self).__init__()

    def preprocess(self):
        num_workers = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for cid in tqdm(self.cids, desc='Preprocessing'):
                future = executor.submit(self.process_single, cid)
                futures.append(future)

            # Wait for all futures to complete
            for future in tqdm(futures, desc='Processing Complete', total=len(futures)):
                future.result()

    def process_single(self, cid):
        raw_path = osp.join(self.root, 'raw', str(cid) + '.graph')
        edge_index, x = process_graph(raw_path, self.gt)
        text = self.description[cid]
        input_ids, attention_mask = process_text(text, self.tokenizer)
        data = Data(x=x, edge_index=edge_index, input_ids=input_ids, attention_mask=attention_mask)
        torch.save(data, osp.join(self.preprocessed_dir, 'data_{}.pt'.format(cid)))

    def len(self):
        return len(self.cids)
    
    def get(self, idx):
        if self.in_memory:
            return self.data[idx]
        else:
            cid = self.cids[idx]
            data = torch.load(osp.join(self.preprocessed_dir, 'data_{}.pt'.format(cid)))
            return data
