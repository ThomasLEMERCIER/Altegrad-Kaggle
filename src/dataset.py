# Standard library imports
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

# Related third-party imports
import torch
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch.utils.data import Dataset as TorchDataset

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
        self.description = (
            pd.read_csv(osp.join(self.root, split + ".tsv"), sep="\t", header=None)
            .set_index(0)[1]
            .to_dict()
        )
        self.cids = list(self.description.keys())

        self.preprocessed_dir = osp.join(
            self.root, "preprocessed", self.nlp_model, self.split
        )
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            self.preprocess()

        if self.in_memory:
            self.data = []
            for cid in tqdm(self.cids, desc="Loading data"):
                self.data.append(
                    torch.load(
                        osp.join(self.preprocessed_dir, "data_{}.pt".format(cid))
                    )
                )

        super(GraphTextDataset, self).__init__()

    def preprocess(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        num_workers = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for cid in tqdm(self.cids, desc="Preprocessing"):
                future = executor.submit(self.process_single, cid)
                futures.append(future)

            # Wait for all futures to complete
            for future in tqdm(futures, desc="Processing Complete", total=len(futures)):
                future.result()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def process_single(self, cid):
        raw_path = osp.join(self.root, "raw", str(cid) + ".graph")
        edge_index, x = process_graph(raw_path, self.gt)
        text = self.description[cid]
        input_ids, attention_mask = process_text(text, self.tokenizer)
        data = Data(
            x=x,
            edge_index=edge_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        torch.save(data, osp.join(self.preprocessed_dir, "data_{}.pt".format(cid)))

    def len(self):
        return len(self.cids)

    def get(self, idx):
        if self.in_memory:
            return self.data[idx]
        else:
            cid = self.cids[idx]
            data = torch.load(osp.join(self.preprocessed_dir, "data_{}.pt".format(cid)))
            return data


class GraphDataset(Dataset):
    def __init__(self, root, gt, split):
        self.root = root
        self.gt = gt
        self.split = split
        self.description = pd.read_csv(
            osp.join(self.root, split + ".txt"), sep="\t", header=None
        )

        self.cids = self.description.iloc[:, 0].tolist()

        self.preprocessed_dir = osp.join(self.root, "preprocessed", "test_graph")
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            self.preprocess()

        super(GraphDataset, self).__init__()

    def preprocess(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        num_workers = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for cid in tqdm(self.cids, desc="Preprocessing"):
                future = executor.submit(self.process_single, cid)
                futures.append(future)

            # Wait for all futures to complete
            for future in tqdm(futures, desc="Processing Complete", total=len(futures)):
                future.result()

        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def process_single(self, cid):
        raw_path = osp.join(self.root, "raw", str(cid) + ".graph")
        edge_index, x = process_graph(raw_path, self.gt)
        data = Data(
            x=x,
            edge_index=edge_index,
        )
        torch.save(data, osp.join(self.preprocessed_dir, "data_{}.pt".format(cid)))

    def len(self):
        return len(self.cids)

    def get(self, idx):
        cid = self.cids[idx]
        data = torch.load(osp.join(self.preprocessed_dir, "data_{}.pt".format(cid)))

        return data


class TextDataset(TorchDataset):
    def __init__(self, root, test_file, tokenizer, nlp_model):
        self.tokenizer = tokenizer
        self.nlp_model = nlp_model
        self.root = root
        self.test_file = test_file + ".txt"
        self.split = "test"
        self.preprocessed_dir = osp.join(
            self.root, "preprocessed", self.nlp_model, self.split
        )
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            self.preprocess()
        else:
            self.length = len(os.listdir(self.preprocessed_dir))

        super(TextDataset, self).__init__()

    def load_sentences(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        return [line.strip() for line in lines]

    def preprocess(self):
        self.sentences = self.load_sentences(osp.join(self.root, self.test_file))
        self.length = len(self.sentences)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        num_workers = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for index in tqdm(range(self.length), desc="Preprocessing"):
                future = executor.submit(self.process_single, index)
                futures.append(future)

            # Wait for all futures to complete
            for future in tqdm(futures, desc="Processing Complete", total=len(futures)):
                future.result()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def process_single(self, index):
        text = self.sentences[index]
        input_ids, attention_mask = process_text(text, self.tokenizer)

        data = {
            "input_ids": input_ids.squeeze(),
            "attention_mask": attention_mask.squeeze(),
        }

        torch.save(data, osp.join(self.preprocessed_dir, "data_{}.pt".format(index)))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = torch.load(osp.join(self.preprocessed_dir, "data_{}.pt".format(idx)))

        return data
