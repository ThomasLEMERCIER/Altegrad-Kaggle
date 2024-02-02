# Standard library imports
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

# Related third-party imports
import torch
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import Data, HeteroData
from torch_geometric.data import Dataset
from torch.utils.data import Dataset as TorchDataset

# Local application/library specific imports
from src.preprocessing import process_graph, process_text


class GraphTextDataset(Dataset):
    """
    Dataset for the graph and text data
    """

    def __init__(
        self,
        root,
        gt,
        split,
        tokenizer,
        nlp_model,
        in_memory=True,
        transform=None,
        transform_params=None,
    ):
        self.root = root
        self.gt = gt
        self.split = split
        self.nlp_model = nlp_model
        self.tokenizer = tokenizer
        self.in_memory = in_memory
        self.data_transform = (
            transform  # not overwriting the transform method of Dataset
        )
        self.data_transform_params = (
            transform_params  # not overwriting the transform method of Dataset
        )
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
            if self.data_transform is not None:
                return self.data_transform(
                    self.data[idx].clone(), self.data_transform_params
                )
            return self.data[idx]
        else:
            cid = self.cids[idx]
            data = torch.load(osp.join(self.preprocessed_dir, "data_{}.pt".format(cid)))
            if self.data_transform is not None:
                return self.data_transform(data, self.data_transform_params)
            return data


class GraphDataset(Dataset):
    def __init__(self, root, gt, split, transform=None, transform_params=None):
        self.root = root
        self.gt = gt
        self.split = split
        self.data_transform = transform
        self.data_transform_params = transform_params

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

        if self.data_transform is not None:
            return self.data_transform(data, self.data_transform_params)
        return data


class TextDataset(TorchDataset):
    def __init__(
        self,
        root,
        test_file,
        tokenizer,
        nlp_model,
        transform=None,
        transform_params=None,
    ):
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

        self.data_transform = transform
        self.data_transform_params = transform_params

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

        if self.data_transform is not None:
            return self.data_transform(data, self.data_transform_params)
        return data

class GraphPretrainingDataset(Dataset):
    def __init__(self, root, gt, transform, transform_params, in_memory=True):
        self.root = root
        self.gt = gt
        self.transform_data = transform
        self.transform_data_params = transform_params
        self.in_memory = in_memory

        self.preprocessed_dir = osp.join(self.root, "preprocessed", "all_graph")
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            self.preprocess()

        self.size = len([file for file in os.listdir(self.preprocessed_dir) if file.endswith(".pt")])
        if self.in_memory:
            self.data = []
            for idx in tqdm(range(self.len()), desc="Loading data"):
                self.data.append(
                    torch.load(
                        osp.join(self.preprocessed_dir, "data_{}.pt".format(idx))
                    )
                )

        super(GraphPretrainingDataset, self).__init__()

    def preprocess(self):
        num_workers = os.cpu_count()
        files = [file for file in os.listdir(osp.join(self.root, "raw")) if file.endswith(".graph")]
        files = [(file, idx) for idx, file in enumerate(files)]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for file_idx in tqdm(files, desc="Preprocessing"):
                future = executor.submit(self.process_single, file_idx)
                futures.append(future)

            # Wait for all futures to complete
            for future in tqdm(futures, desc="Processing Complete", total=len(futures)):
                future.result()

    def process_single(self, file_idx):
        raw_path = osp.join(self.root, "raw", file_idx[0])
        edge_index, x = process_graph(raw_path, self.gt)
        data = Data(
            x=x,
            edge_index=edge_index,
        )
        torch.save(data, osp.join(self.preprocessed_dir, "data_{}.pt".format(file_idx[1])))

    def len(self):
        return self.size

    def get(self, idx):
        if self.in_memory:

            data = self.data[idx].clone()
            u_x, u_edge_index = self.transform_data(data.x, data.edge_index, self.transform_data_params)
            data = self.data[idx].clone()
            v_x, v_edge_index = self.transform_data(data.x, data.edge_index, self.transform_data_params)

            data_u = Data(
                x=u_x,
                edge_index=u_edge_index,
            )

            data_v = Data(
                x=v_x,
                edge_index=v_edge_index,
            )

            return data_u, data_v
        else:
            data = torch.load(osp.join(self.preprocessed_dir, "data_{}.pt".format(idx)))

            data_clone = data.clone()
            u_x, u_edge_index = self.transform_data(data_clone.x, data_clone.edge_index, self.transform_data_params)
            data_clone = data.clone()
            v_x, v_edge_index = self.transform_data(data_clone.x, data_clone.edge_index, self.transform_data_params)

            data_u = Data(
                x=u_x,
                edge_index=u_edge_index,
            )

            data_v = Data(
                x=v_x,
                edge_index=v_edge_index,
            )

            return data_u, data_v

class MultiDataset(Dataset):
    """
    Dataset for the (graph, text) pairs and graph only
    """

    def __init__(
        self,
        root,
        gt,
        split,
        tokenizer,
        nlp_model,
        in_memory=True,
        transform=None,
        transform_params=None,
    ):
        self.root = root
        self.gt = gt
        self.split = split
        self.nlp_model = nlp_model
        self.tokenizer = tokenizer
        self.in_memory = in_memory
        self.data_transform = (
            transform  # not overwriting the transform method of Dataset
        )
        self.data_transform_params = (
            transform_params  # not overwriting the transform method of Dataset
        )
        self.description = (
            pd.read_csv(osp.join(self.root, split + ".tsv"), sep="\t", header=None)
            .set_index(0)[1]
            .to_dict()
        )
        self.cids = list(self.description.keys())

        self.preprocessed_dir = osp.join(
            self.root, "preprocessed", "multi", self.nlp_model, self.split
        )
        if not os.path.exists(self.preprocessed_dir):
            os.makedirs(self.preprocessed_dir)
            self.preprocess()
        
        self.size = len([file for file in os.listdir(self.preprocessed_dir) if file.endswith(".pt")])
        self.labeled_size = len(self.cids)
        if self.in_memory:
            self.data = []
            for idx in tqdm(range(self.len()), desc="Loading data"):
                self.data.append(
                    torch.load(
                        osp.join(self.preprocessed_dir, "data_{}.pt".format(idx))
                    )
                )

        super(MultiDataset, self).__init__()

    def preprocess(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        num_workers = os.cpu_count()
        all_graphs = [file for file in os.listdir(osp.join(self.root, "raw")) if file.endswith(".graph")]
        only_graphs = list(filter(lambda x: x.split(".")[0] in self.cids, all_graphs))
        files_to_process = [(cid, idx, True) for idx, cid in enumerate(self.cids)] + [(file, idx+len(self.cids), False) for idx, file in enumerate(only_graphs)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []

            for file_info in tqdm(files_to_process, desc="Preprocessing"):
                future = executor.submit(self.process_single, file_info)
                futures.append(future)

            # Wait for all futures to complete
            for future in tqdm(futures, desc="Processing Complete", total=len(futures)):
                future.result()
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def process_single(self, file_info):
        if file_info[2]:
            cid = file_info[0]
            raw_path = osp.join(self.root, "raw", str(cid) + ".graph")
            edge_index, x = process_graph(raw_path, self.gt)
            text = self.description[cid]
            input_ids, attention_mask = process_text(text, self.tokenizer)
            data = Data(
                x=x,
                edge_index=edge_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                has_text=True
            )
            torch.save(data, osp.join(self.preprocessed_dir, "data_{}.pt".format(file_info[1])))
        else:
            raw_path = osp.join(self.root, "raw", file_info[0])
            edge_index, x = process_graph(raw_path, self.gt)
            data = Data(
                x=x,
                edge_index=edge_index,
                attention_mask=None,
                input_ids=None,
                has_text=False
            )
            torch.save(data, osp.join(self.preprocessed_dir, "data_{}.pt".format(file_info[1])))

    def len(self):
        return self.size

    def get(self, idx):
        if self.in_memory:
            if self.data_transform is not None:
                data = self.data[idx].clone()
                has_text = data.has_text
                data = Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    input_ids=data.input_ids,
                    attention_mask=data.attention_mask,
                )
                data = self.data_transform(data, self.data_transform_params)
                return Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    input_ids=data.input_ids,
                    attention_mask=data.attention_mask,
                    has_text=has_text
                )
            return self.data[idx]
        else:
            data = torch.load(osp.join(self.preprocessed_dir, "data_{}.pt".format(idx)))

            if self.data_transform is not None:
                has_text = data.has_text
                data = Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    input_ids=data.input_ids,
                    attention_mask=data.attention_mask,
                )
                data = self.data_transform(data, self.data_transform_params)
                return Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    input_ids=data.input_ids,
                    attention_mask=data.attention_mask,
                    has_text=has_text
                )
            return data
