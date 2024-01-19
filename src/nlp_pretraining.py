import pandas as pd
import numpy as np
import os.path as osp
from torch.utils.data import Dataset


def get_corpus_generator(root, split):
    description = (
        pd.read_csv(osp.join(root, split + ".tsv"), sep="\t", header=None)
        .set_index(0)[1]
        .to_dict()
    )

    for cid in description:
        yield description[cid]


class MaskedDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.chunks[idx],
            "labels": self.chunks[idx],
            "attention_mask": np.ones_like(self.chunks[idx]),
        }

        return item


def _get_chunks(corpus_generator, tokenizer, chunk_size=128):
    max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 100000

    concatenated_corpus = [
        np.array(tokenizer(sample)["input_ids"]) for sample in corpus_generator
    ]
    concatenated_corpus = np.concatenate(concatenated_corpus)

    tokenizer.model_max_length = max_length

    chunks = [
        concatenated_corpus[i : i + chunk_size]
        for i in range(0, len(concatenated_corpus), chunk_size)
    ]
    chunks = chunks[:-1]  # drop last

    return chunks


def masked_training_dataset(root, split, tokenizer):
    corpus_generator = get_corpus_generator(root, split)

    chunks = _get_chunks(corpus_generator, tokenizer)

    return MaskedDataset(chunks)
