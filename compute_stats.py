import os
import numpy as np
from tqdm import tqdm

from src.constants import GT_PATH, NODE_FEATURES_SIZE

if __name__ == "__main__":
    gt = np.load(GT_PATH, allow_pickle=True)[()] # id: node features
    n_ids = len(gt)

    features = np.zeros((n_ids, NODE_FEATURES_SIZE))
    for k, id in enumerate(tqdm(gt.keys(), desc="Loading features")):
        features[k] = gt[id]

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    print("Mean: ", mean)
    print("Std: ", std)
