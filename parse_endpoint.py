from glob import glob
import os
import numpy as np
import torch
from utils.dump_helper import dump_results
from igibson.model_util_igbson import IGbsonDatasetConfig


if __name__ == '__main__':
    end_points = {}
    all_filename = glob(os.path.join("./endpoint/"+"*.npy"))
    for i in range(len(all_filename)):
        key = all_filename[i].split("/")[-1][:-4]
        try:
            end_points[key] = torch.from_numpy(np.load(all_filename[i],allow_pickle=True))
        except:
            print("can not load ",key)
            pass

    DATASET_CONFIG = IGbsonDatasetConfig()
    prefix = 'last_'

    DUMP_DIR = "./dump_dir"
    dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

