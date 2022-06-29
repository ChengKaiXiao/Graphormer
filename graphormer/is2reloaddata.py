# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Sequence, Union

import pickle
from functools import lru_cache

import lmdb

import numpy as np
import torch
from torch import Tensor
# from fairseq.data import (
#     FairseqDataset,
#     BaseWrapperDataset,
#     NestedDictionaryDataset,
#     data_utils,
# )
# from fairseq.tasks import FairseqTask, register_task


class LMDBDataset:
    """
    Input dataset

    pos:
    atom: tensor size [N]
        atomic_numbers indicate atom type
    cell: tensor size [1, 3, 3]
        three lattice vectors 
    """
    def __init__(self, db_path):
        super().__init__()
        assert Path(db_path).exists(), f"{db_path}: No such file or directory"
        self.env = lmdb.Environment(
            db_path,
            map_size=(1024 ** 3) * 256,
            subdir=False,
            readonly=True,
            readahead=True,
            meminit=False,
        )
        self.len: int = self.env.stat()["entries"]

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.len:
            raise IndexError
        data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
        return dict(
            pos=torch.as_tensor(data["pos"]).float(),
            pos_relaxed=torch.as_tensor(data["pos_relaxed"]).float(),
            cell=torch.as_tensor(data["cell"]).float().view(3, 3),
            atoms=torch.as_tensor(data["atomic_numbers"]).long(),
            tags=torch.as_tensor(data["tags"]).long(),
            relaxed_energy=data["y_relaxed"],  # python float
        )

db_path = "/home/victor/Downloads/is2re_test_challenge_2021/data.lmdb"
db_path = "/home/victor/Downloads/OCP/is2res_train_val_test_lmdbs/data/is2re/all/10k/train/data.lmdb"

env = lmdb.Environment(
            db_path,
            map_size=(1024 ** 3) * 256,
            subdir=False,
            readonly=True,
            readahead=True,
            meminit=False,
        )
len = env.stat()["entries"]
print(len)
idx = 1
data = pickle.loads(env.begin().get(f"{idx}".encode()))
print(data)
# data =  LMDBDataset(db_path)
# print(data[1])
