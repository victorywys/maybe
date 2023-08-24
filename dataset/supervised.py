from .base import DATASET

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler

import random

from tqdm import tqdm

import os
import glob
import pandas as pd
from itertools import chain
import gc


@DATASET.register_module("supervised")
class SupervisedDataset(Dataset):
    def __init__(
            self, 
            root_dir: str, 
            split: str = "train", 
            debug: bool = False
        ):
        self.data_files = list(glob.glob(os.path.join(root_dir, split, "*", "**", "*.pkl"), recursive=True))
        if debug:
            self.data_files = self.data_files[:2]
        self.file_lengths = {}
        print("init dataset ...")
        if os.path.isfile(os.path.join(root_dir, split, "file_lengths.pkl")):
            self.file_lengths = pd.read_pickle(os.path.join(root_dir, split, "file_lengths.pkl"))
        for fn in tqdm(self.data_files):
            if fn in self.file_lengths:
                continue
            a = pd.read_pickle(fn)
            self.file_lengths[fn] = a["self_info"].shape[0]
            del a
            pd.to_pickle(self.file_lengths, os.path.join(root_dir, split, "file_lengths.pkl"))
        
        self.build_start_end_idx()
        self.file_cache = {}
        self.CACHE_SIZE = 5
        self.RECORD_PAD = torch.zeros(1, 55)
        
    def build_start_end_idx(self):
        self.start_end_idx = []
        start = 0
        for fn in self.data_files:
            end = start + self.file_lengths[fn]
            self.start_end_idx.append((start, end))
            start = end

    @property
    def file_num(self):
        return len(self.data_files)
    
    @property
    def data_num(self):
        return self.start_end_idx[-1][1]

    def suffle_file_name(self):
        random.shuffle(self.data_files)
        self.build_start_end_idx()
        self.file_cache = {}
        gc.collect()


    def get_item(self, data, data_id):
        return (
            torch.Tensor(data["self_info"][data_id]), 
            torch.cat([self.RECORD_PAD, torch.Tensor(data["record"][data_id])], 0),
            torch.Tensor(data["global_info"][data_id]), 
            data["action"][data_id]
        )

    def __getitem__(self, idx):
        for i, (start, end) in enumerate(self.start_end_idx):
            if start <= idx < end:
                file_id = i
                break
        data_id = idx - self.start_end_idx[file_id][0]

        if file_id not in self.file_cache:
            self.file_cache[file_id] = pd.read_pickle(self.data_files[file_id])
            # delete the earliest file
            if len(self.file_cache) > self.CACHE_SIZE:
                del self.file_cache[min(self.file_cache.keys())]
            
        return self.get_item(self.file_cache[file_id], data_id)

    def __len__(self):
        return self.data_num


class RandomPerm():
    def __init__(self, l, r):
        self.l = l
        self.r = r

    def __iter__(self):
        self.permutation = list(range(self.l, self.r))
        random.shuffle(self.permutation)
        return iter(self.permutation)


class MultiFileSampler(Sampler):
    def __init__(self, dataset: SupervisedDataset, random: bool = True):
        self.dataset = dataset
        self.file_num = dataset.file_num
        self.random = random

    def __iter__(self):
        if self.random:
            self.dataset.suffle_file_name()
            return chain(*[RandomPerm(start, end) for start, end in self.dataset.start_end_idx])
        else:
            return iter(range(self.dataset.data_num))

    def __len__(self):
        return len(self.dataset)


def supervised_collate_fn(batch):
    self_info, record, global_info, action = zip(*batch)
    # pad and pack record
    record_lengths = [r.size(0) for r in record]
    sorted_lengths, sorted_idx = torch.sort(torch.Tensor(record_lengths), descending=True)
    sorted_record = [record[i] for i in sorted_idx]
    sorted_self_info = [self_info[i] for i in sorted_idx]
    sorted_global_info = [global_info[i] for i in sorted_idx]
    sorted_action = [action[i] for i in sorted_idx]

    padded_record = nn.utils.rnn.pad_sequence(sorted_record, batch_first=True)
    packed_record = nn.utils.rnn.pack_padded_sequence(padded_record, sorted_lengths, batch_first=True)

    return (
        torch.stack(sorted_self_info), 
        packed_record, 
        torch.stack(sorted_global_info), 
        torch.LongTensor(sorted_action),
    )