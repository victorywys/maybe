from .base import DATASET

import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler, get_worker_info

import random

from tqdm import tqdm

import os
import glob
import pandas as pd
from itertools import chain
import gc

import threading 
import time

def preload_file(file_ids, file_names, dataset, lock):
    for idx in file_ids:
        with lock:
            if idx >= len(file_names) or idx in dataset.file_cache:
                continue
            print("caching idx", idx)
            file_name = file_names[idx]
            dataset.file_cache[idx] = pd.read_pickle(file_name)
            # delete the earliest file
            if len(dataset.file_cache) > dataset.CACHE_SIZE:
                del dataset.file_cache[min(dataset.file_cache.keys())]
        time.sleep(0.1) # allow main thread to load data from cached file
    

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
        self.last_file_id = -1
        self.CACHE_SIZE = 3
        self.cache_lock = threading.Lock()
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

    def shuffle_file_name(self):
        random.shuffle(self.data_files)
        self.build_start_end_idx()
        self.file_cache = {}
        self.last_file_id = -1
        gc.collect()


    def get_item(self, data, data_id):
        return (
            torch.Tensor(data["self_info"][data_id]), 
            torch.cat([self.RECORD_PAD, torch.Tensor(data["record"][data_id])], 0),
            torch.Tensor(data["global_info"][data_id]), 
            data["action"][data_id]
        )

    def __getitem__(self, idx):
        new_file = False
        if self.last_file_id < 0:
            file_id = self.last_file_id = 0
            new_file = True
        else:
            while self.last_file_id < len(self.start_end_idx) - 1 and idx >= self.start_end_idx[self.last_file_id + 1][0]:
                self.last_file_id += 1
                new_file = True
            file_id = self.last_file_id
        
        next_file_id_start = file_id
        next_file_id_end = file_id + self.CACHE_SIZE

        data_id = idx - self.start_end_idx[file_id][0]

        if not (min(next_file_id_end, len(self.start_end_idx) - 1) in self.file_cache) and (new_file):
            thread = threading.Thread(target=preload_file, args=(list(range(next_file_id_start, next_file_id_end)), self.data_files, self, self.cache_lock))
            thread.start()

        if file_id not in self.file_cache:
            with self.cache_lock:
                if file_id not in self.file_cache:
                    self.file_cache[file_id] = pd.read_pickle(self.data_files[file_id])
            
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
            self.dataset.shuffle_file_name()
            return chain(*[RandomPerm(start, end) for start, end in self.dataset.start_end_idx])
        else:
            self.dataset.file_cache = {}
            gc.collect()
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