from typing import List

from .base import DATASET
from .supervised import SupervisedDataset

import torch
import torch.nn as nn
import threading
import time
import pandas as pd
import numpy as np

import glob
from tqdm import tqdm
import os


@DATASET.register_module("tenpai", inherit=True)
class TenpaiDataset(SupervisedDataset):
    def get_item(self, data, data_id):
        return (
            torch.Tensor(data["self_info"][data_id]),
            torch.cat([self.RECORD_PAD, torch.Tensor(data["record"][data_id])], 0),
            torch.Tensor(data["global_info"][data_id]),
            torch.Tensor(data["shimocha_tenpai"][data_id]),
            torch.Tensor(data["toimen_tenpai"][data_id]),
            torch.Tensor(data["kamicha_tenpai"][data_id]),
        )
    
    @property
    def default_collate_fn(self):
        return tenpai_collate_fn
    

TENPAI_DIM = 34
    
def tenpai_collate_fn(batch):
    self_info, record, global_info, shimocha_tenpai, toimen_tenpai, kamicha_tenpai = zip(*batch)
    # pad and pack record
    record_lengths = [r.size(0) for r in record]
    sorted_lengths, sorted_idx = torch.sort(torch.Tensor(record_lengths), descending=True)
    sorted_record = [record[i] for i in sorted_idx]
    sorted_self_info = [self_info[i] for i in sorted_idx]
    sorted_global_info = [global_info[i] for i in sorted_idx]
    
    
    def generate_tenpai_vec(shimocha, toimen, kamicha):
        return torch.cat([torch.any(shimocha, 0, True), shimocha, torch.any(toimen, 0, True), toimen, torch.any(kamicha, 0, True), kamicha])
    
    def generate_tenpai_mask(shimocha, toimen, kamicha):
        return torch.BoolTensor([True, *([torch.any(shimocha)] * TENPAI_DIM), True, *([torch.any(toimen)] * TENPAI_DIM), True, *([torch.any(kamicha)] * TENPAI_DIM)])
    
    
    sorted_tenpai = [generate_tenpai_vec(shimocha_tenpai[i], toimen_tenpai[i], kamicha_tenpai[i]) for i in sorted_idx]
    sorted_tenpai_mask = [generate_tenpai_mask(shimocha_tenpai[i], toimen_tenpai[i], kamicha_tenpai[i]) for i in sorted_idx]

    padded_record = nn.utils.rnn.pad_sequence(sorted_record, batch_first=True)
    packed_record = nn.utils.rnn.pack_padded_sequence(padded_record, sorted_lengths, batch_first=True)

    return (
        torch.stack(sorted_self_info),
        packed_record,
        torch.stack(sorted_global_info),
        torch.stack(sorted_tenpai),
        torch.stack(sorted_tenpai_mask),
    )
