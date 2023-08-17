from typing import Optional

from tqdm import tqdm

import time

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer  import SummaryWriter
from torch.utils.data import DataLoader
from utilsd import use_cuda

from .base import MODEL


@MODEL.register()
class SupervisedMahjong(nn.Module):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        num_worker: int = 8,
        max_epoch: int = 100,
        network: Optional[nn.Module] = None,
        output_dir: Optional[nn.Module] = None,
    ):
        self.batch_szie = batch_size
        self.num_workers = num_worker
        self.max_epoch = max_epoch
        self.network = network
        self._init_optimization(lr, weight_decay)
        self._init_logger(output_dir)
        if use_cuda():
            print("Using GPU")
            self.cuda()

    def _init_logger(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.flush()

    def _init_optimization(self, lr, weight_decay):
        self.loss_fn = nn.CrossEntropyLoss(),
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x):
        return self.network(x)

    def fit(self, trainset, validset=None):
        loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        for epoch in range(self.max_epochs):
            self.train()
            start_time = time.time()
            total_loss = 0.
            total_num = 0.
            total_correct = 0.
            
            with tqdm(total=len(loader), desc=f"Epoch {epoch}") as pbar:
                for _, (data, label) in enumerate(loader):
                    if use_cuda():
                        data = data.cuda()
                        label = label.cuda()
                    self.optimizer.zero_grad()
                    output = self(data)
                    loss = self.loss_fn(output, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item() * data.size(0)
                    total_num += data.size(0)
                    total_correct += (output.argmax(dim=1) == label).sum().item()
                    pbar.set_postfix_str(f"loss: {loss.item()}")
                    pbar.update(1)
                
            train_summary = {
                "time": time.time() - start_time,
                "loss": total_loss / total_num,
                "acc": total_correct / total_num,
            }
            for k, v in train_summary.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)
            self.writer.flush()
            
            print(f"Epoch {epoch} training summary:")
            print(f"\ttime: {train_summary['time']}")
            print(f"\ttrain loss: {train_summary['loss']}")
            print(f"\ttrain acc: {train_summary['acc']}")
            
            if validset is not None:
                valid_summary = self.evaluate(validset)
                for k, v in valid_summary.items():
                    self.writer.add_scalar(f"valid/{k}", v, epoch)
                self.writer.flush()
                
                print(f"Epoch {epoch} validation summary:")
                print(f"\ttime: {valid_summary['time']}")
                print(f"\tvalid acc: {valid_summary['acc']}")
            
                
    def evaluate(self, evalset):
        self.eval()
        loader = DataLoader(
            evalset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        start_time = time.time()
        total_num = 0.
        total_correct = 0.
        with tqdm(total=len(loader), desc="Evaluation") as pbar:
            for _, (data, label) in enumerate(loader):
                if use_cuda():
                    data = data.cuda()
                    label = label.cuda()
                output = self(data)
                total_num += data.size(0)
                total_correct += (output.argmax(dim=1) == label).sum().item()
                pbar.set_postfix_str(f"acc: {total_correct / total_num}")
                pbar.update(1)
        summary = {
            "time": time.time() - start_time,
            "acc": total_correct / total_num,
        }
        return summary


    def predict(self, evalset):
        self.eval()
        loader = DataLoader(
            evalset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        output_list = []
        with tqdm(total=len(loader), desc="Prediction") as pbar:
            for _, (data, label) in enumerate(loader):
                if use_cuda():
                    data = data.cuda()
                    label = label.cuda()
                output = self(data)
                output_list.append(output)
                pbar.update(1)
        return torch.cat(output_list, dim=0)