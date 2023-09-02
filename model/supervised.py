from typing import Optional

from pathlib import Path

from tqdm import tqdm

import time

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer  import SummaryWriter
from torch.utils.data import DataLoader
from utilsd import use_cuda

from .base import MODEL

from dataset import supervised_collate_fn, MultiFileSampler

import os

@MODEL.register_module()
class SupervisedMahjong(nn.Module):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        num_worker: int = 8,
        max_epochs: int = 100,
        save_interval: Optional[int] = None,
        network: Optional[nn.Module] = None,
        output_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        super(SupervisedMahjong, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_worker
        self.max_epochs = max_epochs
        self.network = network
        self.save_interval = save_interval
        self._init_optimization(lr, weight_decay)
        self._init_logger(output_dir)
        self.checkpoint_dir = checkpoint_dir
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if use_cuda():
            print("Using GPU")
            self.cuda()

    def _init_logger(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.flush()

    def _init_optimization(self, lr, weight_decay):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, self_info, record, global_info):
        return self.network(self_info, record, global_info)

    def fit(self, trainset, validset=None):
        loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            sampler=MultiFileSampler(trainset),
            collate_fn=supervised_collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        start_epoch, best_res = self._resume()
        
        for epoch in range(start_epoch, self.max_epochs):
            self.train()
            start_time = time.time()
            total_loss = 0.
            total_num = 0.
            total_correct = 0.
            
            with tqdm(total=len(loader), desc=f"Epoch {epoch}") as pbar:
                for batch_i, (self_info, record, global_info, label) in enumerate(loader):
                    if use_cuda():
                        self_info = self_info.cuda()
                        record = record.cuda()
                        global_info = global_info.cuda()
                        label = label.cuda()
                    self.optimizer.zero_grad()
                    output = self(self_info, record, global_info)
                    loss = self.loss_fn(output, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item() * self_info.size(0)
                    total_num += self_info.size(0)
                    total_correct += (output.argmax(dim=1) == label).sum().item()
                    pbar.set_postfix_str(f"loss: {loss.item():.4f}, acc: {total_correct / total_num:.4f}")
                    pbar.update(1)
                    if (batch_i + 1) % self.save_interval == 0:
                        self._checkpoint(epoch, best_res)
                
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

                if valid_summary["acc"] > best_res.get("acc", 0.):
                    best_res = valid_summary
                    best_res["best_epoch"] = epoch
                    self.best_params = self.state_dict()
                    self.best_network_params = self.network.state_dict()
                    torch.save(self.best_params, f"{self.checkpoint_dir}/model_best.pkl")
                    torch.save(self.best_network_params, f"{self.checkpoint_dir}/network_best.pkl")
            self._checkpoint(epoch, best_res)
            
                
    def evaluate(self, evalset):
        self.eval()
        loader = DataLoader(
            evalset,
            batch_size=self.batch_size,
            sampler=MultiFileSampler(evalset, random=False),
            collate_fn=supervised_collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        start_time = time.time()
        total_num = 0.
        total_correct = 0.
        with tqdm(total=len(loader), desc="Evaluation") as pbar:
            for _, (self_info, record, global_info, label) in enumerate(loader):
                if use_cuda():
                    self_info = self_info.cuda()
                    record = record.cuda()
                    global_info = global_info.cuda()
                    label = label.cuda()
                output = self(self_info, record, global_info)
                total_num += self_info.size(0)
                total_correct += (output.argmax(dim=1) == label).sum().item()
                pbar.set_postfix_str(f"acc: {total_correct / total_num:.4f}")
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
            sampler=MultiFileSampler(evalset, random=False),
            collate_fn=supervised_collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
        )
        output_list = []
        with tqdm(total=len(loader), desc="Prediction") as pbar:
            for _, (self_info, record, global_info, label) in enumerate(loader):
                if use_cuda():
                    self_info = self_info.cuda()
                    record = record.cuda()
                    global_info = global_info.cuda()
                    label = label.cuda()
                output = self(self_info, record, global_info)
                output_list.append(output)
                pbar.update(1)
        return torch.cat(output_list, dim=0)
    

    def _checkpoint(self, cur_epoch, best_res, checkpoint_dir=None):
        torch.save(
            {
                "model": self.state_dict(),
                "optim": self.optimizer.state_dict(),
                "epoch": cur_epoch,
                "best_res": best_res,
                "best_params": self.best_params,
                "best_network_params": self.best_network_params,
            },
            self.checkpoint_dir / "resume.pth" if checkpoint_dir is None else checkpoint_dir / "resume.pth",
        )
        print(f"Checkpoint saved to {self.checkpoint_dir / 'resume.pth' if checkpoint_dir is None else checkpoint_dir / 'resume.pth'}", __name__)

    def _resume(self):
        if (self.checkpoint_dir / "resume.pth").exists():
            print(f"Resume from {self.checkpoint_dir / 'resume.pth'}", __name__)
            checkpoint = torch.load(self.checkpoint_dir / "resume.pth")
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optim"])
            self.best_params = checkpoint["best_params"]
            self.best_network_params = checkpoint["best_network_params"]
            return checkpoint["epoch"], checkpoint["best_res"]
        else:
            print(f"No checkpoint found in {self.checkpoint_dir}", __name__)
            self.best_params = self.state_dict()
            self.best_network_params = self.network.state_dict()
            return 0, {}

