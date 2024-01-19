from typing import Optional

from pathlib import Path

from tqdm import tqdm
import numpy as np

import time

import torch
import torch.nn as nn
from torch.utils.tensorboard.writer  import SummaryWriter
from torch.utils.data import DataLoader
from utilsd import use_cuda

from .base import MODEL

from dataset import supervised_collate_fn, MultiFileSampler

import os
from copy import deepcopy

@MODEL.register_module()
class RLMahjong(nn.Module):
    def __init__(
        self,
        # batch_size: int = 256,
        actor_network: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        output_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        # ranking_rewards: Optional[list] = [45, 5, -15, -35],
    ):
        # self.batch_size = batch_size
        super(RLMahjong, self).__init__()

        self.value_network = value_network
        self.actor_network = actor_network
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) # TODO
        self.target_value_network = deepcopy(value_network)

        self.update_times = 0
    
    def _init_logger(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.writer.flush()

    def update(self, buffer):

        self.train()

        # sample a batch of data from replay buffer
        batch = buffer.sample_batch()
        
        self_infos, others_infos, records, global_infos, actions, action_masks, rewards, dones, lengths = batch

        max_len = int(torch.max(lengths).item())

        self_infos = self_infos[:, :max_len + 1]
        others_infos = others_infos[:, :max_len + 1]
        # records = records[:, :max_len + 1] Records is already PackedSequence
        global_infos = global_infos[:, :max_len + 1]
        actions = actions[:, :max_len]
        action_masks = action_masks[:, :max_len]
        rewards = rewards[:, :max_len]
        dones = dones[:, :max_len]


        # for tmp in [self_infos, others_infos, global_infos]:
        #     tmp = tmp[:, :max_len + 1].clone()
        
        # for tmp in [actions, action_masks, rewards, dones]:
        #     tmp = tmp[:, :max_len].clone()

        # -------- update value network ------------

        # include oracle information to better estimate value function (asymmetric actor critic)
        # hand_infos = torch.cat([self_infos, others_infos], dim=-1)
        
        v = self.value_network(self_infos, others_infos, records, global_infos)

        with torch.no_grad():
            v_tar = self.target_value_network(self_infos, others_infos, records, global_infos).detach()

        # TODO: modify algorithm
        v_grape = rewards + (1 - dones) * v_tar[:, 1:]
        
        advantage = v_grape - v[:, :-1]

        loss_c = advantage.pow(2).mean()

        # -------- update actor network ------------

        logits = self.actor_network(self_infos[:, :-1], records[:, :-1], global_infos[:, :-1])

        # policy gradient algorithm to update actor network
        loss_a = - advantage * torch.log_softmax(logits, dim=-1) * action_masks

        loss_a = loss_a.sum(dim=-1).mean()
        
        # --------- update model parameters ------------
        loss = loss_a + loss_c
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # -------- update target value network ------------
        if self.update_times % 1000 == 0:
            self.target_value_network.load_state_dict(self.value_network.state_dict())

        self.update_times += 1

        self.eval()

        return loss


    def select_action(self, self_info, record, global_info, action_mask, temp=1) -> int:
        
        self.eval()

        logits = self.actor_network(self_info, record, global_info)
        policy_prob = (torch.softmax(temp * logits[0], dim=-1) * action_mask.reshape[-1]).cpu().detach().numpy()

        a = np.random.choice(54, p=policy_prob)

        return a

        
    def _checkpoint(self, checkpoint_dir=None):
        torch.save(
            {
                "model": self.state_dict(),
                "optim": self.optimizer.state_dict(),
                "update_times": self.update_times
            },
            self.checkpoint_dir / f"resume_{self.update_times}.pth" if checkpoint_dir is None else checkpoint_dir / f"resume_{self.update_times}.pth",
        )
        print(f"Checkpoint saved to {self.checkpoint_dir / f'resume_{self.update_times}.pth' if checkpoint_dir is None else checkpoint_dir / f'resume_{self.update_times}.pth'}", __name__)
        torch.save(
            {
                "model": self.state_dict(),
                "optim": self.optimizer.state_dict(),
                "update_times": self.update_times,
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
            self.update_times = checkpoint["update_times"]
            
        else:
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}", __name__)

            
