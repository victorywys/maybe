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


class GRAPE:
    """
    policy evaluation algorithm: 
    """

    def __init__(self, alpha, gamma, lambd):
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

    def compute_targets(self, q, pi, a_t, r_t, mu_t, done_t):
        """
        Compute G Q (s, a) + a A (s, a), and return a tuple of (G Q  (s, a) + a A  (s, a), A (s, \cdot))
        q_t   = s[:-1]
        q_tp1 = s[1:]
        """
        l = q.shape[0] - 1
        pi_t, pi_tp1 = pi[:-1], pi[1:]
        q_t, q_tp1 = q[:-1], q[1:]
        q_t_a = q_t[np.arange(l), a_t]
        v_t, v_tp1 = torch.sum(pi_t * q_t, dim=1), torch.sum(pi_tp1 * q_tp1, dim=1)
        q_t_a_est = r_t + (1. - done_t) * self.gamma * v_tp1
        td_error = q_t_a_est - q_t_a
        rho_t_a = pi_t[np.arange(l), a_t] / mu_t[np.arange(l), a_t]  # importance sampling ratios
        c_t_a = self.lambd * torch.clamp(rho_t_a, 0, 1)

        y_prime = 0  # y'_t
        g_q = torch.zeros([l]).to(q.device) 
        for u in reversed(range(l)):  # l-1, l-2, l-3, ..., 0
            # If s_tp1[u] is from an episode different from s_t[u], y_prime needs to be reset.
            y_prime = 0 if done_t[u] else y_prime  # y'_u
            g_q[u] = q_t_a_est[u] + y_prime

            # y'_{u-1} used in the next step
            y_prime = self.lambd * self.gamma * rho_t_a[u] * td_error[u] + self.gamma * c_t_a[
                u] * y_prime

        targets_q = g_q + self.alpha * (q_t_a - v_t)
        advantages = (1. - self.alpha) * (q_t - v_t.reshape([-1, 1]))

        return targets_q, advantages


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
        device: Optional[str] = "cuda",
        gamma: Optional[float] = 0.999,
    ):
        # self.batch_size = batch_size
        super(RLMahjong, self).__init__()

        self.value_network = value_network
        self.actor_network = actor_network
        self.gamma = gamma

        self.grape = GRAPE(alpha=0.99, gamma=gamma, lambd=0) # TODO: search
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4) # TODO
        self.target_value_network = deepcopy(value_network)

        self.update_times = 0

        self.mse_loss = nn.MSELoss()
        self.device = torch.device(device)
        self.to(device=self.device)
    
    def _init_logger(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.writer.flush()

    def update(self, buffer):

        self.train()

        # sample a batch of data from replay buffer
        batch = buffer.sample_contiguous_batch(num_seq=100)
        
        self_infos, others_infos, records, global_infos, actions, action_masks, policy_probs, rewards, dones, lengths = batch

        action_masks = action_masks.to(torch.float)
        action_masks = torch.zeros_like(action_masks) - (action_masks < 0.5).to(torch.float) * 100
        action_masks_p = torch.cat([action_masks, torch.zeros_like(action_masks[:1])], dim=0)

        # self.infos = self_infos.float()
        # others_infos = others_infos.float()
        # records = records.float()
        # max_len = int(torch.max(lengths).item())

        # self_infos = self_infos[:, :max_len + 1]
        # others_infos = others_infos[:, :max_len + 1]
        # # records = records[:, :max_len + 1] Records is already PackedSequence
        # global_infos = global_infos[:, :max_len + 1]
        # actions = actions[:, :max_len]
        # action_masks = action_masks[:, :max_len]
        # rewards = rewards[:, :max_len]
        # dones = dones[:, :max_len]

        # for tmp in [self_infos, others_infos, global_infos]:
        #     tmp = tmp[:, :max_len + 1].clone()
        
        # for tmp in [actions, action_masks, rewards, dones]:
        #     tmp = tmp[:, :max_len].clone()

        # -------- update value network ------------

        # include oracle information to better estimate value function (asymmetric actor critic)
        # hand_infos = torch.cat([self_infos, others_infos], dim=-1)
        
        q = self.value_network(self_infos, others_infos, records, global_infos) # [batch_size + 1, action_dim]
        with torch.no_grad():
            # q_tar = self.target_value_network(self_infos, others_infos, records, global_infos).detach()
            pi = torch.softmax(self.actor_network(self_infos, records, global_infos).detach() + action_masks_p, dim=-1) 
            pi = pi / pi.sum(dim=-1, keepdim=True)
            q_grape, adv = self.grape.compute_targets(q.detach(), pi, actions, rewards, policy_probs, dones)

        # print(q_grape.shape, adv.shape)  [batch_size],  [batch_size, action_dim]
        
        loss_c = self.mse_loss(q[np.arange(rewards.shape[0]), actions], q_grape.detach())

        # -------- update actor network ------------

        logits = self.actor_network(self_infos, records, global_infos)[:-1]
        
        # from GRAPE ---
        logpi = torch.log_softmax(logits + action_masks, dim=-1)
        pi = torch.softmax(logits + action_masks, dim=-1)

        if np.random.rand() < 0.01:
            print(np.array2string(pi[5].cpu().detach().numpy(), precision=4, suppress_small=True))
            print(np.array2string(adv[5].cpu().detach().numpy(), precision=4, suppress_small=True))

        one_hot_action = nn.functional.one_hot(actions.to(torch.int64), num_classes=pi.shape[-1]) # convert actions to one-hot

        r =  torch.sum((pi / policy_probs.clamp(1e-3, torch.inf)) * one_hot_action, dim=-1)
        adv_a = torch.sum(adv.detach() * one_hot_action, dim=-1)
        obj_pi = r * adv_a

        loss_p = torch.mean(- obj_pi) # policy gradient loss
        _loss_h = torch.sum(pi * logpi, dim=-1)
        loss_h = torch.mean(_loss_h)  # entropy loss

        loss_a = loss_p + 0.001 * loss_h

        # # -------------

        # print(advantage.shape, logits.shape, action_masks.shape)
        # policy gradient algorithm to update actor network

        # loss_a = - advantage * torch.log_softmax(logits, dim=-1) * action_masks
        # loss_a = loss_a.sum(dim=-1).mean()
        
        # --------- update model parameters ------------
        if self.update_times > 10000:
            loss = loss_c + loss_a
        else:
            loss = loss_c
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()

        # -------- update target value network ------------
        if self.update_times % 500 == 0:
            self.target_value_network.load_state_dict(self.value_network.state_dict())

        self.update_times += 1

        self.eval()

        return loss


    def select_action(self, self_info, record, global_info, action_mask, temp=1):
        
        self.eval()

        if isinstance(self_info, np.ndarray):
            self_info = torch.from_numpy(self_info).float().to(self.device)[None, :]
        if isinstance(record, np.ndarray):
            record = torch.from_numpy(record).float().to(self.device)[None, :]
        if isinstance(global_info, np.ndarray):
            global_info = torch.from_numpy(global_info).float().to(self.device)[None, :]
        if isinstance(action_mask, np.ndarray):
            action_mask = torch.from_numpy(action_mask).float().to(self.device)
        
        RECORD_PAD = torch.zeros(1, 1, 55).to(self.device).float()
        
        if record.ndim > 1:
            record = torch.cat([RECORD_PAD, record], dim=1)
        else:
            record = RECORD_PAD

        logits = self.actor_network(self_info, record, global_info)
        policy_prob = (torch.softmax(logits[0] / temp, dim=-1) * action_mask.reshape([-1])).cpu().detach().numpy()

        a = np.random.choice(54, p=policy_prob/policy_prob.sum())

        return a, policy_prob

        
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

    def _resume(self, checkpoint_dir):
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        if (self.checkpoint_dir / "resume.pth").exists():
            print(f"Resume from {self.checkpoint_dir / 'resume.pth'}", __name__)
            checkpoint = torch.load(self.checkpoint_dir / "resume.pth")
            self.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optim"])
            self.update_times = checkpoint["update_times"]
            
        else:
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}", __name__)

            
