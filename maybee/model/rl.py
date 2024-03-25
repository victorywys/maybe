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
from utilsd.config import PythonConfig

from .base import MODEL

from dataset import supervised_collate_fn, MultiFileSampler

import os
from copy import deepcopy

from arena.common import *

class GRAPE:
    """
    policy evaluation algorithm: 
    """

    def __init__(self, alpha, gamma, lambd):
        self.alpha = alpha
        self.gamma = gamma
        self.lambd = lambd

    def compute_targets(self, q, q_tar, pi, a_t, r_t, mu_t, done_t):
        """
        Compute G Q (s, a) + a A (s, a), and return a tuple of (G Q  (s, a) + a A  (s, a), A (s, \cdot))
        q_t   = s[:-1]
        q_tp1 = s[1:]
        """
        l = q.shape[0] - 1
        pi_t, pi_tp1 = pi[:-1], pi[1:]
        q_t, q_tp1 = q[:-1], q_tar[1:]
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

class TwinQNetwork(nn.Module):
    def __init__(self, q1, q2):
        super(TwinQNetwork, self).__init__()
        self.q1 = q1
        self.q2 = q2

    def forward(self, self_info, others_info, records, global_info):
        q1 = self.q1(self_info, others_info, records, global_info)
        # q2 = self.q2(self_info, others_info, records, global_info)
        return q1

@MODEL.register_module()
class RLMahjong(nn.Module):
    def __init__(
        self,
        actor_network: Optional[nn.Module] = None,
        # value_network: Optional[nn.Module] = None,
        # output_dir: Optional[str] = None,
        # checkpoint_dir: Optional[str] = None,
        # ranking_rewards: Optional[list] = [45, 5, -15, -35],
        device: Optional[str] = "cuda",
        config: Optional[PythonConfig] = None,
    ):
        # self.batch_size = batch_size
        super(RLMahjong, self).__init__()

        # self.value_network = value_network
        self.actor_network = actor_network
        self.gamma = config.gamma
        self.config = config
        self.algorithm = config.algorithm

        if self.config.algorithm == "grape":
            self.grape = GRAPE(alpha=config.alpha_grape, gamma=self.gamma, lambd=config.lambd_grape)
            self.value_network = config.value_network.build()
            self.target_value_network = deepcopy(self.value_network)

        elif self.config.algorithm == "dsac":
            self.log_alpha = nn.Parameter(torch.tensor(-3, requires_grad=True, dtype=torch.float32))
            self.optimizer_alpha = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)
            self.value_network_1 = config.value_network.build()
            self.value_network_2 = config.value_network.build()
            self.target_value_network_1 = deepcopy(self.value_network_1)
            self.target_value_network_2 = deepcopy(self.value_network_2)
            self.value_network = TwinQNetwork(self.value_network_1, self.value_network_2)
            self.target_value_network = TwinQNetwork(self.target_value_network_1, self.target_value_network_2)

        self.optimizer_a = torch.optim.Adam(self.actor_network.parameters(), lr=config.lr_actor)
        self.optimizer_v = torch.optim.Adam(self.value_network.parameters(), lr=config.lr_value)  # TODO: two Q networks

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
        batch = buffer.sample_contiguous_batch(num_seq=self.config.batch_seq_num, random_mps_change=self.config.random_mps_change)
        
        self_infos, others_infos, records, global_infos, actions, action_masks, policy_probs, rewards, dones, lengths = batch

        
        one_hot_actions = nn.functional.one_hot(actions.to(torch.int64), num_classes=action_masks.shape[-1]) # convert actions to one-hot

        action_masks = action_masks.to(torch.float)

        n_valid_actions = action_masks.sum(dim=-1).to(torch.float)

        policy_normed = policy_probs # already normed
        entropy_old = - torch.sum(policy_normed * torch.log(policy_normed + 1e-9), dim=-1).detach()
        policy_normed_a = torch.sum(policy_normed * one_hot_actions, dim=-1).clamp(1e-6, 1)

        action_masks_ = torch.zeros_like(action_masks) - (action_masks < 0.5).to(torch.float) # [-1, -1, 0, 0, -1, -1, ....]
        action_masks_p_ = torch.cat([action_masks_, torch.zeros_like(action_masks_[:1])], dim=0)

        if self.algorithm == "dsac":
            
            # -------- critic learning ----------
            q1 = self.value_network_1(self_infos, others_infos, records, global_infos)[:-1] # [batch_size, action_dim]
            q2 = self.value_network_2(self_infos, others_infos, records, global_infos)[:-1] # [batch_size, action_dim]
            with torch.no_grad():
                q_tar_tp1_1 = self.target_value_network_1(self_infos, others_infos, records, global_infos)[1:]
                q_tar_tp1_2 = self.target_value_network_2(self_infos, others_infos, records, global_infos)[1:]
                
                pi_tp1 = torch.softmax(self.actor_network(self_infos, records, global_infos).detach() + action_masks_p_ * 10000, dim=-1)[1:]
                if self.config.use_avg_q:
                    y = rewards.reshape([-1]) + (1. - dones.reshape([-1])) * self.gamma * ((
                        torch.sum(q_tar_tp1_1 * pi_tp1, dim=-1) + torch.sum(q_tar_tp1_2 * pi_tp1, dim=-1)) / 2 - self.log_alpha.detach().exp() * torch.log(torch.cat(
                    [policy_normed_a[1:], torch.ones_like(policy_normed_a[:1])], dim=0)))
                else:
                    y = rewards.reshape([-1]) + (1. - dones.reshape([-1])) * self.gamma * (torch.minimum(
                        torch.sum(q_tar_tp1_1 * pi_tp1, dim=-1), torch.sum(q_tar_tp1_2 * pi_tp1, dim=-1)) - self.log_alpha.detach().exp() * torch.log(torch.cat(
                    [policy_normed_a[1:], torch.ones_like(policy_normed_a[:1])], dim=0)))

                # y - alpha * log(pi(a_{t+1}|s_{t+1}))
                    
            if np.random.rand() < 0.02:
                idx = torch.argwhere(y < -5)
                print("target", y[idx].detach().cpu().numpy())
                print("q1", q1[idx, actions[idx]].detach().cpu().numpy())
                print("policy_normed_a", policy_normed_a[idx].detach().cpu().numpy())

            loss_c_1 = torch.mean(self.mse_loss(torch.sum(q1 * one_hot_actions, dim=-1), y.detach()))
            loss_c_2 = torch.mean(self.mse_loss(torch.sum(q2 * one_hot_actions, dim=-1), y.detach()))
            loss_c = loss_c_1 + loss_c_2

            # debug
            # print("loss_c_1 : %3.3f" % (loss_c_1.detach().cpu().numpy()))
            # print("loss_c_2 : %3.3f" % (loss_c_2.detach().cpu().numpy()))
            # print("loss_c : %3.3f" % (loss_c.detach().cpu().numpy()))
            # print("y : %3.3f" % (y.detach().cpu().numpy().mean()))
            # print("q1 : %3.3f" % (q1.detach().cpu().numpy().mean()))
            # print("q2 : %3.3f" % (q2.detach().cpu().numpy().mean()))
            # print("entropy_old : %3.3f" % (entropy_old.detach().cpu().numpy().mean()))

            # -------- actor learning ----------
            logits = self.actor_network(self_infos, records, global_infos)[:-1]
            
            # 方案1
            # pi = torch.softmax(logits + action_masks_ * 10, dim=-1)
            # logpi = torch.log_softmax(logits + action_masks_ * 10, dim=-1)
            
            # 方案2
            pi = torch.softmax(logits, dim=-1)
            pi_masked = torch.softmax(logits + action_masks_ * 10000, dim=-1)
            logpi_masked = torch.log_softmax(logits + action_masks_ * 10000, dim=-1)
            # pi_masked = pi_masked / (1e-20 + pi_masked.sum(dim=-1, keepdim=True))
            # logpi = torch.log(pi + 1e-20) * action_masks
            a_sampled = torch.distributions.Categorical(pi_masked).sample().detach()

            # 方案3
            # pi = torch.softmax(logits, dim=-1)
            # logpi = torch.log_softmax(logits, dim=-1)

            entropy = - torch.sum(pi_masked * logpi_masked, dim=-1)

            # q_with_penalty_for_invalid_action =  q1.detach() * action_masks_ * 16 # punish 16,000 points for invalid action
                        
            loss_a = - self.log_alpha.detach().exp() * entropy - q1[torch.arange(0, q1.shape[0]), a_sampled].detach() * logpi_masked[torch.arange(0, q1.shape[0]), a_sampled] # pass-response keep its value for debug
            loss_a = torch.mean(loss_a) + self.config.entropy_penalty_beta * torch.mean(self.mse_loss(entropy, entropy_old.detach()))

            # debug
            # print("pi : %3.3f" % (pi.detach().cpu().numpy().mean()))
            # print("logpi : %3.3f" % (logpi.detach().cpu().numpy().mean()))
            # print("loss_a : %3.3f" % (loss_a.detach().cpu().numpy()))
            # print("entropy : %3.3f" % (entropy.detach().cpu().numpy().mean()))
            # print("log_alpha : %3.3f" % (self.log_alpha.detach().cpu().numpy()))
            # print("alpha : %3.3f" % (self.log_alpha.detach().exp().cpu().numpy()))
            
            rnd_idx = np.random.randint(0, int(pi.shape[0]))
            pi_np = pi_masked[rnd_idx].detach().cpu().numpy().reshape([-1])

            if np.random.rand() < 0.01 or pi_np[49] > 0.1:  # contains ron
                print("----------- 抽查 ------------")
                for idx in np.argwhere(action_masks[rnd_idx].detach().cpu().numpy()):
                    print(action_v2_to_human_chinese[int(idx)] + " policy prob : %3.3f" % (pi_np[int(idx)])
                           + "; logit : %3.3f" % (logits[rnd_idx, int(idx)].detach().cpu().numpy())
                           + "; Q : %3.3f" % (q1[rnd_idx, int(idx)].detach().cpu().numpy()))
                    
                print("alpha = {}".format(self.log_alpha.detach().exp().cpu().numpy()))
                print("entropy = %3.3f" % (entropy.detach().cpu().numpy().mean()))
                print("entropy_old = %3.3f" % (entropy_old.detach().cpu().numpy().mean()))

            if self.config.target_entropy is not None:
                target_entropy = self.config.target_entropy
            else:
                eps = torch.tensor(self.config.policy_epsilon, dtype=torch.float32).to(self.device)
                target_entropy = - (1 - eps * (n_valid_actions - 1)) * torch.log(
                    1 - eps * (n_valid_actions - 1)) - (n_valid_actions - 1) * eps * torch.log(eps)
            loss_alpha = - self.log_alpha * torch.mean((target_entropy - entropy.detach()))
            loss_a = loss_a + loss_alpha

        # -------- update value network ------------
        elif self.algorithm == "grape":
            q = self.value_network(self_infos, others_infos, records, global_infos) # [batch_size + 1, action_dim]
            with torch.no_grad():
                q_tar = self.target_value_network(self_infos, others_infos, records, global_infos)
                pi = torch.softmax(self.actor_network(self_infos, records, global_infos).detach() + action_masks_p_ * 10000, dim=-1) 
                pi = pi / pi.sum(dim=-1, keepdim=True)
                q_grape, adv = self.grape.compute_targets(q.detach(), q_tar.detach(), pi, actions, rewards, policy_probs, dones)
            
            loss_c = self.mse_loss(q[np.arange(rewards.shape[0]), actions], q_grape.detach())

            # -------- update actor network ------------

            logits = self.actor_network(self_infos, records, global_infos)[:-1]
            
            # from GRAPE ---
            logpi = torch.log_softmax(logits + action_masks * 100, dim=-1)
            pi = torch.softmax(logits + action_masks * 100, dim=-1)

            if np.random.rand() < 0.02:
                print("----------- 抽查 ------------")
                rnd_idx = np.random.randint(0, int(adv.shape[0]))
                adv_np = adv[rnd_idx].detach().cpu().numpy().reshape([-1])
                pi_np = pi[rnd_idx].detach().cpu().numpy().reshape([-1])
                for idx in np.argwhere(pi_np):
                    print(action_v2_to_human_chinese[int(idx)] + " policy prob : %4.4f" % (pi_np[int(idx)]) + "; advantage : %5.5f" % (adv_np[int(idx)]))

            one_hot_action = nn.functional.one_hot(actions.to(torch.int64), num_classes=pi.shape[-1])  # convert actions to one-hot

            r =  torch.sum((pi / policy_probs.clamp(1e-3, 1)) * one_hot_action, dim=-1)
            adv_a = torch.sum(adv.detach() * one_hot_action, dim=-1)
            obj_pi = r * adv_a

            loss_p = torch.mean(- obj_pi) # policy gradient loss
            _loss_h = torch.sum(pi * logpi, dim=-1)
            loss_h = torch.mean(_loss_h)  # entropy loss

            loss_a = loss_p + self.config.coef_entropy * loss_h
        
        # --------- update model parameters ------------
        if self.config.actor_training_offset > 0:
            if self.update_times > self.config.actor_training_offset:
                loss = loss_c + loss_a
            else:
                loss = loss_c
        else:
            if self.update_times >= - self.config.actor_training_offset:
                loss = loss_c + loss_a
            else:
                loss = loss_a
        
        self.optimizer_a.zero_grad()
        self.optimizer_v.zero_grad()
        if self.algorithm == "dsac":
            self.optimizer_alpha.zero_grad()
        
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 10)

        self.optimizer_a.step()
        self.optimizer_v.step()
        if self.algorithm == "dsac":
            self.optimizer_alpha.step()

        # -------- soft-update target value network ------------
        state_dict = self.value_network.state_dict()
        state_dict_tar = self.target_value_network.state_dict()
        for key in state_dict_tar.keys():
            state_dict_tar[key] = 0.995 * state_dict_tar[key] + 0.005 * state_dict[key]
        self.target_value_network.load_state_dict(state_dict_tar)

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
        action_mask_ = torch.zeros_like(action_mask) - (action_mask < 0.5).to(torch.float) # [-1, -1, 0, 0, -1, -1, ....]
        policy_prob = (torch.softmax(logits[0] / temp + action_mask_ * 1e9, dim=-1).reshape([-1])).cpu().detach().numpy()

        a = np.random.choice(54, p=policy_prob/policy_prob.sum())

        return a, policy_prob
    
    def _save(self, savepath):
        torch.save(
            {
                "model": self.state_dict(),
                "update_times": self.update_times
            },
            savepath)
    
    def _checkpoint(self, checkpoint_dir=None):
        torch.save(
            {
                "model": self.state_dict(),
                # "optim": self.optimizer.state_dict(),
                "update_times": self.update_times
            },
            self.checkpoint_dir / f"resume_{self.update_times}.pth" if checkpoint_dir is None else checkpoint_dir / f"resume_{self.update_times}.pth",
        )
        print(f"Checkpoint saved to {self.checkpoint_dir / f'resume_{self.update_times}.pth' if checkpoint_dir is None else checkpoint_dir / f'resume_{self.update_times}.pth'}", __name__)
        torch.save(
            {
                "model": self.state_dict(),
                # "optim": self.optimizer.state_dict(),
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
            # self.optimizer.load_state_dict(checkpoint["optim"])
            self.update_times = checkpoint["update_times"]
            
        else:
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}", __name__)

            
