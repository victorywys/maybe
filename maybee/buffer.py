import numpy as np
import torch
import warnings
from torch.nn.utils.rnn import *


class MajEncV2ReplayBuffer:
    def __init__(self, sin_shape=[34, 18], oin_shape=[34, 54], rcd_dim=55, gin_dim=15, action_dim=54, max_num_seq=int(1e5), 
                 batch_size=32, device='cuda'):
        super(MajEncV2ReplayBuffer, self).__init__()

        self.max_steps = 50
        self.max_rcd_len = 200

        self.sin_shape = sin_shape
        self.rcd_dim = rcd_dim
        self.gin_dim = gin_dim
 
        self.batch_size = batch_size
        self.max_num_seq = max_num_seq 

        self.device = device

        self.length = torch.zeros((max_num_seq,), dtype=torch.int, device=self.device)

        self.actions = torch.zeros((max_num_seq, self.max_steps), dtype=torch.int, device=self.device)
        self.policy_prob = torch.zeros((max_num_seq, self.max_steps, action_dim), dtype=torch.float32, device=self.device)
        # behavior policy probability, used for computing importance sampling ratio

        self.action_masks = torch.zeros((max_num_seq, self.max_steps, action_dim), dtype=torch.bool, device=self.device)
        self.self_infos = torch.zeros((max_num_seq, self.max_steps + 1, *sin_shape), dtype=torch.bool, device=self.device)
        self.others_infos = torch.zeros((max_num_seq, self.max_steps + 1, *oin_shape), dtype=torch.bool, device=self.device)
        # info of other players, used as oracle observation

        self.records = torch.zeros((max_num_seq, self.max_steps + 1, self.max_rcd_len, rcd_dim), dtype=torch.bool, device=self.device)
        self.global_infos = torch.zeros((max_num_seq, self.max_steps, gin_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((max_num_seq, self.max_steps), dtype=torch.float32, device=self.device)

        self.dones = torch.zeros((max_num_seq, self.max_steps), dtype=torch.float32, device=self.device)
        self.tail, self.size = 0, 0

        self.sum_steps = 0
        self.min_length = 0
        self.max_length = 0

        # Allocate GPU space for sampling batch data
        # self.length_b = torch.zeros((batch_size,), dtype=torch.int, device=self.device)

        # self.actions_b = torch.zeros((batch_size, self.max_steps), dtype=torch.int, device=self.device)
        # self.policy_prob_b = torch.zeros((batch_size, self.max_steps, action_dim), dtype=torch.float32, device=self.device)
        # self.action_masks_b = torch.zeros((batch_size, self.max_steps, action_dim), dtype=torch.float32, device=self.device)
        # self.self_infos_b = torch.zeros((batch_size, self.max_steps + 1, *sin_shape), dtype=torch.float32, device=self.device)
        # self.others_infos_b = torch.zeros((batch_size, self.max_steps + 1, *oin_shape), dtype=torch.float32, device=self.device)
        # # self.records_b = torch.zeros((batch_size, self.max_steps, self.max_rcd_len, rcd_dim), dtype=torch.float32, device=self.device)
        # self.global_infos_b = torch.zeros((batch_size, self.max_steps, gin_dim), dtype=torch.float32, device=self.device)
        # self.rewards_b = torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=self.device)
        # self.dones_b = torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=self.device)


    def append_episode(self, sin, oin, rcd, gin, actions, policy_probs, action_masks, r, done, length):

        if length < 1:
            warnings.warn("Episode length < 1, not recorded!")
            
        if self.length[self.tail] != 0:
            self.sum_steps -= self.length[self.tail]
        
        self.length[self.tail] = length
        self.sum_steps += length
        self.min_length = min(self.min_length, length)
        self.max_length = max(self.max_length, length)

        self.length[self.tail] = length

        self.dones[self.tail][:length] = torch.from_numpy(done[:length]).to(device=self.device)
        self.dones[self.tail][length:] = 1
        
        self.self_infos[self.tail][:length + 1] = torch.from_numpy(sin[:length + 1]).to(device=self.device)
        self.self_infos[self.tail][length + 1:] = 0

        self.others_infos[self.tail][:length + 1] = torch.from_numpy(oin[:length + 1]).to(device=self.device)
        self.others_infos[self.tail][length + 1:] = 0

        self.records[self.tail][:length + 1] = torch.from_numpy(rcd[:length + 1]).to(device=self.device)
        self.records[self.tail][length + 1:] = 0

        self.global_infos[self.tail][:length] = torch.from_numpy(gin[:length]).to(device=self.device)
        self.global_infos[self.tail][length:] = 0
        
        self.actions[self.tail][:length] = torch.from_numpy(actions[:length]).to(device=self.device)
        self.actions[self.tail][length:] = 0

        self.policy_prob[self.tail][:length] = torch.from_numpy(policy_probs[:length]).to(device=self.device)
        self.policy_prob[self.tail][length:] = 0

        self.action_masks[self.tail][:length] = torch.from_numpy(action_masks[:length]).to(device=self.device)
        self.action_masks[self.tail][length:] = 0

        self.rewards[self.tail][:length] = torch.from_numpy(r[:length]).to(device=self.device)
        self.rewards[self.tail][length:] = 0

        self.tail = (self.tail + 1) % self.max_num_seq
        self.size = min(self.size + 1, self.max_num_seq)

    def sample_contiguous_batch(self, num_seq=32):
        # sample num_seq episodes, each episode is a contiguous sequence, concatenate them to a batch
        sampled_episodes = torch.from_numpy(np.random.choice(self.size, [num_seq])).to(torch.int64)

        length_b = self.length[sampled_episodes]
        
        length_bp1 = length_b.clone()
        length_bp1[-1] += 1
        
        batch_size = torch.sum(length_b).cpu().item()

        actions_b =  torch.cat([tmp[:n] for (tmp, n) in zip(self.actions[sampled_episodes], length_b)], dim=0)
        action_masks_b = torch.cat([tmp[:n] for (tmp, n) in zip(self.action_masks[sampled_episodes], length_b)], dim=0)
        policy_prob_b = torch.cat([tmp[:n] for (tmp, n) in zip(self.policy_prob[sampled_episodes], length_b)], dim=0)
        rewards_b = torch.cat([tmp[:n] for (tmp, n) in zip(self.rewards[sampled_episodes], length_b)], dim=0)
        dones_b = torch.cat([tmp[:n] for (tmp, n) in zip(self.dones[sampled_episodes], length_b)], dim=0)

        self_infos_b = torch.cat([tmp[:n] for (tmp, n) in zip(self.self_infos[sampled_episodes], length_bp1)], dim=0).float()
        others_infos_b = torch.cat([tmp[:n] for (tmp, n) in zip(self.others_infos[sampled_episodes], length_bp1)], dim=0).float()
        global_infos_b = torch.cat([tmp[:n] for (tmp, n) in zip(self.global_infos[sampled_episodes], length_bp1)], dim=0)
        records_b = torch.cat([tmp[:n] for (tmp, n) in zip(self.records[sampled_episodes], length_bp1)], dim=0).float()
        
        rcd_lens = []
        
        rcd_sum  = torch.sum(torch.abs(records_b), dim=-1)  # [batch_size, max_rcd_len]

        for i in range(batch_size + 1):
            rcd_lens.append(rcd_sum[i].nonzero().size(0) + 1)  # + 1 due to the start token

        # print(rcd_lens)

        records_b = pack_padded_sequence(records_b, rcd_lens, batch_first=True, enforce_sorted=False)

        return self_infos_b, others_infos_b, records_b, global_infos_b, actions_b, action_masks_b, policy_prob_b, rewards_b, dones_b, length_b

    # def sample_seq_batch(self):
    #     sampled_episodes = torch.from_numpy(np.random.choice(self.size, [self.batch_size])).to(torch.int64)

    #     self.actions_b.fill_(0)
    #     self.action_masks_b.fill_(0)
    #     self.policy_prob_b.fill_(0)
    #     self.self_infos_b.fill_(0)
    #     self.others_infos_b.fill_(0)
    #     self.global_infos_b.fill_(0)
    #     self.rewards_b.fill_(0)
    #     self.dones_b.fill_(0)
        
    #     self.length_b[:] = self.length[sampled_episodes]
    #     self.actions_b[:] = self.actions[sampled_episodes]
    #     self.policy_prob_b[:] = self.policy_prob[sampled_episodes]
    #     self.action_masks_b[:] = self.action_masks[sampled_episodes]
    #     self.rewards_b[:] = self.rewards[sampled_episodes]
    #     self.dones_b[:] = self.dones[sampled_episodes]

    #     self.self_infos_b[:] = self.self_infos[sampled_episodes]
    #     self.others_infos_b[:] = self.others_infos[sampled_episodes]
    #     self.global_infos_b[:] = self.global_infos[sampled_episodes]

    #     self.records_b = pack_sequence([self.records[sampled_episodes[b]][:self.length_b[b]] for b in range(self.batch_size)], enforce_sorted=False)
    #     # records_b is a PackedSequence

    #     return self.self_infos_b, self.others_infos_b, self.records_b, self.global_infos_b, self.actions_b, self.action_masks_b, self.rewards_b, self.dones_b, self.length_b
    
    def sample_batch(self, batch_size=256):

        sampled_episodes = torch.from_numpy(np.random.choice(self.size, [batch_size])).to(torch.int64)

        length_b = self.length[sampled_episodes]
        
        sampled_steps = torch.zeros([batch_size], dtype=torch.int64)
        for b in range(batch_size):
            sampled_steps[b] = np.random.choice(length_b[b].cpu().item())

        actions_b = self.actions[sampled_episodes, sampled_steps]

        policy_prob_b = self.policy_prob[sampled_episodes, sampled_steps]
        action_masks_b = self.action_masks[sampled_episodes, sampled_steps]
        rewards_b = self.rewards[sampled_episodes, sampled_steps]
        dones_b = self.dones[sampled_episodes, sampled_steps]

        self_infos_b = self.self_infos[sampled_episodes, sampled_steps].float()
        others_infos_b = self.others_infos[sampled_episodes, sampled_steps].float()
        global_infos_b = self.global_infos[sampled_episodes, sampled_steps].float()
        records_b = self.records[sampled_episodes, sampled_steps].float()

        records_b = pack_padded_sequence(records_b, length_b + 1, batch_first=True, enforce_sorted=False)
        
        return self_infos_b, others_infos_b, records_b, global_infos_b, actions_b, action_masks_b, policy_prob_b, rewards_b, dones_b, length_b
    
