import numpy as np
import torch
import warnings
from torch.nn.utils.rnn import *


class MajEncV2ReplayBuffer:
    def __init__(self, sin_shape=[34, 18], rcd_dim=55, gin_dim=15, action_dim=54, max_num_seq=int(1e5), 
                 batch_size=32, device='cpu'):
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
        self.action_masks = torch.zeros((max_num_seq, self.max_steps, action_dim), dtype=torch.bool, device=self.device)
        self.self_infos = torch.zeros((max_num_seq, self.max_steps + 1, *sin_shape), dtype=torch.bool, device=self.device)
        self.records = torch.zeros((max_num_seq, self.max_rcd_len, rcd_dim), dtype=torch.bool, device=self.device)
        self.global_infos = torch.zeros((max_num_seq, self.max_steps, gin_dim), dtype=torch.float32, device=self.device)

        self.rewards = torch.zeros((max_num_seq, self.max_steps), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((max_num_seq, self.max_steps), dtype=torch.float32, device=self.device)
        self.tail, self.size = 0, 0

        self.sum_steps = 0
        self.min_length = 0
        self.max_length = 0

        # Allocate GPU space for sampling batch data
        self.length_b = torch.zeros((batch_size,), dtype=torch.int, device=self.device)

        self.actions_b = torch.zeros((batch_size, self.max_steps), dtype=torch.int, device=self.device)
        self.action_masks_b = torch.zeros((batch_size, self.max_steps, action_dim), dtype=torch.bool, device=self.device)
        self.self_infos_b = torch.zeros((batch_size, self.max_steps + 1, *sin_shape), dtype=torch.bool, device=self.device)
        # self.records_b = torch.zeros((batch_size, self.max_steps, self.max_rcd_len, rcd_dim), dtype=torch.float32, device=self.device)
        self.global_infos_b = torch.zeros((batch_size, self.max_steps, gin_dim), dtype=torch.float32, device=self.device)
        self.rewards_b = torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=self.device)
        self.dones_b = torch.zeros((batch_size, self.max_steps), dtype=torch.float32, device=self.device)


    def append_episode(self, sin, rcd, gin, actions, action_masks, r, done, length):

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

        self.records[self.tail][:rcd.shape[0]] = torch.from_numpy(rcd).to(device=self.device) # length is different
        self.records[self.tail][rcd.shape[0]:] = 0

        self.global_infos[self.tail][:length] = torch.from_numpy(gin[:length]).to(device=self.device)
        self.global_infos[self.tail][length:] = 0
        
        self.actions[self.tail][:length] = torch.from_numpy(actions[:length]).to(device=self.device)
        self.actions[self.tail][length:] = 0

        self.action_masks[self.tail][:length] = torch.from_numpy(action_masks[:length]).to(device=self.device)
        self.action_masks[self.tail][length:] = 0

        self.rewards[self.tail][:length] = torch.from_numpy(r[:length]).to(device=self.device)
        self.rewards[self.tail][length:] = 0

        self.tail = (self.tail + 1) % self.max_num_seq
        self.size = min(self.size + 1, self.max_num_seq)

    def sample_batch(self):
        sampled_episodes = torch.from_numpy(np.random.choice(self.size, [self.batch_size])).to(torch.int64)

        self.actions_b.fill_(0)
        self.self_infos_b.fill_(0)
        self.global_infos_b.fill_(0)
        self.rewards_b.fill_(0)
        self.dones_b.fill_(0)

        self.length_b[:] = self.length[sampled_episodes]
        self.actions_b[:] = self.actions[sampled_episodes]
        self.action_masks_b[:] = self.action_masks[sampled_episodes]
        self.rewards_b[:] = self.rewards[sampled_episodes]
        self.dones_b[:] = self.dones[sampled_episodes]

        self.self_infos_b = self.self_infos[sampled_episodes]
        self.global_infos_b = self.global_infos[sampled_episodes]

        self.records_b = pack_sequence([self.records[sampled_episodes[b]][:self.length_b[b]] for b in range(self.batch_size)], enforce_sorted=False)
        # records_b is a PackedSequence

        return self.self_infos_b, self.records_b, self.global_infos_b, self.actions_b, self.action_masks_b, self.rewards_b, self.dones_b, self.length_b
    
