import torch
import torch.nn as nn

from .base import NETWORK
from .encoder import SelfInfoEncoder, RecordEncoder, GlobalInfoEncoder


@NETWORK.register_module()
class MahjongPlayer(nn.Module):
    def __init__(self):
        super(MahjongPlayer, self).__init__()
        action_dim = 47
        self.self_info_encoder = SelfInfoEncoder()
        self.record_encoder = RecordEncoder()
        self.global_info_encoder = GlobalInfoEncoder()
        self.policy_head = nn.Sequential(
            nn.Linear(self.self_info_encoder.out_dim + self.record_encoder.out_dim + self.global_info_encoder.out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, self_info, record, global_info):
        self_info = self.self_info_encoder(self_info)
        record = self.record_encoder(record)
        global_info = self.global_info_encoder(global_info)
        x = torch.cat([self_info, record, global_info], dim=1)
        return self.policy_head(x)