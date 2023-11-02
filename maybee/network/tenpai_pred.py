import torch
import torch.nn as nn

from .base import NETWORK
from .encoder import SelfInfoEncoder, RecordEncoder, GlobalInfoEncoder


@NETWORK.register_module()
class TenpaiPred(nn.Module):
    def __init__(self):
        super(TenpaiPred, self).__init__()
        # action_dim = 47
        pred_dim = (1 + 34) * 3 # (tenpai_or_not(1) + tiles(34)) * num_players(3)
        self.self_info_encoder = SelfInfoEncoder()
        self.record_encoder = RecordEncoder()
        self.global_info_encoder = GlobalInfoEncoder()
        self.pred_head = nn.Sequential(
            nn.Linear(self.self_info_encoder.out_dim + self.record_encoder.out_dim + self.global_info_encoder.out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, pred_dim),
        )

    def forward(self, self_info, record, global_info):
        self_info = self.self_info_encoder(self_info)
        record = self.record_encoder(record)
        global_info = self.global_info_encoder(global_info)
        x = torch.cat([self_info, record, global_info], dim=1)
        return self.pred_head(x)