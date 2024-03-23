import torch
import torch.nn as nn


from .base import NETWORK
from .encoder import RecordEncoder, GlobalInfoEncoder


class AllInfoEncoder(nn.Module):
    def __init__(self):
        super(AllInfoEncoder, self).__init__()
        n_channels = 72 # include self_info of other players

        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.out_dim = 32 * 34
        
    def forward(self, x: torch.Tensor):
        # x: shape [batch_size, 34, 72]
        return self.encoder(x.permute(0, 2, 1))
    

@NETWORK.register_module()
class QNetwork(nn.Module):
    def __init__(self, hidden_size: int = 512):
        super(QNetwork, self).__init__()

        action_dim = 54
        self.info_encoder = AllInfoEncoder()
        self.record_encoder = RecordEncoder()
        self.global_info_encoder = GlobalInfoEncoder()
        self.v_net = nn.Sequential(
            nn.Linear(self.info_encoder.out_dim + self.record_encoder.out_dim + self.global_info_encoder.out_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.a_net = nn.Sequential(
            nn.Linear(self.info_encoder.out_dim + self.record_encoder.out_dim + self.global_info_encoder.out_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, self_info, others_info, records, global_info) -> torch.Tensor:
        
        x1 = self.info_encoder(torch.cat([self_info, others_info], dim=-1))
        x2 = self.record_encoder(records)
        x3 = self.global_info_encoder(global_info)
        y = torch.cat([x1, x2, x3], dim=1)

        value = self.v_net(y)
        advantages = self.a_net(y)
        # Q = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        # q_values = self.q_net(y)

        return q_values

