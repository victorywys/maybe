import torch
import torch.nn as nn


class SelfInfoEncoder(nn.Module):
    def __init__(self):
        super(SelfInfoEncoder, self).__init__()
        n_channels = 18

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
        # x: shape [batch_size, 34, 18]
        return self.encoder(x.permute(0, 2, 1))
    
    
class RecordEncoder(nn.Module):
    def __init__(self):
        super(RecordEncoder, self).__init__()
        input_size = 55
        self.lstm = nn.LSTM(input_size, 128, 4, batch_first=True, bidirectional=True)
        self.out_dim = 128 * 2
    
    def forward(self, x):
        # x: shape [batch_size, record_step, 55]
        x, _ = self.lstm(x)
        if isinstance(x, nn.utils.rnn.PackedSequence):
            unpacked, unpacked_lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            idx = (unpacked_lengths - 1).unsqueeze(-1).expand(-1, unpacked.size(2))
            idx = idx.to(unpacked.device)
            last_states = unpacked.gather(1, idx.unsqueeze(1)).squeeze(1)
        else:
            last_states = x[:, -1, :]
        return last_states
    
    
class GlobalInfoEncoder(nn.Module):
    def __init__(self):
        super(GlobalInfoEncoder, self).__init__()
        input_size = 15
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.out_dim = 64
        
    def forward(self, x):
        # x: shape [batch_size, 15]
        return self.encoder(x)