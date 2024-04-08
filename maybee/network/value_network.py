import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

from .base import NETWORK
from .encoder import RecordEncoder, GlobalInfoEncoder


class FourierTransformInfoEmbedding(nn.Module):
    def __init__(self, n_dim: int = 18):
        super(FourierTransformInfoEmbedding, self).__init__()
        self.n_dim = n_dim
    
    def forward(self, x: torch.Tensor):
        # x: shape [batch_size, 34, n_dim]
        # return: shape [batch_size, 34, n_embedding * 2]
        x_shape = x.shape
        y = torch.view_as_real(fft.fft(x)).reshape([*x.shape[:-1], self.n_dim * 2])
        
        return y


class InfoEncoder(nn.Module):
    def __init__(self, n_channels=72):
        super(InfoEncoder, self).__init__()
        # n_channels = 36 # after FFT

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
        # x: shape [batch_size, 34, n_channels]
        return self.encoder(x.permute(0, 2, 1))

# TODO: change to conv2D for speed

class FTInfoEncoder(nn.Module):
    def __init__(self):
        super(FTInfoEncoder, self).__init__()
        self.encoder = InfoEncoder(18)
        self.fft = FourierTransformInfoEmbedding(18)
        self.out_dim = self.encoder.out_dim * 4
    
    def forward(self, x: torch.Tensor):
        # x: shape [batch_size, 34, 18]
        self_info = self.fft(x[:, :, :18])
        next_info = self.fft(x[:, :, 18:36])
        opposite_info = self.fft(x[:, :, 36:54])
        last_info = self.fft(x[:, :, 54:72])

        return torch.cat([self.encoder(self_info),
                          self.encoder(next_info),
                          self.encoder(opposite_info),
                          self.encoder(last_info)], dim=-1)


@NETWORK.register_module()
class QNetwork(nn.Module):
    def __init__(self, hidden_size: int = 1024, dropout: float = 0.1, hand_encoder: str="transformer"):
        super(QNetwork, self).__init__()

        action_dim = 54
        self.hand_encoder = hand_encoder
        if self.hand_encoder == "transformer":
            self.info_encoder = TransformerRegression(input_dim=72, out_dim=hidden_size)
        elif self.hand_encoder == "cnn":
            self.info_encoder = InfoEncoder(72)
        elif self.hand_encoder == "fftcnn":
            self.info_encoder = FTInfoEncoder()
        self.record_encoder = RecordEncoder()
        self.global_info_encoder = GlobalInfoEncoder()

        self.last_layer_v = nn.Linear(hidden_size, 1)
        self.last_layer_a = nn.Linear(hidden_size, action_dim)

        self.v_net = nn.Sequential(
            nn.Linear(self.info_encoder.out_dim + self.record_encoder.out_dim + self.global_info_encoder.out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.last_layer_v
        )
        self.a_net = nn.Sequential(
            nn.Linear(self.info_encoder.out_dim + self.record_encoder.out_dim + self.global_info_encoder.out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            self.last_layer_a
        )

    def reset_last_layer(self):
        self.last_layer_a.reset_parameters()
        self.last_layer_v.reset_parameters()

    def forward(self, self_info, others_info, records, global_info) -> torch.Tensor:
        
        x1 = self.info_encoder(torch.cat([self_info, others_info], dim=-1))
        x2 = self.record_encoder(records)
        x3 = self.global_info_encoder(global_info)
        y = torch.cat([x1, x2, x3], dim=-1)

        value = self.v_net(y)
        advantages = self.a_net(y)
        # Q = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        # q_values = self.q_net(y)

        return q_values


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=34):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(10 / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class TransformerRegression(nn.Module):
    def __init__(self, input_dim=72, out_dim=2048, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerRegression, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)  # Linear layer to project input to d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout, 34)  # Positional encoding layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_linear = nn.Linear(d_model, out_dim)  # Output layer
        self.d_model = d_model
        self.out_dim = out_dim
        self.sqrt_d_model = self.d_model ** 0.5

    def forward(self, src):
        """
        src: Tensor, shape [batch_size, seq_length, feature_dim]
        """
        # Project input to model dimension
        src = self.input_linear(src) * self.sqrt_d_model
        src = self.pos_encoder(src)

        # Apply transformer encoder
        output = self.transformer_encoder(src)
        # Pool the transformer outputs along the seq_length dimension
        output = torch.mean(output, dim=1)  # Mean pooling
        # Project to output dimension
        output = self.output_linear(output)
        return output  # shape [batch_size, output_dim]
