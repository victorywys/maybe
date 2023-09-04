from typing import Optional

from .Base import BasePlayer, PLAYER
from network import MahjongPlayer

import torch
import numpy as np


RECORD_PAD = torch.zeros(1, 55)

@PLAYER.register_module(name="supervised")
class SupervisedPlayer(BasePlayer):
    def __init__(
            self, 
            name: str,
            weight_path: Optional[str] = None,
            gpu: Optional[int] = 0,
            share_model: Optional[str] = None,
        ):
        super().__init__(name)
        self.model = MahjongPlayer()
        self.model.load_state_dict(torch.load(weight_path)["best_network_params"])
        self.gpu = gpu
        if gpu is not None:
            self.model.to(gpu)
        self.model.eval()

    def play(
            self, 
            self_info: np.ndarray,
            record_info: np.ndarray,
            global_info: np.ndarray,
            valid_actions_mask: np.ndarray,
        ):
        self_info = torch.Tensor(self_info).unsqueeze(0)
        record_info = torch.Tensor(record_info)
        if record_info.ndim > 1:
            record_info = torch.cat([RECORD_PAD, record_info], dim=0).unsqueeze(0)
        else:
            record_info = RECORD_PAD.unsqueeze(0)
        global_info = torch.Tensor(global_info).unsqueeze(0)
        if self.gpu is not None:
            self_info = self_info.to(self.gpu)
            record_info = record_info.to(self.gpu)
            global_info = global_info.to(self.gpu)
        with torch.no_grad():
            pred = self.model(self_info, record_info, global_info)
            pred = torch.softmax(pred[0], dim=-1).cpu().detach().numpy()
        a = (pred * valid_actions_mask).argmax(-1)
        return a
