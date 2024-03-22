try:
    from typing import Optional
except:
    from typing_extensions import Optional

from collections import OrderedDict

from .Base import BasePlayer, PLAYER
from network import MahjongPlayer, TenpaiPred
from ..common import tile_to_human

import torch
import numpy as np


RECORD_PAD = torch.zeros(1, 55)
SHIMOCHA_TENPAI_POS = 0
TOIMEN_TENPAI_POS = 35
KAMICHA_TENPAI_POS = 70

@PLAYER.register_module(name="supervised")
class SupervisedPlayer(BasePlayer):
    def __init__(
            self, 
            name: str,
            weight_path: Optional[str] = None,
            tenpai_pred_weight_path: Optional[str] = None,
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
        if tenpai_pred_weight_path is not None:
            self.log_details = True
            self.tenpai_pred_model = TenpaiPred()
            self.tenpai_pred_model.load_state_dict(torch.load(tenpai_pred_weight_path)["best_network_params"])
            self.tenpai_pred_model.eval()
            if gpu is not None:
                self.tenpai_pred_model.to(gpu)
        else:
            self.log_details = False

    def play(
            self, 
            self_info: np.ndarray,
            record_info: np.ndarray,
            global_info: np.ndarray,
            valid_actions_mask: np.ndarray,
            return_policy: bool = False
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
        
        if self.log_details:
            print(a, pred[48], pred[52])
            self.log_tenpai_pred(self_info, record_info, global_info)
        
        if return_policy:
            return a, pred
        else:
            return a

    def log_tenpai_pred(self, self_info, record_info, global_info):
        tenpai_pred = self.tenpai_pred_model(self_info, record_info, global_info)
        tenpai_pred = torch.sigmoid(tenpai_pred).cpu().detach().numpy()
        print("Shimocha Tenpai:", f"{tenpai_pred[0][SHIMOCHA_TENPAI_POS] * 100:.2f}%")
        for i in range(34):
            print(f"{tile_to_human[i]}: {tenpai_pred[0][SHIMOCHA_TENPAI_POS + i + 1] * 100:.2f}%", end="\t")
            if i % 9 == 8:
                print()
        print()
        print("Toimen Tenpai:", f"{tenpai_pred[0][TOIMEN_TENPAI_POS] * 100:.2f}%")
        for i in range(34):
            print(f"{tile_to_human[i]}: {tenpai_pred[0][TOIMEN_TENPAI_POS + i + 1] * 100:.2f}%", end="\t")
            if i % 9 == 8:
                print()
        print()
        print("Kamicha Tenpai:", f"{tenpai_pred[0][KAMICHA_TENPAI_POS] * 100:.2f}%")
        for i in range(34):
            print(f"{tile_to_human[i]}: {tenpai_pred[0][KAMICHA_TENPAI_POS + i + 1] * 100:.2f}%", end="\t")
            if i % 9 == 8:
                print()
        print()
        