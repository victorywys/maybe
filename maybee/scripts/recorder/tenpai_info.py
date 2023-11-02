from .supervised import SupervisedRecorder
import numpy as np
import os
import pandas as pd

from copy import deepcopy


TENPAI_DIM = 34 # 1~9m, 1~9p, 1~9s, 1~7z

class TenpaiRecorder(SupervisedRecorder):
    def __init__(self, save_path="./data/supervised"):
        super().__init__(save_path)
        
        self.shimocha_tenpai = []
        self.toimen_tenpai = []
        self.kamicha_tenpai = []
        
    def check_capacity(self):
        assert len(self.self_infos) == len(self.records) \
            == len(self.global_infos) == len(self.actions) \
            == len(self.action_mask) == len(self.shimocha_tenpai) \
            == len(self.toimen_tenpai) == len(self.kamicha_tenpai), \
        "Lengths of self_infos, records, global_infos, actions, action_mask are not equal.\n" \
        "self_infos: {}, records: {}, global_infos: {}, actions: {}, action_mask: {}, shimocha_tenpai: {}, "\
        "toimen_tenpai: {}, kamicha_tenpai: {}".format(
            len(self.self_infos), len(self.records), len(self.global_infos), len(self.actions), len(self.action_mask),
            len(self.shimocha_tenpai), len(self.toimen_tenpai), len(self.kamicha_tenpai)
        )
        if len(self.self_infos) >= self._max_allowed_samples:
            self.save(self.save_path)
            self.self_infos = []
            self.records = []
            self.global_infos = []
            self.actions = []
            self.action_mask = []
            self.shimocha_tenpai = []
            self.toimen_tenpai = []
            self.kamicha_tenpai = []
            
    def drop_last_obs(self):
        super().drop_last_obs()
        self.shimocha_tenpai.pop()
        self.toimen_tenpai.pop()
        self.kamicha_tenpai.pop()
        
    def before_selection(self, table, pid, riichi_step2_tile=None):
        super().before_selection(table, pid, riichi_step2_tile)
        
        def get_tenpai_vector(tiles):
            vec = np.zeros(TENPAI_DIM, bool)
            for tile in tiles:
                vec[int(tile)] = True
            return vec
            
        if self.record_this_game:
            if (len(self.aval_actions) > 1 and self.curr_player == pid) or riichi_step2_tile is not None:
                shimocha_id = (pid + 1) % 4
                self.shimocha_tenpai.append(get_tenpai_vector(table.players[shimocha_id].atari_tiles))
                toimen_id = (pid + 2) % 4
                self.toimen_tenpai.append(get_tenpai_vector(table.players[toimen_id].atari_tiles))
                kamicha_id = (pid + 3) % 4
                self.kamicha_tenpai.append(get_tenpai_vector(table.players[kamicha_id].atari_tiles))
                
    def save(self, path=None):
        if path is None:
            path = self.save_path
            
        data = {}
        data["shimocha_tenpai"] = np.stack(self.shimocha_tenpai)
        data["toimen_tenpai"] = np.stack(self.toimen_tenpai)
        data["kamicha_tenpai"] = np.stack(self.kamicha_tenpai)
        data["self_info"] = np.stack(self.self_infos)
        data["global_info"] = np.stack(self.global_infos)
        # self.pad_record()
        data["record"] = deepcopy(self.records)
        data["action"] = np.array(self.actions)
        data["action_mask"] = np.stack(self.action_mask)
        if not os.path.isdir(path):
            os.makedirs(path)
        pd.to_pickle(data, os.path.join(path, f"supervised_v2_{self.batch_num}.pkl"))
        print(f"batch saved to {os.path.join(path, f'supervised_v2_{self.batch_num}.pkl')}")
        self.batch_num += 1
        del data