from typing import List
from .player import BasePlayer
from .logger import TenhouJsonLogger

import pymahjong as pm

import numpy as np


class Game():
    def __init__(
        self,
        players: List[BasePlayer],
        th_logger: TenhouJsonLogger = None,
    ):
        assert len(players) == 4, "There should be 4 players in a game."
        self.players = players
        self.env = pm.MahjongEnv()
        self.th_logger = th_logger
        self.scores = [25000, 25000, 25000, 25000]

    def start_new(
        self,
        oya=0,
        game_wind="east",
        scores=[25000, 25000, 25000, 25000],
        honba=0,
        kyoutaku=0,
    ):
        print(oya, game_wind, scores, honba, kyoutaku)
        self.scores = scores
        self.env.reset(
            oya=oya,
            game_wind=game_wind,
            scores=scores,
            honba=honba,
            kyoutaku=kyoutaku,
        )

        self.te = pm.TableEncoder(self.env.t)
        self.te.init()
        self.te.update()
        self.th_logger.start_game(self.env.t, self.te)

    def play(
        self,
        record_buffer=None,
    ):
        if record_buffer is not None:
            sin_array = np.zeros([51, 34, 18], dtype=bool)
            oin_array = np.zeros([51, 34, 54], dtype=bool)
            gin_array = np.zeros([51, 15], dtype=bool)
            actions = np.zeros([50], dtype=int)
            action_masks = np.zeros([50, 54], dtype=bool)
            policy_probs = np.zeros([50, 54], dtype=np.float32)
            rs = np.zeros([50], dtype=np.float32)
            dones = np.zeros([50], dtype=np.float32)
        
        step = 0

        while not self.env.is_over():
            curr_player_id = self.env.get_curr_player_id()
            valid_actions_mask = self.env.get_valid_actions(nhot=True)
            obs = np.array(self.te.self_infos[curr_player_id]).reshape([18, 34]).swapaxes(0, 1)
            rcd = np.array(self.te.records[curr_player_id])
            gin = np.array(self.te.global_infos[curr_player_id])

            # --------- make decision -------------
            a, policy_prob = self.players[curr_player_id].play(obs, rcd, gin, valid_actions_mask, return_policy=True)

            self.env.step(curr_player_id, a)

            if curr_player_id == 0 and record_buffer is not None:   # only record player 0 (the RL agent)
                sin_array[step] = obs
                oin = np.zeros([34, 54], dtype=bool)
                for i in range(1, 4): # oracle information (others' hands)
                    oin[:, (i - 1) * 18 : i * 18] = np.array(self.te.self_infos[(curr_player_id + i) % 4]).reshape([18, 34]).swapaxes(0, 1)
                oin_array[step] = oin
                gin_array[step] = gin
                actions[step] = a
                # rs[step] = 0
                # dones[step] = 0
                action_masks[step] = valid_actions_mask
                policy_probs[step] = policy_prob

                step += 1

            # ------- update state encoding ------------
            if not self.env.is_over():
                self.te.update()
        # ----------------------- get result ---------------------------------
        self.th_logger.end_game(self.env.t, self.te)

        if record_buffer is not None and step >= 1:
            #  ------- record final step info ----------
            self.te.update()

            rcd_array = np.array(self.te.records[0]) # only record player 0 (the RL agent)
            if rcd_array.ndim == 1:
                rcd_array = np.zeros([0, 55], dtype=bool)

            dones[step - 1] = 1

            rs[step - 1] = (self.env.t.get_result().score[0]  - self.scores[0])/ 1000  # normalized by 1
            # only only consider straight points (no ranking points)
            
            sin_array[step] = np.array(self.te.self_infos[0]).reshape([18, 34]).swapaxes(0, 1)
            
            oin = np.zeros([34, 54], dtype=bool)
            for i in range(1, 4): # oracle information (others' hands)
                oin[:, (i - 1) * 18 : i * 18] = np.array(self.te.self_infos[(curr_player_id + i) % 4]).reshape([18, 34]).swapaxes(0, 1)
            oin_array[step] = oin
            
            gin_array[step] = np.array(self.te.global_infos[0])

            # --------- append to record buffer -------------
        
            self.scores = self.env.t.get_result().score
            record_buffer.append_episode(sin_array, oin_array, rcd_array, gin_array, actions, policy_probs, action_masks, rs, dones, step)

        return self.env.t.get_result()
