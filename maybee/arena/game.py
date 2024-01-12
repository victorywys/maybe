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

    def start_new(
        self,
        oya=0,
        game_wind="east",
        scores=[25000, 25000, 25000, 25000],
        honba=0,
        kyoutaku=0,
    ):
        print(oya, game_wind, scores, honba, kyoutaku)
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
    ):
        while not self.env.is_over():
            curr_player_id = self.env.get_curr_player_id()
            valid_actions_mask = self.env.get_valid_actions(nhot=True)
            obs = np.array(self.te.self_infos[curr_player_id]).reshape([18, 34]).swapaxes(0, 1)
            rcd = np.array(self.te.records[curr_player_id])
            gin = np.array(self.te.global_infos[curr_player_id])

            # --------- make decision -------------
            a = self.players[curr_player_id].play(obs, rcd, gin, valid_actions_mask)

            self.env.step(curr_player_id, a)

            # ------- update state encoding ------------
            if not self.env.is_over():
                self.te.update()

        # ----------------------- get result ---------------------------------
        # self.te.update()
        self.th_logger.end_game(self.env.t, self.te)
        return self.env.t.get_result()
