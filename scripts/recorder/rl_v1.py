import scipy.io as sio
import numpy as np
import warnings

import pymahjong as mp
from pymahjong import BaseAction

from .base import BaseRecorder
from utils import *

import gc


max_ten_diff = 25000  # 排除点数差距过大时的非正常打法
min_ten = 10000  # 排除点数差距过大时的非正常打法

PLAYER_OBS_DIM = 93
ORACLE_OBS_DIM = 18
ACTION_DIM = 47

class RLDataRecorderV1(BaseRecorder):
    def __init__(self, max_all_steps=500000):
        self.max_all_steps = max_all_steps
        self.x = np.zeros([max_all_steps, PLAYER_OBS_DIM, 34], dtype=bool)
        self.o = np.zeros([max_all_steps, ORACLE_OBS_DIM, 34], dtype=bool)
        self.m = np.zeros([max_all_steps, ACTION_DIM], dtype=bool)  # valid actions mask
        self.a = np.zeros([max_all_steps], dtype=np.uint8) + 255
        self.r = np.zeros([max_all_steps], dtype=np.float32)
        self.v = np.zeros([max_all_steps], dtype=bool)
        self.d = np.zeros([max_all_steps], dtype=bool)
        self.step = -1
        self.curr_episode_step_start = self.step + 1
        self.batch = 0

        self.obs_container = np.zeros([PLAYER_OBS_DIM + ORACLE_OBS_DIM, 34], dtype=np.int8)
        self.act_container = np.zeros([ACTION_DIM], dtype=np.int8)

        self.riichi_tiles = []
        self.aval_actions = []
        self.curr_player = -1
        self.record_this_game = True
        
    def init(self, replayer):
        scores = []
        for player in replayer.table.players:
            scores.append(player.score)                 
                    
        if max(scores) - min(scores) > max_ten_diff or min(scores) < min_ten:
            self.record_this_game = False  # not record this whole game
        else:
            self.record_this_game = True
        

    def clear_curr_episode_data(self):
        for tmp in [self.x, self.o, self.m, self.r, self.v, self.d]:
            tmp[self.curr_episode_step_start:self.step + 1] = 0
        self.a[self.curr_episode_step_start:self.step + 1] = 255
        self.step = self.curr_episode_step_start - 1

    def check_capacity(self):
        if self.step >= self.max_all_steps - 1:
            # save data
            self.save()
            print("Batch {} recording finish!".format(self.batch))
            # re-init
            self.x = np.zeros([self.max_all_steps, PLAYER_OBS_DIM, 34], dtype=bool)
            self.o = np.zeros([self.max_all_steps, ORACLE_OBS_DIM, 34], dtype=bool)
            self.m = np.zeros([self.max_all_steps, ACTION_DIM], dtype=bool)  # valid actions mask
            self.a = np.zeros([self.max_all_steps], dtype=np.uint8) + 255
            self.r = np.zeros([self.max_all_steps], dtype=np.float32)
            self.v = np.zeros([self.max_all_steps], dtype=bool)
            self.d = np.zeros([self.max_all_steps], dtype=bool)
            self.step = -1
            gc.collect()

    def end_episode(self, main_player_payoff):
        if self.record_this_game:
            if main_player_payoff != 0 and (main_player_payoff % 100) == 0:
                warnings.warn("!!!!! Make sure the reward is point / 100 !!!!!!!!!!!")
            # the unit here is divided by 100
            self.d[self.step] = 1
            self.r[self.step] = main_player_payoff
            self.step += 1
            self.check_capacity()
            self.riichi_tiles = []
            self.aval_actions = []
            self.curr_player = -1
            self.curr_episode_step_start = self.step + 1

    def make_selection(self, table, pid, made_action, chi_info=None):
        if self.record_this_game and len(self.aval_actions) > 1 and self.curr_player == pid:
            if len(self.riichi_tiles) > 0 and table.get_selected_action_tile() is not None and \
                    table.get_selected_action_tile().tile in self.riichi_tiles and \
                    made_action.action in [BaseAction.Riichi, BaseAction.Discard]:
                # 2-stages riichi
                self.a[self.step] = int(table.get_selected_action_tile().tile)  # first discard the tile
                self.before_selection(table, pid, riichi_step2_tile=self.a[self.step])  # step += 1
                if made_action.action == BaseAction.Riichi:
                    self.a[self.step] = RIICHI  # riichi
                elif made_action.action == BaseAction.Discard:
                    self.a[self.step] = PASS_RIICHI  # pass
                else:
                    self.clear_curr_episode_data()
                    raise MahjongException("============ self action exception ==============")
            else:
                if made_action.action == BaseAction.Pass:
                    self.a[self.step] = PASS_RESPONSE
                elif made_action.action == BaseAction.Discard:
                    self.a[self.step] = int(table.get_selected_action_tile().tile)
                elif made_action.action == BaseAction.Chi:
                    if chi_info == 'left':
                        self.a[self.step] = CHILEFT
                    elif chi_info == 'middle':
                        self.a[self.step] = CHIMIDDLE
                    elif chi_info == 'right':
                        self.a[self.step] = CHIRIGHT
                    else:
                        self.clear_curr_episode_data()
                        raise MahjongException("============ 不能判断吃哪个 ==============")

                elif made_action.action == BaseAction.Pon:
                    self.a[self.step] = PON
                elif made_action.action == BaseAction.AnKan:
                    self.a[self.step] = ANKAN
                elif made_action.action == BaseAction.Kan:
                    self.a[self.step] = MINKAN
                elif made_action.action == BaseAction.KaKan:
                    self.a[self.step] = KAKAN
                elif made_action.action in [BaseAction.Ron, BaseAction.ChanKan, BaseAction.ChanAnKan]:
                    self.a[self.step] = RON
                elif made_action.action == BaseAction.Tsumo:
                    self.a[self.step] = TSUMO
                elif made_action.action == BaseAction.Kyushukyuhai:
                    self.a[self.step] = PUSH
                else:
                    print("phase = ", table.get_phase())
                    print("current player =", self.curr_player)
                    print(table.get_selected_base_action())
                    print(self.riichi_tiles)
                    print(table.get_selected_action_tile().tile)
                    print("[Player 0]")
                    print(table.players[0].to_string())
                    print("[Player 1]")
                    print(table.players[1].to_string())
                    print("[Player 2]")
                    print(table.players[2].to_string())
                    print("[Player 3]")
                    print(table.players[3].to_string())

                    self.clear_curr_episode_data()
                    raise MahjongException("============ response action exception ==============")

            if self.m[self.step, self.a[self.step]] != 1:
                print("============  self.m[self.step, self.a[self.step]] != 1 ==============")
                print("made action = ", made_action.to_string())
                print("action = ", self.a[self.step])
                print("mask = ", self.m[self.step])
                print("allowed actions = ", [action_i for action_i in range(self.m[self.step].shape[-1]) if self.m[self.step][action_i]])
                print("phase = ", table.get_phase())
                print("current player =", self.curr_player)
                print(table.get_selected_base_action())
                print(self.riichi_tiles)
                print(table.get_selected_action_tile().tile)
                print("[Player 0]")
                print(table.players[0].to_string())
                print("[Player 1]")
                print(table.players[1].to_string())
                print("[Player 2]")
                print(table.players[2].to_string())
                print("[Player 3]")
                print(table.players[3].to_string())

                self.clear_curr_episode_data()
                raise MahjongException("========== Wrong action encoding!! ================")

    def before_selection(self, table, pid, riichi_step2_tile=None):

        if self.record_this_game:
            phase = table.get_phase()
            self.curr_player = phase % 4
            if phase < 4:  # play stage
                self.aval_actions = table.get_self_actions()
                self.riichi_tiles = mp.encv1_get_riichi_tiles(table)
            elif phase < 16:
                self.aval_actions = table.get_response_actions()
            else:
                self.aval_actions = [-1]

            if (len(self.aval_actions) > 1 and self.curr_player == pid) or riichi_step2_tile is not None:
                self.check_capacity()
                self.step += 1
                if riichi_step2_tile is None:
                    self.obs_container.fill(0)
                    self.act_container.fill(0)
                    mp.encv1_encode_action(table, pid, self.act_container)  # no need zeros
                    mp.encv1_encode_table(table, pid, True, self.obs_container)
                    self.act_container[RIICHI] = 0
                    self.act_container[PASS_RIICHI] = 0
                    if not self.act_container.sum() > 1:
                        print("[Player 0]")
                        print(table.players[0].to_string())
                        print("[Player 1]")
                        print(table.players[1].to_string())
                        print("[Player 2]")
                        print(table.players[2].to_string())
                        print("[Player 3]")
                        print(table.players[3].to_string())
                else:
                    mp.encv1_encode_table_riichi_step2(table, riichi_step2_tile, self.obs_container)
                    self.act_container.fill(0)
                    self.act_container[RIICHI] = 1
                    self.act_container[PASS_RIICHI] = 1
                    self.riichi_tiles = []

                self.x[self.step] = self.obs_container[:93, :]
                self.o[self.step] = self.obs_container[-18:, :]
                self.r[self.step] = 0
                self.v[self.step] = 1
                self.m[self.step] = self.act_container

    def save(self):
        data = {}
        data["X"] = self.x
        data["O"] = self.o
        data["M"] = self.m
        data["A"] = self.a
        data["R"] = self.r
        data["D"] = self.d
        data["V"] = self.v
        sio.savemat("./mahjong-offline-data-batch-{}.mat".format(self.batch), data)
        self.batch += 1
        del data
