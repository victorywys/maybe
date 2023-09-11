import pymahjong as pm
from pymahjong import BaseAction, BaseTile

from copy import deepcopy

import numpy as np
from utils import *

import pandas as pd

import os

RECORD_DIM = 55

class SupervisedRecorder():
    def __init__(self, save_path="./data/supervised"):
        self.save_path = save_path

        self.record_this_game = True

        self.self_infos = []
        self.records = []
        self.global_infos = []
        self.actions = []
        self.action_mask = []

        self.act_container = np.zeros([ACTION_DIM], dtype=np.int8)
        self.batch_num = 0
        self._max_allowed_samples = 50000
        self.curr_player = -1

    def init(self, replayer):
        self.te = pm.TableEncoder(replayer.table)
        self.te.init()
        self.te.update()

    def check_capacity(self):
        assert len(self.self_infos) == len(self.records) == len(self.global_infos) == len(self.actions) == len(self.action_mask), \
        "Lengths of self_infos, records, global_infos, actions, action_mask are not equal.\n"
        "self_infos: {}, records: {}, global_infos: {}, actions: {}, action_mask: {}".format(
            len(self.self_infos), len(self.records), len(self.global_infos), len(self.actions), len(self.action_mask)
        )
        if len(self.self_infos) >= self._max_allowed_samples:
            self.save(self.save_path)
            self.self_infos = []
            self.records = []
            self.global_infos = []
            self.actions = []
            self.action_mask = []

    def end_episode(self, main_player_payoff):
        if self.record_this_game:
            self.check_capacity()
        self.riichi_tiles = []
        self.aval_actions = []
        self.curr_player = -1

    def drop_last_obs(self):
        self.action_mask.pop()
        self.global_infos.pop()
        self.self_infos.pop()
        self.records.pop()

    def make_selection(self, table, pid, made_action, chi_info=None, use_aka_to_naru=False):
        if self.record_this_game and len(self.aval_actions) > 1 and self.curr_player == pid:
            if len(self.riichi_tiles) > 0 and table.get_selected_action_tile() is not None and \
                    table.get_selected_action_tile().tile in self.riichi_tiles and \
                    made_action.action in [BaseAction.Riichi, BaseAction.Discard]:
                # 2-stages riichi
                # FIXME: in state encoding v2, if we're using 2-stage riichi for supervised learning,
                #  then it seems that the tile selection action and the riichi action share the same state
                #  it might cause the problem where the same x corresponds to different ys.
                action = int(table.get_selected_action_tile().tile)  # first discard the tile
                self.actions.append(action)
                self.before_selection(table, pid, riichi_step2_tile=action)  # step += 1
                if made_action.action == BaseAction.Riichi:
                    action = RIICHI  # riichi
                elif made_action.action == BaseAction.Discard:
                    action = PASS_RIICHI  # pass
                else:
                    self.drop_last_obs()
                    raise MahjongException("============ self action exception ==============")
            else:
                if made_action.action == BaseAction.Pass:
                    action = PASS_RESPONSE
                elif made_action.action == BaseAction.Discard:
                    tile = table.get_selected_action_tile()
                    if tile.red_dora:
                        if tile.tile == BaseTile._5m:
                            action = DISCARD_0M
                        elif tile.tile == BaseTile._5p:
                            action = DISCARD_0P
                        elif tile.tile == BaseTile._5s:
                            action = DISCARD_0S
                        else:
                            raise MahjongException("============ 不能判断红宝牌是哪个 ==============")
                    else:
                        action = int(tile.tile)
                elif made_action.action == BaseAction.Chi:
                    if chi_info == 'left':
                        action = CHILEFT_AKA if use_aka_to_naru else CHILEFT
                    elif chi_info == 'middle':
                        action = CHIMIDDLE_AKA if use_aka_to_naru else CHIMIDDLE
                    elif chi_info == 'right':
                        action = CHIRIGHT_AKA if use_aka_to_naru else CHIRIGHT
                    else:
                        self.drop_last_obs()
                        raise MahjongException("============ 不能判断吃哪个 ==============")

                elif made_action.action == BaseAction.Pon:
                    action = PON_AKA if use_aka_to_naru else PON
                elif made_action.action == BaseAction.AnKan:
                    action = ANKAN
                elif made_action.action == BaseAction.Kan:
                    action = MINKAN
                elif made_action.action == BaseAction.KaKan:
                    action = KAKAN
                elif made_action.action in [BaseAction.Ron, BaseAction.ChanKan, BaseAction.ChanAnKan]:
                    action = RON
                elif made_action.action == BaseAction.Tsumo:
                    action = TSUMO
                elif made_action.action == BaseAction.Kyushukyuhai:
                    action = PUSH
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

                    self.drop_last_obs()
                    raise MahjongException("============ response action exception ==============")

            if self.action_mask[-1][action] != 1:
                print("============  self.m[self.step, self.a[self.step]] != 1 ==============")
                print("made action = ", made_action.to_string())
                print("action = ", action)
                print("mask = ", self.action_mask[-1])
                print("allowed actions = ", [action_i for action_i in range(self.action_mask[-1].shape[-1]) if self.action_mask[-1][action_i]])
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

                self.drop_last_obs()
                raise MahjongException("========== Wrong action encoding!! ================")

            self.actions.append(action)

    def before_selection(self, table, pid, riichi_step2_tile=None):
        self.te.update()
        if self.record_this_game:
            phase = table.get_phase()
            self.curr_player = phase % 4
            if phase < 4: # play stage
                self.aval_actions = table.get_self_actions()
                self.riichi_tiles = pm.encv1_get_riichi_tiles(table)
            elif phase < 16:
                self.aval_actions = table.get_response_actions()
            else:
                self.aval_actions = [-1]

            if (len(self.aval_actions) > 1 and self.curr_player == pid) or riichi_step2_tile is not None:
                self.check_capacity()
                self_info = np.array(self.te.self_infos[self.curr_player]).reshape([18, 34]).swapaxes(0, 1)
                record = np.array(self.te.records[self.curr_player])
                if record.ndim == 1:
                    record = np.zeros([0, RECORD_DIM])
                global_info = np.array(self.te.global_infos[self.curr_player])
                if riichi_step2_tile is None:
                    self.act_container.fill(0)
                    pm.encv1_encode_action(table, pid, self.act_container)
                    self.act_container[RIICHI] = 0
                    self.act_container[PASS_RIICHI] = 0
                else:
                    self.act_container.fill(0)
                    self.act_container[RIICHI] = 1
                    self.act_container[PASS_RIICHI] = 1
                    self.riichi_tiles = []

                self.self_infos.append(self_info)
                self.global_infos.append(global_info)
                self.records.append(record)
                self.action_mask.append(self.act_container)

    def pad_record(self):
        max_len = max([len(record) for record in self.records])
        for i in range(len(self.records)):
            self.records[i] = np.pad(self.records[i], ((0, max_len - len(self.records[i])), (0, 0)))

    def save(self, path=None):
        if path is None:
            path = self.save_path
        data = {}
        data["self_info"] = np.stack(self.self_infos)
        data["global_info"] = np.stack(self.global_infos)
        # self.pad_record()
        data["record"] = deepcopy(self.records)
        data["action"] = np.array(self.actions)
        data["action_mask"] = np.stack(self.action_mask)
        if not os.path.isdir(path):
            os.makedirs(path)
        pd.to_pickle(data, os.path.join(path, f"supervised_{self.batch_num}.pkl"))
        print(f"batch saved to {os.path.join(path, f'supervised_{self.batch_num}.pkl')}")
        self.batch_num += 1
        del data
