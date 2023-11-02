from enum import Flag
from copy import deepcopy
import re
import os
import time
import numpy as np
import warnings
import shutil
import sys
import gc

# 下面这些包用于下载和解析牌谱
# import eventlet
import xml.etree.ElementTree as ET
import urllib.request
import gzip

import pymahjong as mp
from pymahjong import BaseAction

from utils import *
from recorder import BaseRecorder, RLDataRecorderV1, NullRecorder, SupervisedRecorder, TenpaiRecorder

import joblib
# eventlet.monkey_patch()



# -------------- Hyper-parameters ------------------

# main_player_no = 0


min_dan = 16  # 最低段位限制，可以排除三麻的局（三麻的缺省player的dan=0）

at_least_how_many_tiles_remain_can_riichi = 4


def paipu_link(paipu):
    paipu = paipu[:-4]
    return r'http://tenhou.net/0/?log=' + paipu


class PaipuReplay:
    def __init__(self, data_recorder: BaseRecorder):
        self.num_games = 0
        self.success = 0
        self.total_games = 0
        self.errors = list()
        self.logger = logger()
        self.log_cache = ""
        self.write_log = False
        self.data_recorder = data_recorder

    def log(self, *info):
        # self.logger.log(*info)
        for s in info:
            self.log_cache += str(s)
            self.log_cache += '\n'

    def set_log(self, log):
        self.write_log = log

    def progress(self):
        print('Games {}/{}/{}'.format(self.success, self.num_games, self.total_games))

    def _paipu_replay(self, path, paipu):

        if not paipu.endswith('txt'):
            raise RuntimeError(f"Cannot read paipu {paipu}")
        filename = os.path.join(path, paipu)
        # log(filename)

        try:
            tree = ET.parse(filename)
        except Exception as e:
            raise RuntimeError(e.__str__(), f"Cannot read paipu {filename}")
        root = tree.getroot()
        self.log("解析牌谱为ElementTree成功！")

        replayer = None
        riichi_status = False
        after_kan = False
        for child in root:
            if child.tag == "BYE":
                return None  # not record this whole game

        self.data_recorder.record_this_game = True

        for main_player_no in range(4):

            for child_no, child in enumerate(root):
                if child.tag == "SHUFFLE":
                    seed_str = child.get("seed")
                    prefix = 'mt19937ar-sha512-n288-base64,'
                    if not seed_str.startswith(prefix):
                        self.log('Bad seed string')
                        continue
                    seed = seed_str[len(prefix):]
                    inst = mp.TenhouShuffle.instance()
                    inst.init(seed)

                elif child.tag == "GO":  # 牌桌规则和等级等信息.
                    # self.log(child.attrib)
                    try:
                        type_num = int(child.get("type"))
                        tmp = str(bin(type_num))

                        game_info = dict()
                        game_info["is_pvp"] = int(tmp[-1])
                        if not game_info["is_pvp"]:
                            break

                        game_info["no_aka"] = int(tmp[-2])
                        if game_info["no_aka"]:
                            break

                        game_info["no_kuutan"] = int(tmp[-3])
                        if game_info["no_kuutan"]:
                            break

                        game_info["is_hansou"] = int(tmp[-4])
                        # no requirement

                        game_info["is_3ma"] = int(tmp[-5])
                        if game_info["is_3ma"]:
                            break

                        game_info["is_pro"] = int(tmp[-6])
                        if not game_info["is_pro"]:
                            break

                        game_info["is_fast"] = int(tmp[-7])
                        if game_info["is_fast"]:
                            break

                        game_info["is_joukyu"] = int(tmp[-8])

                    except Exception as e:
                        self.log(e)
                        continue

                    for key in game_info:
                        self.log(key, game_info[key])

                    # 0x01	如果是PVP对战则为1
                    # 0x02	如果没有赤宝牌则为1
                    # 0x04	如果无食断则为1
                    # 0x08	如果是半庄则为1
                    # 0x10	如果是三人麻将则为1
                    # 0x20	如果是特上卓或凤凰卓则为1
                    # 0x40	如果是速卓则为1
                    # 0x80	如果是上级卓则为1

                elif child.tag == "TAIKYOKU":
                    self.log("=========== 此场开始 =========")
                    is_new_round = False
                    round_no = 0

                elif child.tag == "UN":
                    # 段位信息
                    if "dan" in child.attrib:
                        dans_str = child.get("dan").split(',')
                        dans = [int(tmp) for tmp in dans_str]

                        if np.any(np.array(dans) < min_dan):
                            return None

                elif child.tag == "INIT":
                    self.log("-----   此局开始  -------")

                    riichi_status = False
                    # 开局时候的各家分数
                    scores_str = child.get("ten").split(',')
                    scores = [int(tmp) * 100 for tmp in scores_str]
                    self.log("开局各玩家分数：", scores)

                    # Oya ID
                    oya_id = int(child.get("oya"))
                    self.log("庄家是玩家{}".format(oya_id))

                    # 什么局

                    game_order = int(child.get("seed").split(",")[0])
                    # self.log("此局是{}{}局".format(winds[game_order // 4], chinese_numbers[game_order % 4]))

                    # 本场和立直棒
                    honba = int(child.get("seed").split(",")[1])
                    riichi_sticks = int(child.get("seed").split(",")[2])
                    # self.log("此局是{}本场, 有{}根立直棒".format(honba, riichi_sticks))

                    # 骰子数字
                    dice_numbers = [int(child.get("seed").split(",")[3]) + 1, int(child.get("seed").split(",")[4]) + 1]
                    self.log("骰子的数字是", dice_numbers)

                    # 牌山
                    inst = mp.TenhouShuffle.instance()
                    yama = inst.generate_yama()
                    # self.log("牌山是: ", yama)

                    # 利用PaiPuReplayer进行重放
                    replayer = mp.PaipuReplayer()
                    if self.write_log:
                        replayer.set_write_log(True)
                    # self.log(f'Replayer.init: {yama} {scores} {riichi_sticks} {honba} {game_order // 4} {oya_id}')
                    replayer.init(yama, scores, riichi_sticks, honba, game_order // 4, oya_id)
                    # self.log('Init over.')

                    # Initialize data recorder with replayer
                    self.data_recorder.init(replayer)

                    # 开局的dora
                    dora_tiles = [int(child.get("seed").split(",")[5])]
                    self.log("开局DORA：{}".format(dora_tiles[-1]))

                    # 开局的手牌信息:
                    hand_tiles = []
                    for pid in range(4):
                        tiles_str = child.get("hai{}".format(pid)).split(",")
                        hand_tiles_player = [int(tmp) for tmp in tiles_str]
                        hand_tiles_player.sort()
                        hand_tiles.append(hand_tiles_player)
                        self.log("玩家{}的开局手牌是{}".format(pid, get_tiles_from_id(hand_tiles_player)))

                    game_has_init = True  # 表示这一局开始了
                    remaining_tiles = 70

                    chi_info = None  # for juding chi left , middle or right

                # ------------------------- 对局过程中信息 ---------------------------
                elif child.tag == "DORA":
                    dora_tiles.append(int(child.get("hai")))
                    self.log("翻DORA：{}".format(dora_tiles[-1]))

                elif child.tag == "REACH":
                    # 立直
                    player_id = int(child.get("who"))
                    if int(child.get("step")) == 1:
                        riichi_status = True
                        self.log("玩家{}宣布立直".format(player_id))

                    if int(child.get("step")) == 2:
                        riichi_status = False
                        self.log("玩家{}立直成功".format(player_id))
                        scores[player_id] -= 1000

                elif child.tag[0] in ["T", "U", "V", "W"] and child.attrib == {}:  # 摸牌
                    player_id = "TUVW".find(child.tag[0])
                    remaining_tiles -= 1
                    obtained_tile = int(child.tag[1:])
                    self.log("玩家{}摸牌{}".format(player_id, get_tile_from_id(obtained_tile)))
                    # 进行4次放弃动作
                    if after_kan:
                        after_kan = False
                    else:
                        if not (child_no - 1 < 0 or
                                root[child_no - 1].tag == "INIT"):
                            for _ in range(4):
                                self.data_recorder.before_selection(replayer.table, main_player_no)
                                aval_actions = replayer.table.get_response_actions()
                                made_action = aval_actions[0]
                                replayer.make_selection(0)
                                self.log(made_action.to_string())
                                self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                     made_action=made_action)

                elif child.tag[0] in ["D", "E", "F", "G"] and child.attrib == {}:  # 打牌
                    player_id = "DEFG".find(child.tag[0])
                    discarded_tile = int(child.tag[1:])
                    self.log("玩家{}舍牌{}".format(player_id, get_tile_from_id(discarded_tile)))
                    phase = int(replayer.table.get_phase())
                    self.log(f'phase {phase}')

                    self.log('SelfAction Options')
                    self_actions = replayer.get_self_actions()
                    for sa in self_actions:
                        self.log(sa.to_string() + '|')
                    self.log('SelfAction Options End')
                    if riichi_status:
                        selection = replayer.get_selection_from_action(BaseAction.Riichi, [discarded_tile])
                        self.log(f'Select: {selection}')

                        self.data_recorder.before_selection(replayer.table, main_player_no)
                        aval_actions = replayer.table.get_self_actions()
                        made_action = aval_actions[selection]
                        ret = replayer.make_selection(selection)
                        self.data_recorder.make_selection(replayer.table, main_player_no, made_action=made_action)

                        if not ret:
                            self.log('phase', int(phase))
                            self.log(f'要打 {get_tile_from_id(discarded_tile)}, Fail.\n')

                            raise ActionException('立直打牌', paipu, game_order, honba)
                    else:
                        selection = replayer.get_selection_from_action(BaseAction.Discard, [discarded_tile])
                        # print("selection:", selection, "disarded_tile:", discarded_tile)
                        self.log(f'Select: {selection}')

                        self.data_recorder.before_selection(replayer.table, main_player_no)
                        aval_actions = replayer.table.get_self_actions()
                        made_action = aval_actions[selection]
                        # print("actions:", [action.to_string() for action in aval_actions], selection, made_action)
                        ret = replayer.make_selection(selection)
                        self.data_recorder.make_selection(replayer.table, main_player_no, made_action=made_action)

                        if not ret:
                            self.log('phase', int(phase))
                            self.log(f'要打 {get_tile_from_id(discarded_tile)}, Fail.')

                            raise ActionException('打牌', paipu, game_order, honba)

                elif child.tag == "N":  # 鸣牌 （包括暗杠）
                    naru_player_id = int(child.get("who"))
                    player_id = naru_player_id
                    naru_tiles_int = int(child.get("m"))

                    # self.log("==========  Naru =================")
                    side_tiles_added_by_naru, hand_tiles_removed_by_naru, naru_is_aka, use_aka_to_naru, naru_type, opened = decodem(
                        naru_tiles_int, naru_player_id)

                    # opened = True表示是副露，否则是暗杠

                    self.log("玩家{}用手上的{}进行了一个{}，形成了{}".format(
                        naru_player_id, hand_tiles_removed_by_naru, naru_type, side_tiles_added_by_naru))
                    for i in range(4):
                        if replayer.get_phase() > int(mp.PhaseEnum.P4_ACTION):  # 回复阶段，除该人之外
                            if i != naru_player_id:
                                self.log(f'Select: {0}')

                                self.data_recorder.before_selection(replayer.table, main_player_no)
                                aval_actions = replayer.table.get_response_actions()
                                made_action = aval_actions[0]
                                replayer.make_selection(0)
                                self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                    made_action=made_action,
                                                                    use_aka_to_naru=use_aka_to_naru)

                        # else: #自己暗杠或者鸣牌
                        if i == naru_player_id:
                            response_types = ['Chi', 'Pon', 'Min-Kan']
                            action_types = {'Chi': BaseAction.Chi, 'Pon': BaseAction.Pon,
                                            'Min-Kan': BaseAction.Kan,
                                            'An-Kan': BaseAction.AnKan, 'Ka-Kan': BaseAction.KaKan}

                            if naru_type == 'Min-Kan':
                                after_kan = True

                            if naru_type in response_types:
                                self.log('Response Options')
                                response_actions = replayer.get_response_actions()
                                for ra in response_actions:
                                    self.log(ra.to_string() + "|")
                                self.log('Response Options End')
                            else:
                                self.log('SelfAction Options')
                                self_actions = replayer.get_self_actions()
                                for sa in self_actions:
                                    self.log(sa.to_string() + "|")
                                self.log('SelfAction Options End')

                            selection = replayer.get_selection_from_action(action_types[naru_type],
                                                                           hand_tiles_removed_by_naru)
                            self.log(f'Select: {selection}')

                            if naru_type == "Chi":
                                side_tiles_added_by_naru_id = [tt[0] // 4 for tt in side_tiles_added_by_naru]
                                start_index = min(side_tiles_added_by_naru_id)
                                hand_tiles_removed_by_naru_id = [tt // 4 for tt in hand_tiles_removed_by_naru]
                                chi_relative_order = sum(side_tiles_added_by_naru_id) - sum(
                                    hand_tiles_removed_by_naru_id) - start_index
                                chi_info = ['left', 'middle', 'right'][chi_relative_order]

                            self.data_recorder.before_selection(replayer.table, main_player_no)
                            if replayer.table.get_phase() < 4:
                                aval_actions = replayer.table.get_self_actions()
                            else:
                                aval_actions = replayer.table.get_response_actions()

                            made_action = aval_actions[selection]
                            ret = replayer.make_selection(selection)
                            self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                 made_action=made_action, chi_info=chi_info,
                                                                 use_aka_to_naru=use_aka_to_naru)

                            if not ret:
                                self.log(f'要{naru_type} {get_tiles_from_id(hand_tiles_removed_by_naru)}, Fail.\n'
                                         f'{replayer.table.players[naru_player_id].to_string()}')
                                raise ActionException(f'{naru_type}', paipu, game_order, honba)
                            if naru_type not in response_types:
                                break

                elif child.tag == "BYE":  # 掉线
                    bye_player_id = int(child.get("who"))
                    self.log("### 玩家{}掉线！！ ".format(bye_player_id))

                elif child.tag == "RYUUKYOKU" or child.tag == "AGARI":
                    self.log('~Game Over~')
                    score_info_str = child.get("sc").split(",")
                    score_info = [int(tmp) for tmp in score_info_str]
                    score_changes = [score_info[1] * 100, score_info[3] * 100, score_info[5] * 100, score_info[7] * 100]

                    score_changes1 = [_ for _ in score_changes]
                    score_changes2 = None
                    double_ron = False
                    if child.tag == "RYUUKYOKU":
                        if child.get('type') == 'yao9':
                            self.log("九种九牌！")

                            self.data_recorder.before_selection(replayer.table, main_player_no)
                            aval_actions = replayer.table.get_self_actions()
                            made_action = aval_actions[14]
                            replayer.make_selection(14)
                            self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                 made_action=made_action)

                        elif child.get('type') == 'ron3':
                            self.log('三家和牌！')
                            print("3 players RON!!!!!!!")
                            self.data_recorder.end_episode(score_changes[main_player_no] // 100)
                            continue
                        else:
                            for _ in range(4):
                                self.data_recorder.before_selection(replayer.table, main_player_no)
                                aval_actions = replayer.table.get_response_actions()
                                made_action = aval_actions[0]
                                replayer.make_selection(0)
                                self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                     made_action=made_action)

                        self.log("本局结束: 结果是流局")
                        if replayer.get_phase() != int(mp.PhaseEnum.GAME_OVER):
                            raise ActionException('牌局未结束', paipu, game_order, honba)

                        result = replayer.get_result()
                        result_score = result.score
                        target_score = [score_changes[i] + scores[i] for i in range(4)]
                        result_score_change = [result_score[i] - scores[i] for i in range(4)]
                        self.log(score_changes1, score_changes2, scores, result_score)
                        self.log(result.to_string())
                        for i in range(4):
                            if score_changes[i] + scores[i] == result_score[i]:
                                continue
                            else:
                                raise ScoreException(
                                    f'Now: {result_score}({result_score_change}) Expect: {target_score}({score_changes}) Original: {scores}',
                                    paipu, game_order, honba)

                        self.log('OK!')

                    elif child.tag == "AGARI":
                        who_agari = []
                        if child_no + 1 < len(root) and root[child_no + 1].tag == "AGARI":
                            double_ron = True
                            self.log("这局是Double Ron!!!!!!!!!!!!")
                            who_agari.append(int(root[child_no + 1].get("who")))

                            score_info_str2 = root[child_no + 1].get("sc").split(",")
                            score_info2 = [int(tmp) for tmp in score_info_str2]
                            score_changes2 = [score_info2[1] * 100, score_info2[3] * 100, score_info2[5] * 100,
                                              score_info2[7] * 100]
                            for i in range(4):
                                score_changes[i] += score_changes2[i]

                        agari_player_id = int(child.get("who"))
                        who_agari.append(int(child.get("who")))

                        for i in range(4):
                            phase = replayer.get_phase()
                            self.log(f'phase {phase}')
                            ret = True
                            if phase <= int(mp.PhaseEnum.P4_ACTION):
                                self.log('SelfAction Options')
                                self_actions = replayer.get_self_actions()
                                for sa in self_actions:
                                    self.log(sa.to_string() + "|")
                                self.log('SelfAction Options End')
                                selection = replayer.get_selection_from_action(BaseAction.Tsumo, [])
                                self.log(f'Select: {selection}')

                                self.data_recorder.before_selection(replayer.table, main_player_no)
                                aval_actions = replayer.table.get_self_actions()
                                made_action = aval_actions[selection]
                                ret = replayer.make_selection(selection)
                                self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                     made_action=made_action)

                                if not ret:
                                    self.log(replayer.table.players[int(phase)].to_string(),
                                             '听牌:' + replayer.table.players[int(phase)].tenpai_to_string()
                                             )

                                    raise ActionException('自摸', paipu, game_order, honba)
                                break

                            else:
                                if i not in who_agari:
                                    self.log('Select: 0')
                                    self.data_recorder.before_selection(replayer.table, main_player_no)
                                    aval_actions = replayer.table.get_response_actions()
                                    made_action = aval_actions[0]
                                    replayer.make_selection(0)
                                    self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                         made_action=made_action)

                                else:

                                    self.log('Response Options')
                                    response_actions = replayer.get_response_actions()
                                    for ra in response_actions:
                                        self.log(ra.to_string() + "|")
                                    self.log('Response Options End')
                                    if phase <= int(mp.PhaseEnum.P4_RESPONSE):
                                        selection = replayer.get_selection_from_action(BaseAction.Ron, [])
                                        self.log(f'Select: {selection}')

                                        self.data_recorder.before_selection(replayer.table, main_player_no)
                                        aval_actions = replayer.table.get_response_actions()
                                        made_action = aval_actions[selection]
                                        ret = replayer.make_selection(selection)
                                        self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                             made_action=made_action)

                                    elif phase <= int(mp.PhaseEnum.P4_chankan):
                                        selection = replayer.get_selection_from_action(BaseAction.ChanKan, [])
                                        self.log(f'Select: {selection}')

                                        self.data_recorder.before_selection(replayer.table, main_player_no)
                                        aval_actions = replayer.table.get_response_actions()
                                        made_action = aval_actions[selection]
                                        ret = replayer.make_selection(selection)
                                        self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                             made_action=made_action)

                                    elif phase <= int(mp.PhaseEnum.P4_chanankan):
                                        selection = replayer.get_selection_from_action(BaseAction.ChanAnKan, [])
                                        self.log(f'Select: {selection}')

                                        self.data_recorder.before_selection(replayer.table, main_player_no)
                                        aval_actions = replayer.table.get_response_actions()
                                        made_action = aval_actions[selection]
                                        ret = replayer.make_selection(selection)
                                        self.data_recorder.make_selection(replayer.table, main_player_no,
                                                                             made_action=made_action)

                                    if not ret:
                                        raise ActionException('荣和', paipu, game_order, honba)
                        if replayer.get_phase() != int(mp.PhaseEnum.GAME_OVER):
                            raise ActionException('牌局未结束', paipu, game_order, honba)

                        result = replayer.get_result()
                        result_score = result.score
                        target_score = [score_changes[i] + scores[i] for i in range(4)]
                        result_score_change = [result_score[i] - scores[i] for i in range(4)]
                        self.log(score_changes1, score_changes2, scores, result_score)
                        self.log(result.to_string())
                        for i in range(4):
                            if score_changes[i] + scores[i] == result_score[i]:
                                continue
                            else:
                                raise ScoreException(
                                    f'Now: {result_score}({result_score_change}) Expect: {target_score}({score_changes}) Original: {scores}',
                                    paipu, game_order, honba)

                        self.log('OK!')

                        # from_who = int(child.get("fromWho"))
                        # agari_tile = int(child.get("machi"))
                        # honba = int(child.get("ba").split(",")[0])
                        # riichi_sticks = int(child.get("ba").split(",")[1])

                        # if from_who == agari_player_id:
                        #     info = "自摸"
                        # else:
                        #     info = "玩家{}的点炮".format(from_who)

                        # self.log("玩家{} 通过{} 用{} 和了{}点 ({}本场,{}根立直棒)".format(
                        #     agari_player_id, info, agari_tile, score_changes[agari_player_id] , honba, riichi_sticks))
                        # self.log("和牌时候的手牌 (不包含副露):", [int(hai) for hai in child.get("hai").split(",")])
                        if double_ron:
                            # print("Double RON !!!!!!!!!!!!!!!!!")
                            assert score_changes[main_player_no] / 100 - score_changes[main_player_no] // 100 == 0
                            self.data_recorder.end_episode(score_changes[main_player_no] // 100)
                            break

                    # self.log("本局各玩家的分数变化是", score_changes)
                    assert score_changes[main_player_no] / 100 - score_changes[main_player_no] // 100 == 0
                    self.data_recorder.end_episode(score_changes[main_player_no] // 100)

                    if not (child_no + 1 < len(root) and root[child_no + 1].tag == "AGARI"):
                        game_has_end = True  # 这一局结束了
                        # num_games += 1

                        # if num_games % 100 == 0:
                        #     self.log("****************** 已解析{}局 ***************".format(num_games))

                    # if "owari" in child.attrib:
                    #     owari_scores = child.get("owari").split(",")
                    #     self.log("========== 此场终了，最终分数 ==========")
                    #     self.log(int(owari_scores[0]) * 100, int(owari_scores[2]) * 100,
                    #         int(owari_scores[4]) * 100, int(owari_scores[6]) * 100)
                else:
                    raise MahjongException(child.tag, child.attrib, "Unexpected Element!")

    def paipu_replay(self, basepath=".", part=0, mode='debug'):
        # -------- 读取2020年所有牌谱的url ---------------
        # 参考 https://m77.hatenablog.com/entry/2017/05/21/214529

        num_games = 0
        is_new_round = False

        # ----------------- start ---------------------
        self.part = part

        path = basepath

        if mode == 'mark':
            from datetime import datetime
            timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
            logfilename = './error_logs/error_' + timestr + '.log'
            fp = open(logfilename, 'w+')
            fp.close()
        files = os.listdir(path)  # 得到文件夹下的所有文件名称
        self.total_games = len(files)
        file_per_part = self.total_games // 4
        part_start = part * file_per_part
        part_end = (part + 1) * file_per_part if part < 3 else self.total_games

        self.total_games = part_end - part_start
        t_start = time.time()
        for num, paipu in enumerate(files[part_start:part_end]):

            if not paipu.endswith('txt'):
                continue

            if '0000' != paipu.split('-')[2]:
                continue

            self.num_games += 1
            t_now = time.time()
            t_eps = t_now - t_start
            t_eps_h = int(t_eps / 3600)
            t_eps_min = int((t_eps - 3600 * t_eps_h) / 60)
            t_eps_s = t_eps - 3600 * t_eps_h - 60 * t_eps_min
            t_eps_format = f"{t_eps_h:02d}:{t_eps_min:02d}:{t_eps_s:.02f}"

            t_eta = t_eps / self.num_games * self.total_games - t_eps
            t_eta_h = int(t_eta / 3600)
            t_eta_min = int((t_eta - 3600 * t_eta_h) / 60)
            t_eta_s = t_eta - 3600 * t_eta_h - 60 * t_eta_min
            t_eta_format = f"{t_eta_h:02d}:{t_eta_min:02d}:{t_eta_s:.02f}"

            print(f"{num}/{self.num_games}/{self.total_games} {paipu} elapse: {t_eps_format} eta: {t_eta_format}")
            try:
                self.log_cache = ""
                self._paipu_replay(path, paipu)
                self.success += 1
            except MahjongException as e:
                if mode == 'debug':
                    print(self.log_cache)
                    raise e
                elif mode == 'test':
                    self.errors.append(paipu)
                elif mode == 'mark':
                    fp = open(logfilename, 'a+')
                    print('MahjongException: ' + str(e) + ' ' + paipu, file=fp)
                    self.errors.append(paipu)
                    fp.close()
                else:
                    self.errors.append(paipu)
                continue

            except RuntimeError as e:
                if mode == 'debug':
                    print(self.log_cache)
                    raise e
                elif mode == 'test':
                    self.errors.append(paipu)
                elif mode == 'mark':
                    fp = open(logfilename, 'a+')
                    print('RuntimeError: ' + str(e) + ' ' + paipu, file=fp)
                    self.errors.append(paipu)
                    fp.close()
                else:
                    self.errors.append(paipu)
                continue

            except AttributeError as e:
                if mode == 'debug':
                    print(self.log_cache)
                    raise e
                elif mode == 'test':
                    self.errors.append(paipu)
                elif mode == 'mark':
                    fp = open(logfilename, 'a+')
                    print('RuntimeError: ' + str(e) + ' ' + paipu, file=fp)
                    self.errors.append(paipu)
                    fp.close()
                else:
                    self.errors.append(paipu)
                continue

    def paipu_replay_1(self, paipu_name):
        # basepath = os.path.abspath(__file__)
        # path = os.path.join(basepath, "paipuxmls")
        self._paipu_replay(".", paipu_name)


def paipu_replay(data_recorder, basepath, part, mode='debug'):
    if mode == 'debug':
        _logger = logger(fp='stdout')
    else:
        _logger = logger()
    replayer = PaipuReplay(data_recorder)
    replayer.logger = _logger
    replayer.paipu_replay(basepath=basepath, part=part, mode=mode)
    print(replayer.progress())
    return replayer


def paipu_replay_1(filename, data_recorder):
    replayer = PaipuReplay(data_recorder)
    replayer.set_log(True)
    replayer.logger = logger(fp='stdout')
    try:
        replayer.paipu_replay_1(filename)
        # print(replayer.log_cache)
    except Exception as e:
        print(replayer.log_cache)
        print(e)
    return replayer

def run_recorder(batch_id):
    month = batch_id % 12 + 1
    part = batch_id // 12
    batch_id = f"2020{month:02d}"
    print(batch_id, part)
    data_recorder = TenpaiRecorder(save_path=f"/data/yansen/mahjong/supervised_v2/2020/{batch_id}/{part}/")
    paipu_replay(data_recorder, os.path.join("/data/yansen/mahjong/paipuxmls/", f"{batch_id}"), part, mode='mark')
    data_recorder.save()

if __name__ == "__main__":
    joblib.Parallel(n_jobs=48)(joblib.delayed(run_recorder)(batch_id) for batch_id in range(48))
    # run_recorder(0)
