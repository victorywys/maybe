try:
    from typing import List
except:
    from typing_extensions import List

import json
import pymahjong as pm

from .common import tile_name_to_tenhou, tile_to_tenhou, get_base_tile, yaku_to_tenhou

import numpy as np

class TenhouJsonLogger():
    def __init__(
        self,
    ):
        pass
    
    def init_match(
        self, 
        player_names: List[str] = ["player 1", "player 2", "player 3", "player 4"],
        game_desc1: str = "",
        game_desc2: str = "",
        rule_desc: str = "1 game",
    ):
        self.title = [game_desc1, game_desc2]
        self.name = player_names
        self.rule = {
            "disp": rule_desc,
            "aka": 1,
        }
        self.logs = []
        self.log = []
        
    def start_game(
        self, 
        t: pm.Table,
        te: pm.TableEncoder,
    ):
        self.log = []
        global_info = te.global_infos[0]
        ju = global_info[0]
        benchang = global_info[2]
        changgong = global_info[3]
        start_points = [p * 100 for p in global_info[6:10]]
        # print(ju, benchang, changgong, start_points)
        self.log.append([ju, benchang, changgong])
        self.log.append(start_points)
        self.start_hand = [[] for _ in range(4)]
        self.draw_tiles = [[] for _ in range(4)]
        self.discard_tiles = [[] for _ in range(4)]
        for i in range(4):
            self_info = np.array(te.self_infos[i]).reshape([18, 34]).swapaxes(0, 1)

            hands = list(np.argwhere(self_info[:, 0]).reshape([-1])) + list(np.argwhere(self_info[:, 1]).reshape([-1])) + list(np.argwhere(self_info[:, 2]).reshape([-1])) + list(np.argwhere(self_info[:, 3]).reshape([-1]))
            hands.sort()

            zimopai = list(np.argwhere(self_info[:, 9]).reshape([-1]))
            if len(zimopai) > 0:
                hands.remove(zimopai[0])
                self.draw_tiles[i].append(tile_to_tenhou[zimopai[0]])

            hand_akas = list(np.argwhere(self_info[:, 6]).reshape([-1]))
            if 4 in hand_akas:
                if 4 in hands:
                    hands.remove(4)
                    hands.append(34)
                else:
                    self.draw_tiles[i][0] = tile_to_tenhou[34]
            if 13 in hand_akas:
                if 13 in hands:
                    hands.remove(13)
                    hands.append(35)
                else:
                    self.draw_tiles[i][0] = tile_to_tenhou[35]
            if 22 in hand_akas:
                if 22 in hands:
                    hands.remove(22)
                    hands.append(36)
                else:
                    self.draw_tiles[i][0] = tile_to_tenhou[36]

            hands = [tile_to_tenhou[tile] for tile in hands]
            self.start_hand[i] = hands
    
    def parse_actions(
        self,
        t: pm.Table,
        te: pm.TableEncoder,
    ):
        for player_i in range(4):
            records = te.records[player_i]
            for record_i, record in enumerate(records):
                a = np.argwhere(np.array(record)).reshape([-1])
                if 51 in a: # 关联自家
                    tile = a[0]
                    if 37 in a or 38 in a: # 摸牌，摸杠牌
                        self.draw_tiles[player_i].append(tile_to_tenhou[tile])
                    elif 39 in a: # 手切
                        self.discard_tiles[player_i].append(tile_to_tenhou[tile])
                    elif 40 in a: # 摸切
                        self.discard_tiles[player_i].append(60)
                    elif 41 in a or 42 in a or 43 in a: # 吃L M R
                        last_record = records[record_i - 1]
                        last_tile = np.argwhere(np.array(last_record)).reshape([-1])[0]
                        if last_tile > 36:
                            last_record = records[record_i - 2]
                            last_tile = np.argwhere(np.array(last_record)).reshape([-1])[0]

                        chi_tile1 = a[0]
                        chi_tile2 = a[1]
                        if get_base_tile(chi_tile2) < get_base_tile(chi_tile1):
                            chi_tile1, chi_tile2 = chi_tile2, chi_tile1
                        self.draw_tiles[player_i].append(f"c{tile_to_tenhou[last_tile]}{tile_to_tenhou[chi_tile1]}{tile_to_tenhou[chi_tile2]}")
                    elif 44 in a: # 碰
                        last_record = records[record_i - 1]
                        last_tile = np.argwhere(np.array(last_record)).reshape([-1])[0]

                        if last_tile > 36:
                            last_record = records[record_i - 2]
                            last_tile = np.argwhere(np.array(last_record)).reshape([-1])[0]

                        last_player = np.argwhere(np.array(last_record)).reshape([-1])[-1] - 51
                        pon_tile1 = a[0]
                        if a[1] > 36:
                            pon_tile2 = a[0]
                        else:
                            pon_tile2 = a[1]
                        if last_player == 1: # 下家
                            self.draw_tiles[player_i].append(f"{tile_to_tenhou[pon_tile1]}{tile_to_tenhou[pon_tile2]}p{tile_to_tenhou[last_tile]}")
                        elif last_player == 2: # 对家
                            self.draw_tiles[player_i].append(f"{tile_to_tenhou[pon_tile1]}p{tile_to_tenhou[last_tile]}{tile_to_tenhou[pon_tile2]}")
                        elif last_player == 3: # 上家
                            self.draw_tiles[player_i].append(f"p{tile_to_tenhou[last_tile]}{tile_to_tenhou[pon_tile1]}{tile_to_tenhou[pon_tile2]}")
                    elif 45 in a: # 明杠
                        last_record = records[record_i - 1]
                        last_tile = np.argwhere(np.array(last_record)).reshape([-1])[0]
                        last_player = np.argwhere(np.array(last_record)).reshape([-1])[-1] - 51

                        if last_tile > 36:
                            last_record = records[record_i - 2]
                            last_tile = np.argwhere(np.array(last_record)).reshape([-1])[0]
                            last_player = np.argwhere(np.array(last_record)).reshape([-1])[-1] - 51

                        kan_tile1 = kan_tile2 = last_tile
                        if tile_to_tenhou[last_tile] == 15:
                            kan_tile3 = 34
                        elif tile_to_tenhou[last_tile] == 25:
                            kan_tile3 = 35
                        elif tile_to_tenhou[last_tile] == 35:
                            kan_tile3 = 36
                        else:
                            kan_tile3 = last_tile
                        if last_player == 1: # 下家
                            self.draw_tiles[player_i].append(f"{tile_to_tenhou[kan_tile1]}{tile_to_tenhou[kan_tile2]}{tile_to_tenhou[kan_tile3]}m{tile_to_tenhou[last_tile]}")
                        elif last_player == 2: # 对家
                            self.draw_tiles[player_i].append(f"{tile_to_tenhou[kan_tile1]}m{tile_to_tenhou[last_tile]}{tile_to_tenhou[kan_tile2]}{tile_to_tenhou[kan_tile3]}")
                        elif last_player == 3: # 上家
                            self.draw_tiles[player_i].append(f"m{tile_to_tenhou[last_tile]}{tile_to_tenhou[kan_tile1]}{tile_to_tenhou[kan_tile2]}{tile_to_tenhou[kan_tile3]}")
                        self.discard_tiles[player_i].append(0)
                    elif 46 in a: # 暗杠
                        tile = a[0]
                        if tile_to_tenhou[tile] == 15 or tile_to_tenhou[tile] == 51:
                            kan_tile1 = kan_tile2 = kan_tile3 = 4
                            kan_tile4 = 34
                        elif tile_to_tenhou[tile] == 25 or tile_to_tenhou[tile] == 52:
                            kan_tile1 = kan_tile2 = kan_tile3 = 13
                            kan_tile4 = 35
                        elif tile_to_tenhou[tile] == 35 or tile_to_tenhou[tile] == 53:
                            kan_tile1 = kan_tile2 = kan_tile3 = 22
                            kan_tile4 = 36
                        else:
                            kan_tile1 = kan_tile2 = kan_tile3 = kan_tile4 = tile
                        self.draw_tiles[player_i].append(f"a{tile_to_tenhou[kan_tile1]}{tile_to_tenhou[kan_tile2]}{tile_to_tenhou[kan_tile3]}{tile_to_tenhou[kan_tile4]}")
                        self.discard_tiles[player_i].append(0)
                    elif 47 in a: # 加杠
                        tile = a[0]
                        kan_tile = tile
                        base_tile = get_base_tile(tile)
                        for record_j in range(record_i):
                            related_record = np.argwhere(np.array(records[record_j])).reshape([-1])
                            if 44 in related_record:
                                related_tile = related_record[0]
                                related_tile = get_base_tile(related_tile)
                                if related_tile == base_tile:
                                    pon_tile1 = related_record[0]
                                    if related_record[1] < 37:
                                        pon_tile2 = related_record[1]
                                    else:
                                        pon_tile2 = related_record[0]
                                    source_record = records[record_j - 1]
                                    source_player = np.argwhere(np.array(source_record)).reshape([-1])[-1] - 51
                                    source_tile = np.argwhere(np.array(source_record)).reshape([-1])[0]
                                    if source_tile > 36:
                                        source_record = records[record_j - 2]
                                        source_tile = np.argwhere(np.array(source_record)).reshape([-1])[0]
                                        source_player = np.argwhere(np.array(source_record)).reshape([-1])[-1] - 51

                                    if source_player == 1: # 下家
                                        self.discard_tiles[player_i].append(f"{tile_to_tenhou[pon_tile1]}{tile_to_tenhou[pon_tile2]}k{tile_to_tenhou[kan_tile]}{tile_to_tenhou[source_tile]}")
                                    elif source_player == 2: # 对家
                                        self.discard_tiles[player_i].append(f"{tile_to_tenhou[pon_tile1]}k{tile_to_tenhou[kan_tile]}{tile_to_tenhou[source_tile]}{tile_to_tenhou[pon_tile2]}")
                                    elif source_player == 3: # 上家
                                        self.discard_tiles[player_i].append(f"k{tile_to_tenhou[kan_tile]}{tile_to_tenhou[source_tile]}{tile_to_tenhou[pon_tile1]}{tile_to_tenhou[pon_tile2]}")
                                    break
                    elif 48 in a: # 手切立直
                        tile = a[0]
                        self.discard_tiles[player_i].append(f"r{tile_to_tenhou[tile]}")
                    elif 49 in a: # 摸切立直
                        self.discard_tiles[player_i].append("r60")
    
    
    
    def result_details(
        self,
        counterResult: pm.CounterResult,
        is_tsumo: bool,
        is_oya: bool,
    ):
        def fan_fu_str(fan, fu):
            if fan >= 13:
                return "役満"
            elif fan >= 11:
                return "三倍満"
            elif fan >= 8:
                return "倍満"
            elif fan >= 6:
                return "跳満"
            elif (fan == 5) or (fan == 4 and fu > 30) or (fan == 3 and fu > 70):
                return "満貫"
            else:
                return f"{fu}符{fan}飜"
        ret = []
        fan, fu = counterResult.fan, counterResult.fu
        score1 = counterResult.score1
        score2 = counterResult.score2
        yakus = counterResult.yakus
        if is_tsumo:
            if is_oya:
                ret.append(f"{fan_fu_str(fan, fu)}{score1}点∀")
            else:
                ret.append(f"{fan_fu_str(fan, fu)}{score2}-{score1}点")
        else:
            ret.append(f"{fan_fu_str(fan, fu)}{score1}点")
            
        dora_num = 0
        # show_ura = False
        ura_num = 0
        aka_num = 0
        for yaku in yakus:
            if yaku == pm.Yaku.Dora:
                dora_num += 1
            elif yaku == pm.Yaku.UraDora:
                ura_num += 1
            elif yaku == pm.Yaku.AkaDora:
                aka_num += 1
            else:
                if yaku == pm.Yaku.Riichi:
                    # show_ura = True
                    pass
                ret.append(yaku_to_tenhou[yaku])
        if dora_num > 0:
            ret.append(f"ドラ({dora_num}飜)")
        if aka_num > 0:
            ret.append(f"赤ドラ({aka_num}飜)")
        if ura_num > 0:
            ret.append(f"裏ドラ({ura_num}飜)")
            
        return ret
    
    def parse_results(
        self,
        t: pm.Table,
        te: pm.TableEncoder,
    ):
        score_last_round = [score * 100 for score in te.global_infos[0][6:10]]
        score_last_round = [score_last_round[0], score_last_round[3], score_last_round[2], score_last_round[1]]
        # print(score_last_round)
        result = t.get_result()
        # print(result.score)
        result_type = result.result_type
        if result_type == pm.ResultType.RonAgari:
            oya = t.oya
            self.result = [
                "和了"
            ]
            loser = result.loser[0]
            first_ron = True
            for i in range(4):
                player_id = (loser + i) % 4
                if player_id in result.results:
                    score_change = [0, 0, 0, 0]
                    winner = player_id
                    if first_ron:
                        score_change[winner] = result.results[winner].score1 + t.riichibo * 1000 + t.honba * 300
                        score_change[loser] = -result.results[winner].score1 - t.honba * 300
                        first_ron = False
                    else:
                        score_change[winner] = result.results[winner].score1
                        score_change[loser] = -result.results[winner].score1
                    self.result.append(score_change)
                    self.result.append([winner, loser, winner, *(self.result_details(result.results[winner], False, winner == oya))])
        elif result_type == pm.ResultType.TsumoAgari:
            oya = t.oya
            winner = result.winner[0]
            self.result = [
                "和了", 
                [result.score[i] - score_last_round[i] for i in range(4)], 
                [winner, winner, winner, *(self.result_details(result.results[winner], True, winner == oya))],
            ]
        elif result_type == pm.ResultType.NoTileRyuuKyoku:
            self.result = ["流局", [result.score[i] - score_last_round[i] for i in range(4)]]
        elif result_type == pm.ResultType.NagashiMangan:
            self.result = ["流し満貫", [result.score[i] - score_last_round[i] for i in range(4)]]
        elif result_type == pm.ResultType.Ryukyouku_Interval_9Hai:
            self.result = ["九種九牌"]
        elif result_type == pm.ResultType.Ryukyouku_Interval_4Wind:
            self.result = ["四風連打"]
        elif result_type == pm.ResultType.Ryukyouku_Interval_4Kan:
            self.result = ["四槓散了"]
        elif result_type == pm.ResultType.Ryukyouku_Interval_4Riichi:
            self.result = ["四家立直"]
        elif result_type == pm.ResultType.Ryukyouku_Interval_3Ron:
            self.result = ["三家和了"]
    
    def end_game(
        self,
        t: pm.Table,
        te: pm.TableEncoder,
    ):
        dora_indicator = []
        ura_indicator = []
        for i in range(t.n_active_dora):
            dora_indicator.append(tile_name_to_tenhou[t.dora_indicator[i].to_string()])
        result = t.get_result()
        results = result.results
        reveal_ura_indicator = False
        for player_i in results:
            for yaku in results[player_i].yakus:
                if yaku == pm.Yaku.Riichi:
                    reveal_ura_indicator = True
                    break
        if reveal_ura_indicator:
            for i in range(t.n_active_dora):
                ura_indicator.append(tile_name_to_tenhou[t.uradora_indicator[i].to_string()])
        
        self.parse_actions(t, te)
        self.parse_results(t, te)
        
         
        self.log.append(dora_indicator)
        self.log.append(ura_indicator)
        self.log.append(self.start_hand[0])
        self.log.append(self.draw_tiles[0])
        self.log.append(self.discard_tiles[0])
        self.log.append(self.start_hand[1])
        self.log.append(self.draw_tiles[1])
        self.log.append(self.discard_tiles[1])
        self.log.append(self.start_hand[2])
        self.log.append(self.draw_tiles[2])
        self.log.append(self.discard_tiles[2])
        self.log.append(self.start_hand[3])
        self.log.append(self.draw_tiles[3])
        self.log.append(self.discard_tiles[3])
        self.log.append(self.result)
        self.logs.append(self.log)
    
    def dump_game(self):
        json_obj = {
            "title": self.title,
            "name": self.name,
            "rule": self.rule,
            "log": [self.log],
        }
        return json.dumps(json_obj)
    
    def dump_match(self):
        json_obj = {
            "title": self.title,
            "name": self.name,
            "rule": self.rule,
            "log": self.logs,
        }
        return json.dumps(json_obj)
    
    def dump_urls(self):
        return ['https://tenhou.net/6/#json=' + json.dumps({
            'title': self.title,
            'name': self.name,
            'rule': self.rule,
            'log': [log],
        }, ensure_ascii=False, separators=(',', ':')) for log in self.logs]