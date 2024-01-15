from typing import List, Optional

import pandas as pd

from dataclasses import dataclass, field
from utilsd.config import Registry
import numpy as np

import pymahjong as pm
from pymahjong import Yaku
from arena.common import yaku_to_tenhou

import os


class PLAYER(metaclass=Registry, name="player"):
    pass


@dataclass
class Statistics():
    stat_path: str = None

    # ---- game stats ----
    total_games: int = 0
    total_agali: int = 0
    total_tsumo: int = 0
    total_furi: int = 0
    total_notile: int = 0
    total_notile_tenpai: int = 0
    total_riichi: int = 0
    total_menzen: int = 0
    agali_scores: List = field(
        default_factory=lambda: []
    )
    furi_scores: List = field(
        default_factory=lambda: []
    )
    yakus: dict = field(
        default_factory= lambda: {
            yaku: 0 for yaku in yaku_to_tenhou
        }
    )

    # ---- match stats ----
    total_match: int = 0
    total_1st: int = 0
    scores_1st: List = field(
        default_factory=lambda: []
    )
    total_2nd: int = 0
    scores_2nd: List = field(
        default_factory=lambda: []
    )
    total_3rd: int = 0
    scores_3rd: List = field(
        default_factory=lambda: []
    )
    total_4th: int = 0
    scores_4th: List = field(
        default_factory=lambda: []
    )
    total_strikeout: int = 0
    rating: float = 1500.

    def update_yakus(self, yakus: List[Yaku]):
        for yaku in yakus:
            self.yakus[yaku] += 1

    def update(self, t: pm.Table, player_id: int):
        self.total_games += 1
        if t.players[player_id].is_riichi():
            self.total_riichi += 1
        if t.players[player_id].is_menzen():
            self.total_menzen += 1
        result = t.get_result()
        result_type = result.result_type
        if result_type == pm.ResultType.TsumoAgari:
            winner = result.winner[0]
            if winner == player_id: # 自摸
                self.total_agali += 1
                self.total_tsumo += 1
                is_oya = t.oya == player_id
                # 计算本场数，不计算立直棒
                if is_oya:
                    score = result.results[winner].score1 * 3 + t.honba * 300
                else:
                    score = result.results[winner].score1 + result.results[winner].score2 * 2 + t.honba * 300
                self.agali_scores.append(score)
                self.update_yakus(result.results[player_id].yakus)
        elif result_type == pm.ResultType.RonAgari:
            if player_id in result.results: # 荣胡
                self.total_agali += 1
                loser = result.loser[0]
                for i in range(4):
                    if (pid := (loser + i) % 4) in result.results:
                        first_ron = pid
                        break
                if first_ron == player_id: # 头跳有场供
                    score = result.results[player_id].score1 + t.honba * 300
                else:
                    score = result.results[player_id].score1
                self.agali_scores.append(score)
                self.update_yakus(result.results[player_id].yakus)
            elif player_id == result.loser[0]: # 放铳
                self.total_furi += 1
                score = t.honba * 300
                for winner in result.results:
                    score += result.results[winner].score1
                self.furi_scores.append(score)
        elif result_type == pm.ResultType.NagashiMangan:
            # TODO: add nagashi mangan
            pass
        elif result_type == pm.ResultType.NoTileRyuuKyoku:
            self.total_notile += 1
            if t.players[player_id].is_tenpai():
                self.total_notile_tenpai += 1

    def update_match(self, match, player_id):
        if match.match_result["score"][player_id] < 0:
            self.total_strikeout += 1
        paired_score = zip(match.match_result["score"], range(4))
        paired_score = sorted(paired_score, key=lambda x: x[0], reverse=True)
        for i, (score, index) in enumerate(paired_score):
            if index == player_id:
                if i == 0:
                    self.total_1st += 1
                    self.scores_1st.append(score)
                    rating_base = 30
                elif i == 1:
                    self.total_2nd += 1
                    self.scores_2nd.append(score)
                    rating_base = 10
                elif i == 2:
                    self.total_3rd += 1
                    self.scores_3rd.append(score)
                    rating_base = -10
                elif i == 3:
                    self.total_4th += 1
                    self.scores_4th.append(score)
                    rating_base = -30
                break
        # use tenhou rating system
        num_match_modifier = (1 - self.total_match * 0.002) if self.total_match < 400 else 0.2
        ave_rating = sum([player.stat.rating for player in match.players]) / 4
        if ave_rating < 1500:
            ave_rating = 1500
        ave_rating_modifier = (ave_rating - self.rating) / 40
        self.rating += num_match_modifier * (rating_base + ave_rating_modifier)

        self.total_match += 1 


    def dump_stats(self):
        lines = [
            f"------------",
            f"总场数：{self.total_match}",
            f"Rating: {self.rating:.2f}",
            f"一位率：{self.match_1st_rate * 100:.2f}%",
            f"二位率：{self.match_2nd_rate * 100:.2f}%",
            f"三位率：{self.match_3rd_rate * 100:.2f}%",
            f"四位率：{self.match_4th_rate * 100:.2f}%",
            f"被飞率：{self.strikeout_rate * 100:.2f}%",
            f"均点：{self.mean_score:.2f}",
            f"一位均点：{self.mean_1st_score:.2f}",
            f"二位均点：{self.mean_2nd_score:.2f}",
            f"三位均点：{self.mean_3rd_score:.2f}",
            f"四位均点：{self.mean_4th_score:.2f}",
            f"------------",
            f"总局数：{self.total_games}",
            f"和牌率：{self.agali_rate * 100:.2f}%",
            f"自摸率：{self.tsumo_rate * 100:.2f}%",
            f"放铳率：{self.furi_rate * 100:.2f}%",
            f"流局率：{self.notile_rate * 100:.2f}%",
            f"流局听牌率: {self.notile_tenpai_rate * 100:.2f}%",
            f"立直率：{self.riichi_rate * 100:.2f}%",
            f"副露率：{self.naki_rate * 100:.2f}%",
            f"平均和牌点数：{self.mean_agali_score:.2f}",
            f"平均放铳点数：{self.mean_furi_score:.2f}",
            f"------------",
            f"役种："
        ] 
        for yaku in self.yakus:
            if self.yakus[yaku] != 0:
                lines.append(f"\t{yaku_to_tenhou[yaku]}: {self.yakus[yaku]} ({float(self.yakus[yaku]) / self.total_agali * 100:.2f}%)")
        return "\n".join(lines)
    
    def save_to_file(self, path=None):
        pd.to_pickle(self, path or self.stat_path)
       
    @classmethod     
    def load_from_file(cls, path=None) -> "Statistics":
        return pd.read_pickle(path)

    # ---- game properties ----
    @property
    def agali_rate(self):
        return float(self.total_agali) / self.total_games if self.total_games != 0 else 0.
    
    @property
    def furi_rate(self):
        return float(self.total_furi) / self.total_games if self.total_games != 0 else 0.
    
    @property
    def tsumo_rate(self):
        return float(self.total_tsumo) / self.total_agali if self.total_agali != 0 else 0.
    
    @property
    def notile_rate(self):
        return float(self.total_notile) / self.total_games if self.total_games != 0 else 0.
    
    @property
    def notile_tenpai_rate(self):
        return float(self.total_notile_tenpai) / self.total_notile if self.total_notile != 0 else 0.
    
    @property
    def riichi_rate(self):
        return float(self.total_riichi) / self.total_games if self.total_games != 0 else 0.
    
    @property
    def naki_rate(self):
        return 1 - float(self.total_menzen) / self.total_games if self.total_games != 0 else 0.
    
    @property
    def mean_agali_score(self):
        return np.mean(self.agali_scores) if len(self.agali_scores) > 0 else 0.
    
    @property
    def mean_furi_score(self):
        return np.mean(self.furi_scores) if len(self.furi_scores) > 0 else 0.

    # ---- match properties ----
    @property
    def strikeout_rate(self):
        return float(self.total_strikeout) / self.total_match if self.total_match != 0 else 0.

    @property
    def mean_score(self):
        return np.mean(self.scores_1st + self.scores_2nd + self.scores_3rd + self.scores_4th) if len(self.scores_1st + self.scores_2nd + self.scores_3rd + self.scores_4th) > 0 else 0.

    @property
    def match_1st_rate(self):
        return float(self.total_1st) / self.total_match if self.total_match != 0 else 0.
    
    @property
    def mean_1st_score(self):
        return np.mean(self.scores_1st) if len(self.scores_1st) > 0 else 0.

    @property
    def match_2nd_rate(self):
        return float(self.total_2nd) / self.total_match if self.total_match != 0 else 0.

    @property
    def mean_2nd_score(self):
        return np.mean(self.scores_2nd) if len(self.scores_2nd) > 0 else 0.

    @property
    def match_3rd_rate(self):
        return float(self.total_3rd) / self.total_match if self.total_match != 0 else 0.
    
    @property
    def mean_3rd_score(self):
        return np.mean(self.scores_3rd) if len(self.scores_3rd) > 0 else 0.

    @property
    def match_4th_rate(self):
        return float(self.total_4th) / self.total_match if self.total_match != 0 else 0.
    
    @property
    def mean_4th_score(self):
        return np.mean(self.scores_4th) if len(self.scores_4th) > 0 else 0.
    

@PLAYER.register_module(name="base")
class BasePlayer():
    def __init__(
        self, 
        name: str,
        stat_path: Optional[str] = None
    ):
        self.name = name
        if stat_path is not None and os.path.exists(stat_path):
            self.stat = Statistics.load_from_file(stat_path)
            self.stat.stat_path = stat_path
        else:
            self.stat = Statistics()
            self.stat.stat_path = stat_path
            
    def play(
        self, 
        self_info: np.ndarray,
        record_info: np.ndarray,
        global_info: np.ndarray,
    ):
        raise NotImplementedError

    def update_stats(self, t: pm.Table, player_id: int):
        self.stat.update(t, player_id)

    def update_match_stats(self, match, player_id):
        self.stat.update_match(match, player_id)  

    def dump_stats(self):
        return self.stat.dump_stats()
    
    def save_stats(self, path=None):
        path = path or self.stat.stat_path
        return self.stat.save_to_file(path)
    
    def load_stats(self, path=None):
        path = path or self.stat.stat_path
        return self.stat.load_from_file(path)
    

@PLAYER.register_module(name="random")
class RandomPlayer(BasePlayer):
    def __init__(
        self, 
        name: str,
        stat_path: Optional[str] = None
    ):
        super().__init__(name, stat_path)

    def play(
        self, 
        self_info: np.ndarray,
        record_info: np.ndarray,
        global_info: np.ndarray,
        valid_actions_mask: np.ndarray,
    ):
        return np.random.choice(np.argwhere(valid_actions_mask).reshape([-1]))