from typing import List

from dataclasses import dataclass, field
from utilsd.config import Registry
import numpy as np

import pymahjong as pm
from pymahjong import Yaku
from arena.common import yaku_to_tenhou


class PLAYER(metaclass=Registry, name="player"):
    pass


@dataclass
class Statistics():
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

    def dump_stats(self):
        lines = [
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
            f"役种："
        ] 
        for yaku in self.yakus:
            if self.yakus[yaku] != 0:
                lines.append(f"\t{yaku_to_tenhou[yaku]}: {self.yakus[yaku]} ({float(self.yakus[yaku]) / self.total_agali * 100:.2f}%)")
        return "\n".join(lines)
    
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


@PLAYER.register_module(name="base")
class BasePlayer():
    def __init__(
        self, 
        name: str
    ):
        self.name = name
        self.stat = Statistics()

    def play(
        self, 
        self_info: np.ndarray,
        record_info: np.ndarray,
        global_info: np.ndarray,
    ):
        raise NotImplementedError

    def update_stats(self, t: pm.Table, player_id: int):
        self.stat.update(t, player_id)

    def update_match_stats(self, match):
        # TODO
        pass    

    def dump_stats(self):
        return self.stat.dump_stats()
    

@PLAYER.register_module(name="random")
class RandomPlayer(BasePlayer):
    def __init__(
        self, 
        name: str
    ):
        super().__init__(name)

    def play(
        self, 
        self_info: np.ndarray,
        record_info: np.ndarray,
        global_info: np.ndarray,
        valid_actions_mask: np.ndarray,
    ):
        return np.random.choice(np.argwhere(valid_actions_mask).reshape([-1]))