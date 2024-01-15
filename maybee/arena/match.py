from typing import List, Literal
from dataclasses import dataclass
import random


from .game import Game
from .player import BasePlayer
from .logger import TenhouJsonLogger


@dataclass
class GameName:
    id: int = 0
    
    @property
    def wind(self):
        return ["east", "south", "west", "north"][self.id // 4]
    
    @property
    def oya(self):
        return self.id % 4
    
    def next(self):
        self.id += 1


class Match:
    def __init__(
        self,
        players: List[BasePlayer],
        match_type: Literal["hanchan", "tonpuusen"] = "hanchan",
        shuffle_players: bool = True,
        match_desc: str = "",
    ):
        self.players = players
        assert len(self.players) == 4, "There should be 4 players in a match."
        if shuffle_players:
            random.shuffle(self.players)
        self.match_type = match_type
        assert self.match_type in ["hanchan", "tonpuusen"], "Invalid game type."
        self.th_logger = TenhouJsonLogger()
        self.th_logger.init_match(
            player_names=[player.name for player in self.players],
            game_desc1=match_desc,
            rule_desc=match_type,
        )
        self.game = Game(self.players, self.th_logger)
        
        
    def play_match(self):
        game_name = GameName()
        score = [25000, 25000, 25000, 25000]
        renchan = False
        honba = 0
        kyoutaku = 0
        while not self.is_over(game_name, score, renchan):
            self.game.start_new(
                oya=game_name.oya,
                game_wind=game_name.wind,
                scores=score,
                honba=honba,
                kyoutaku=kyoutaku,
            )
            result = self.game.play()
            for i, player in enumerate(self.players):
                player.update_stats(self.game.env.t, i)
            honba = result.n_honba
            kyoutaku = result.n_riichibo
            score = result.score
            renchan = result.renchan
            if not renchan:
                game_name.next()
        if kyoutaku > 0:
            # kyoutaku is given to the player who has the highest score
            score[self.top_player_index(score)] += kyoutaku * 1000
        self.match_result = {
            "score": score,
        }
        for i, player in enumerate(self.players):
            player.update_match_stats(self, i)
        
    def top_player_index(self, score: List[int]):
        return score.index(max(score))
        
    def is_over(self, game_name: GameName, score: List[int], renchan: bool):
        if min(score) < 0:
            return True
        if self.match_type == "hanchan":
            if game_name.id >= 12:
                return True
            elif game_name.id == 7:
                if renchan and score[game_name.oya] >= 30000 and game_name.oya == self.top_player_index(score):
                    return True
            elif game_name.id >= 8:
                if renchan and score[game_name.oya] >= 30000 and game_name.oya == self.top_player_index(score):
                    return True
                elif max(score) >= 30000:
                    return True
        elif self.match_type == "tonpuusen":
            if game_name.id >= 8:
                return True
            elif game_name.id == 3:
                if renchan and score[game_name.oya] >= 30000 and game_name.oya == self.top_player_index(score):
                    return True
            elif game_name.id >= 4:
                if renchan and score[game_name.oya] >= 30000 and game_name.oya == self.top_player_index(score):
                    return True
                elif max(score) >= 30000:
                    return True
        else:
            raise ValueError("Invalid game type.")
        return False