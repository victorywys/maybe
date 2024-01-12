from typing import Literal, List, Optional

from .player import BasePlayer
from .match import Match


class Arena():
    def __init__(
        self,
        match_number: int,
        players: Optional[List[BasePlayer]] = None,
        match_type: str = "hanchan",
    ):
        assert players is not None, "Players not set."
        self.players = players
        self.match_number = match_number
        self.match_type = match_type
    
    def start(self):
        for i in range(self.match_number):
            print(f"Match {i + 1} start.")
            match = Match(self.players, match_type=self.match_type, match_desc=f"Match {i + 1}")
            match.play_match()
            print(match.th_logger.dump_match())
            for url in match.th_logger.dump_urls():
                print(url)
            print(f"Match {i + 1} end with final score {match.match_result['score']}.")
            print()
    
    def print_player_stats(self):
        for player in self.players:
            print(player.name)
            print(player.dump_stats())