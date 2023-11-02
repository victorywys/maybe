import time
import numpy as np
import pymahjong as pm
from pymahjong import MahjongEnv
import traceback

from utilsd.config import configclass, RegistryConfig, PythonConfig

from network import MahjongPlayer
from arena.common import render_global_info, tile_to_tenhou, get_base_tile
from arena.logger import TenhouJsonLogger
from arena.player import PLAYER

import torch
import json


@configclass
class ArenaConfig(PythonConfig):
    player1: RegistryConfig[PLAYER]
    player2: RegistryConfig[PLAYER]
    player3: RegistryConfig[PLAYER]
    player4: RegistryConfig[PLAYER]


if __name__ == "__main__":
    config = ArenaConfig.fromcli()
    players = [
        config.player1.build(),
        config.player2.build(),
        config.player3.build(),
        config.player4.build(),
    ]

    env = MahjongEnv()
    num_games = 100

    start_time = time.time()
    game = 0
    success_games = 0

    winds = ["east", "south", "west", "north"]

    while game < num_games:
        try:
            env.reset(oya=game % 4, game_wind=winds[game % 3])
            # env.reset(oya=2, game_wind='south', debug_mode=1)

            th_log = TenhouJsonLogger()
            th_log.init_match(
                player_names=[p.name for p in players],
                game_desc1="Test play with random players",
                rule_desc="1 game test",
            )

            # encoder
            te = pm.TableEncoder(env.t)
            te.init()
            te.update()
            
            th_log.start_game(env.t, te)

            while not env.is_over():
                curr_player_id = env.get_curr_player_id()
                valid_actions_mask = env.get_valid_actions(nhot=True)
                obs = np.array(te.self_infos[curr_player_id]).reshape([18, 34]).swapaxes(0, 1)
                rcd = np.array(te.records[curr_player_id])
                gin = np.array(te.global_infos[curr_player_id])
                
                # --------- make decision -------------
                a = players[curr_player_id].play(obs, rcd, gin, valid_actions_mask)

                env.step(curr_player_id, a)

                # ------- update state encoding ------------
                if not env.is_over():
                    te.update()

            # ----------------------- get result ---------------------------------
            payoffs = np.array(env.get_payoffs())
            print("Game {}, payoffs: {}".format(game, payoffs))

            th_log.end_game(env.t, te)
            print(th_log.dump_game())
            print()

            for i, p in enumerate(players):
                p.update_stats(env.t, i)

            success_games += 1
            game += 1

        except Exception as inst:
            game += 1
            time.sleep(0.1)
            print(
                "-------------- execption in game {} -------------------------".format(game))
            print('Exception: ', inst)
            print("----------------- Traceback ---------------------------------")
            traceback.print_exc()
            env.render()
            # print("-------------- replayable log -------------------------------")
            # env.t.print_debug_replay()
            continue
    
    print("Total time: {:.2f}s".format(time.time() - start_time))
    print("Success games: {}".format(success_games))
    for i, p in enumerate(players):
        print("Player {} stats: \n{}".format(i, p.dump_stats()))
