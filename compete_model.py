import time
import numpy as np
import pymahjong as pm
from pymahjong import MahjongEnv
import traceback

from network import MahjongPlayer
from arena.common import render_global_info, tile_to_tenhou, get_base_tile
from arena.logger import TenhouJsonLogger

import torch

import json


player = MahjongPlayer()
player.load_state_dict(torch.load("output/test_fixed/checkpoint/resume.pth")["best_network_params"])
player.to(1)
player.eval()
# print(player)

env = MahjongEnv()
num_games = 1000



RECORD_PAD = torch.zeros(1, 55)


start_time = time.time()
game = 0
success_games = 0

win_game = 0
lose_game = 0
tie_game = 0

winds = ["east", "south", "west", "north"]

while game < num_games:

    try:
        env.reset(oya=game % 4, game_wind=winds[game % 3], debug_mode=1)
        # env.reset(oya=2, game_wind='south', debug_mode=1)

        th_log = TenhouJsonLogger()
        th_log.init_match(
            ["Supervised 1", "random 1", "Supervised 2", "random 2"],
            game_desc="Test play with random players",
            rule_desc="1 game test",
        )

        # encoder
        te = pm.TableEncoder(env.t)
        te.init()
        te.update()
        
        th_log.start_game(env.t, te)

        while not env.is_over():
            curr_player_id = env.get_curr_player_id()

            # --------- get decision information -------------

            valid_actions_mask = env.get_valid_actions(nhot=True)
            executor_obs = env.get_obs(curr_player_id)

            oracle_obs = env.get_oracle_obs(curr_player_id)
            full_obs = env.get_full_obs(curr_player_id)
            full_obs = np.concatenate([executor_obs, oracle_obs], axis=0)

            # --------- make decision -------------

            if curr_player_id in [0, 2]:
                obs = np.array(te.self_infos[curr_player_id]).reshape([18, 34]).swapaxes(0, 1)
                rcd = np.array(te.records[curr_player_id])

                gin = np.array(te.global_infos[curr_player_id])
                obs = torch.Tensor(obs).to(1).unsqueeze(0)
                if rcd.ndim > 1:
                    rcd = torch.cat([RECORD_PAD, torch.Tensor(rcd)], 0).to(1).unsqueeze(0)
                else:
                    rcd = RECORD_PAD.to(1).unsqueeze(0)
                gin = torch.Tensor(gin).to(1).unsqueeze(0)
                pred = player(obs, rcd, gin)
                pred = torch.softmax(pred[0], -1).cpu().detach().numpy()
                a = (pred * valid_actions_mask).argmax(-1)
            else:
                a = np.random.choice(np.argwhere(
                    valid_actions_mask).reshape([-1]))

            env.step(curr_player_id, a)

            # ------- update state encoding ------------
            # print("render")
            # env.render()
            # print("te update")
            if not env.is_over():
                te.update()

        # ----------------------- get result ---------------------------------

        payoffs = np.array(env.get_payoffs())
        print("Game {}, payoffs: {}".format(game, payoffs))
        # env.render()

        th_log.end_game(env.t, te)
        print(th_log.dump_game())
        print()

        max_score = np.max(payoffs)
        player_max = (payoffs[0] == max_score) or (payoffs[2] == max_score)
        oppo_max = (payoffs[1] == max_score) or (payoffs[3] == max_score)
        if player_max and oppo_max:
            tie_game += 1
        elif player_max:
            win_game += 1
        else:
            lose_game += 1

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
        print("-------------- replayable log -------------------------------")
        env.t.print_debug_replay()
        continue

print("Total {} random-play games, {} games without error, takes {} s".format(
    num_games, success_games, time.time() - start_time))
print("win: {}, lose: {}, tie: {}".format(win_game, lose_game, tie_game))
print("win rate: {}".format(win_game / success_games))
print("lose rate: {}".format(lose_game / success_games))
print("tie rate: {}".format(tie_game / success_games))
