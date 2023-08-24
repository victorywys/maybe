import time
import numpy as np
import pymahjong as pm
from pymahjong import MahjongEnv
import traceback

from network import MahjongPlayer

import torch

import json

start_hand = [[] for _ in range(4)]
draw_tiles = [[] for _ in range(4)]
discard_tiles = [[] for _ in range(4)]


player = MahjongPlayer()
player.load_state_dict(torch.load("output/test/checkpoint/resume.pth")["best_network_params"])
player.to(1)
player.eval()
print(player)

env = MahjongEnv()
num_games = 1
UNICODE_TILES = """
    🀇 🀈 🀉 🀊 🀋 🀌 🀍 🀎 🀏 
    🀙 🀚 🀛 🀜 🀝 🀞 🀟 🀠 🀡
    🀐 🀑 🀒 🀓 🀔 🀕 🀖 🀗 🀘
    🀀 🀁 🀂 🀃
    🀆 🀅 🀄
    🀋 🀝 🀔
""".split()

tile_to_tenhou = list(range(11, 20)) + list(range(21, 30)) + list(range(31, 40)) + list(range(41, 48)) + [51, 52, 53]

RECORD_PAD = torch.zeros(1, 55)



ACTIONS = ["摸牌", "摸杠牌", "手切", "摸切", "吃L", "吃M", "吃R", "碰", "明杠", "暗杠", "加杠", "摸切立直", "手切立直", "立直通过", "关联自家", "关联下家", "关联对家", "关联上家"]

def render_global_info(global_info):
    explains = ["局数", "最终局", "本场数", "场供数", "自风牌", "场风牌", "自家点数", "下家点数", "对面点数", "上家点数", "自家一发", "下家一发", "对面一发", "上家一发", "剩余牌数"]
    for i, e in enumerate(explains):
        print(e + ":" + str(global_info[i]))
        
def render_encoding_record(record):
    a = np.argwhere(np.array(record)).reshape([-1])
    action_strs = []
    for i in a:
        if i < 37:
            action_strs.append(UNICODE_TILES[i])
        else:
            action_strs.append(ACTIONS[i - 37])
            
    print(", ".join(action_strs))
    
def render_encoding_self_info(self_info):
    # 0-3
    
    hands = list(np.argwhere(self_info[:, 0]).reshape([-1])) + list(np.argwhere(self_info[:, 1]).reshape([-1])) + list(np.argwhere(self_info[:, 2]).reshape([-1])) + list(np.argwhere(self_info[:, 3]).reshape([-1]))
    hands.sort()
    print("-------- Hand -------------")
    print("".join([UNICODE_TILES[i] for i in hands]))
    
    hand_akas = list(np.argwhere(self_info[:, 6]).reshape([-1])) 
    print("-------- Aka -------------")
    print("".join([UNICODE_TILES[i] for i in hand_akas]))
    
    
    doras = list(np.argwhere(self_info[:, 4]).reshape([-1]))
    print("-------- Dora -------------")
    print("".join([UNICODE_TILES[i] for i in doras]))
    
    dora_indicators = list(np.argwhere(self_info[:, 5]).reshape([-1]))
    print("-------- Dora Indicator -------------")
    print("".join([UNICODE_TILES[i] for i in dora_indicators]))
    
    changfengs = list(np.argwhere(self_info[:, 7]).reshape([-1]))
    print("-------- Game Wind -------------")
    print("".join([UNICODE_TILES[i] for i in changfengs]))
    
    zifengs = list(np.argwhere(self_info[:, 8]).reshape([-1]))
    print("-------- Self Wind -------------")
    print("".join([UNICODE_TILES[i] for i in zifengs]))
    
    
    zimopai = list(np.argwhere(self_info[:, 9]).reshape([-1]))
    print("-------- Tsumo Tiles -------------")
    print("".join([UNICODE_TILES[i] for i in zimopai]))
    
    tmp = list(np.argwhere(self_info[:, 10]).reshape([-1]))
    print("-------- Self Discarded -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))
    
    tmp = list(np.argwhere(self_info[:, 11]).reshape([-1]))
    print("-------- Next Discarded -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))
    
    tmp = list(np.argwhere(self_info[:, 12]).reshape([-1]))
    print("-------- Opposite Discarded -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))
    
    tmp = list(np.argwhere(self_info[:, 13]).reshape([-1]))
    print("-------- Previous Discarded -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))
    
    
    tmp = list(np.argwhere(self_info[:, 14]).reshape([-1])) + list(np.argwhere(self_info[:, 15]).reshape([-1])) + list(np.argwhere(self_info[:, 16]).reshape([-1])) + list(np.argwhere(self_info[:, 17]).reshape([-1]))
    tmp.sort()
    print("-------- Disclosed Tiles -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))
    
start_time = time.time()
game = 0
success_games = 0

win_game = 0
lose_game = 0
tie_game = 0

winds = ["east", "south", "west", "north"]

print("begin test")

while game < num_games:
    
    try:
        env.reset(oya=game % 4, game_wind=winds[game % 3], debug_mode=1)
        # env.reset(oya=2, game_wind='south', debug_mode=1)

        # encoder
        te = pm.TableEncoder(env.t)
        
        te.init()
        
        te.update()

        # # print(np.array(te.self_infos[0]).reshape([18, 34]).swapaxes(0, 1))
        for i in range(4):
            print("东风局，庄家是 {}, player {} global info".format(game % 4, i), te.global_infos[i])
            
        for i in range(4):
            
            print("==========================================================")
            print("==================== Turn {} Player {} =====================".format(env.t.turn, i))
            print("==========================================================")
            
            print("---------- Self Info --------------")
            self_info = np.array(te.self_infos[i]).reshape([18, 34]).swapaxes(0, 1)
            
            hands = list(np.argwhere(self_info[:, 0]).reshape([-1])) + list(np.argwhere(self_info[:, 1]).reshape([-1])) + list(np.argwhere(self_info[:, 2]).reshape([-1])) + list(np.argwhere(self_info[:, 3]).reshape([-1]))
            hands.sort()
            if i == 0:
                zimopai = list(np.argwhere(self_info[:, 9]).reshape([-1]))
                hands.remove(zimopai[0])
                draw_tiles[i].append(tile_to_tenhou[zimopai[0]])
                
            hand_akas = list(np.argwhere(self_info[:, 6]).reshape([-1]))
            if 4 in hand_akas:
                hands.remove(4)
                hands.append(34)
            if 13 in hand_akas:
                hands.remove(13)
                hands.append(35)
            if 22 in hand_akas:
                hands.remove(22)
                hands.append(36)
            
            hands = [tile_to_tenhou[tile] for tile in hands]
            start_hand[i] = hands
            # print("---------- Records Info --------------")
            # rcd = np.array(te.records[i][-1])
            # render_encoding_record(rcd)
            
            # print("---------- Global Info --------------")
            # gin = np.array(te.global_infos[i])
            # render_global_info(gin)
            
            # print("东风局，庄家是 {}, player {} global info".format(game % 4, i), te.global_infos[i])
            # print(a)
        
        global_info = te.global_infos[0]
        ju = global_info[0]
        benchang = global_info[2]
        changgong = global_info[3]
        
        while not env.is_over():

            
            curr_player_id = env.get_curr_player_id()

            # --------- get decision information -------------

            valid_actions_mask = env.get_valid_actions(nhot=True)
            executor_obs = env.get_obs(curr_player_id)

            oracle_obs = env.get_oracle_obs(curr_player_id)
            full_obs = env.get_full_obs(curr_player_id)
            full_obs = np.concatenate([executor_obs, oracle_obs], axis=0)

            # --------- make decision -------------

            if curr_player_id == 0:
                obs = np.array(te.self_infos[0]).reshape([18, 34]).swapaxes(0, 1)
                rcd = np.array(te.records[0])
                
                gin = np.array(te.global_infos[0])
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
        
        for player_i in range(4):
            print("==========================================================")
            records = te.records[player_i]
            for record_i, record in enumerate(records):
                a = np.argwhere(np.array(record)).reshape([-1])
                if 51 in a: # 关联自家
                    render_encoding_record(record)
                    tile = a[0]
                    if 37 in a or 38 in a: # 摸牌，摸杠牌
                        draw_tiles[player_i].append(tile_to_tenhou[tile])
                    elif 39 in a: # 手切
                        discard_tiles[player_i].append(tile_to_tenhou[tile])
                    elif 40 in a: # 摸切
                        discard_tiles[player_i].append(60)
                    elif 41 in a or 42 in a or 43 in a: # 吃L M R
                        last_record = records[record_i - 1]
                        print("相关动作：", end="")
                        render_encoding_record(last_record)
                        last_tile = np.argwhere(np.array(last_record)).reshape([-1])[0]
                        chi_tile1 = a[0]
                        chi_tile2 = a[1]
                        draw_tiles[player_i].append(f"c{tile_to_tenhou[last_tile]}{tile_to_tenhou[chi_tile1]}{tile_to_tenhou[chi_tile2]}")
                    elif 44 in a: # 碰
                        last_record = records[record_i - 1]
                        print("相关动作：", end="")
                        render_encoding_record(last_record)
                        last_tile = np.argwhere(np.array(last_record)).reshape([-1])[0]
                        last_player = np.argwhere(np.array(last_record)).reshape([-1])[-1] - 51
                        pon_tile1 = a[0]
                        if a[1] > 36:
                            pon_tile2 = a[0]
                        else:
                            pon_tile2 = a[1]
                        if last_player == 1: # 下家
                            draw_tiles[player_i].append(f"{tile_to_tenhou[pon_tile1]}{tile_to_tenhou[pon_tile2]}p{tile_to_tenhou[last_tile]}")
                        elif last_player == 2: # 对家
                            draw_tiles[player_i].append(f"{tile_to_tenhou[pon_tile1]}p{tile_to_tenhou[last_tile]}{tile_to_tenhou[pon_tile2]}")
                        elif last_player == 3: # 上家
                            draw_tiles[player_i].append(f"p{tile_to_tenhou[last_tile]}{tile_to_tenhou[pon_tile1]}{tile_to_tenhou[pon_tile2]}")
                    elif 45 in a: # 明杠
                        last_record = records[record_i - 1]
                        print("相关动作：", end="")
                        render_encoding_record(last_record)
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
                            draw_tiles[player_i].append(f"{tile_to_tenhou[kan_tile1]}{tile_to_tenhou[kan_tile2]}{tile_to_tenhou[kan_tile3]}m{tile_to_tenhou[last_tile]}")
                        elif last_player == 2: # 对家
                            draw_tiles[player_i].append(f"{tile_to_tenhou[kan_tile1]}m{tile_to_tenhou[last_tile]}{tile_to_tenhou[kan_tile2]}{tile_to_tenhou[kan_tile3]}")
                        elif last_player == 3: # 上家
                            draw_tiles[player_i].append(f"m{tile_to_tenhou[last_tile]}{tile_to_tenhou[kan_tile1]}{tile_to_tenhou[kan_tile2]}{tile_to_tenhou[kan_tile3]}")
                        discard_tiles[player_i].append(0)
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
                        draw_tiles[player_i].append(f"a{tile_to_tenhou[kan_tile1]}{tile_to_tenhou[kan_tile2]}{tile_to_tenhou[kan_tile3]}{tile_to_tenhou[kan_tile4]}")
                        discard_tiles[player_i].append(0)
                    elif 47 in a: # 加杠
                        tile = a[0]
                        kan_tile = tile
                        base_tile = tile
                        if base_tile == 34:
                            base_tile = 4
                        elif base_tile == 35:
                            base_tile = 13
                        elif base_tile == 36:
                            base_tile = 22
                        for record_j in range(record_i):
                            related_record_raw = records[record_j]
                            related_record = np.argwhere(np.array(records[record_j])).reshape([-1])
                            if 44 in related_record:
                                related_tile = related_record[0]
                                if related_tile == 34:
                                    related_tile = 4
                                elif related_tile == 35:
                                    related_tile = 13
                                elif related_tile == 36:
                                    related_tile = 22
                                if related_tile == base_tile:
                                    pon_tile1 = related_record[0]
                                    if related_record[1] < 37:
                                        pon_tile2 = related_record[1]
                                    else:
                                        pon_tile2 = related_record[0]
                                    source_record = records[record_j - 1]
                                    print("相关动作1：", end="")
                                    render_encoding_record(related_record_raw)
                                    print("相关动作2：", end="")
                                    render_encoding_record(source_record)
                                    source_player = np.argwhere(np.array(source_record)).reshape([-1])[-1] - 51
                                    source_tile = np.argwhere(np.array(source_record)).reshape([-1])[0]
                                    if source_player == 1: # 下家
                                        discard_tiles[player_i].append(f"{tile_to_tenhou[pon_tile1]}{tile_to_tenhou[pon_tile2]}k{tile_to_tenhou[kan_tile]}{tile_to_tenhou[source_tile]}")
                                    elif source_player == 2: # 对家
                                        discard_tiles[player_i].append(f"{tile_to_tenhou[pon_tile1]}k{tile_to_tenhou[kan_tile]}{tile_to_tenhou[source_tile]}{tile_to_tenhou[pon_tile2]}")
                                    elif source_player == 3: # 上家
                                        discard_tiles[player_i].append(f"k{tile_to_tenhou[kan_tile]}{tile_to_tenhou[source_tile]}{tile_to_tenhou[pon_tile1]}{tile_to_tenhou[pon_tile2]}")
                                    break
                    elif 48 in a: # 摸切立直
                        discard_tiles[player_i].append("r60")
                    elif 49 in a: # 手切立直
                        tile = a[0]
                        discard_tiles[player_i].append(f"r{tile_to_tenhou[tile]}")
        

        dora_info = te.self_infos[0]
        dora_indicators = list(np.argwhere(self_info[:, 5]).reshape([-1]))
        dora_indicator_tile = [tile_to_tenhou[tile] for tile in dora_indicators]
        
        json_obj = {
            "title": ["Test play with random players", "player 0 is supervised learned model"],
            "name": ["Supervised", "random 1", "random 2", "random 3"],
            "rule": {
                "disp": "1 game test",
                "aka": 1,
            },
            "log": [
                [
                    [ju, benchang, changgong],
                    [25000, 25000, 25000, 25000],
                    dora_indicator_tile, # dora indicator
                    [], # ura dora indicator
                    start_hand[0], # start hand for player 0
                    draw_tiles[0], # draw tiles for player 0
                    discard_tiles[0], # discard tiles for player 0
                    start_hand[1], # start hand for player 1
                    draw_tiles[1], # draw tiles for player 1
                    discard_tiles[1], # discard tiles for player 1
                    start_hand[2], # start hand for player 2
                    draw_tiles[2], # draw tiles for player 2
                    discard_tiles[2], # discard tiles for player 2
                    start_hand[3], # start hand for player 3
                    draw_tiles[3], # draw tiles for player 3
                    discard_tiles[3], # discard tiles for player 3
                    ["不明"], # result
                ]
            ]
        }
        print(json_obj)
        print(json.dumps(json_obj))
        
        max_score = np.max(payoffs)
        player_max = payoffs[0] == max_score
        oppo_max = (payoffs[1] == max_score) or (payoffs[2] == max_score) or (payoffs[3] == max_score)
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