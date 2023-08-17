import time
import numpy as np
import pymahjong as pm
from pymahjong import MahjongEnv
import traceback

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
    
    
    zifengs = list(np.argwhere(self_info[:, 9]).reshape([-1]))
    print("-------- Tsumo Tiles -------------")
    print("".join([UNICODE_TILES[i] for i in zifengs]))
    
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
            
        while not env.is_over():

            
            curr_player_id = env.get_curr_player_id()

            # --------- get decision information -------------

            valid_actions_mask = env.get_valid_actions(nhot=True)
            executor_obs = env.get_obs(curr_player_id)

            oracle_obs = env.get_oracle_obs(curr_player_id)
            full_obs = env.get_full_obs(curr_player_id)
            full_obs = np.concatenate([executor_obs, oracle_obs], axis=0)

            # --------- make decision -------------

            a = np.random.choice(np.argwhere(
                valid_actions_mask).reshape([-1]))
            
            env.step(curr_player_id, a)
            
            # ------- update state encoding ------------
            # print("render")
            # env.render()
            # print("te update")
            if not env.is_over():
                te.update()
            for i in range(4):
                
                print("==========================================================")
                print("==================== Turn {} Player {} =====================".format(env.t.turn, i))
                print("==========================================================")
               
                print("---------- Self Info --------------")
                obs = np.array(te.self_infos[i]).reshape([18, 34]).swapaxes(0, 1)
                print(obs)
                render_encoding_self_info(obs)
                
                print("---------- Records Info --------------")
                rcd = np.array(te.records[i][-1])
                print(rcd)
                render_encoding_record(rcd)
                
                print("---------- Global Info --------------")
                gin = np.array(te.global_infos[i])
                print(gin)
                render_global_info(gin)
                
                print("---------- Info Size ----------------")
                print("self info:", np.array(te.self_infos[i]).shape)
                print("record:", np.array(te.records[i]).shape)
                print("global info:", np.array(te.global_infos[i]).shape)
                
                print("东风局，庄家是 {}, player {} global info".format(game % 4, i), te.global_infos[i])
            
            # break

        # ----------------------- get result ---------------------------------

        payoffs = np.array(env.get_payoffs())
        print("Game {}, payoffs: {}".format(game, payoffs))
        # env.render()

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