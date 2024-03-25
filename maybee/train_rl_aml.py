import pymahjong as pm
import numpy as np
import torch
import time
import os
import traceback
import logging

from utilsd import setup_experiment, get_output_dir, get_checkpoint_dir
from utilsd.experiment import print_config
# from utilsd.logging import print_log, setup_logger
from utilsd.config import configclass, RegistryConfig, PythonConfig, ClassConfig, RuntimeConfig
from arena.player import PLAYER
from buffer import MajEncV2ReplayBuffer
from network import NETWORK
from model import MODEL
from pymahjong import MahjongEnv
from arena.logger import TenhouJsonLogger
from arena.common import render_global_info, tile_to_tenhou, get_base_tile, action_v2_to_human_chinese
from dataclasses import field

# import scipy.io as sio

logging.basicConfig(level=logging.INFO)

# ==================== arg parse & hyper-parameter setting ==================
savepath = os.getenv('AMLT_OUTPUT_DIR', '/tmp') + '/data/'

def is_prime(n):
    # 小于2的数不是素数
    if n <= 1:
        return False
    # 2是最小的素数
    elif n == 2:
        return True
    # 所有偶数（除了2）都不是素数
    elif n % 2 == 0:
        return False
    # 检查从3到sqrt(n)的所有奇数是否能整除n
    sqrt_n = int(n**0.5) + 1
    for i in range(3, sqrt_n, 2):
        if n % i == 0:
            return False
    return True


@configclass
class RLConfig(PythonConfig):
    player1: RegistryConfig[PLAYER]
    player2: RegistryConfig[PLAYER]
    player3: RegistryConfig[PLAYER]
    player4: RegistryConfig[PLAYER]
    value_network: RegistryConfig[NETWORK]
    agent: RegistryConfig[MODEL]
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # online learning setting
    save_interval: int = 10000
    stat_interval: int = 4000
    resume: bool = False  # whether to resume from checkpoint

    # RL general
    algorithm: str = "dsac"  # dsac, grape
    gamma: float = 0.999
    train_start: int = 4000
    buffer_size: int = 10000  # for 16G GPU memory
    batch_seq_num: int = 40  # for 16G GPU memory
    grad_step_num_per_game: int = 1
    actor_training_offset: int = 4000  # how many steps of value training before policy training
    lr_value: float = 3e-5
    lr_actor: float = 1e-5
    random_mps_change: int = 1

    # Discrete SAC 
    lr_alpha: float = 3e-4
    clip_q_epsilon: float = 1.0
    target_entropy: float = 0.7
    entropy_penalty_beta: float = 0.1    
    use_avg_q: int = 0

    # GRAPE
    alpha_grape: float = 0.99
    lambd_grape: float = 0.5
    temp: float = 1.0  # policy temperature
    coef_entropy: float = 0.01
    
    
if __name__ == "__main__":

    config = RLConfig.fromcli()

    setup_experiment(config.runtime)
    print_config(config)

    if os.path.exists(savepath):
        logging.info('{} exists (possibly so do data).'.format(savepath))
    else:
        try: os.makedirs(savepath)
        except: pass

    players = [
        config.player1.build(),
        config.player2.build(),
        config.player3.build(),
        config.player4.build(),
    ]
    
    agent = config.agent.build(actor_network=players[0].model,
                               config=config,
                               device='cuda'
                              )
    if config.resume:
        agent._resume("./resume.pth")

    record_buffer = MajEncV2ReplayBuffer(max_num_seq=config.buffer_size, device='cuda') 

    env = MahjongEnv()
    num_games = int(35000)

    start_time = time.time()
    game = 0
    success_games = 0


    winds = ["east", "south", "west", "north"]

    while game < num_games:

        try:

            env.reset(oya=game % 4, game_wind=winds[game % 2], kyoutaku=0, honba=0)
            # env.reset(oya=2, game_wind='south', debug_mode=1)

            # encoder
            te = pm.TableEncoder(env.t)
            te.init()
            te.update()

            sin_array = np.zeros([51, 34, 18], dtype=bool)
            oin_array = np.zeros([51, 34, 54], dtype=bool)
            rcd_array = np.zeros([51, 200, 55], dtype=bool)
            gin_array = np.zeros([51, 15], dtype=bool)
            actions = np.zeros([50], dtype=int)
            action_masks = np.zeros([50, 54], dtype=bool)
            policy_probs = np.zeros([50, 54], dtype=np.float32)
            rs = np.zeros([50], dtype=np.float32)
            dones = np.zeros([50], dtype=np.float32)
            scores = [25000, 25000, 25000, 25000]
            
            step = 0

            while not env.is_over():
                curr_player_id = env.get_curr_player_id()
                valid_actions_mask = env.get_valid_actions(nhot=True)
                obs = np.array(te.self_infos[curr_player_id]).reshape([18, 34]).swapaxes(0, 1)
                rcd = np.array(te.records[curr_player_id])
                
                if rcd.ndim == 1:
                    rcd = np.zeros([0, 55], dtype=bool)
                
                gin = np.array(te.global_infos[curr_player_id])

                # --------- make decision -------------                

                if curr_player_id == 0:   # only record player 0 (the RL agent)
                    a, policy_prob = agent.select_action(obs, rcd, gin, valid_actions_mask, temp=config.temp)

                    sin_array[step] = obs
                    oin = np.zeros([34, 54], dtype=bool)
                    for i in range(1, 4): # oracle information (others' hands)
                        oin[:, (i - 1) * 18 : i * 18] = np.array(te.self_infos[(curr_player_id + i) % 4]).reshape([18, 34]).swapaxes(0, 1)
                    oin_array[step] = oin
                    gin_array[step] = gin
                    rcd_array[step][1 : rcd.shape[0] + 1] = rcd  # with start token
                    actions[step] = a
                    # rs[step] = 0
                    # dones[step] = 0
                    action_masks[step] = valid_actions_mask
                    policy_probs[step] = policy_prob

                    step += 1
                
                else:
                    a = players[curr_player_id].play(obs, rcd, gin, valid_actions_mask)  # AI player does padding interiorly 
                                        
                env.step(curr_player_id, a)

                # test Q value, for debugging
                if env.is_over() and is_prime(game) and step > 1:
                    
                    q = agent.value_network(torch.from_numpy(sin_array[step - 1: step]).cuda().float(),
                                            torch.from_numpy(oin_array[step - 1: step]).cuda().float(),
                                            torch.from_numpy(rcd_array[step - 1: step]).cuda().float(),
                                            torch.from_numpy(gin_array[step - 1: step]).cuda().float())

                    print("Q value = ", q[0, actions[step - 1]].cpu().item())
            
                # ------- update state encoding ------------
                te.update()
            
            # ----------------------- get result ---------------------------------

            if step >= 1:
                #  ------- record final step info ----------
                # te.update()

                dones[step - 1] = 1

                rs[step - 1] = (env.t.get_result().score[0]  - scores[0])/ 1000  # normalized by 1000
                # only only consider straight points (no ranking points)
                
                sin_array[step] = np.array(te.self_infos[0]).reshape([18, 34]).swapaxes(0, 1)
                
                oin = np.zeros([34, 54], dtype=bool)
                for i in range(1, 4): # oracle information (others' hands)
                    oin[:, (i - 1) * 18 : i * 18] = np.array(te.self_infos[(0 + i) % 4]).reshape([18, 34]).swapaxes(0, 1)
                oin_array[step] = oin

                if np.array(te.records[0]).ndim > 1:
                    rcd_array[step][1 : np.array(te.records[0]).shape[0] + 1] = np.array(te.records[0])  # with start token

                gin_array[step] = np.array(te.global_infos[0])

                # --------- append to record buffer -------------
            
                # scores = env.t.get_result().score
                record_buffer.append_episode(sin_array, oin_array, rcd_array, gin_array, actions, policy_probs, action_masks, rs, dones, step)

            # ----------------------- get result ---------------------------------
            payoffs = np.array(env.get_payoffs())
            
            if is_prime(game):
                print("Game {}, payoffs: {}".format(game, payoffs))

            for i, p in enumerate(players):
                p.update_stats(env.t, i)

            success_games += 1
            game += 1

            # if record_buffer.size > min(config.save_interval, record_buffer.max_num_seq // 10):
            
            if game > config.train_start:
                for _ in range(config.grad_step_num_per_game):
                    agent.update(record_buffer)


        except Exception as inst:
            
            game += 1
            time.sleep(0.1)
            logging.info(
                "-------------- execption in game {} -------------------------".format(game))
            logging.info('Exception: ', inst)
            logging.info("----------------- Traceback ---------------------------------")
            traceback.print_exc()
            env.render()
            # logging.info("-------------- replayable log -------------------------------")
            # env.t.print_debug_replay()
            continue
        
        if game % config.stat_interval == 0:
            
            for p in players:
                p.reset_stats()
            
            for game_test in range(config.stat_interval):

                try:
                    env.reset(oya=game_test % 4, game_wind=winds[game_test % 2], kyoutaku=0, honba=0)
                    
                    # encoder
                    te = pm.TableEncoder(env.t)
                    te.init()
                    te.update()

                    if game_test % 200 == 0:
                        th_logger = TenhouJsonLogger()
                        th_logger.init_match(
                            player_names=[p.name for p in players],
                            game_desc1="Test play with SL players",
                            rule_desc="1 game test",
                        )
                        th_logger.start_game(env.t, te)
                    
                    while not env.is_over():
                        curr_player_id = env.get_curr_player_id()
                        valid_actions_mask = env.get_valid_actions(nhot=True)
                        obs = np.array(te.self_infos[curr_player_id]).reshape([18, 34]).swapaxes(0, 1)
                        rcd = np.array(te.records[curr_player_id])
                        
                        if rcd.ndim == 1:
                            rcd = np.zeros([0, 55], dtype=bool)
                        
                        gin = np.array(te.global_infos[curr_player_id])

                        # --------- make decision -------------                

                        if curr_player_id == 0:   # only record player 0 (the RL agent)
                            a, policy_prob = agent.select_action(obs, rcd, gin, valid_actions_mask, temp=0.01) # greedy                        
                        else:
                            a = players[curr_player_id].play(obs, rcd, gin, valid_actions_mask)  # AI player does padding interiorly 
                                                
                        env.step(curr_player_id, a)
                        te.update()
                    
                    for i, p in enumerate(players):
                        p.update_stats(env.t, i)
                    
                    if game_test % 200 == 0:
                        th_logger.end_game(env.t, te)
                        logging.info("game {}".format(game_test)) 
                        logging.info(th_logger.dump_urls())
                
                except Exception as inst:
                    try:
                        print("-------------- execption in game {} -------------------------".format(game_test))
                        print('Exception: ', inst)
                        print("----------------- Traceback ---------------------------------")
                        traceback.print_exc()
                    except:
                        pass
                    
                # end for
                
            for i, p in enumerate(players[:1]):
                logging.info("Player {} stats: \n{}".format(i, p.dump_stats()))
            
        if game % config.save_interval == 0:

            logging.info("==================================================================================")
            
            logging.info("Total time: {:.2f}s".format(time.time() - start_time))
            logging.info("Success games: {}".format(success_games))
            
            # save model
            agent._save(os.path.join(savepath, "rl_game_{}.pth".format(game)))
            
            logging.info("==================================================================================")