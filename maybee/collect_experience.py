import pymahjong as pm
import numpy as np
import torch
import time
import os
import traceback
import logging

from utilsd import setup_experiment, get_output_dir, get_checkpoint_dir
from utilsd.experiment import print_config
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


# ==================== arg parse & hyper-parameter setting ==================

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
    stat_interval: int = 5500
    resume: bool = False  # whether to resume from checkpoint
    

    # RL general
    num_games: int = 1050000
    algorithm: str = "dsac"  # dsac, grape
    gamma: float = 0.999
    buffer_size: int = 10000  # for 16G GPU memory
    batch_seq_num: int = 40  # for 16G GPU memory
    grad_step_num_v_per_epoch: int = 5000
    grad_step_num_a_per_game: int = 1
    lr_value: float = 2e-4
    lr_actor: float = 1e-5
    random_mps_change: int = 0

    action_mask_mode: int = 1

    epoch_reset: int = 2  # Resetting the last layer of Q network

    hand_encoder: str = "cnn"
    init_log_alpha: float = -3

    # Discrete SAC 
    lr_alpha: float = 3e-4
    clip_q_epsilon: float = 1.0
    target_entropy: float = 0.7 # for action_mask_mode = 2
    policy_epsilon: float = 0.02 # for action_mask_mode = 1
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
    logging.basicConfig(filename=os.path.join(get_output_dir(), "stdout.log"), level=logging.INFO)
    
    savepath = os.path.join(get_output_dir(), "replay_buffer")

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
    
    record_buffer = MajEncV2ReplayBuffer(max_num_seq=config.buffer_size, device='cuda') 

    env = MahjongEnv()
    num_games = config.num_games

    start_time = time.time()
    game = 0
    success_games = 0

    epoch = 99

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

                if record_buffer.size >= record_buffer.max_num_seq:
                    torch.save(record_buffer, os.path.join(savepath, "replay_buffer_{}.pth".format(epoch)))
                    epoch += 1
                    del record_buffer
                    record_buffer = MajEncV2ReplayBuffer(max_num_seq=config.buffer_size, device='cuda') 
                    logging.info("========== Replay buffer saved at epoch {} ============".format(epoch))

            # ----------------------- get result ---------------------------------
            payoffs = np.array(env.get_payoffs())
            
            if 1:
                logging.info("Game {}, {}".format(game, payoffs))

            for i, p in enumerate(players):
                p.update_stats(env.t, i)

            success_games += 1
            game += 1

            
        except Exception as inst:
            
            game += 1
            time.sleep(0.1)
            logging.info(
                "-------------- execption in game {} -------------------------".format(game))
            logging.info(inst)
            logging.info("----------------- Traceback ---------------------------------")
            traceback.print_exc()
            env.render()
            # logging.info("-------------- replayable log -------------------------------")
            # env.t.print_debug_replay()
            continue