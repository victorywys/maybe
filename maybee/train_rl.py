import pymahjong as pm
import numpy as np
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
from arena.common import render_global_info, tile_to_tenhou, get_base_tile
from dataclasses import field

logging.basicConfig(level=logging.INFO)

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
    save_interval: int = 10000

    
if __name__ == "__main__":

    config = RLConfig.fromcli()

    setup_experiment(config.runtime)
    print_config(config)

    if not os.path.exists(get_checkpoint_dir()):
        # create checkpoint dir
        os.makedirs(get_checkpoint_dir())
   
    players = [
        config.player1.build(),
        config.player2.build(),
        config.player3.build(),
        config.player4.build(),
    ]
    
    value_network=config.value_network.build()

    
    agent = config.agent.build(actor_network=players[0].model,
                               value_network=value_network
                              )


    record_buffer = MajEncV2ReplayBuffer(max_num_seq=1000)

    env = MahjongEnv()
    num_games = int(1e6)

    start_time = time.time()
    game = 0
    success_games = 0

    winds = ["east", "south", "west", "north"]

    while game < num_games:

        try:

            env.reset(oya=game % 4, game_wind=winds[game % 3])
            # env.reset(oya=2, game_wind='south', debug_mode=1)

            th_logger = TenhouJsonLogger()
            th_logger.init_match(
                player_names=[p.name for p in players],
                game_desc1="Test play with random players",
                rule_desc="1 game test",
            )

            # encoder
            te = pm.TableEncoder(env.t)
            te.init()
            te.update()

            th_logger.start_game(env.t, te)

            sin_array = np.zeros([51, 34, 18], dtype=bool)
            oin_array = np.zeros([51, 34, 54], dtype=bool)
            gin_array = np.zeros([51, 15], dtype=bool)
            actions = np.zeros([50], dtype=int)
            action_masks = np.zeros([50, 54], dtype=bool)
            policy_probs = np.zeros([50, 54], dtype=np.float32)
            rs = np.zeros([50], dtype=np.float32)
            dones = np.zeros([50], dtype=np.float32)
            scores=[25000, 25000, 25000, 25000]
            
            step = 0

            while not env.is_over():
                curr_player_id = env.get_curr_player_id()
                valid_actions_mask = env.get_valid_actions(nhot=True)
                obs = np.array(te.self_infos[curr_player_id]).reshape([18, 34]).swapaxes(0, 1)
                rcd = np.array(te.records[curr_player_id])
                gin = np.array(te.global_infos[curr_player_id])

                # --------- make decision -------------
                # TODO: epsilon greedy
                a, policy_prob = players[curr_player_id].play(obs, rcd, gin, valid_actions_mask, return_policy=True)

                env.step(curr_player_id, a)

                if curr_player_id == 0 and record_buffer is not None:   # only record player 0 (the RL agent)
                    sin_array[step] = obs
                    oin = np.zeros([34, 54], dtype=bool)
                    for i in range(1, 4): # oracle information (others' hands)
                        oin[:, (i - 1) * 18 : i * 18] = np.array(te.self_infos[(curr_player_id + i) % 4]).reshape([18, 34]).swapaxes(0, 1)
                    oin_array[step] = oin
                    gin_array[step] = gin
                    actions[step] = a
                    # rs[step] = 0
                    # dones[step] = 0
                    action_masks[step] = valid_actions_mask
                    policy_probs[step] = policy_prob

                    step += 1

                # ------- update state encoding ------------
                if not env.is_over():
                    te.update()
            # ----------------------- get result ---------------------------------
            th_logger.end_game(env.t, te)

            if step >= 1:
                #  ------- record final step info ----------
                te.update()

                rcd_array = np.array(te.records[0]) # only record player 0 (the RL agent)
                if rcd_array.ndim == 1:
                    rcd_array = np.zeros([0, 55], dtype=bool)

                dones[step - 1] = 1

                rs[step - 1] = (env.t.get_result().score[0]  - scores[0])/ 1000  # normalized by 1
                # only only consider straight points (no ranking points)
                
                sin_array[step] = np.array(te.self_infos[0]).reshape([18, 34]).swapaxes(0, 1)
                
                oin = np.zeros([34, 54], dtype=bool)
                for i in range(1, 4): # oracle information (others' hands)
                    oin[:, (i - 1) * 18 : i * 18] = np.array(te.self_infos[(curr_player_id + i) % 4]).reshape([18, 34]).swapaxes(0, 1)
                oin_array[step] = oin
                
                gin_array[step] = np.array(te.global_infos[0])

                # --------- append to record buffer -------------
            
                scores = env.t.get_result().score
                record_buffer.append_episode(sin_array, oin_array, rcd_array, gin_array, actions, policy_probs, action_masks, rs, dones, step)

            # ----------------------- get result ---------------------------------
            payoffs = np.array(env.get_payoffs())
            if is_prime(game):
                print("Game {}, payoffs: {}".format(game, payoffs))

            th_logger.end_game(env.t, te)
            # print(th_logger.dump_game())
            # print()

            for i, p in enumerate(players):
                p.update_stats(env.t, i)

            success_games += 1
            game += 1

            if record_buffer.size > min(config.save_interval, record_buffer.max_num_seq // 10):
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
        
        if game % config.save_interval == 0:

            logging.info("==================================================================================")
            
            logging.info("Total time: {:.2f}s".format(time.time() - start_time))
            logging.info("Success games: {}".format(success_games))
            
            # TODO: save model
            agent._checkpoint(get_checkpoint_dir())

            for i, p in enumerate(players[:1]):
                logging.info("Player {} stats: \n{}".format(i, p.dump_stats()))
                p.reset_stats()
            
            logging.info("==================================================================================")
