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


@configclass
class RLConfig(PythonConfig):

    player1: RegistryConfig[PLAYER]
    player2: RegistryConfig[PLAYER]
    player3: RegistryConfig[PLAYER]
    player4: RegistryConfig[PLAYER]
    value_network: RegistryConfig[NETWORK]
    agent: RegistryConfig[MODEL]
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


    algorithm: str = "dsac"  # dsac, grape
    gamma: float = 0.999
    n_buffer: int = 88
    batch_seq_num: int = 88

    hand_encoder: str = "transformer"
    random_mps_change: int = 0
    action_mask_mode: int = 1
    lr_value: float = 1e-3
    # epoch_reset: int = 2  # Resetting the last layer of Q network
    
    # Discrete SAC 
    use_avg_q: int = 1
    init_log_alpha: float = -1e9

    # GRAPE
    alpha_grape: float = 0.99
    lambd_grape: float = 0.5

    save_interval: int = 10000

    # (not used in value learning)
    lr_alpha: float = 3e-4
    clip_q_epsilon: float = 1.0  # not used
    target_entropy: float = 0.7  # for action_mask_mode = 2
    policy_epsilon: float = 0.02  # for action_mask_mode = 1
    entropy_penalty_beta: float = 0.1 
    lr_actor: float = 1e-5

    temp: float = 1.0  # policy temperature
    coef_entropy: float = 0.01
    
    
if __name__ == "__main__":

    config = RLConfig.fromcli()

    buffer_dir = "./output/experience/replay_buffer"

    setup_experiment(config.runtime)
    print_config(config)
    logging.basicConfig(level=logging.INFO)
    
    savepath = os.path.join(os.getenv('AMLT_OUTPUT_DIR', get_output_dir()), '/data/')

    if os.path.exists(savepath):
        logging.info('{} exists (possibly so do data).'.format(savepath))
    else:
        try: os.makedirs(savepath)
        except: pass

    agent = config.agent.build(actor_network=config.player1.build().model,
                               config=config,
                               device='cuda')

    grad_step = 0

    n_buffer = config.n_buffer
    test_data_id = config.n_buffer
    
    test_record_buffer = torch.load(os.path.join(buffer_dir, "replay_buffer_{}.pth".format(test_data_id)))
    test_record_buffer.epoch = 0
    loss_c_trains = []
    loss_c_tests = []

    # SGDR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=agent.optimizer_v, eta_min=1e-5, T_0=1000)

    train_buffer_ids = list()
    for n in range(10000):
        train_buffer_ids = train_buffer_ids + list(np.random.permutation(n_buffer))
    
    grad_step = 0
    train_data_id = 0

    while grad_step <= 100000:
        # Load the selected file
        train_buffer_id = train_buffer_ids[train_data_id]
        train_record_buffer = torch.load(os.path.join(buffer_dir, "replay_buffer_{}.pth".format(train_buffer_id)))
        
        train_record_buffer.epoch = 0
        while train_record_buffer.epoch < 1:

            loss_c_train = agent.update(train_record_buffer, actor_training=False, critic_training=True)
            loss_c_trains.append(loss_c_train)
            
            # test loss
            if grad_step % 5 == 0:
                with torch.no_grad():
                    loss_c_test = agent.update(test_record_buffer, actor_training=False, critic_training=False)
                loss_c_tests.append(loss_c_test)
            
            # report loss
            if grad_step >= 50 and grad_step % 50 == 0:
                logging.info(" -------- total grad step: %6d, buffer epoch: %3.3f ---------- " % (grad_step, train_record_buffer.epoch))
                logging.info("Train loss: %6.6f" % np.mean(loss_c_trains[-50:]))
                logging.info("Test loss: %6.6f" % np.mean(loss_c_tests[-10:]))
                
            
            # save model
            if (grad_step + 1) % config.save_interval == 0:

                logging.info("==================================================================================")
                
                # save modeln
                agent._save(os.path.join(savepath, "critic_learning_{}.pth".format(grad_step + 1)))
                
                logging.info("==================================================================================")

            grad_step += 1
        
        scheduler.step()
        train_data_id += 1

        
