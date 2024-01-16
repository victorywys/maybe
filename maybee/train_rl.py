from arena import Arena

from utilsd.config import configclass, RegistryConfig, PythonConfig, ClassConfig
from arena.player import PLAYER
from buffer import MajEncV2ReplayBuffer

@configclass
class ArenaConfig(PythonConfig):
    player1: RegistryConfig[PLAYER]
    player2: RegistryConfig[PLAYER]
    player3: RegistryConfig[PLAYER]
    player4: RegistryConfig[PLAYER]
    arena: ClassConfig[Arena]


if __name__ == "__main__":
    config = ArenaConfig.fromcli()
    print(config)
    buffer = MajEncV2ReplayBuffer(batch_size=32, max_num_seq=5000, device='cuda')
    players = [
        config.player1.build(),
        config.player2.build(),
        config.player3.build(),
        config.player4.build(),
    ]
    arena = config.arena.build(players=players)
    arena.start(buffer, verbose=1)

    print(buffer.size)
    print(buffer.sample_batch())
