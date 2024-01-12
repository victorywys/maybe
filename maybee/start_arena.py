from arena import Arena

from utilsd.config import configclass, RegistryConfig, PythonConfig, ClassConfig
from arena.player import PLAYER


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
    players = [
        config.player1.build(),
        config.player2.build(),
        config.player3.build(),
        config.player4.build(),
    ]
    arena = config.arena.build(players=players)
    arena.start()