from utilsd.config import Registry
import numpy as np

class PLAYER(metaclass=Registry, name="player"):
    pass


@PLAYER.register_module(name="base")
class BasePlayer():
    def __init__(
        self, 
        name: str
    ):
        self.name = name

    def play(
        self, 
        self_info: np.ndarray,
        record_info: np.ndarray,
        global_info: np.ndarray,
    ):
        raise NotImplementedError
    

@PLAYER.register_module(name="random")
class RandomPlayer(BasePlayer):
    def __init__(
        self, 
        name: str
    ):
        super().__init__(name)

    def play(
        self, 
        self_info: np.ndarray,
        record_info: np.ndarray,
        global_info: np.ndarray,
        valid_actions_mask: np.ndarray,
    ):
        return np.random.choice(np.argwhere(valid_actions_mask).reshape([-1]))