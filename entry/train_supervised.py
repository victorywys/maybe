from utilsd import setup_experiment
from utilsd.config import configclass, RegistryConfig, RuntimeConfig, PythonConfig
from utilsd.experiment import print_config

from ..model import MODEL
from ..network import NETWORK


@configclass
class SupervisedMahjongConfig(PythonConfig):
    # data = RegistryConfig[DATASET]
    network: RegistryConfig[NETWORK]
    model: RegistryConfig[MODEL]
    runtime: RuntimeConfig = RuntimeConfig()


def run_supervised(config):
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(dataset="train")
    testset = config.data.build(dataset="test")
    network = config.network.build()
    model = config.model.build(
        network=network,
    )
    model.fit(trainset, testset)


if __name__ == "__main__":
    _config = SupervisedMahjongConfig.fromcli()
    run_supervised(_config)
