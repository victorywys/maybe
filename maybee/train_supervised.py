from utilsd import setup_experiment, get_output_dir, get_checkpoint_dir
from utilsd.config import configclass, RegistryConfig, RuntimeConfig, PythonConfig
from utilsd.experiment import print_config

from dataset import DATASET
from model import MODEL
from network import NETWORK


@configclass
class SupervisedMahjongConfig(PythonConfig):
    data: RegistryConfig[DATASET]
    network: RegistryConfig[NETWORK]
    model: RegistryConfig[MODEL]
    runtime: RuntimeConfig = RuntimeConfig()


def run_supervised(config):
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(split="train", debug=config.runtime.debug)
    testset = config.data.build(split="test", debug=config.runtime.debug)
    network = config.network.build()
    model = config.model.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
    )
    model.fit(trainset, testset)


if __name__ == "__main__":
    _config = SupervisedMahjongConfig.fromcli()
    run_supervised(_config)
