# Ref: https://mit-ll-responsible-ai.github.io/hydra-zen/tutorials.html

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, instantiate

Config = make_config('player1', 'player2')

# Register our config with Hydra's config store
cs = ConfigStore.instance()
cs.store(name="hydrazen_sandbox0_configs", node=Config)

# Tell Hydra which config to use for our task function
@hydra.main(config_path=None, config_name="hydrazen_sandbox0_configs")
def task_function(cfg: Config):
    obj = instantiate(cfg)

    # access player names from configs
    p1 = obj.player1
    p2 = obj.player2

    # write a log file of the players
    with open("hydrazen_sandbox.log", 'w') as f:
        f.write("Game Session Log\n")
        f.write("Player 1: {}\n".format(p1))
        f.write("Player 2: {}\n".format(p2))

    return p1, p2

# Executing this script will run the task function
if __name__ == "__main__":
    task_function()