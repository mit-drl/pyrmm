# Ref: https://mit-ll-responsible-ai.github.io/hydra-zen/tutorials/hierarchy.html
import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import builds, make_config, instantiate
from game_library import Character, inventory

_CONFIG_NAME = "sandbox_app"

# Compose sub-level configs
InventoryConf = builds(inventory, populate_full_signature=True)
starter_gear = InventoryConf(gold=10, weapon='stick', costume='tunic')

CharacterConf = builds(Character, populate_full_signature=True, inventory=starter_gear)

# Top-level config
Config = make_config(player=CharacterConf)

# Store the top level config for command line interface
cs = ConfigStore.instance()
cs.store(_CONFIG_NAME, node=Config)


# define task function with command line interface
@hydra.main(config_path=None, config_name=_CONFIG_NAME)
def task_function(cfg: Config):
    obj = instantiate(cfg)

    player = obj.player # defined in the top-level make_config

    with open("player_log.txt", "w") as f:
        f.write("Game session log:\n")
        f.write(f"Player: {player}\n")

    return player

if __name__ == "__main__":
    task_function()

