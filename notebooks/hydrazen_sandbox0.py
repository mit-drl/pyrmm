from hydra_zen import make_config, instantiate

Config = make_config('player1', 'player2')

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