# Ref: https://mit-ll-responsible-ai.github.io/hydra-zen/tutorials/hierarchy.html

class Character:
    def __init__(self, name: str, level: int = 1, inventory=None):
        self.name = name
        self.level = level
        self.inventroy = inventory

    def __repr__(self):
        out = ""
        out += f"{self.name}, "
        out += f"lvl: {self.level}, "
        out += f"has: {self.inventroy}"
        return out

def inventory(gold: int, weapon: str, costume: str):
    return {"gold": gold, "weapon": weapon, "costume": costume}