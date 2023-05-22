import pytorch_lightning as pl


def set_random_state(seed: int):
    pl.seed_everything(seed)
