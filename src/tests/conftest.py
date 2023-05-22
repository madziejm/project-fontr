import pytest
import pytorch_lightning as pl


@pytest.fixture(autouse=True)
def seed_everything():
    # TODO make this in line with seed from configuration?
    # TODO: test whether the seed from the configuration is actually set
    pl.seed_everything(seed=42)
