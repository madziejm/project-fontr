import pytest
import pytorch_lightning as pl


@pytest.fixture(autouse=True)
def seed_everything():
    pl.seed_everything(seed=42)
