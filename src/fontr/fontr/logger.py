from pytorch_lightning.loggers import WandbLogger
from kedro.config import ConfigLoader, MissingConfigException
from kedro.framework.project import settings
import os


class TorchLogger:
    logger = None

    @classmethod
    def getLogger(cls):
        if TorchLogger.logger:
            return TorchLogger.logger

        api_key = TorchLogger.obtainCredentials()
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            TorchLogger.logger = WandbLogger(project="fontr")
            return TorchLogger.logger

    @classmethod
    def obtainCredentials(cls):
        conf_loader = ConfigLoader(conf_source=settings.CONF_SOURCE)
        try:
            return conf_loader["credentials"]["api_key"]
        except (KeyError, MissingConfigException):
            raise Exception(
                "No credentials for wandb found!\n"
                + "Make sure that <root>/conf/local/credentials_wandb.yml file exists\n"
                + "It should contain a single key `api_key`"
            )
