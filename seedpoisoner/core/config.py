import os

from pydantic import BaseSettings, Field

"""
Generic API Parameters
"""

APP_NAME = "SEEDPosioner"
APP_VERSION = "1.0"

SEED_POISONER_URL_PREFIX = "/SEEDProtector/poisoner"

"""
Common Parameters for 
1. Attack Strategies and their Training Tasks
2. Defense Paradigms
"""


class SEEDPoisonerSetting(BaseSettings):
    DEBUG: bool = True

    LOGGING_LEVEL: str = Field(default="INFO", env="LOGGING_LEVEL")

    # ADD DATABASE PARAMETERS [ PREFERRABLY POSTGRESQ or a CLOUD VECTOR DATABASE ]

    """
    Database Credentials: UserName, Password, Datasource etc.
    """

    # ADD MODEL BASED PARAMETERS FOR TRAINING AND INFERENCE

    MODEL_TYPE: str = Field(default=None, env="roberta")

    TASK_NAME: str = Field(default=None, env="codesearch")

    DO_TRAIN: bool = False

    DO_EVAL: bool = False

    DO_PREDICT: bool = False

    LEARNING_RATE: str = Field(default=None, env="1e-5")

    NUM_EPOCHS: int = 4

    CUDA_ID: int = 0

    class Config:
        env_prefix = ""
        case_sensitive = True


config = {"seedpoisoner": SEEDPoisonerSetting()}
