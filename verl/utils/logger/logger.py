# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..py_functional import convert_dict_to_str, flatten_dict, is_package_available, unflatten_dict
from .gen_logger import AggregateGenerationsLogger


if is_package_available("mlflow"):
    import mlflow  # type: ignore


if is_package_available("tensorboard"):
    from torch.utils.tensorboard import SummaryWriter


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


class Logger(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None: ...

    @abstractmethod
    def log(self, data: Dict[str, Any], step: int) -> None: ...

    def finish(self) -> None:
        pass


class ConsoleLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        print("Config\n" + convert_dict_to_str(config))

    def log(self, data: Dict[str, Any], step: int) -> None:
        print(f"Step {step}\n" + convert_dict_to_str(unflatten_dict(data)))


class MlflowLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        mlflow.start_run(run_name=config["trainer"]["experiment_name"])
        mlflow.log_params(flatten_dict(config))

    def log(self, data: Dict[str, Any], step: int) -> None:
        mlflow.log_metrics(metrics=data, step=step)


class SwanlabLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        swanlab_key = os.getenv("SWANLAB_API_KEY")
        swanlab_dir = os.getenv("SWANLAB_DIR", "swanlab_log")
        swanlab_mode = os.getenv("SWANLAB_MODE", "cloud")
        if swanlab_key:
            swanlab.login(swanlab_key)

        swanlab.init(
            project=config["trainer"]["project_name"],
            experiment_name=config["trainer"]["experiment_name"],
            config={"UPPERFRAMEWORK": "EasyR1", "FRAMEWORK": "veRL", **config},
            logdir=swanlab_dir,
            mode=swanlab_mode,
        )

    def log(self, data: Dict[str, Any], step: int) -> None:
        swanlab.log(data=data, step=step)

    def finish(self) -> None:
        swanlab.finish()


class TensorBoardLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        tensorboard_dir = os.getenv("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)
        config_dict = {}
        for key, value in flatten_dict(config).items():
            if isinstance(value, (int, float, str, bool, torch.Tensor)):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)

        self.writer.add_hparams(hparam_dict=config_dict, metric_dict={"placeholder": 0})

    def log(self, data: Dict[str, Any], step: int) -> None:
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)

    def finish(self):
        self.writer.close()


class WandbLogger(Logger):
    def __init__(self, config: Dict[str, Any]) -> None:
        wandb.init(
            project=config["trainer"]["project_name"],
            name=config["trainer"]["experiment_name"],
            config=config,
        )

    def log(self, data: Dict[str, Any], step: int) -> None:
        wandb.log(data=data, step=step)

    def finish(self) -> None:
        wandb.finish()


LOGGERS = {
    "console": ConsoleLogger,
    "mlflow": MlflowLogger,
    "swanlab": SwanlabLogger,
    "tensorboard": TensorBoardLogger,
    "wandb": WandbLogger,
}


class Tracker:
    def __init__(self, loggers: Union[str, List[str]] = "console", config: Optional[Dict[str, Any]] = None):
        if isinstance(loggers, str):
            loggers = [loggers]

        self.loggers: List[Logger] = []
        for logger in loggers:
            if logger not in LOGGERS:
                raise ValueError(f"{logger} is not supported.")

            self.loggers.append(LOGGERS[logger](config))

        self.gen_logger = AggregateGenerationsLogger(loggers)

    def log(self, data: Dict[str, Any], step: int) -> None:
        for logger in self.loggers:
            logger.log(data=data, step=step)

    def log_generation(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        self.gen_logger.log(samples, step)

    def __del__(self):
        for logger in self.loggers:
            logger.finish()
