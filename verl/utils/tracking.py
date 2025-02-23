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
from typing import List, Union

from verl.utils.logger.aggregate_logger import LocalLogger


class Tracking:
    supported_backend = ["wandb", "mlflow", "swanlab", "console"]

    def __init__(self, project_name, experiment_name, default_backend: Union[str, List[str]] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]

        for backend in default_backend:
            assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "wandb" in default_backend:
            import wandb  # type: ignore

            wandb.init(project=project_name, name=experiment_name, config=config)
            self.logger["wandb"] = wandb

        if "mlflow" in default_backend:
            import mlflow  # type: ignore

            mlflow.start_run(run_name=experiment_name)
            mlflow.log_params(config)
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import swanlab  # type: ignore

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config=config,
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "console" in default_backend:
            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)

        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow  # type: ignore

        mlflow.log_metrics(metrics=data, step=step)
