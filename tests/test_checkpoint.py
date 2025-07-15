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


import json
import os
import shutil
import uuid

import pytest

from verl.utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt


@pytest.fixture
def save_checkpoint_path():
    ckpt_dir = os.path.join("checkpoints", str(uuid.uuid4()))
    os.makedirs(ckpt_dir, exist_ok=True)
    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_find_latest_ckpt(save_checkpoint_path):
    with open(os.path.join(save_checkpoint_path, CHECKPOINT_TRACKER), "w") as f:
        json.dump({"last_global_step": 10}, f, ensure_ascii=False, indent=2)

    assert find_latest_ckpt(save_checkpoint_path) is None
    os.makedirs(os.path.join(save_checkpoint_path, "global_step_10"), exist_ok=True)
    assert find_latest_ckpt(save_checkpoint_path) == os.path.join(save_checkpoint_path, "global_step_10")


def test_remove_obsolete_ckpt(save_checkpoint_path):
    for step in range(5, 30, 5):
        os.makedirs(os.path.join(save_checkpoint_path, f"global_step_{step}"), exist_ok=True)

    remove_obsolete_ckpt(save_checkpoint_path, global_step=30, best_global_step=10, save_limit=3)
    for step in range(5, 30, 5):
        is_exist = step in [10, 25]
        assert os.path.exists(os.path.join(save_checkpoint_path, f"global_step_{step}")) == is_exist
