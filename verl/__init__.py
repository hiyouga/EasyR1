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

import os

from .utils.py_functional import is_package_available


if is_package_available("modelscope"):
    from modelscope.utils.hf_util import patch_hub  # type: ignore


__version__ = "0.3.1.dev0"


if os.getenv("USE_MODELSCOPE_HUB", "0").lower() in ["true", "y", "1"]:
    # Patch hub to download models from modelscope to speed up.
    if not is_package_available("modelscope"):
        raise ImportError("You are using the modelscope hub, please install modelscope by `pip install modelscope`.")

    patch_hub()
