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
Contain small python utility functions
"""

import importlib.util
from functools import lru_cache
from typing import Any, Dict, List, Union

import numpy as np
import yaml
from yaml import Dumper


def numpy_representer(dumper: Dumper, value: Union[np.float32, np.float64]):
    value = str(round(value, 3))
    return dumper.represent_scalar("tag:yaml.org,2002:float", value)


yaml.add_representer(np.float32, numpy_representer)
yaml.add_representer(np.float64, numpy_representer)


@lru_cache
def is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def union_two_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Union two dict. Will throw an error if there is an item not the same object with the same key."""
    for key in dict2.keys():
        if key in dict1:
            assert dict1[key] == dict2[key], f"{key} in dict1 and dict2 are not the same object"

        dict1[key] = dict2[key]

    return dict1


def append_to_dict(data: Dict[str, List[Any]], new_data: Dict[str, Any]) -> None:
    """Append dict to a dict of list."""
    for key, val in new_data.items():
        if key not in data:
            data[key] = []

        data[key].append(val)


def unflatten_dict(data: Dict[str, Any], sep: str = "/") -> Dict[str, Any]:
    unflattened = {}
    for key, value in data.items():
        pieces = key.split(sep)
        pointer = unflattened
        for piece in pieces[:-1]:
            if piece not in pointer:
                pointer[piece] = {}

            pointer = pointer[piece]

        pointer[pieces[-1]] = value

    return unflattened


def flatten_dict(data: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    flattened = {}
    for key, value in data.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key, sep=sep))
        else:
            flattened[new_key] = value

    return flattened


def convert_dict_to_str(data: Dict[str, Any]) -> str:
    return yaml.dump(data, indent=2)
