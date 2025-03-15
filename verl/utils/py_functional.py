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

from typing import Any, Dict, List


def union_two_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Union two dict. Will throw an error if there is an item not the same object with the same key."""
    for key in dict2.keys():
        if key in dict1:
            assert dict1[key] == dict2[key], f"{key} in meta_dict1 and meta_dict2 are not the same object"

        dict1[key] = dict2[key]

    return dict1


def append_to_dict(data: Dict[str, List[Any]], new_data: Dict[str, Any]) -> None:
    for key, val in new_data.items():
        if key not in data:
            data[key] = []

        data[key].append(val)
