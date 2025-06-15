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
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import torch

from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto


def _get_data_proto(
    tensors: Optional[Dict[str, List[Any]]] = None,
    non_tensors: Optional[Dict[str, List[Any]]] = None,
    meta_info: Optional[Dict[str, Any]] = None,
) -> DataProto:
    if tensors is None and non_tensors is None:
        tensors = {"obs": [1, 2, 3, 4, 5, 6]}
        non_tensors = {"labels": ["a", "b", "c", "d", "e", "f"]}

    if tensors is not None:
        tensors = {k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in tensors.items()}

    if non_tensors is not None:
        non_tensors = {
            k: np.array(v, dtype=object) if not isinstance(v, np.ndarray) else v for k, v in non_tensors.items()
        }

    meta_info = meta_info or {"info": "test_info"}
    return DataProto.from_dict(tensors=tensors, non_tensors=non_tensors, meta_info=meta_info)


def _assert_equal(data1: DataProto, data2: Optional[DataProto] = None):
    data2 = data2 or _get_data_proto()
    if data1.batch is not None:
        assert data1.batch.keys() == data2.batch.keys()
        for key in data1.batch.keys():
            assert torch.all(data1.batch[key] == data2.batch[key])
    else:
        assert data2.batch is None

    if data1.non_tensor_batch is not None:
        assert data1.non_tensor_batch.keys() == data2.non_tensor_batch.keys()
        for key in data1.non_tensor_batch.keys():
            assert np.all(data1.non_tensor_batch[key] == data2.non_tensor_batch[key])
    else:
        assert data2.non_tensor_batch is None

    assert data1.meta_info == data2.meta_info


def test_tensor_dict_constructor():
    obs = torch.randn(100, 10)
    act = torch.randn(100, 10, 3)
    data = DataProto.from_dict(tensors={"obs": obs, "act": act})
    assert len(data) == 100

    with pytest.raises(AssertionError):
        data = DataProto.from_dict(tensors={"obs": obs, "act": act}, num_batch_dims=2)

    with pytest.raises(AssertionError):
        data = DataProto.from_dict(tensors={"obs": obs, "act": act}, num_batch_dims=3)

    labels = np.array(["a", "b", "c"], dtype=object)
    data = DataProto.from_dict(non_tensors={"labels": labels})
    assert len(data) == 3


def test_getitem():
    data = _get_data_proto()
    assert data[0].batch["obs"] == torch.tensor(1)
    assert data[0].non_tensor_batch["labels"] == "a"
    _assert_equal(data[1:3], _get_data_proto({"obs": [2, 3]}, {"labels": ["b", "c"]}))
    _assert_equal(data[[0, 2]], _get_data_proto({"obs": [1, 3]}, {"labels": ["a", "c"]}))
    _assert_equal(data[torch.tensor([1])], _get_data_proto({"obs": [2]}, {"labels": ["b"]}))


def test_select_pop():
    obs = torch.randn(100, 10)
    act = torch.randn(100, 3)
    dataset = _get_data_proto(tensors={"obs": obs, "act": act}, meta_info={"p": 1, "q": 2})
    selected_dataset = dataset.select(batch_keys=["obs"], meta_info_keys=["p"])

    assert selected_dataset.batch.keys() == {"obs"}
    assert selected_dataset.meta_info.keys() == {"p"}
    assert dataset.batch.keys() == {"obs", "act"}
    assert dataset.meta_info.keys() == {"p", "q"}

    popped_dataset = dataset.pop(batch_keys=["obs"], meta_info_keys=["p"])
    assert popped_dataset.batch.keys() == {"obs"}
    assert popped_dataset.meta_info.keys() == {"p"}
    assert dataset.batch.keys() == {"act"}
    assert dataset.meta_info.keys() == {"q"}


def test_chunk_concat_split():
    data = _get_data_proto()
    with pytest.raises(AssertionError):
        data.chunk(5)

    chunked_data = data.chunk(2)

    assert len(chunked_data) == 2
    expected_data = _get_data_proto({"obs": [1, 2, 3]}, {"labels": ["a", "b", "c"]})
    _assert_equal(chunked_data[0], expected_data)

    concat_data = DataProto.concat(chunked_data)
    _assert_equal(concat_data, data)

    splitted_data = data.split(2)
    assert len(splitted_data) == 3
    expected_data = _get_data_proto({"obs": [1, 2]}, {"labels": ["a", "b"]})
    _assert_equal(splitted_data[0], expected_data)


def test_reorder():
    data = _get_data_proto()
    data.reorder(torch.tensor([3, 4, 2, 0, 1, 5]))
    expected_data = _get_data_proto({"obs": [4, 5, 3, 1, 2, 6]}, {"labels": ["d", "e", "c", "a", "b", "f"]})
    _assert_equal(data, expected_data)


@pytest.mark.parametrize("interleave", [True, False])
def test_repeat(interleave: bool):
    data = _get_data_proto({"obs": [1, 2]}, {"labels": ["a", "b"]})
    repeated_data = data.repeat(repeat_times=2, interleave=interleave)
    expected_tensors = {"obs": [1, 1, 2, 2] if interleave else [1, 2, 1, 2]}
    expected_non_tensors = {"labels": ["a", "a", "b", "b"] if interleave else ["a", "b", "a", "b"]}
    _assert_equal(repeated_data, _get_data_proto(expected_tensors, expected_non_tensors))


@pytest.mark.parametrize("size_divisor", [2, 3])
def test_dataproto_pad_unpad(size_divisor: int):
    data = _get_data_proto({"obs": [1, 2, 3]}, {"labels": ["a", "b", "c"]})
    # test size_divisor=2
    padded_data, pad_size = pad_dataproto_to_divisor(data, size_divisor=size_divisor)
    unpadded_data = unpad_dataproto(padded_data, pad_size=pad_size)

    if size_divisor == 2:
        assert pad_size == 1
        expected_tensors = {"obs": [1, 2, 3, 1]}
        expected_non_tensors = {"labels": ["a", "b", "c", "a"]}
        expected_data = _get_data_proto(expected_tensors, expected_non_tensors)
    else:
        assert pad_size == 0
        expected_data = data

    _assert_equal(padded_data, expected_data)
    _assert_equal(unpadded_data, data)


def test_data_proto_save_load():
    data = _get_data_proto()
    data.save_to_disk("test_data.pt")
    loaded_data = DataProto.load_from_disk("test_data.pt")
    os.remove("test_data.pt")
    _assert_equal(data, loaded_data)


def test_union_tensor_dict():
    obs = torch.randn(100, 10)
    data1 = _get_data_proto({"obs": obs, "act": torch.randn(100, 3)})
    data2 = _get_data_proto({"obs": obs, "rew": torch.randn(100)})
    data1.union(data2)

    data1 = _get_data_proto({"obs": obs, "act": torch.randn(100, 3)})
    data2 = _get_data_proto({"obs": obs + 1, "rew": torch.randn(100)})
    with pytest.raises(ValueError):
        data1.union(data2)
