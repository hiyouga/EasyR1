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

import random

from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions


# Partitions recorded from the previous padded implementation.
_FIXED = [
    ([4, 1, 8, 3, 2, 7], 3, [[4, 5], [0, 1, 3], [2]]),
    ([5, 5, 5, 5, 5, 5], 2, [[0, 2, 5], [1, 3, 4]]),
    ([0, 9, 0, 4], 2, [[1], [0, 2, 3]]),
    (list(range(12)), 5, [[1, 5, 8], [0, 6, 7], [2, 11], [3, 10], [4, 9]]),
]


def test_unequal_partitions_match_recorded_examples():
    for lengths, k, expected in _FIXED:
        assert get_seqlen_balanced_partitions(lengths, k, equal_size=False) == expected


def test_unequal_partitions_cover_every_index():
    rng = random.Random(0)
    for n in range(1, 18):
        for k in range(1, n + 1):
            lengths = [rng.randint(0, 40) for _ in range(n)]
            partitions = get_seqlen_balanced_partitions(lengths, k, equal_size=False)
            assert sorted(i for part in partitions for i in part) == list(range(n))
            assert all(part == sorted(part) and part for part in partitions)


def test_equal_size_partitions_cover_every_index():
    lengths = [4, 1, 8, 3, 2, 7, 9, 5]
    partitions = get_seqlen_balanced_partitions(lengths, 4, equal_size=True)
    assert sorted(i for part in partitions for i in part) == list(range(len(lengths)))
    assert {len(part) for part in partitions} == {2}
