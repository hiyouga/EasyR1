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

import torch


HALF_LIST = ["fp16", "float16"]
FLOAT_LIST = ["fp32", "float32"]
BFLOAT_LIST = ["bf16", "bfloat16"]


class PrecisionType:
    """Type of precision used."""

    @staticmethod
    def is_fp16(precision: str) -> bool:
        return precision in HALF_LIST

    @staticmethod
    def is_fp32(precision: str) -> bool:
        return precision in FLOAT_LIST

    @staticmethod
    def is_bf16(precision: str) -> bool:
        return precision in BFLOAT_LIST

    @staticmethod
    def to_dtype(precision: str) -> torch.dtype:
        if precision in HALF_LIST:
            return torch.float16
        elif precision in FLOAT_LIST:
            return torch.float32
        elif precision in BFLOAT_LIST:
            return torch.bfloat16
        else:
            raise RuntimeError(f"Unexpected precision: {precision}")

    @staticmethod
    def to_str(precision: torch.dtype) -> str:
        if precision == torch.float16:
            return "float16"
        elif precision == torch.float32:
            return "float32"
        elif precision == torch.bfloat16:
            return "bfloat16"
        else:
            raise RuntimeError(f"Unexpected precision: {precision}")
