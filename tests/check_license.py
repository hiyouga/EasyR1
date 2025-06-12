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

import sys
from pathlib import Path


KEYWORDS = ("Copyright", "2024", "Bytedance")


def main():
    path_list: list[Path] = []
    for check_dir in sys.argv[1:]:
        path_list.extend(Path(check_dir).glob("**/*.py"))

    for path in path_list:
        with open(path.absolute(), encoding="utf-8") as f:
            file_content = f.read().strip().split("\n")
            license = "\n".join(file_content[:5])
            if not license:
                continue

            print(f"Check license: {path}")
            assert all(keyword in license for keyword in KEYWORDS), f"File {path} does not contain license."


if __name__ == "__main__":
    main()
