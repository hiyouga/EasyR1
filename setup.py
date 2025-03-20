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
import re

from setuptools import find_packages, setup


def get_version() -> str:
    with open(os.path.join("verl", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"__version__\W*=\W*\"([^\"]+)\""
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


extra_require = {
    "dev": ["pre-commit", "ruff"],
}


def main():
    setup(
        name="verl",
        version=get_version(),
        description="An Efficient, Scalable, Multi-Modality RL Training Framework based on veRL",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author="verl",
        author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk, hiyouga@buaa.edu.cn",
        license="Apache 2.0 License",
        url="https://github.com/volcengine/verl",
        package_dir={"": "."},
        packages=find_packages(where="."),
        python_requires=">=3.9.0",
        install_requires=get_requires(),
        extras_require=extra_require,
    )


if __name__ == "__main__":
    main()
