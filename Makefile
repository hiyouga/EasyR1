.PHONY: build commit license quality style test

check_dirs := examples scripts tests verl setup.py

code_dirs := scripts tests verl setup.py

RUN := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "")
BUILD := $(shell command -v uv >/dev/null 2>&1 && echo "uv build" || echo "python -m build")
TOOL := $(shell command -v uv >/dev/null 2>&1 && echo "uvx" || echo "")

build:
	$(RUN) python3 setup.py sdist bdist_wheel

commit:
	$(TOOL) pre-commit install
	$(TOOL) pre-commit run --all-files

license:
	$(RUN) python3 tests/check_license.py $(code_dirs)

quality:
	$(TOOL) ruff check $(check_dirs)
	$(TOOL) ruff format --check $(check_dirs)

style:
	$(TOOL) ruff check $(check_dirs) --fix
	$(TOOL) ruff format $(check_dirs)

test:
	$(RUN) pytest -vv tests/
