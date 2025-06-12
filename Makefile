.PHONY: build commit license quality style test

check_dirs := examples scripts tests verl setup.py

build:
	python3 setup.py sdist bdist_wheel

commit:
	pre-commit install
	pre-commit run --all-files

license:
	python3 tests/check_license.py $(check_dirs)

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	pytest -vv tests/
