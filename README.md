# Quaternion Neural Networks

This project provides a robust implementation of Quaternion Neural Networks in PyTorch, heavily inspired by and built upon the excellent [Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks) by Orkis Research. It extends the original work with improved software engineering practices, type hinting, and strict code quality enforcement.

## Package Management

This package uses `uv` for lightning-fast project dependency management and environment synchronization. `uv sync` is used to strictly resolve dependencies and create reproducible environments.

## Development Commands

This project uses a `Makefile` to provide a convenient interface for common development tasks. You can also run the commands directly with `uv run`.

## Dependencies

Dependencies are defined in [`pyproject.toml`](./pyproject.toml). To synchronize your local virtual environment and install all development tools:

```shell
make edit-install
```
Alternatively, you can run `uv sync` directly:

```shell
uv sync --all-extras
```
After syncing, you can execute any script or test seamlessly using `uv run`, which automatically uses the managed environment:
```shell
uv run pytest
```

To upgrade all dependencies to their latest versions, you can edit the `pyproject.toml` file and then run the install command again.

## Packaging

This project is designed as a Python package, meaning that it can be bundled up and redistributed as a single compressed file. Packaging is configured by:

- [`pyproject.toml`](./pyproject.toml)

To package the project as both a [source distribution](https://packaging.python.org/en/latest/flow/#the-source-distribution-sdist) and a [wheel](https://packaging.python.org/en/latest/specifications/binary-distribution-format/):

First, install the `build` package:
```shell
pip install build
```

Then, run the build command:
```shell
python -m build
```

This will generate `dist/quaternion_neural_networks-0.0.1.tar.gz` and `dist/quaternion_neural_networks-0.0.1-py3-none-any.whl`.

## Enforcing Code Quality

Automated code quality checks are performed using [Ruff](https://docs.astral.sh/ruff/).

## Unit Testing

Unit testing is performed with [pytest](https://pytest.org/).

To run unit tests:

```shell
make test
```
Alternatively, you can run `pytest` directly:
```shell
pytest
```

Code coverage is provided by the [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) plugin.

## Code Style Checking

[PEP 8](https://peps.python.org/pep-0008/) is the universally accepted style guide for Python code. PEP 8 code compliance is verified using [Ruff](https://docs.astral.sh/ruff/). Ruff is configured in the `[tool.ruff]` section of [`pyproject.toml`](./pyproject.toml).

To lint code, run:

```shell
make lint
```
Alternatively, you can run `ruff` directly:
```shell
ruff check .
```

To automatically fix fixable lint errors, run:

```shell
ruff check . --fix
```

## Automated Code Formatting

[Ruff](https://docs.astral.sh/ruff/) is used to automatically format code and group and sort imports.

To automatically format code, run:

```shell
make format
```
Alternatively, you can run `ruff` directly:
```shell
ruff format .
```

## Type Checking

[Type annotations](https://docs.python.org/3/library/typing.html) allows developers to include optional static typing information to Python source code. This allows static analyzers such as [mypy](http://mypy-lang.org/) to check that functions are used with the correct types before runtime.

mypy is configured in [`pyproject.toml`](./pyproject.toml). To type check code, run:

```shell
make typecheck
```
Alternatively, you can run `mypy` directly:
```shell
mypy .
```

## Project Structure

This project uses a flat layout. This results in a directory structure like:

```
quaternion_neural_networks
├── quaternion_neural_networks
│   ├── __init__.py
│   ├── cli.py
│   └── schema.py
├── tests
│   └── test_example.py
└── pyproject.toml
```

## Licensing

Licensing for the project is defined in:

- [`LICENSE`](./LICENSE)
- [`pyproject.toml`](./pyproject.toml)

This project uses the GNU General Public License v3.0 (GPL-3.0), as it inherits from the original Pytorch-Quaternion-Neural-Networks codebase.

## Container

[Docker](https://www.docker.com/) is a tool that allows for software to be packaged into isolated containers.

The Docker configuration in this repository is optimized for small size and increased security. Docker is configured in:

- [`Dockerfile`](./Dockerfile)
- [`.dockerignore`](./.dockerignore)

To build the container image:

```shell
docker build --tag quaternion_neural_networks .
```

To run the image in a container:

```shell
docker run --rm --interactive --tty quaternion_neural_networks
```