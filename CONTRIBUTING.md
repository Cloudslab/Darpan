# Contributing to Darpan

Thank you for improving Darpan. Contributions should preserve the shared
Physical/Twin execution contract and keep public behavior backed by tests.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e packages/darpan-rl
```

## Before submitting a change

1. Open an issue for API changes or new stable Actions so the contract can be
   agreed before implementation.
2. Keep Physical and Twin semantics aligned. A new stable Action requires a
   schema, validation, canonical Events, Reducer behavior, backend semantics,
   and contract tests.
3. Add the smallest test that demonstrates the intended behavior and the
   failure case it prevents.
4. Run formatting, compilation, and the relevant test suites.

```bash
python -m ruff check src packages/darpan-rl/src tests examples
python -m compileall -q src packages/darpan-rl/src
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -p pytest_asyncio.plugin tests/unit tests/contract \
  --ignore=tests/contract/test_evaluation_v2_inputs.py
```

## Pull requests

Keep pull requests focused. Describe the problem, the execution contract that
changes, tests performed, and any compatibility or security implications. Do
not commit credentials, machine inventories, generated results, virtual
environments, caches, or large binary artifacts.
