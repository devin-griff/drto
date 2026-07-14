# drto -- Makefile wrapper around the Python toolchain.
# Each target mirrors one CI job, so local green means CI green.
#
#   make help          # list targets
#   make dev           # editable install with dev extras
#   make fmt           # format with black
#   make fmt-check     # check formatting (CI gate)
#   make typos         # spell-check with typos
#   make lint          # fmt-check + typos
#   make test          # run pytest with coverage
#   make check-imports # import drto with only base deps present
#   make min-deps      # install the pyomo floor, then test

.PHONY: help dev fmt fmt-check typos lint test check-imports min-deps

help:
	@sed -n 's/^#   //p' Makefile

dev:
	python -m pip install -e ".[dev]"

fmt:
	black src/ tests/

fmt-check:
	black --check --diff src/ tests/

typos:
	typos

lint: fmt-check typos

test:
	python -m pytest -q --cov=drto --cov-report=term-missing

check-imports:
	python -c "import drto; print('drto', drto.__version__)"

min-deps:
	python -m pip install "pyomo==6.8.1"
	python -m pytest -q
