.PHONY: setup data test lint typecheck serve stop pilot bench report clean

UV ?= uv

setup:
	$(UV) sync --extra dev

data:
	$(UV) run python -m cmbench.data.seed --out ./data/crm.db --seed 42

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check src tests

typecheck:
	$(UV) run mypy src

# make serve MODEL=gpt-oss-20b
serve:
	bash scripts/serve_vllm.sh $(MODEL)

stop:
	bash scripts/stop_vllm.sh

pilot:
	$(UV) run cmbench run --config configs/experiment.yaml --pilot

bench:
	bash scripts/run_all.sh

report:
	$(UV) run python -m cmbench.report.aggregate
	$(UV) run python -m cmbench.report.plots
	$(UV) run python -m cmbench.report.render

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__
