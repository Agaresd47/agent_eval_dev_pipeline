PYTHON ?= python
SOURCE ?= examples/source/example_pair_promotion.py
SOURCE_ROOT ?= examples/source
LIMIT ?= 4

.PHONY: test run run-no-critics batch

test:
	$(PYTHON) -m pytest tests -q

run:
	$(PYTHON) -m task_forge_v2.run_pipeline --source $(SOURCE)

run-no-critics:
	$(PYTHON) -m task_forge_v2.run_pipeline --source $(SOURCE) --pipeline-variant no_critics

batch:
	$(PYTHON) -m task_forge_v2.run_batch --source-root $(SOURCE_ROOT) --limit $(LIMIT)
