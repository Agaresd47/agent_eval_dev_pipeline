PYTHON ?= python
SOURCE ?= examples/source/example_pair_promotion.py
SOURCE_ROOT ?= examples/source
LIMIT ?= 4
ARTIFACTS ?=
RUN_PREFIX ?= dryrun_plan_check

.PHONY: test run run-no-critics batch dryrun-check

test:
	$(PYTHON) -m pytest tests -q

run:
	$(PYTHON) -m task_forge_v2.run_pipeline --source $(SOURCE)

run-no-critics:
	$(PYTHON) -m task_forge_v2.run_pipeline --source $(SOURCE) --pipeline-variant no_critics

batch:
	$(PYTHON) -m task_forge_v2.run_batch --source-root $(SOURCE_ROOT) --limit $(LIMIT)

dryrun-check:
	@if [ -z "$(ARTIFACTS)" ]; then \
		echo "Set ARTIFACTS, e.g.: make dryrun-check ARTIFACTS=\"task_forge_20260518_v8_full_review_01 task_forge_20260518_v8_no_critics_02\""; \
		exit 1; \
	fi
	$(PYTHON) -m scripts.run_dryrun_plan_check --artifacts $(ARTIFACTS) --run-prefix $(RUN_PREFIX)
