PYTHON ?= python3

.PHONY: test run bench clean

test:
	$(PYTHON) -m pytest tests/ -v

run:
	$(PYTHON) -m uvicorn backend.app:app --host 127.0.0.1 --port 8080 --reload

bench:
	$(PYTHON) evaluation/benchmark.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage dist build *.egg-info
