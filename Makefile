.PHONY: .venv

run-test:
ifdef dst
	PYTHONPATH=.:src python -m pytest $(dst) -v
else
	PYTHONPATH=.:src python -m pytest -v
endif