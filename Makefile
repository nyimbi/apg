install:
	pip install -e .

test:
	python -m pytest tests/ -q

lint:
	python -m ruff check compiler/ cli/

dist:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info
