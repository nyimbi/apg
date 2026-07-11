install:
	pip install -e .

test:
	python -m pytest tests/ -q

lint:
	python -m flake8 compiler/ cli/ --max-line-length=120 || true

dist:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info
