.PHONY: test coverage echo clean fmt

run:
	python3 -m src.main

test:
	pytest

coverage:
	pytest --cov --cov-config=.coveragerc

fmt:
	black .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm -rf htmlcov/ .vscode/
	rm -f .coverage