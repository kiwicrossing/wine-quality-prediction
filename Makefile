.PHONY: test coverage echo clean fmt

#test:

fmt:
	black .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
# 	rm -rf htmlcov/
# 	rm -f .coverage