.PHONY: install test eval run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

eval:
	python evals/eval_suite.py

run:
	python main.py

run-ollama:
	python main.py --provider ollama --model mistral

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -f logs/trace.jsonl evals/results.json
