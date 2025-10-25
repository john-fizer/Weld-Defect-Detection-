.PHONY: help install dev test lint format clean backtest dashboard

help:
	@echo "Available commands:"
	@echo "  make install    - Install production dependencies"
	@echo "  make dev        - Install development dependencies"
	@echo "  make test       - Run test suite"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code with black"
	@echo "  make backtest   - Run backtests"
	@echo "  make dashboard  - Launch Streamlit dashboard"
	@echo "  make clean      - Clean generated files"

install:
	pip install poetry
	poetry install --only main

dev:
	pip install poetry
	poetry install
	mkdir -p data/vectorstore logs

test:
	poetry run pytest

lint:
	poetry run ruff check .
	poetry run mypy .

format:
	poetry run black .
	poetry run ruff check --fix .

backtest:
	poetry run jupyter nbconvert --execute --to html notebooks/backtest_iron_condor.ipynb
	poetry run jupyter nbconvert --execute --to html notebooks/backtest_long_straddle.ipynb
	poetry run jupyter nbconvert --execute --to html notebooks/backtest_wheel.ipynb

dashboard:
	poetry run streamlit run dashboards/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache
	rm -rf *.egg-info dist build
