export PYTHONPATH := $(CURDIR)

.PHONY: train predict precision

train:
	uv run --directory training python main.py

predict:
	uv run --directory prediction python main.py

precision:
	uv run --directory precision python main.py
