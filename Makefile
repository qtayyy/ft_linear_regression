.PHONY: train predict precision test

train:
	uv run python -m training.main

predict:
	uv run python -m prediction.main

precision:
	uv run python -m precision.main

test:
	uv run python -m pytest
