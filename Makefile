.PHONY: train predict precision

train:
	uv run python -m training.main

predict:
	uv run python -m prediction.main

precision:
	uv run python -m precision.main
