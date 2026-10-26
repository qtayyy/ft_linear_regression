.PHONY: train predict

train:
	uv run --directory training python main.py

predict:
	uv run --directory prediction python prediction.py
