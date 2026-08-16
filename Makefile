.PHONY: sync ui bench version

# Avoid editable installs: under Documents/iCloud, Python 3.13 skips hidden .pth files.
export UV_NO_EDITABLE := 1

sync:
	uv sync --no-editable

ui:
	uv run --no-editable python run.py ui

bench:
	uv run --no-editable python run.py bench $(ARGS)

version:
	uv run --no-editable python run.py version
