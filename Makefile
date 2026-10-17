# Recipes are plain commands so they run the same from PowerShell, cmd and Git Bash.

.PHONY: sync lock check format

# Create .venv and install the versions pinned in uv.lock
sync:
	uv sync

lock:
	uv lock

# Report lint and formatting issues without changing files
check:
	uv sync --locked
	uv run ruff check .
	uv run ruff format --check .

# Apply ruff's automatic fixes, then reformat
format:
	uv run ruff check --fix .
	uv run ruff format .
