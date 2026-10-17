# Recipes are plain commands so they run the same from PowerShell, cmd and Git Bash.

.PHONY: sync lock data pipeline app check format

# Create .venv and install the versions pinned in uv.lock
sync:
	uv sync

lock:
	uv lock

# Download daily XU100, VIX and USD/TRY history from Yahoo Finance into data/raw/
data:
	uv run python -m volforecast.data

# Score every model and forecast from the latest close into data/output/
pipeline:
	uv run python -m volforecast.pipeline

# Start the Streamlit app in the browser
app:
	uv run streamlit run app/streamlit_app.py

# Report lint and formatting issues without changing files
check:
	uv sync --locked
	uv run ruff check .
	uv run ruff format --check .

# Apply ruff's automatic fixes, then reformat
format:
	uv run ruff check --fix .
	uv run ruff format .
