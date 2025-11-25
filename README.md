# QRZ Log Visualizer

A lightweight cross-platform Python app that fetches your QRZ logbook, caches it locally, and renders an interactive map of your contacts with customizable themes. The app minimizes memory usage by streaming downloads into a SQLite cache and only loading filtered records for visualization.

## Features
- **Minimal setup**: only requires your QRZ Logbook API key.
- **Memory-friendly**: streaming fetch + SQLite cache to avoid holding full logs in RAM.
- **Cross-platform**: pure Python, works on Linux/Mac/Windows.
- **Configurable themes**: define color themes in `config.toml` and switch via CLI.
- **Date filtering**: visualize QSOs from a selected start date.

## Quickstart
1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
   (or install the minimal set manually: `requests`, `adif_io`, `pandas`, `plotly`).

2. Create a config and store your API key:
   ```bash
   python qrz_viz.py --configure
   ```
   - Paste your **QRZ Logbook API key** when prompted.
   - Pick a theme (default: `dark`).
   - Choose an output folder (defaults to `output/`).

3. Fetch and visualize in one step:
   ```bash
   python qrz_viz.py --start-date 2024-01-01
   ```
   The script downloads new QSOs into the cache and writes an HTML map in the chosen output directory.

### Separate fetch/visualize steps
- Fetch new records only (useful for automation):
  ```bash
  python qrz_viz.py --fetch-only
  ```
- Visualize previously cached data (no network call):
  ```bash
  python qrz_viz.py --visualize-only --start-date 2023-01-01
  ```

## Configuration guide (`config.toml`)
The file is created on first run. You can edit it manually:
```toml
[credentials]
api_key = "YOUR_QRZ_LOGBOOK_KEY"
user_agent = "KC2QJA-LogbookViz/2.0"

[paths]
cache = "cache/qrz_cache.sqlite3"
output = "output"

[ui]
theme = "dark"

[themes.dark]
background = "#0b0c10"
land = "#1f2833"
ocean = "#0b1e2d"
text = "#66fcf1"
title = "#45a29e"

[themes.light]
background = "#f7f7f7"
land = "#e0e0e0"
ocean = "#d5e7f5"
text = "#1f1f1f"
title = "#005f73"
```

## Running on any OS
- Use a Python 3.10+ interpreter.
- No system-level dependencies are required; all needed packages are pure Python.
- Paths in the config are relative to the project directory and work on Linux, macOS, or Windows.

## Notes on caching & memory
- Records are streamed in batches and written straight to SQLite, so memory remains bounded.
- The visualizer loads only the filtered rows (by `--start-date`), not the entire logbook.
- Re-running with the same cache only pulls new entries (based on the highest `LOGID`).

## Generating output
- The map is saved as `logbook_<timestamp>.html` in the configured output folder.
- Open the file in any modern browser.
- Summary stats are printed to the console during visualization.
