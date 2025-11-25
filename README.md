# jupi_log_viz

A cross-platform QRZ logbook visualizer that fetches your QSOs, caches them in
SQLite to stay memory-friendly, and produces an interactive Plotly world map.
Only your QRZ API key is required; everything else can be customized in a
simple YAML config (including themes and date filters).

## Features
- Incremental QRZ fetch with a small SQLite cache so runs stay fast and memory-safe
- Date filtering to generate focused reports
- Theme support (dark/light by default) for consistent visuals
- Export to a standalone HTML file that works on Linux, macOS, or Windows

## Quickstart
1. Install dependencies (Python 3.10+ recommended):
   ```bash
   pip install pandas plotly requests adif-io pyyaml
   ```
2. Copy the example config and add your credentials:
   ```bash
   cp config.example.yaml config.yaml
   # edit config.yaml and set api_key: "<YOUR_QRZ_API_KEY>"
   ```
3. Generate a map (optionally limit by date):
   ```bash
   python qrz_viz.py --start-date 2024-01-01 --end-date 2024-12-31 --output my_log.html
   ```
   The script will fetch any new QSOs, cache them at `~/.cache/qrz_viz`, and
   write the visualization to `my_log.html`.

## Configuration
- **api_key**: Your QRZ Logbook API key (or set `QRZ_API_KEY` env var).
- **user_agent**: Custom UA string for the QRZ API.
- **batch_size**: Max records per QRZ API page (default 250).
- **cache_path**: SQLite file for cached QSOs.
- **title**: Figure title.
- **origin**: `lat`/`lon` of your station for distance lines.
- **themes**: Named color sets; select one via `--theme`.

Example themes are provided in `config.example.yaml`; add more as needed.

## Notes
- The cache stores only parsed QSO JSON and keeps memory bounded by loading
  only the date range you request.
- Output HTML is self-contained and can be opened in any modern browser.
