"""Cross-platform QRZ logbook visualizer with caching and theming.

Usage examples:
    python qrz_viz.py --configure
    python qrz_viz.py --start-date 2024-01-01
    python qrz_viz.py --fetch-only
    python qrz_viz.py --visualize-only --theme light
"""
from __future__ import annotations

import argparse
import getpass
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import adif_io
import pandas as pd
import plotly.graph_objects as go
import requests

DEFAULT_CONFIG_PATH = Path("config.toml")

DEFAULT_CONFIG = {
    "credentials": {
        "api_key": "",
        "user_agent": "KC2QJA-LogbookViz/2.0",
    },
    "paths": {
        "cache": "cache/qrz_cache.sqlite3",
        "output": "output",
    },
    "ui": {
        "theme": "dark",
    },
    "themes": {
        "dark": {
            "background": "#0b0c10",
            "land": "#1f2833",
            "ocean": "#0b1e2d",
            "text": "#66fcf1",
            "title": "#45a29e",
        },
        "light": {
            "background": "#f7f7f7",
            "land": "#e0e0e0",
            "ocean": "#d5e7f5",
            "text": "#1f1f1f",
            "title": "#005f73",
        },
    },
}

BATCH = 250


@dataclass
class Config:
    api_key: str
    user_agent: str
    cache_path: Path
    output_dir: Path
    theme_name: str
    theme: dict

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "Config":
        import tomllib

        if not path.exists():
            write_default_config(path)
        data = tomllib.loads(path.read_text())
        theme_name = data.get("ui", {}).get("theme", "dark")
        theme = data.get("themes", {}).get(theme_name)
        if not theme:
            theme_name = "dark"
            theme = DEFAULT_CONFIG["themes"]["dark"]
        return cls(
            api_key=data.get("credentials", {}).get("api_key", ""),
            user_agent=data.get("credentials", {}).get("user_agent", DEFAULT_CONFIG["credentials"]["user_agent"]),
            cache_path=Path(data.get("paths", {}).get("cache", DEFAULT_CONFIG["paths"]["cache"])),
            output_dir=Path(data.get("paths", {}).get("output", DEFAULT_CONFIG["paths"]["output"])),
            theme_name=theme_name,
            theme=theme,
        )

    def save_api_key(self, key: str, path: Path = DEFAULT_CONFIG_PATH) -> None:
        import tomllib
        import tomli_w

        data = tomllib.loads(path.read_text()) if path.exists() else DEFAULT_CONFIG.copy()
        data.setdefault("credentials", {})["api_key"] = key
        with path.open("wb") as fh:
            tomli_w.dump(data, fh)


def write_default_config(path: Path = DEFAULT_CONFIG_PATH) -> None:
    import tomli_w

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        tomli_w.dump(DEFAULT_CONFIG, fh)


def prompt_for_api_key(existing: str | None) -> str:
    if existing:
        return existing
    print("Enter your QRZ Logbook API key (input hidden):")
    return getpass.getpass(prompt="API key: ").strip()


def grid2latlon(gs: str) -> tuple[float | None, float | None]:
    gs = gs.strip().upper()
    if len(gs) < 4 or not gs[:4].isalnum():
        return None, None
    try:
        lon = (ord(gs[0]) - 65) * 20 - 180 + int(gs[2]) * 2 + 1
        lat = (ord(gs[1]) - 65) * 10 - 90 + int(gs[3]) + 0.5
    except ValueError:
        return None, None
    if len(gs) >= 6 and gs[4:6].isalpha():
        lon += (ord(gs[4]) - 65) * 5 / 60 + 5 / 120
        lat += (ord(gs[5]) - 65) * 2.5 / 60 + 2.5 / 120
    return lat, lon


def _parse_coord(val: str | None) -> float | None:
    if val is None:
        return None
    val = str(val).strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def iter_records(api_key: str, after: int, user_agent: str) -> Iterator[dict]:
    while True:
        opt = f"MAX:{BATCH}" + (f",AFTERLOGID:{after}" if after else "")
        resp = requests.post(
            "https://logbook.qrz.com/api",
            data={"KEY": api_key, "ACTION": "FETCH", "OPTION": opt},
            headers={"User-Agent": user_agent},
            timeout=30,
        )
        resp.raise_for_status()
        if "ADIF=" not in resp.text:
            break
        adif_raw = resp.text.split("ADIF=", 1)[1]
        if not adif_raw.strip():
            break
        qsos, _ = adif_io.read_from_string("<adif_ver:5>3.1.0<eoh>\n" + adif_raw + "\n")
        if not qsos:
            break
        for qso in qsos:
            yield qso
        after = max(int(q["app_qrzlog_logid"]) for q in qsos) + 1
        if len(qsos) < BATCH:
            break


def ensure_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS qso (
            log_id INTEGER PRIMARY KEY,
            qso_date TEXT,
            band TEXT,
            mode TEXT,
            call TEXT,
            gridsquare TEXT,
            lat REAL,
            lon REAL,
            raw_json TEXT
        )
        """
    )
    return conn


def insert_records(conn: sqlite3.Connection, records: Iterable[dict]) -> int:
    cur = conn.cursor()
    inserted = 0
    for rec in records:
        log_id = int(rec.get("app_qrzlog_logid"))
        qso_date = str(rec.get("qso_date"))
        band = str(rec.get("band", "")).upper() or None
        mode = str(rec.get("mode", "")).upper() or None
        call = str(rec.get("call", "")).upper() or None
        lat = _parse_coord(rec.get("lat")) or _parse_coord(rec.get("my_lat"))
        lon = _parse_coord(rec.get("lon")) or _parse_coord(rec.get("my_lon"))
        if (lat is None or lon is None) and rec.get("gridsquare"):
            lat, lon = grid2latlon(str(rec["gridsquare"]))
        cur.execute(
            """
            INSERT OR IGNORE INTO qso (log_id, qso_date, band, mode, call, gridsquare, lat, lon, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (log_id, qso_date, band, mode, call, rec.get("gridsquare"), lat, lon, json.dumps(rec)),
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def highest_log_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(log_id), 0) FROM qso").fetchone()
    return int(row[0]) if row and row[0] else 0


def fetch_to_cache(config: Config) -> int:
    conn = ensure_db(config.cache_path)
    after = highest_log_id(conn)
    print(f"Fetching QRZ logbook starting after log_id {after}...")
    records = iter_records(config.api_key, after=after, user_agent=config.user_agent)
    inserted = insert_records(conn, records)
    if inserted:
        print(f"Cached {inserted} new QSOs.")
    else:
        print("No new QSOs available.")
    return inserted


def load_dataframe(conn: sqlite3.Connection, start_date: str | None) -> pd.DataFrame:
    query = "SELECT * FROM qso"
    params: list[str] = []
    if start_date:
        query += " WHERE qso_date >= ?"
        params.append(start_date.replace("-", ""))
    df = pd.read_sql_query(query, conn, params=params)
    df = df.dropna(subset=["lat", "lon"])
    return df


def summarize(df: pd.DataFrame) -> dict:
    summary = {
        "total": len(df),
        "bands": df["band"].nunique() if "band" in df else 0,
        "modes": df["mode"].nunique() if "mode" in df else 0,
        "calls": df["call"].nunique() if "call" in df else 0,
    }
    return summary


def visualize(df: pd.DataFrame, config: Config, start_date: str | None) -> Path:
    if df.empty:
        raise SystemExit("No QSOs with location data to plot. Try fetching data or expanding the date range.")

    colors = {
        "80M": "#00FF7F",
        "40M": "#FFFF55",
        "20M": "#FF5555",
        "17M": "#DA70D6",
        "15M": "#FF9F1C",
        "10M": "#FFD166",
        "6M": "#7FFF00",
        "30M": "#00CED1",
    }
    band_colors = [colors.get(band, "#888888") for band in df.get("band", [])]

    hover = [
        f"{row.call or '??'}<br>{row.band or ''} {row.mode or ''}<br>{row.qso_date}"
        for row in df.itertuples()
    ]

    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon=df["lon"],
                lat=df["lat"],
                text=hover,
                mode="markers",
                marker=dict(size=6, color=band_colors, opacity=0.75),
            )
        ]
    )

    theme = config.theme
    fig.update_layout(
        title=f"QRZ Logbook – since {start_date or 'beginning'}",
        geo=dict(
            projection_type="natural earth",
            showland=True,
            landcolor=theme.get("land", "#1f2833"),
            oceancolor=theme.get("ocean", "#0b1e2d"),
            showocean=True,
            bgcolor=theme.get("background", "#0b0c10"),
        ),
        paper_bgcolor=theme.get("background", "#0b0c10"),
        plot_bgcolor=theme.get("background", "#0b0c10"),
        font=dict(color=theme.get("text", "#66fcf1")),
        title_font=dict(color=theme.get("title", "#45a29e")),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = config.output_dir / f"logbook_{timestamp}.html"
    fig.write_html(out_path)
    return out_path


def configure(path: Path = DEFAULT_CONFIG_PATH) -> None:
    try:
        cfg = Config.load(path)
    except Exception:
        write_default_config(path)
        cfg = Config.load(path)
    key = prompt_for_api_key(cfg.api_key)
    cfg.save_api_key(key, path)
    theme = cfg.theme_name
    print(f"Stored API key. Current theme: {theme}. Edit {path} to change themes or paths.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="QRZ logbook visualizer")
    parser.add_argument("--configure", action="store_true", help="Prompt for and store API key in config.toml")
    parser.add_argument("--api-key", dest="api_key", help="Override API key from config")
    parser.add_argument("--start-date", dest="start_date", help="YYYY-MM-DD; only visualize QSOs on/after this date")
    parser.add_argument("--theme", dest="theme", help="Theme name from config")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch/cached data, skip visualization")
    parser.add_argument("--visualize-only", action="store_true", help="Only visualize cached data")
    args = parser.parse_args(argv)

    if args.configure:
        configure()
        return

    cfg = Config.load()
    if args.theme and args.theme in DEFAULT_CONFIG["themes"]:
        cfg.theme_name = args.theme
        cfg.theme = DEFAULT_CONFIG["themes"][args.theme]

    api_key = args.api_key or cfg.api_key
    needs_fetch = not args.visualize_only
    if needs_fetch:
        api_key = prompt_for_api_key(api_key)
        cfg.api_key = api_key
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if needs_fetch:
        fetch_to_cache(cfg)

    if args.fetch_only:
        return

    conn = ensure_db(cfg.cache_path)
    df = load_dataframe(conn, start_date=args.start_date)
    summary = summarize(df)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    out_path = visualize(df, cfg, start_date=args.start_date)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
