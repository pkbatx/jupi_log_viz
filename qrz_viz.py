"""Command-line QRZ logbook visualizer.

This module fetches logbook entries from the QRZ Logbook API, caches them
in a small SQLite database to avoid unbounded memory growth, and produces an
interactive Plotly map plus basic metrics. The app is cross-platform and only
requires users to provide their QRZ API key and optional preferences via a
YAML config file.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

import adif_io
import pandas as pd
import plotly.graph_objects as go
import requests

DEFAULT_BATCH = 250
DEFAULT_UA = "LogbookViz/2.0"
DEFAULT_CACHE = "~/.cache/qrz_viz/cache.sqlite"
DEFAULT_OUTPUT = "logbook.html"
DEFAULT_THEME = {
    "background": "#000000",
    "land": "#141414",
    "ocean": "#002b36",
    "text": "#00FF7F",
    "accent": "#FF5555",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an interactive QRZ logbook map.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file with API key and preferences.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Optional start date (YYYY-MM-DD) to filter QSOs.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Optional end date (YYYY-MM-DD) to filter QSOs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="Output HTML file for the map.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="default",
        help="Theme name from the config file to apply to the visualization.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    import yaml

    if not path.exists():
        raise FileNotFoundError(
            f"Config file '{path}' was not found. Copy config.example.yaml and set your API key."
        )
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def normalize_date(val: str | None) -> str | None:
    if not val:
        return None
    val = str(val).strip()
    if not val:
        return None
    if len(val) == 8 and val.isdigit():
        return f"{val[:4]}-{val[4:6]}-{val[6:]}"
    try:
        parsed = datetime.fromisoformat(val)
        return parsed.date().isoformat()
    except ValueError:
        return None


def parse_coord(val: str | None) -> float | None:
    if not val:
        return None
    val = str(val).strip()
    if not val:
        return None
    if val[0] in "NSEW":
        parts = val[1:].strip().split()
        if len(parts) == 2:
            try:
                deg, minutes = float(parts[0]), float(parts[1])
                out = deg + minutes / 60.0
                return -out if val[0] in "SW" else out
            except ValueError:
                return None
    try:
        return float(val)
    except ValueError:
        return None


def grid_to_latlon(gs: str) -> tuple[float | None, float | None]:
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


class LogCache:
    def __init__(self, path: str | Path):
        resolved = Path(os.path.expanduser(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(resolved)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS qso (
                log_id INTEGER PRIMARY KEY,
                qso_date TEXT,
                data TEXT NOT NULL
            )
            """
        )

    def max_log_id(self) -> int:
        cur = self.conn.execute("SELECT MAX(log_id) FROM qso")
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def add_records(self, records: Iterable[dict]) -> int:
        to_insert = []
        for rec in records:
            log_id = int(rec.get("app_qrzlog_logid", 0) or 0)
            if not log_id:
                continue
            qso_date = normalize_date(rec.get("qso_date")) or normalize_date(rec.get("qso_date_off"))
            to_insert.append((log_id, qso_date, json.dumps(rec)))
        if not to_insert:
            return 0
        with self.conn:
            self.conn.executemany("INSERT OR IGNORE INTO qso(log_id, qso_date, data) VALUES (?, ?, ?)", to_insert)
        return len(to_insert)

    def load_records(self, start: str | None, end: str | None) -> list[dict]:
        clauses = []
        params: list[str] = []
        if start:
            clauses.append("qso_date >= ?")
            params.append(start)
        if end:
            clauses.append("qso_date <= ?")
            params.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self.conn.execute(f"SELECT data FROM qso {where}", params)
        return [json.loads(row[0]) for row in cur.fetchall()]


def fetch_new_records(api_key: str, user_agent: str, batch: int, after: int) -> list[dict]:
    after_log = after
    collected: list[dict] = []
    while True:
        opt = f"MAX:{batch}" + (f",AFTERLOGID:{after_log}" if after_log else "")
        response = requests.post(
            "https://logbook.qrz.com/api",
            data={"KEY": api_key, "ACTION": "FETCH", "OPTION": opt},
            headers={"User-Agent": user_agent},
            timeout=30,
        )
        response.raise_for_status()
        if "ADIF=" not in response.text:
            break
        adif_raw = html.unescape(response.text.split("ADIF=", 1)[1])
        if not adif_raw.strip():
            break
        qsos, _ = adif_io.read_from_string("<adif_ver:5>3.1.0<eoh>\n" + adif_raw + "\n")
        if not qsos:
            break
        collected.extend(qsos)
        after_log = max(int(q.get("app_qrzlog_logid", 0) or 0) for q in qsos) + 1
        if len(qsos) < batch:
            break
    return collected


def enrich_dataframe(records: list[dict], origin_lat: float, origin_lon: float) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    df.columns = [str(c).strip().lower() for c in df.columns]
    df["mode"] = df.get("mode", "").astype(str).str.upper()
    df["band"] = df.get("band", "").astype(str).str.upper()

    lats, lons = [], []
    for _, row in df.iterrows():
        lat = parse_coord(row.get("lat"))
        lon = parse_coord(row.get("lon"))
        if lat is None or lon is None:
            gs = str(row.get("gridsquare") or row.get("my_gridsquare") or "")
            lat, lon = grid_to_latlon(gs) if gs else (None, None)
        lats.append(lat)
        lons.append(lon)
    df["lat"], df["lon"] = lats, lons
    df = df.dropna(subset=["lat", "lon"])

    def hav(lat1, lon1, lat2, lon2):
        from math import radians, sin, cos, asin, sqrt

        dlat, dlon = map(radians, [lat2 - lat1, lon2 - lon1])
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return 2 * 6371 * asin(sqrt(a))

    df["distance_km"] = df.apply(lambda r: hav(origin_lat, origin_lon, r.lat, r.lon), axis=1)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_qsos": 0,
            "unique_dxcc": 0,
            "us_states": 0,
            "avg_ft8_rcvd": float("nan"),
            "avg_ft8_sent": float("nan"),
        }
    total_qsos = len(df)
    dxcc_cnt = df.get("country", pd.Series([], dtype=str)).astype(str).str.upper().nunique()
    us_mask = df.get("country", pd.Series([], dtype=str)).astype(str).str.upper().isin(
        ["USA", "UNITED STATES", "UNITED STATES OF AMERICA"]
    )
    states = df.loc[us_mask, "state"].astype(str).str.upper().nunique()
    ft8 = df[df["mode"] == "FT8"]
    avg_ft8_rcvd = pd.to_numeric(ft8.get("rst_rcvd"), errors="coerce").mean()
    avg_ft8_sent = pd.to_numeric(ft8.get("rst_sent"), errors="coerce").mean()
    return {
        "total_qsos": total_qsos,
        "unique_dxcc": dxcc_cnt,
        "us_states": states,
        "avg_ft8_rcvd": avg_ft8_rcvd,
        "avg_ft8_sent": avg_ft8_sent,
    }


def build_figure(df: pd.DataFrame, theme: dict, origin_lat: float, origin_lon: float, title: str) -> go.Figure:
    fig = go.Figure()
    if not df.empty:
        for band, grp in df.groupby("band"):
            fig.add_trace(
                go.Scattergeo(
                    lon=[origin_lon, *grp["lon"].tolist()],
                    lat=[origin_lat, *grp["lat"].tolist()],
                    mode="lines",
                    line=dict(width=0.6, color=theme.get("accent", DEFAULT_THEME["accent"])),
                    name=f"{band} paths",
                    opacity=0.6,
                )
            )
            fig.add_trace(
                go.Scattergeo(
                    lon=grp["lon"],
                    lat=grp["lat"],
                    mode="markers",
                    marker=dict(size=6, color=theme.get("text", DEFAULT_THEME["text"])),
                    name=f"{band} QSOs",
                    text=grp.get("call"),
                    hovertemplate="<b>%{text}</b><br>%{lat:.2f}, %{lon:.2f}<extra></extra>",
                )
            )

    metrics = compute_metrics(df)
    metrics_lines = [
        f"Total QSOs : {metrics['total_qsos']}",
        f"Unique DXCC : {metrics['unique_dxcc']}",
        f"US States   : {metrics['us_states']}",
        f"FT8 Avg RCVD: {metrics['avg_ft8_rcvd']:.2f}" if pd.notna(metrics["avg_ft8_rcvd"]) else "FT8 Avg RCVD: N/A",
        f"FT8 Avg SENT: {metrics['avg_ft8_sent']:.2f}" if pd.notna(metrics["avg_ft8_sent"]) else "FT8 Avg SENT: N/A",
        f"Last update : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    fig.add_annotation(
        text="<br>".join(metrics_lines),
        align="left",
        showarrow=False,
        x=0.98,
        y=0.02,
        xanchor="right",
        yanchor="bottom",
        bgcolor=theme.get("background", DEFAULT_THEME["background"]),
        font=dict(color=theme.get("text", DEFAULT_THEME["text"])),
        bordercolor=theme.get("text", DEFAULT_THEME["text"]),
        borderwidth=1,
    )

    fig.update_layout(
        title=title,
        paper_bgcolor=theme.get("background", DEFAULT_THEME["background"]),
        plot_bgcolor=theme.get("background", DEFAULT_THEME["background"]),
        font=dict(color=theme.get("text", DEFAULT_THEME["text"])),
        geo=dict(
            showcountries=True,
            showcoastlines=True,
            projection_type="natural earth",
            bgcolor=theme.get("background", DEFAULT_THEME["background"]),
            landcolor=theme.get("land", DEFAULT_THEME["land"]),
            oceancolor=theme.get("ocean", DEFAULT_THEME["ocean"]),
        ),
        legend=dict(bgcolor=theme.get("background", DEFAULT_THEME["background"])),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def load_theme(config: dict, name: str) -> dict:
    themes = config.get("themes", {})
    if name == "default":
        return {**DEFAULT_THEME, **themes.get(name, {})}
    return {**DEFAULT_THEME, **themes.get(name, {})}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    api_key = os.getenv("QRZ_API_KEY") or config.get("api_key")
    if not api_key:
        raise ValueError("QRZ API key is required. Set QRZ_API_KEY or api_key in config.yaml.")

    user_agent = config.get("user_agent", DEFAULT_UA)
    batch = int(config.get("batch_size", DEFAULT_BATCH))
    origin = config.get("origin", {})
    origin_lat = float(origin.get("lat", 0))
    origin_lon = float(origin.get("lon", 0))
    cache_path = config.get("cache_path", DEFAULT_CACHE)
    title = config.get("title", "QRZ Logbook")

    cache = LogCache(cache_path)
    last_log_id = cache.max_log_id()
    new_records = fetch_new_records(api_key, user_agent, batch, last_log_id)
    inserted = cache.add_records(new_records)
    if inserted:
        print(f"Cached {inserted} new QSOs (through log ID {cache.max_log_id()}).")

    start = normalize_date(args.start_date)
    end = normalize_date(args.end_date)
    if start and end and start > end:
        raise ValueError("Start date must be before end date.")

    records = cache.load_records(start, end)
    df = enrich_dataframe(records, origin_lat, origin_lon)
    theme = load_theme(config, args.theme)

    fig = build_figure(df, theme, origin_lat, origin_lon, title)
    fig.write_html(args.output)
    print(f"Wrote visualization to {args.output.resolve()}")


if __name__ == "__main__":
    main()
