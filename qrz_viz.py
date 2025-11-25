# Databricks notebook source
# MAGIC %pip install adif_io cartopy shapely pyproj

# COMMAND ----------

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# Cell 1 – imports & constants
import html, re, requests, adif_io, pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, timezone
from pathlib import Path
from pyspark.sql import SparkSession

API_KEY = "api_key"
UA      = "KC2QJA-LogbookViz/1.2"
BATCH   = 250
ORIGIN_LAT, ORIGIN_LON = 30.4145, -97.7411
BAND_COLORS = {"80M":"#00FF7F","40M":"#FFFF55","20M":"#FF5555","17M":"#DA70D6","6M":"#7FFF00","30M":"#00CED1"}
BG, LAND, OCEAN, TXT, TITLE = "#000000", "#141414", "#002b36", "#00FF7F", "#FF5555"
FONT = "'Share Tech Mono','IBM Plex Mono',monospace"

# COMMAND ----------
# Cell 2 – fetch QRZ logbook
def fetch_records(key: str, ua: str = UA, batch: int = BATCH) -> list[dict]:
    after, recs = 0, []
    while True:
        opt = f"MAX:{batch}" + (f",AFTERLOGID:{after}" if after else "")
        r = requests.post("https://logbook.qrz.com/api",
                          data={"KEY": key, "ACTION": "FETCH", "OPTION": opt},
                          headers={"User-Agent": ua}, timeout=30)
        r.raise_for_status()
        if "ADIF=" not in r.text:
            break
        adif_raw = html.unescape(r.text.split("ADIF=", 1)[1])
        if not adif_raw.strip():
            break
        qsos, _ = adif_io.read_from_string("<adif_ver:5>3.1.0<eoh>\n" + adif_raw + "\n")
        if not qsos:
            break
        recs.extend(qsos)
        after = max(int(q["app_qrzlog_logid"]) for q in qsos) + 1
        if len(qsos) < batch:
            break
    return recs

records = fetch_records(API_KEY)

# COMMAND ----------
# Cell 3 – DataFrames & display
pdf = pd.DataFrame(records, dtype=str).replace({pd.NA: None})
pdf.columns = [re.sub(r"[^0-9A-Za-z_]", "_", c.strip()) for c in pdf.columns]

spark = SparkSession.builder.getOrCreate()
sdf = spark.createDataFrame(pdf)
df = pdf.copy()

# COMMAND ----------
# Cell 4 – helper functions
def _load_csv(path: str, url: str = None, **kw) -> pd.DataFrame:
    return pd.read_csv(path, **kw) if Path(path).exists() else (pd.read_csv(url, **kw) if url else pd.DataFrame())

def grid2latlon(gs: str):
    gs = gs.strip().upper()
    if len(gs) < 4 or not gs[:4].isalnum():
        return None, None
    try:
        lon = (ord(gs[0])-65)*20 - 180 + int(gs[2])*2 + 1
        lat = (ord(gs[1])-65)*10 - 90 + int(gs[3]) + 0.5
    except ValueError:
        return None, None
    if len(gs) >= 6 and gs[4:6].isalpha():
        lon += (ord(gs[4])-65)*5/60 + 5/120
        lat += (ord(gs[5])-65)*2.5/60 + 2.5/120
    return lat, lon

def hav(lat1, lon1, lat2, lon2):
    dlat, dlon = map(radians, [lat2-lat1, lon2-lon1])
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*6371*asin(sqrt(a))

def _parse_coord(val: str):
    val = str(val).strip()
    if not val:
        return None
    if val[0] in "NSEW":
        parts = val[1:].strip().split()
        if len(parts) != 2:
            return None
        try:
            deg, minutes = float(parts[0]), float(parts[1])
            out = deg + minutes/60.0
            return -out if val[0] in "SW" else out
        except ValueError:
            return None
    try:
        return float(val)
    except ValueError:
        return None

# COMMAND ----------
# Cell 5 – enrich DataFrame & derive metrics
us_cent = _load_csv("/dbfs/FileStore/pbuch/us_state_centroids.csv",
                    "https://raw.githubusercontent.com/plotly/datasets/master/us-state-centroids.csv")
ca_cent = _load_csv("/dbfs/FileStore/pbuch/ca_province_centroids.csv")
cent_all = (pd.concat([us_cent, ca_cent], ignore_index=True)
            .rename(columns=str.lower)
            .set_index("region").T.to_dict()) if not us_cent.empty else {}

world_coords = _load_csv("/dbfs/FileStore/pbuch/world_coords.csv", dtype=str)
if not world_coords.empty:
    world_coords = world_coords.apply(lambda s: s.str.upper())

df.columns = [c.lower() for c in df.columns]

def _coords(row):
    lat, lon = _parse_coord(row.get("lat")), _parse_coord(row.get("lon"))
    if lat is not None and lon is not None:
        return pd.Series([lat, lon])
    gs = str(row.get("gridsquare") or row.get("my_gridsquare") or "")
    if gs:
        lat, lon = grid2latlon(gs)
        if lat is not None:
            return pd.Series([lat, lon])
    st = str(row.get("state") or "").upper()
    if st and st in cent_all:
        return pd.Series([cent_all[st]["lat"], cent_all[st]["lon"]])
    c = str(row.get("country") or "").upper()
    if not world_coords.empty:
        m = world_coords[(world_coords.country_name == c) |
                         (world_coords.iso2 == c) |
                         (world_coords.iso3 == c)]
        if not m.empty:
            return pd.Series([float(m.lat.iloc[0]), float(m.lon.iloc[0])])
    return pd.Series([np.nan, np.nan])

df[["lat", "lon"]] = df.apply(_coords, axis=1)
df = df.dropna(subset=["lat", "lon"])

df["mode"] = df.get("mode", "").astype(str).str.upper()
df["band"] = df.get("band", "").astype(str).str.upper()
df["km"]   = df.apply(lambda r: hav(ORIGIN_LAT, ORIGIN_LON, r.lat, r.lon), axis=1)

sent_cols = [c for c in df.columns if c in {"rst_sent", "rst_snt", "rst_out"}]
rcvd_cols = [c for c in df.columns if c in {"rst_rcvd", "rst_rcv", "rst_in"}]
df["rst_sent_num"] = pd.to_numeric(df[sent_cols[0]], errors="coerce") if sent_cols else np.nan
df["rst_rcvd_num"] = pd.to_numeric(df[rcvd_cols[0]], errors="coerce") if rcvd_cols else np.nan

total_qsos = len(df)
dxcc_cnt   = df["country"].str.upper().nunique()

def _us_mask(frame):
    mask = pd.Series(False, index=frame.index)
    if "country" in frame.columns:
        mask |= frame["country"].str.upper().isin(
            ["USA", "UNITED STATES", "UNITED STATES OF AMERICA"]
        )
    if "dxcc" in frame.columns:
        mask |= frame["dxcc"].astype(str).str.strip() == "291"
    return mask

us_states = (df.loc[_us_mask(df), "state"]
              .astype(str)
              .str.upper()
              .nunique())

ft8          = df[df["mode"] == "FT8"]
avg_ft8_rcvd = ft8["rst_rcvd_num"].mean()
avg_ft8_sent = ft8["rst_sent_num"].mean()
last_upd     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# COMMAND ----------
# Cell 6 – Robinson 4 K wallpaper (map restored, tighter metrics box)

# MAGIC %pip install -q cartopy shapely pyproj

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

Path("/dbfs/FileStore/pbuch/logbook").mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(32.0, 16.0), dpi=300, facecolor=BG)
gs  = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.02)

# map axis
ax = fig.add_subplot(gs[0], projection=ccrs.Robinson())
ax.set_global()
ax.set_facecolor(BG)
ax.add_feature(cfeature.LAND.with_scale("50m"),  facecolor=LAND,  edgecolor="none")
ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN, edgecolor="none")
edge = {"edgecolor": "#555", "linewidth": 0.4}
ax.add_feature(cfeature.BORDERS.with_scale("50m"), **edge)
ax.add_feature(cfeature.COASTLINE.with_scale("50m"), **edge)

for band, grp in df.groupby("band", sort=False):
    clr = BAND_COLORS.get(band, TXT)
    for _, r in grp.iterrows():
        ax.plot([ORIGIN_LON, r.lon], [ORIGIN_LAT, r.lat],
                color=clr, linewidth=0.8, transform=ccrs.Geodetic())
    ax.scatter(grp["lon"], grp["lat"], s=35, color=clr, edgecolor=BG,
               transform=ccrs.PlateCarree(), label=band, zorder=3)

ax.set_title("KC2QJA Logbook", fontsize=36, color=TITLE, family="monospace", pad=25)
leg = ax.legend(facecolor=BG, edgecolor="none", fontsize=18,
                labelcolor=TXT, loc="lower left")
for t in leg.get_texts():
    t.set_color(TXT)

# metrics axis
ax_info = fig.add_subplot(gs[1])
ax_info.set_facecolor(BG)
ax_info.axis("off")
metrics = (
    f"Total QSOs : {total_qsos}\n"
    f"Unique DXCC : {dxcc_cnt}\n"
    f"US States   : {us_states}\n"
    f"FT8 Avg RCVD: {avg_ft8_rcvd:.2f}\n"
    f"FT8 Avg SENT: {avg_ft8_sent:.2f}\n"
    f"Last update : {last_upd}"
)
ax_info.text(0.02, 0.5, metrics, ha="left", va="center",
             fontsize=15, color=TXT, family="monospace",
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.6", fc=BG, ec=TXT, lw=1.1))

fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)
out_path = "/dbfs/FileStore/pbuch/logbook/7_22.png"
fig.savefig(out_path, facecolor=BG)
plt.close(fig)
displayHTML(f'<img src="/files/pbuch/logbook/7_22.png" style="width:100%">')

