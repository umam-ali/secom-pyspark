"""Download the UCI SECOM semiconductor dataset into data/raw (run once)."""
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
RAW = Path(__file__).parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

zpath = RAW / "secom.zip"
if not zpath.exists():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(zpath, "wb") as f:
        f.write(r.read())

with zipfile.ZipFile(zpath) as z:
    print(z.namelist())
    z.extractall(RAW)
print("done:", sorted(p.name for p in RAW.iterdir()))
