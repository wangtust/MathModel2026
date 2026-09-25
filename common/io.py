from __future__ import annotations
import csv, json, pickle
from pathlib import Path

def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def read_pickle(path):
    with Path(path).open("rb") as f: return pickle.load(f)
def write_pickle(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("wb") as f: pickle.dump(value,f)
def write_csv(path, rows, fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    if not rows: return
    fields=fields or list(rows[0])
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
