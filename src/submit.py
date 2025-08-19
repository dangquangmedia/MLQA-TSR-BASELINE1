
from __future__ import annotations
import json, os, zipfile
from typing import List, Dict

def write_submission(task1_preds: List[Dict], task2_preds: List[Dict], out_zip_path: str):
    tmpdir = os.path.join(os.path.dirname(out_zip_path), "_submit_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    t1_path = os.path.join(tmpdir, "submission_task1.json")
    t2_path = os.path.join(tmpdir, "submission_task2.json")
    with open(t1_path, "w", encoding="utf-8") as f: json.dump(task1_preds, f, ensure_ascii=False, indent=2)
    with open(t2_path, "w", encoding="utf-8") as f: json.dump(task2_preds, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(t1_path, arcname="submission_task1.json")
        z.write(t2_path, arcname="submission_task2.json")
    try:
        os.remove(t1_path); os.remove(t2_path); os.rmdir(tmpdir)
    except Exception:
        pass
