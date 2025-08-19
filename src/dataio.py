
from __future__ import annotations
import json, zipfile
from dataclasses import dataclass
from typing import List

@dataclass
class LawArticle:
    law_id: str
    article_id: str
    title: str
    text: str

def _read_json_from_zip(z: zipfile.ZipFile, inner_path: str):
    with z.open(inner_path) as f:
        return json.load(f)

def load_dataset(dataset_zip_path: str):
    # Reads key JSON files from the BTC dataset zip without manual extraction.
    with zipfile.ZipFile(dataset_zip_path, "r") as z:
        base = "VLSP 2025 - MLQA-TSR Data Release/"
        train_json = base + "train_data/vlsp_2025_train.json"
        law_json   = base + "law_db/vlsp2025_law_new.json"
        public_task1_json = base + "public_test/vlsp_2025_public_test_task1.json"
        public_task2_json = base + "public_test/vlsp_2025_public_test_task2.json"
        train = _read_json_from_zip(z, train_json)
        laws  = _read_json_from_zip(z, law_json)
        public1 = _read_json_from_zip(z, public_task1_json)
        public2 = _read_json_from_zip(z, public_task2_json)
    return {"train": train, "laws": laws, "public_task1": public1, "public_task2": public2}

def flatten_law_db(law_db: dict) -> List[LawArticle]:
    items: List[LawArticle] = []
    if isinstance(law_db, dict):
        if "articles" in law_db:
            lid = law_db.get("law_id") or law_db.get("title","")
            for art in law_db["articles"]:
                items.append(LawArticle(law_id=str(lid), article_id=str(art.get("id","")), title=str(art.get("title","")), text=str(art.get("text",""))))
        elif "laws" in law_db:
            for law in law_db["laws"]:
                lid = law.get("law_id") or law.get("title","")
                for art in law.get("articles", []):
                    items.append(LawArticle(law_id=str(lid), article_id=str(art.get("id","")), title=str(art.get("title","")), text=str(art.get("text",""))))
    elif isinstance(law_db, list):
        for law in law_db:
            lid = law.get("law_id") or law.get("title","")
            for art in law.get("articles", []):
                items.append(LawArticle(law_id=str(lid), article_id=str(art.get("id","")), title=str(art.get("title","")), text=str(art.get("text",""))))
    return items
