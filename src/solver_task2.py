from __future__ import annotations
from typing import List, Dict
import numpy as np
from .retriever import build_law_corpus, retrieve
from .dataio import LawArticle
from .text_utils import preprocess

def _score_pair(index, text: str, law_rows: list):
    text = str(text) if text is not None else ""
    if getattr(index, "kind", "tfidf") == "tfidf":
        qv = index.vectorizer.transform([preprocess(text)])
        if not law_rows:
            return 0.0
        subX = index.X[law_rows, :]
        sims = (qv @ subX.T).toarray().ravel()   # <- FIX: dùng toarray()
        return float(np.mean(sims)) if sims.size else 0.0
    else:
        # BM25
        tokens = preprocess(text).split()
        scores_all = index.bm25.get_scores(tokens)
        scores = [scores_all[r] for r in law_rows] if law_rows else []
        return float(np.mean(scores)) if scores else 0.0

def solve_task2(train_or_public_data: List[dict], law_articles: List[LawArticle], topk: int = 5, retriever: str = "tfidf"):
    index = build_law_corpus(law_articles, method=retriever)
    preds = []
    for ex in train_or_public_data:
        q = str(ex.get("question", "") or "")
        choices = ex.get("choices", {}) or {}

        # bước 1: truy xuất bài viết theo câu hỏi
        hits = retrieve(index, q, topk=topk)
        rows = [index.meta.index(h[1]) for h in hits]

        # bước 2: chấm điểm cho từng lựa chọn
        best_key, best_score = None, -1e18
        scores = {}
        for key, raw_text in choices.items():
            text = str(raw_text) if raw_text is not None else ""   # <- FIX: ép chuỗi
            s = _score_pair(index, q + " " + text, rows)
            scores[key] = s
            if s > best_score:
                best_key, best_score = key, s

        preds.append({"id": ex.get("id"), "answer": best_key or "A", "scores": scores})
    return preds

def accuracy_on_train(train_data: List[dict], pred_list: List[dict]) -> float:
    gt = {ex["id"]: str(ex.get("answer","")).strip() for ex in train_data if "answer" in ex}
    hit, tot = 0, 0
    for p in pred_list:
        if p["id"] in gt:
            tot += 1
            if str(p["answer"]).strip() == gt[p["id"]]:
                hit += 1
    return (hit/tot) if tot else 0.0
