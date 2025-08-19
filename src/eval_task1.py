
from __future__ import annotations
from typing import List, Dict, Tuple
from dataclasses import dataclass
from .dataio import LawArticle
from .retriever import build_law_corpus, retrieve

@dataclass
class RetrievalPred:
    id: str
    preds: List[Tuple[str,str]]  # (law_id, article_id)

def f2_score(y_true: List[List[Tuple[str,str]]], y_pred: List[List[Tuple[str,str]]]) -> float:
    def f2_per(gt: set, pr: set) -> float:
        if not gt and not pr:
            return 1.0
        if not gt and pr:
            return 0.0
        tp = len(gt & pr); fp = len(pr - gt); fn = len(gt - pr)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        denom = 4*precision + recall
        return 5*precision*recall/denom if denom else 0.0
    scores = []
    for gt_list, pr_list in zip(y_true, y_pred):
        scores.append(f2_per(set(gt_list), set(pr_list)))
    return float(sum(scores)/len(scores)) if scores else 0.0

def evaluate_task1(train_data: List[dict], law_articles: List[LawArticle], topk: int = 3) -> Dict:
    index = build_law_corpus(law_articles)
    y_true, y_pred, logs = [], [], []
    for ex in train_data:
        q = ex.get("question","")
        gt_pairs = [(str(ra.get("law_id","")), str(ra.get("article_id",""))) for ra in ex.get("relevant_articles", [])]
        hits = retrieve(index, q, topk=topk)
        pred_pairs = [(h[1].law_id, h[1].article_id) for h in hits]
        y_true.append(gt_pairs); y_pred.append(pred_pairs)
        logs.append({"id": ex.get("id"), "question": q, "gt": gt_pairs, "pred_topk": pred_pairs})
    return {"f2": f2_score(y_true, y_pred), "detail": logs}
