from __future__ import annotations
import argparse, json, os
from typing import List, Dict
from .dataio import load_dataset, flatten_law_db, LawArticle
from .eval_task1 import evaluate_task1
from .solver_task2 import solve_task2, accuracy_on_train
from .retriever import build_law_corpus, retrieve
from .submit import write_submission

def to_task1_submission(public_task1: List[dict], law_articles: List[LawArticle], topk: int = 3, retriever: str = "tfidf"):
    index = build_law_corpus(law_articles, method=retriever)
    out = []
    for ex in public_task1:
        q = ex.get("question","")
        hits = retrieve(index, q, topk=topk)
        pred_pairs = [{"law_id": h[1].law_id, "article_id": h[1].article_id} for h in hits]
        out.append({"id": ex.get("id"), "relevant_articles": pred_pairs})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_zip", required=True)
    ap.add_argument("--eval_train", action="store_true")
    ap.add_argument("--make_submission", action="store_true")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--retriever", type=str, default="tfidf", choices=["tfidf","bm25"])
    args = ap.parse_args()

    os.makedirs("runs", exist_ok=True)
    data = load_dataset(args.dataset_zip)
    law_articles = flatten_law_db(data["laws"])

    if args.eval_train:
        t1 = evaluate_task1(data["train"], law_articles, topk=args.topk, retriever=args.retriever)
        with open("runs/task1_train_f2.json", "w", encoding="utf-8") as f: json.dump(t1, f, ensure_ascii=False, indent=2)
        print(f"[Task1] F2 on train: {t1['f2']:.4f}")

        preds_train = solve_task2(data["train"], law_articles, topk=5, retriever=args.retriever)
        acc = accuracy_on_train(data["train"], preds_train)
        with open("runs/task2_train_acc.json", "w", encoding="utf-8") as f: json.dump({"acc": acc, "detail": preds_train}, f, ensure_ascii=False, indent=2)
        print(f"[Task2] Accuracy on train: {acc:.4f}")

    if args.make_submission:
        sub1 = to_task1_submission(data["public_task1"], law_articles, topk=args.topk, retriever=args.retriever)
        sub2 = solve_task2(data["public_task2"], law_articles, topk=5, retriever=args.retriever)
        out_zip = "runs/submission.zip"
        write_submission(sub1, sub2, out_zip)
        print(f"Submission written to {out_zip}")

if __name__ == "__main__":
    main()
