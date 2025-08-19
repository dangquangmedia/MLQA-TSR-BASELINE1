from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rank_bm25 import BM25Okapi
from .text_utils import preprocess
from .dataio import LawArticle

@dataclass
class CorpusIndexTFIDF:
    vectorizer: TfidfVectorizer
    X: any
    meta: List[LawArticle]
    kind: str = "tfidf"

@dataclass
class CorpusIndexBM25:
    bm25: BM25Okapi
    tokens: List[List[str]]
    meta: List[LawArticle]
    kind: str = "bm25"

def build_law_corpus(articles: List[LawArticle], use_title: bool = True, ngram: Tuple[int,int]=(1,2), method: Literal["tfidf","bm25"]="tfidf"):
    docs = []
    for a in articles:
        txt = (a.title + " " if use_title else "") + (a.text or "")
        docs.append(preprocess(txt))

    if method == "bm25":
        tokens = [d.split() for d in docs]
        bm25 = BM25Okapi(tokens)
        return CorpusIndexBM25(bm25=bm25, tokens=tokens, meta=articles)
    else:
        # <- FIX: min_df=1 để giữ từ hiếm (tăng recall)
        vectorizer = TfidfVectorizer(ngram_range=ngram, min_df=1)
        X = vectorizer.fit_transform(docs)
        return CorpusIndexTFIDF(vectorizer=vectorizer, X=X, meta=articles)

def retrieve(corpus, query: str, topk: int = 5):
    q = preprocess(query)
    if getattr(corpus, "kind", "tfidf") == "bm25":
        scores = corpus.bm25.get_scores(q.split())
        idx = np.argsort(-np.asarray(scores))[:topk]
        return [(float(scores[i]), corpus.meta[i]) for i in idx]
    else:
        qv = corpus.vectorizer.transform([q])
        sims = cosine_similarity(qv, corpus.X).ravel()
        idx = np.argsort(-sims)[:topk]
        return [(float(sims[i]), corpus.meta[i]) for i in idx]
