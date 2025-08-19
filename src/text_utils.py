
import re, unicodedata

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_punct(s: str) -> str:
    s = re.sub(r"[^\w\sÀ-ỹ]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess(s: str) -> str:
    return strip_punct(normalize_text(s))
