
# VLSP 2025 — MLQA‑TSR Baseline (Python, text-only)

See usage instructions in this README. Place the **official dataset zip** inside `data/` and run:

```bash
python -m venv .venv
# activate venv...
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt', quiet=True)"
python -m src.main --dataset_zip "data/<your_dataset_zip>.zip" --eval_train
python -m src.main --dataset_zip "data/<your_dataset_zip>.zip" --make_submission --topk 3
```
