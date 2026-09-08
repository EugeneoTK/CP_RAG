"""CP_RAG context layer — population-level "flight information" for GP care-protocol answers.

Phase 0 (spike): fetches verified key-free Singapore open-data signals around a clinic
point and emits a JSON snapshot, including which corpus care protocols each active
signal is relevant to (see linkage.py).

Stdlib only (urllib/json/math). Python 3.9 compatible. No API keys required;
DGS_API_KEY in .env is used if present (optional, higher rate limits only).

Run:
    venv/bin/python -m context --lat 1.430893 --lon 103.775213 --name "Woodlands Polycline (test)"
    venv/bin/python -m context --postcode 738579
    venv/bin/python -m context ... --json
"""

__version__ = "0.1.0"
