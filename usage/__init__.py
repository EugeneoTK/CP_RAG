"""Ecosystem Insights (Track A usage layer).

Append-only, key-free, stdlib-only. One JSONL file per day under
``usage/`` (gitignored):

- ``event: "chat"``     — one per successful /api/chat turn
- ``event: "feedback"`` — one per /api/feedback (linked via query_id)

Insights are aggregated server-side by ``insights()`` — no LLM calls,
no embeddings, zero credits. The data is the raw material for the
"listening system" (Track A -> Track B loop in the IMDA framing).
"""

import json
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

USAGE_DIR = Path(__file__).resolve().parent.parent / "usage"

FEEDBACK_TAGS = ("missing", "wrong", "unclear", "other")

_ANSWER_START_RE = re.compile(r"^\s*(I\s+(don't|dont|can't|cannot)|I'm not sure|Sorry,? I)\b",
                              re.IGNORECASE)


def _daily_file(day=None):
    d = day or datetime.now().astimezone().date()
    return USAGE_DIR / ("queries-%s.jsonl" % d.isoformat())


def _append(event):
    USAGE_DIR.mkdir(exist_ok=True)
    with _daily_file().open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def log_chat(question, provider, answer, sources):
    """One chat turn. Never raises — usage logging must not break a chat.
    Returns the turn's query_id (for /api/feedback) or None."""
    try:
        query_id = uuid.uuid4().hex
        _append({
            "event": "chat",
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "query_id": query_id,
            "question": question,
            "provider": provider,
            "answer_chars": len(answer or ""),
            "starts_disclaimer": bool(_ANSWER_START_RE.match(answer or "")),
            "sources": list(sources),
        })
        return query_id
    except Exception:
        return None


def log_feedback(query_id, rating, tag=None):
    """Clinician rating on a chat turn. tag: one of FEEDBACK_TAGS (bad only)."""
    try:
        _append({
            "event": "feedback",
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "query_id": query_id,
            "rating": rating,
            "tag": tag,
        })
    except Exception:
        pass


def _iter_events(days):
    today = datetime.now().astimezone().date()
    for i in range(days - 1, -1, -1):
        p = _daily_file(today - timedelta(days=i))
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue  # a torn line never breaks aggregation
        except OSError:
            continue


def _question_key(q):
    """Group near-duplicate questions by their first 8 words."""
    words = re.sub(r"[^0-9a-zA-Z]+", " ", (q or "").lower()).split()
    return " ".join(words[:8]) if words else (q or "").strip()


def insights(days=30):
    """Aggregate the last ``days`` days (free — file reads only)."""
    chats, feedback = [], {}
    for ev in _iter_events(days):
        if ev.get("event") == "chat":
            chats.append(ev)
        elif ev.get("event") == "feedback" and ev.get("query_id"):
            feedback[ev["query_id"]] = ev  # last rating wins (UI enforces it)

    for c in chats:
        c["_rating"] = feedback.get(c.get("query_id"), {}).get("rating")

    ratings = Counter(c["_rating"] for c in chats if c["_rating"])
    bad_tagged = [c for c in chats if c["_rating"] == "bad"]
    gap = []
    for c in sorted(bad_tagged, key=lambda c: c.get("ts", "")):
        gap.append({
            "question": c["question"],
            "ts": c.get("ts", ""),
            "tag": feedback.get(c.get("query_id"), {}).get("tag"),
        })
    # "don't know" disclaimers are a corpus-gap proxy — surface the latest
    disclaimers = [c for c in chats if c.get("starts_disclaimer")]

    src_counter = Counter(s for c in chats for s in c.get("sources", []))
    top_questions = [
        {"question": k, "count": n}
        for k, n in Counter(_question_key(c["question"]) for c in chats).most_common(10)
        if n >= 1
    ]

    return {
        "window_days": days,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "totals": {
            "queries": len(chats),
            "good": ratings.get("good", 0),
            "bad": ratings.get("bad", 0),
            "rated": ratings.get("good", 0) + ratings.get("bad", 0),
            "disclaimers": len(disclaimers),
            "flagged": len(bad_tagged),
        },
        "by_provider": dict(Counter(c.get("provider", "unknown") for c in chats)),
        "top_questions": top_questions[:10],
        "gap_signals": {
            "flagged": gap[:20],
            "disclaimer_questions": [
                {"question": c["question"], "ts": c.get("ts", "")}
                for c in disclaimers[-10:]
            ],
        },
        "most_cited": [{"source": s, "count": n}
                       for s, n in src_counter.most_common(10)],
    }
