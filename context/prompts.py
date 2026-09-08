"""Render the live snapshot as labelled RAG prompt sections (Phase 2, F3).

Provenance rule (docs/signal-research.md §12, corpus audit 2026-09-08):
  - basis notes ("Protocol content" section): ONLY links whose basis is
    corpus/partial, each carrying its basis_note (what the protocol text
    does and does not support).
  - local context section: ALL active links as live signals, labelled as
    environmental observations — never attributable to the protocols.
    `derived`/`none` links appear ONLY here, never as protocol content.
"""


def format_live_context(snapshot):
    """snapshot: build_snapshot() dict.

    Returns (basis_notes, local_context, clinic, as_of); basis_notes is None
    when no corpus/partial link is active.
    """
    meta = snapshot.get("meta") or {}
    clinic = (meta.get("clinic") or {}).get("name") or "n/a"
    as_of = meta.get("generated_at") or "n/a"
    links = (snapshot.get("protocol_links") or {}).get("active") or []

    basis_lines = []
    signal_lines = []
    counselling = []
    for l in links:
        signal = l.get("signal", "?")
        detail = l.get("detail", "")
        if l.get("basis") in ("corpus", "partial"):
            basis_lines.append(
                "- %s [%s]: %s" % (l.get("basis_note", ""), signal, detail))
        line = "- %s [%s]" % (detail, signal)
        if l.get("protocols"):
            line += " — related protocols: " + ", ".join(l["protocols"])
        if l.get("clinical_focus"):
            line += " — clinical framing: " + l["clinical_focus"]
        signal_lines.append(line)
        if l.get("population_counselling"):
            counselling.append("- " + l["population_counselling"])

    if not signal_lines:
        signal_lines.append("- no elevated signals: all context signals within normal range")
    local = (
        "Live population-level signals (NEA / data.gov.sg) — observations "
        "about the area right now, NOT protocol content; never present them "
        "as if the protocols said so. Use them to frame the answer "
        "(environmental triggers, sick-day rules, counselling).\n"
        + "\n".join(signal_lines)
    )
    if counselling:
        local += (
            "\nVisit-level counselling (applies to all patients, independent "
            "of the question):\n" + "\n".join(counselling)
        )
    basis = "\n".join(basis_lines) if basis_lines else None
    return basis, local, clinic, as_of