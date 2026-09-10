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
    local_lines = signal_lines
    # WIDB (Phase 3): national weekly infectious disease counts (CDA bulletin) —
    # baseline context, not an "elevated signal", so it never suppresses the
    # normal-range line above; it is still observation, never protocol content.
    dw = snapshot.get("disease_week") or {}
    if dw.get("epi_week"):
        line = "- WIDB %s (%s), CDA weekly bulletin — national counts: " % (
            dw["epi_week"], dw.get("date_range") or "?")
        line += "; ".join(dw.get("notable") or ["table parsed, no narrative fields"])
        local_lines = signal_lines + [line]

    # URA (Phase 4/8): island-wide planning-decision signal — baseline context,
    # same treatment as WIDB; a written permission is explicitly NOT an
    # opened facility, so the line says so. Phase 8 adds the category split
    # and the (rough) near-clinic count.
    cc = snapshot.get("catchment_change") or {}
    if cc.get("healthcare_decisions_90d_count"):
        examples = "; ".join(
            "%s %s — %s%s" % (
                a.get("date") or "?", a.get("address") or "?",
                (a.get("what") or "?")[:80],
                (" [" + a["decision_type"] + "]") if a.get("decision_type") else "")
            for a in (cc.get("healthcare_decisions_90d") or [])[:3])
        k = cc.get("category_counts") or {}
        bits = ", ".join("%d %s" % (k[c], c.lower())
                         for c in ("Senior care", "Nursing home", "Child care",
                                   "Polyclinic", "Clinic", "Medical", "Other")
                         if k.get(c))
        extra = (" (%s)" % bits) if bits else ""
        if cc.get("near_clinic_count"):
            extra += ", %d within ~2 km of the clinic" % cc["near_clinic_count"]
        local_lines = local_lines + [
            "- URA planning decisions %s (island-wide written permissions, "
            "NOT protocol content): %d healthcare-related%s, e.g. %s. A written "
            "permission is NOT an opened facility — never present one as an "
            "existing service." % (
                cc.get("window") or "last 90 days",
                cc["healthcare_decisions_90d_count"], extra, examples)]

    # NTUC Active Ageing (Phase 9): community (NON-clinical) referral signal —
    # baseline context like WIDB/URA; the monthly programme calendars are
    # answered from the Community resources corpus section.
    aa = snapshot.get("active_ageing") or {}
    if aa.get("count"):
        near = [c for c in (aa.get("centres") or []) if c.get("near")]
        names = ", ".join("%s (%.1f km)" % (c["name"], c["km"]) for c in near)
        local_lines = local_lines + [
            "- NTUC Health Active Ageing Centres (community exercise/social/"
            "digital-skills programmes for older adults — NON-clinical): %d "
            "island-wide (calendar month: %s); within ~2 km of the clinic: %s. "
            "Their monthly programme calendars are in the Community resources "
            "corpus section — use that to answer questions about nearby "
            "activities; never present them as clinical services." % (
                aa.get("count", 0),
                ", ".join(aa.get("calendar_months") or []) or "?",
                names or "none")]
    local = (
        "Live population-level signals (NEA / data.gov.sg / CDA) — observations "
        "about the area right now, NOT protocol content; never present them "
        "as if the protocols said so. Use them to frame the answer "
        "(environmental triggers, sick-day rules, counselling).\n"
        + "\n".join(local_lines)
    )
    if counselling:
        local += (
            "\nVisit-level counselling (applies to all patients, independent "
            "of the question):\n" + "\n".join(counselling)
        )
    basis = "\n".join(basis_lines) if basis_lines else None
    return basis, local, clinic, as_of