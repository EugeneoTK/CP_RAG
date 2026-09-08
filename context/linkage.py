"""Signal -> care-protocol linkage map (the core of the POC).

Each *active* live signal is mapped to the chronic-care protocols in the RAG
corpus (primarycarepages.sg, 17 protocols) with the clinical reasoning a GP
would use. Phase 0 emits structured links + a prose "context brief" that
Phase 2 will prepend to the RAG system prompt.

Mechanisms (how context changes protocol usage):
  A. Interpretation framing — a symptom question is answered with the active
     environmental/infectious context in mind (e.g. breathlessness + high PM2.5
     -> environmental trigger; review inhaler adherence / action plan).
  B. Retrieval steering     — active links add keywords to the query so the
     retriever surfaces the right protocol chunks (no re-embedding).
  C. Population counselling — visit-level nudges (e.g. do-the-mozzie-wipeout
     while a dengue cluster is active nearby).

Provenance (corpus audit 2026-09-08; direct grep of all 3,785 indexed chunks,
see docs/signal-research.md §12): every rule carries a `basis` tag so nothing
is ever presented as protocol content that the protocol text does not contain:
  corpus   — claim is in the protocol text (may be cited as protocol content)
  partial  — the mechanism is in the protocol text, the specific trigger is not
  derived  — clinical synthesis of the live signal + protocol; NOT protocol text
  none     — no clinical claim (population counselling only)
Phase 2 must inject `derived` links only as labelled local context, never as
protocol content.

Corpus protocols (verified list, /Healthier-SG/Care-Protocols/Chronic):
Allergic Rhinitis, Asthma, Benign Prostatic Hyperplasia, Chronic Hepatitis B,
Chronic Kidney Disease, Chronic Obstructive Pulmonary Disease, Diabetes
Mellitus, Generalised Anxiety Disorder, Gout, Hypertension, Ischaemic Heart
Disease, Lipid Disorders, Major Depressive Disorder, Multimorbidity (Diabetes,
Hypertension and Hyperlipidaemia), Osteoarthritis, Pre-Diabetes Mellitus,
Stable Stroke.
"""

PROTOCOLS = [
    "Allergic Rhinitis", "Asthma", "Benign Prostatic Hyperplasia",
    "Chronic Hepatitis B", "Chronic Kidney Disease",
    "Chronic Obstructive Pulmonary Disease", "Diabetes Mellitus",
    "Generalised Anxiety Disorder", "Gout", "Hypertension",
    "Ischaemic Heart Disease", "Lipid Disorders", "Major Depressive Disorder",
    "Multimorbidity (Diabetes, Hypertension and Hyperlipidaemia)",
    "Osteoarthritis", "Pre-Diabetes Mellitus", "Stable Stroke",
]

_RULES = [
    {
        "signal": "pm25",
        "protocols": ["Asthma", "Chronic Obstructive Pulmonary Disease",
                      "Ischaemic Heart Disease", "Hypertension"],
        "basis": "derived",
        "basis_note": (
            "Air pollution is not mentioned in any of the 17 protocols; the "
            "PM2.5 -> respiratory/cardiovascular exacerbation association is "
            "established clinical knowledge, not protocol content."
        ),
        "clinical_focus": (
            "Elevated PM2.5 is a trigger for airway inflammation and cardiovascular "
            "stress. Expect more exacerbations this week; for symptomatic patients "
            "review inhaler technique/adherence, reinforce the asthma/COPD action "
            "plan, and consider the environment (windows, outdoor exercise timing) "
            "when interpreting worsening symptoms."
        ),
        "population_counselling": (
            "Advise patients with respiratory/cardiovascular disease to avoid "
            "prolonged outdoor exertion on high-pollution days."
        ),
    },
    {
        "signal": "psi",
        "protocols": ["Asthma", "Chronic Obstructive Pulmonary Disease",
                      "Ischaemic Heart Disease"],
        "basis": "derived",
        "basis_note": (
            "PSI bands are not mentioned in the protocol corpus (same basis as "
            "PM2.5: clinical synthesis, not protocol content)."
        ),
        "clinical_focus": (
            "PSI in the unhealthy band: same trigger framing as PM2.5; document "
            "environmental context when a respiratory presentation is ambiguous."
        ),
        "population_counselling": None,
    },
    {
        "signal": "heat",
        "protocols": ["Diabetes Mellitus", "Chronic Kidney Disease",
                      "Hypertension", "Gout",
                      "Multimorbidity (Diabetes, Hypertension and Hyperlipidaemia)"],
        "basis": "partial",
        "basis_note": (
            "In protocol text: dehydration as a gout flare trigger; CKD "
            "hydration advice (euglycaemic DKA risk); SGLT2 sick-day rules "
            "(CKD + multimorbidity protocols). NOT in protocol text: 'heat'/ "
            "'WBGT' as a trigger, and ACEi/ARB review in dehydration."
        ),
        "clinical_focus": (
            "Elevated WBGT/heat stress: dehydration risk. In diabetes, increased "
            "hypoglycaemia risk and need for more fluid intake; in CKD, risk of "
            "prerenal AKI — review ACEi/ARB and diuretic use if dehydrated, check "
            "electrolytes/creatinine in symptomatic patients; heat also "
            "precipitates gout flares via urate concentration."
        ),
        "population_counselling": (
            "Counsel on hydration, midday heat avoidance, and recognising "
            "dehydration in the elderly — especially on diuretics."
        ),
    },
    {
        "signal": "rain",
        "protocols": ["Hypertension", "Diabetes Mellitus", "Chronic Kidney Disease"],
        "basis": "derived",
        "basis_note": (
            "Access/logistics argument (medication supply, delayed reviews); "
            "not mentioned in protocol text."
        ),
        "clinical_focus": (
            "Heavy rain/flood conditions disrupt travel: expect missed medication "
            "collections and delayed reviews; verify chronic-tier supply before "
            "discharging chronic patients, and note access barriers in "
            "multimorbid or elderly patients."
        ),
        "population_counselling": (
            "Rainy periods seed dengue vector breeding (1-2 week lag); pair with "
            "dengue-prevention messaging."
        ),
    },
    {
        "signal": "dengue_cluster",
        "protocols": ["Diabetes Mellitus", "Chronic Kidney Disease",
                      "Hypertension", "Ischaemic Heart Disease", "Stable Stroke"],
        "basis": "partial",
        "basis_note": (
            "In protocol text: SGLT2 sick-day rules (stop while acutely unwell "
            "— CKD + multimorbidity protocols). NOT in protocol text: dengue "
            "itself, the NS1 differential, and NSAID avoidance in "
            "antiplatelet/anticoagulated patients (standard practice)."
        ),
        "clinical_focus": (
            "Active dengue cluster nearby: for any FEBRILE patient, consider "
            "dengue in the differential (NS1/CRP). Sick-day rules apply — stop "
            "SGLT2 inhibitors while acutely unwell (CKD + multimorbidity "
            "protocols); avoid NSAIDs (bleeding/thrombocytopaenia risk) in "
            "patients on antiplatelets/anticoagulants (IHD, stroke) — use "
            "paracetamol instead."
        ),
        "population_counselling": (
            "Do the Mozzie Wipeout: eliminate standing water at home; report new "
            "clusters to NEA. Applies to every patient in the catchment."
        ),
    },
    {
        "signal": "aedes_area",
        "protocols": [],
        "basis": "none",
        "basis_note": (
            "Population counselling only; no clinical claim, no protocols "
            "linked."
        ),
        "clinical_focus": None,
        "population_counselling": (
            "Clinic sits inside a NEA high-Aedes-population area: attach dengue "
            "prevention messaging to every consultation this fortnight."
        ),
    },
    {
        "signal": "humidity",
        "protocols": ["Allergic Rhinitis", "Asthma"],
        "basis": "partial",
        "basis_note": (
            "In protocol text: allergic rhinitis names house dust mite and "
            "indoor moulds as common allergens. NOT in protocol text: humidity "
            "as an explicit trigger; the asthma protocol covers trigger "
            "avoidance generally without naming mites/moulds."
        ),
        "clinical_focus": (
            "Sustained very high humidity favours dust-mite/fungal load indoors — "
            "relevant when rhinitis or asthma control seems to drift without an "
            "obvious other trigger."
        ),
        "population_counselling": None,
    },
]


def build_protocol_links(air, weather, dengue):
    """air: {pm25_peak, psi_peak}; weather/dengue: snapshot blocks.
    Returns (links, brief_lines)."""
    links = []
    brief = []

    def add(rule_key, detail):
        for r in _RULES:
            if r["signal"] == rule_key:
                links.append({
                    "signal": rule_key,
                    "detail": detail,
                    "protocols": r["protocols"],
                    "basis": r["basis"],
                    "basis_note": r["basis_note"],
                    "clinical_focus": r["clinical_focus"],
                    "population_counselling": r["population_counselling"],
                })
                parts = []
                if r["protocols"]:
                    parts.append("protocols affected: " + ", ".join(r["protocols"]))
                if r["clinical_focus"]:
                    parts.append(r["clinical_focus"])
                brief.append("- " + detail + (" (" + "; ".join(parts) + ")"))
                break

    pm25_peak = air.get("pm25_peak")
    if pm25_peak is not None and pm25_peak >= 25:
        add("pm25", "peak regional PM2.5 %.0f ug/m3 (unhealthy for sensitive groups)"
            % pm25_peak)
    elif pm25_peak is not None and pm25_peak >= 15:
        add("pm25", "peak regional PM2.5 %.0f ug/m3 (moderate)" % pm25_peak)

    psi_peak = air.get("psi_peak")
    if psi_peak is not None and psi_peak >= 101:
        band = "very unhealthy" if psi_peak >= 201 else "unhealthy"
        add("psi", "peak regional PSI %.0f (%s)" % (psi_peak, band))

    wbgt = (weather or {}).get("wbgt") or {}
    max_w = wbgt.get("max_wbgt_c")
    stress = wbgt.get("max_heat_stress")
    if (max_w is not None and max_w >= 31) or stress in ("Moderate", "High"):
        add("heat", "WBGT %.1f C max, heat stress %s" % (max_w or 0.0, stress or "n/a"))

    days = ((weather or {}).get("outlook_4day") or {}).get("days") or []
    rain_days = [d.get("day") for d in days
                 if d.get("text") and any(w in d["text"].lower()
                                          for w in ("thunder", "shower"))]
    if rain_days:
        add("rain", "rain/thunder expected: " + ", ".join(x or "?" for x in rain_days[:3]))

    flood = (weather or {}).get("flood") or {}
    if flood.get("active_alerts"):
        add("rain", "%d active flood alert(s)" % len(flood["active_alerts"]))

    nearby = (dengue or {}).get("nearby_clusters") or []
    if nearby:
        c = nearby[0]
        add("dengue_cluster",
            "active dengue cluster %s (%d cases) %.1f km from clinic, updated %s"
            % (c.get("locality"), c.get("case_size") or 0, c.get("km") or 0.0,
               c.get("updated") or "?"))

    if (dengue or {}).get("in_high_aedes_area"):
        add("aedes_area", "clinic inside a NEA high-Aedes-population area")

    humidity_high = ((weather or {}).get("today") or {}).get("humidity_pct") or {}
    hum = humidity_high.get("high")
    if hum is not None and hum >= 95:
        add("humidity", "humidity up to %d pct today" % hum)

    if not links:
        brief.append("- no elevated signals: all context signals within normal range")
    return links, brief