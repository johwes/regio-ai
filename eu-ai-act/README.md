# EU AI Act — Short Review for Regio-AI

> **Disclaimer:** This is not legal advice. It is a brief, informal review prepared without legal counsel or a formal conformity assessment. It is intended to help technical teams understand which questions the EU AI Act raises for a system like this, not to substitute for a proper compliance process. Consult qualified legal and regulatory experts before drawing compliance conclusions.

---

## Context

Regio-AI is a demo pipeline that uses satellite imagery and AI models to flag potential *strandskydd* (Swedish shoreline protection law) violations. If a system of this type were adopted by a county authority (*länsstyrelsen*) in an administrative oversight or enforcement workflow, the EU AI Act would likely apply.

---

## Risk classification

The EU AI Act uses a risk-based framework. The classification depends on how the system output is used.

**If a public authority uses this system to triage which properties to inspect**, it likely falls under **Annex III, Article 6(2) — high-risk AI**:

> *"AI systems intended to be used by or on behalf of competent public authorities… to evaluate the eligibility of natural persons for public benefits and services, to grant, reduce, revoke or reclaim such benefits."*

Administrative enforcement under strandskydd — fines, demolition orders, injunctions — is an exercise of public authority that affects individuals' property rights. A triage system feeding into that process is plausibly in scope.

**If the system is only used for internal expert prioritisation** (inspectors deciding which areas to visit, with no output reaching a citizen without independent human review), there is an argument for limited-risk or minimal-risk classification. That argument requires airtight process documentation demonstrating that no automated output constitutes or substantially drives an administrative decision.

---

## What high-risk classification requires

The table below maps the Act's high-risk obligations (Chapter 3) to the current state of the demo.

| Requirement | Article | Current state | Gap |
|---|---|---|---|
| **Risk management system** | Art. 9 | None | Full gap — no documented risk identification, evaluation, or mitigation process |
| **Data governance** | Art. 10 | Data sources documented (ESA Sentinel-2, Naturvårdsverket WMS) | Largely covered; training data provenance for Prithvi should be referenced |
| **Technical documentation** | Art. 11 | README + notebook comments | Needs formal documentation covering model architecture, training data, known failure modes, and accuracy figures |
| **Logging and auditability** | Art. 12 | No persistent logging | Full gap — no per-analysis audit trail is stored |
| **Transparency to users** | Art. 13 | "Not legal evidence" disclaimer in README | Needs in-application disclosure visible to every operator session |
| **Human oversight** | Art. 14 | Mentioned as limitation | Needs a formal workflow gate: the system must make explicit that no output should trigger enforcement without human verification |
| **Accuracy and robustness** | Art. 15 | Limitations section describes qualitative weaknesses | Needs quantitative accuracy benchmarks against a reference dataset |

---

## Specific gaps for this pipeline

### 1. No persistent audit log

Article 12 requires that high-risk AI systems automatically log events sufficient to enable post-hoc review. The current pipeline logs to stdout only and retains nothing between sessions. A minimal fix is structured JSON logging per analysis run, stored durably (e.g., object storage), covering: timestamp, input date ranges, Prithvi output statistics, NDBI change statistics, and violation pixel count.

### 2. Borderline signal handling

The NDBI change threshold (0.05) is hardcoded with no uncertainty communication. A system feeding administrative decisions should distinguish between high-confidence signals and borderline cases, and route borderline cases to mandatory human review rather than returning an undifferentiated violation flag.

### 3. No in-application disclosure

The README disclaimer is not visible to an operator running the Gradio UI. Article 13 requires that operators are clearly informed they are interacting with an AI system and of its limitations. A persistent disclosure panel in the UI would address this.

### 4. No quantitative accuracy statement

The limitations section correctly describes qualitative failure modes (tidal variation, seasonal reflectance, NDBI false positives from bare rock). Article 15 asks for accuracy metrics. Even a small manually-verified test set — a handful of known violation and non-violation sites — would produce a precision/recall figure that is far better than no figure.

---

## What the system gets right

- **Human-in-the-loop by design**: the README explicitly states outputs are not legal evidence and require a *handläggare* (case officer) review. This reflects the spirit of Article 14.
- **Data provenance**: Sentinel-2 (ESA, open licence), Naturvårdsverket WMS (official authority data), and Microsoft Planetary Computer are all documented and traceable.
- **Transparent limitations**: the README enumerates known failure modes — hardcoded POI, single scene comparison, water mask accuracy, NDBI false positives — which is consistent with the technical documentation spirit of Article 11.
- **No autonomous enforcement**: the pipeline produces a map and a natural-language summary; it does not write to any database, send notifications, or trigger any downstream action. This limits the blast radius of errors.

---

## Practical next steps

These steps are addressable without changing the core pipeline:

1. **Add structured logging** to `tools.py` — emit a JSON record per analysis run to a persistent location.
2. **Add a UI disclosure panel** in `app.py` — a visible, non-dismissable banner stating the system is AI-assisted and that outputs require human verification before any enforcement action.
3. **Document the threshold choice** — record why 0.05 was chosen for NDBI and under what conditions it should be reviewed.
4. **Implement two-tier output** — separate `HIGH_CONFIDENCE` from `REQUIRES_REVIEW` signals, based on combined Prithvi water probability and NDBI delta magnitude.
5. **Commission a small reference evaluation** — validate the pipeline against a set of known sites to produce at least a indicative precision/recall figure.

---

## Summary

| Question | Short answer |
|---|---|
| Does the EU AI Act apply? | Likely yes, as high-risk, if used by a public authority to support enforcement decisions |
| Is the demo compliant today? | No — logging, quantitative accuracy, and formal risk management are missing |
| Are the gaps large? | The technical gaps are addressable; the process gaps (risk management system, conformity assessment) require organisational effort beyond code changes |
| Is the core design sound? | Yes — human-in-the-loop framing, transparent limitations, and no autonomous action are the right foundations |
