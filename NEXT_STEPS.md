# Next Steps — Making Regio-AI Production Ready

This document outlines the concrete steps needed to take the demo pipeline toward a system that could be operated responsibly in a real administrative or enforcement context. Items are grouped by theme and ordered roughly by dependency.

---

## 1. Better construction detection — fine-tune Prithvi

The current pipeline uses NDBI (Normalized Difference Built-up Index) as a proxy for new construction. NDBI is a spectral formula applied pixel by pixel with no understanding of context. It produces false positives from bare rock, seasonal vegetation die-off, cleared land, and shadow edges.

A fine-tuned Prithvi checkpoint for built-up surface change would be significantly more robust.

**Steps:**

1. **Assemble a labelled dataset.** Source Sentinel-2 patch pairs (before/after) over Swedish municipalities. Rasterise Lantmäteriet building footprints and impervious surface maps to 10 m resolution to create pixel-level change labels. OpenStreetMap building footprints and DynamicWorld land cover can supplement coverage in areas where Lantmäteriet data is sparse.

2. **Fine-tune using TerraTorch.** Keep the pre-trained Prithvi backbone weights. Replace the Sen1Floods11 segmentation head with a binary change detection head. Train using `SemanticSegmentationTask`. The backbone already understands Sentinel-2 spectral and spatial structure — fine-tuning converges in 1–3 GPU-days on an A100.

3. **Evaluate on held-out Swedish coastal patches.** Report precision and recall specifically for small rural structures, which are the hardest case. Use these numbers to satisfy the EU AI Act accuracy documentation requirement (see section 5).

4. **Deploy as a second KServe endpoint.** The existing `predictor.py` and manifest pattern can be reused with the new checkpoint. Add a `run_prithvi_building_detection` tool to the agent loop alongside the existing water detection step.

**Important constraint:** Sentinel-2 has 10 m spatial resolution. A typical summer house (80–150 m²) is 1–2 pixels. Fine-tuning will not change this. The improvement is in *contextual understanding* — the model can learn that a new impervious surface cluster surrounded by forest and adjacent to an existing structure is more likely construction than a formula-based signal of similar magnitude. Individual small buildings remain below the reliable detection threshold at this resolution.

---

## 2. Higher resolution imagery

For reliable detection of individual small structures, 10 m Sentinel-2 resolution is insufficient. Commercial options:

| Source | Resolution | Notes |
|---|---|---|
| Planet SuperDove | 3 m | Daily revisit; subscription required |
| Airbus Pléiades | 0.5 m | On-demand tasking; expensive per km² |
| Maxar WorldView | 0.3–0.5 m | Highest resolution commercially available |
| Lantmäteriet orthophoto | 0.25 m | Sweden-specific; periodic capture (not continuous); free for government use |

Lantmäteriet's orthophotos are the most practical option for a Swedish government agency — they are free for public sector use, cover all of Sweden, and provide enough resolution to resolve individual houses clearly. The limitation is that they are captured on a multi-year cycle rather than continuously.

A hybrid approach — Sentinel-2 for frequent change screening, Lantmäteriet orthophoto for confirmation — matches the workflow a human inspector already follows.

---

## 3. Temporal consensus — reduce false positives

The current pipeline compares one before-scene to one after-scene. A single-pair comparison is sensitive to atmospheric artefacts, seasonal reflectance differences, and tidal variation.

Replace the single after-scene with a consensus across multiple independent scenes:

- Require a change signal to appear in at least 3 of 5 after-scenes drawn from different months and years
- This eliminates transient signals (snow melt, algae bloom, temporary shadow) while preserving genuine permanent changes
- The agent's `search_and_fetch_scenes` tool would need to be extended to fetch and process multiple scenes; the NDBI change mask would become a vote count rather than a binary flag

---

## 4. Full area scanning — beyond the hardcoded POI

The current pipeline analyses a single 224×224 px patch (2.24 km²) centred on a fixed coordinate. The full Orust coastline is approximately 200 km.

**Steps to enable full-area scanning:**

1. Use **Dask** to tile the full AOI into overlapping 224×224 px patches with a small overlap to avoid edge artefacts
2. Run Prithvi inference across all tiles — this is embarrassingly parallel
3. Merge tile results and dissolve boundaries
4. Allow the agent to accept arbitrary coordinates or administrative boundaries as input rather than a hardcoded POI

Full-island scanning at Sentinel-2 resolution would require on the order of 100–200 tiles. With GPU inference at ~5 seconds per tile, a full scan takes 10–15 minutes. On CPU it would be impractical at scale.

---

## 5. EU AI Act compliance

If this system is used by a public authority to support enforcement decisions it likely qualifies as high-risk AI under Annex III of the EU AI Act. The gaps and remediation steps are documented in detail in [`eu-ai-act/README.md`](eu-ai-act/README.md). The items most directly addressable in code are:

- **Audit logging (Art. 12):** Add structured JSON logging per analysis run — timestamp, input parameters, Prithvi output statistics, NDBI delta, violation pixel count — written to persistent storage.
- **In-application disclosure (Art. 13):** Add a non-dismissable banner in the Gradio UI stating that outputs are AI-generated and require human verification before any enforcement action.
- **Two-tier output:** Distinguish `HIGH_CONFIDENCE` signals (strong Prithvi water probability + large NDBI delta) from `REQUIRES_REVIEW` signals (borderline). Only surface high-confidence results as actionable.
- **Accuracy benchmarking (Art. 15):** Run the pipeline against a manually-verified reference set of known violation and non-violation sites. Publish precision and recall figures in the technical documentation.

---

## 6. Application hardening

The current application is a single-user demo. Before multi-user deployment:

- **Per-request session isolation.** State is currently held in a module-level dict. Concurrent requests will overwrite each other. Move session state into Gradio's `gr.State` object or a request-scoped context — the scaffolding is already in place in `app.py`, but `tools.py` uses a module-level `_SESSION`.
- **Authentication.** The Gradio UI is publicly accessible to anyone with the route URL. Add OpenShift OAuth proxy as a sidecar, or enable Gradio's built-in authentication.
- **Rate limiting.** Each analysis run consumes LLM tokens and triggers Prithvi inference. Add per-user rate limiting at the application layer.
- **Timeout and error surfacing.** The agent loop has a 12-iteration cap but no wall-clock timeout. A hung Prithvi request will block a user session indefinitely. Add a timeout on the KServe call and surface it clearly in the UI.

---

## 7. Model serving improvements

- **GPU nodes.** Prithvi currently runs on CPU, taking 1–3 minutes per patch. With a GPU node, inference drops to under 5 seconds. For full-area scanning this is the difference between minutes and hours.
- **Horizontal scaling.** The KServe InferenceService is configured as a single replica. For concurrent multi-user use, configure `minReplicas` and `maxReplicas` to allow autoscaling.
- **Model versioning.** The container image uses `:latest`. Pin image tags and model checkpoint versions so that deployments are reproducible and rollbacks are straightforward.

---

## 8. Building permit cross-reference

Swedish building permits (*bygglovsansökningar*) are public records under *offentlighetsprincipen*. A property flagged as a potential violation could automatically be cross-referenced against the municipal permit registry:

1. Convert POI coordinates to a *fastighetsbeteckning* (property ID) via Lantmäteriet's coordinate lookup API
2. Query the municipality's ByggR permit system for permits associated with that property
3. Return permit status — found / not found, date, type — as part of the agent's summary

This would let the system distinguish between a flagged site that has an approved permit and one that does not, reducing unnecessary follow-up for legitimate construction. Implementation depends on whether the relevant municipality exposes a machine-readable ByggR endpoint.
