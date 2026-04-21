# Notebooks

Two notebooks form the experimental foundation of the pipeline, each building on the previous.

---

## Stage 1 — Data Exploration (`stage1_data_exploration.ipynb`)

Establishes the full analysis pipeline using **classical remote sensing** methods.

Water is detected using **NDWI** (Normalized Difference Water Index):

```
NDWI = (Green − NIR) / (Green + NIR)
```

This is a formula, not a model. It works because water absorbs near-infrared light and reflects green, so the ratio is predictably high over water — a principle that has been used in satellite analysis since the 1990s. It is fast, transparent, and requires no training data, but it operates pixel by pixel with no understanding of spatial context. A dark shadow, wet roof, or shallow tidal flat can produce false readings.

From the NDWI water mask the notebook derives the strandskydd 100 m buffer, then uses **NDBI change detection** to find new built-up surfaces, and intersects the two to flag potential violations.

---

## Stage 2 — Prithvi Inference (`stage2_prithvi_inference.ipynb`)

Replaces the NDWI water detection step with the **IBM/NASA Prithvi-EO-2.0-300M** foundation model.

Prithvi is a 300M parameter Vision Transformer trained on large volumes of satellite imagery. It performs semantic segmentation — it looks at the entire 224×224 px patch and classifies every pixel, but with an understanding of spatial context learned from many training examples. It has seen what complex coastal shorelines, archipelago inlets, sea rocks, and shallow bays look like across many images, not just what individual pixel ratios look like.

The strandskydd buffer is then derived from the Prithvi water mask instead of NDWI. NDBI change detection remains the same as Stage 1.

---

## What the two stages have in common

Both notebooks:
- Source Sentinel-2 L2A imagery from Microsoft Planetary Computer
- Apply SCL cloud and shadow masking
- Derive the strandskydd 100 m buffer from the water mask
- Use NDBI spectral change as a proxy for new construction
- Intersect the buffer with the NDBI change mask to flag violations

The construction detection step uses classical NDBI in both stages. No AI model for building footprint detection exists in the publicly available Prithvi checkpoint family. This is noted as a known limitation.

---

## How to explain the difference

> *Stage 1 uses a formula — it exploits a known physical property of light: water absorbs infrared. Plug in two spectral bands, apply the equation. It works, but it is blind to context.*
>
> *Stage 2 replaces that formula with a foundation model trained on millions of satellite images. It does not just compare two numbers per pixel — it understands what it is looking at. A complex archipelago shoreline with rocks, shallow bays, and reed beds is much harder for the formula than for the model.*

The honest nuance: Stage 2 uses AI where it helps most — drawing an accurate shoreline boundary, which defines the legal protection zone. The construction signal still comes from a classical spectral index, because the field has not yet produced a public model for that task. A demo that claims AI does everything is less credible than one that is precise about where AI adds value and where it does not.
