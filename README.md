# Regio-AI — Strandskydd Violation Detector

> **This is a demo.** It is designed to show what a production geospatial AI pipeline could look like on OpenShift AI, not to produce legally actionable results. See [Limitations](#limitations) for details.

A staged demonstration pipeline that uses IBM/NASA Prithvi-EO-2.0 and Sentinel-2 satellite imagery to detect potential *strandskydd* (Swedish shoreline protection law) violations — buildings constructed within 100 metres of water — on **Orust island, Bohuslän, Sweden**.

Built on Red Hat OpenShift AI. Models served via KServe. LLM via LiteLLM Model-as-a-Service.

---

## What is strandskydd?

Swedish law (MB 7 kap. 15 §) prohibits construction within 100 metres of any water body — sea, lakes, rivers, and streams. The protected zone is derived from the waterline and enforced by county authorities (*länsstyrelsen*).

---

## Repository layout

```
regio-ai/
├── notebooks/
│   ├── stage1_data_exploration.ipynb   # Data pipeline & interactive map
│   └── stage2_prithvi_inference.ipynb  # Prithvi inference & violation detection
└── stage3/
    ├── Makefile                        # Build, push, deploy, teardown
    ├── prithvi-server/                 # KServe model serving container
    │   ├── Containerfile
    │   ├── predictor.py                # FastAPI v2 inference endpoint
    │   └── requirements.txt
    ├── regio-agent/                    # Gradio + agent application container
    │   ├── Containerfile
    │   ├── app.py                      # Gradio UI + OpenAI tool-calling loop
    │   ├── tools.py                    # Four pipeline tool functions
    │   └── requirements.txt
    └── manifests/                      # OpenShift / KServe deployment manifests
        ├── 00-secret.yaml              # LiteLLM API key (fill before applying)
        ├── 01-pvc-models.yaml          # PVC for HuggingFace model weight cache
        ├── 02-serving-runtime.yaml     # Custom KServe ServingRuntime
        ├── 03-inference-service.yaml   # KServe InferenceService (Prithvi)
        ├── 04-agent-deployment.yaml    # regio-agent Deployment + Service
        └── 05-agent-route.yaml         # OpenShift Route (public HTTPS)
```

---

## Stages

### Stage 1 — Data exploration (`notebooks/stage1_data_exploration.ipynb`)

Explores the full Orust AOI using Sentinel-2 L2A imagery from Microsoft Planetary Computer.

- Searches for best cloud-free scenes before (2017–2018) and after (2022–2023) 2019
- Applies SCL cloud/shadow masking
- Computes NDWI (water index) and derives the strandskydd 100 m buffer
- Computes NDBI change detection to flag new built-up surfaces
- Renders an interactive Folium map with the official Naturvårdsverket WMS overlay

### Stage 2 — Prithvi inference (`notebooks/stage2_prithvi_inference.ipynb`)

Runs IBM/NASA Prithvi-EO-2.0-300M on a 224×224 px patch centred on the Point of Interest.

- Loads `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11` via TerraTorch
- Normalises Sentinel-2 DN values using the Sen1Floods11 training statistics
- Runs semantic segmentation to produce a water mask
- Derives the strandskydd buffer from the Prithvi water mask
- Intersects NDBI change with the buffer to flag potential violations
- Runs fully on CPU (1–3 min per 224×224 patch)

### Stage 3 — Production deployment (`stage3/`)

Deploys the pipeline as two containerised services on OpenShift AI.

**`prithvi-server`** — KServe InferenceService

A FastAPI application that speaks the KServe v2 REST inference protocol. It downloads Prithvi-EO-2.0-300M-TL-Sen1Floods11 from HuggingFace Hub at startup (cached to a PVC) and exposes:

```
POST /v2/models/prithvi-water/infer
  Input : {"inputs": [{"name":"patch","shape":[1,6,224,224],"datatype":"FP32","data":[...]}]}
  Output: {"outputs": [{"name":"preds",...}, {"name":"probs",...}]}
GET  /v2/health/ready
```

**`regio-agent`** — Gradio chat application

A natural-language interface backed by a tool-calling agent loop using the OpenAI SDK against the LiteLLM MaaS endpoint (Qwen3-14B). The agent orchestrates four tools in sequence:

| Tool | What it does |
|---|---|
| `search_and_fetch_scenes` | Searches Planetary Computer, downloads Sentinel-2 scenes, extracts POI patch |
| `run_prithvi_water_detection` | Calls the KServe endpoint, derives strandskydd buffer |
| `compute_ndbi_change` | NDBI change detection, intersects with strandskydd zone |
| `generate_violation_map` | Builds interactive Folium map, returns HTML |

---

## Deploying Stage 3

### Prerequisites

- OpenShift AI cluster with KServe installed
- `oc` CLI logged in to the cluster
- `podman` logged in to quay.io
- LiteLLM MaaS endpoint + API key

### 1. Fill in the secret

```bash
# Edit 00-secret.yaml and replace the placeholder with the real key
vi stage3/manifests/00-secret.yaml
oc apply -f stage3/manifests/00-secret.yaml -n <your-namespace>
```

### 2. Build and push images

```bash
make build   # builds prithvi-server and regio-agent
make push    # pushes both to quay.io/jwesterl/
```

### 3. Deploy

```bash
make deploy  # applies manifests 01–05
make status  # watch pod and route status
```

### 4. Tear down

```bash
make teardown
```

### Useful commands

```bash
make logs-prithvi   # follow Prithvi model server logs
make logs-agent     # follow Gradio agent logs
```

---

## Key technical choices

| Choice | Reason |
|---|---|
| **Sentinel-2 L2A via Planetary Computer** | Free, no credentials, global coverage, COG format enables lazy partial reads |
| **EPSG:32633 (UTM Zone 33N)** | Metric CRS covering Orust; required for accurate 100 m buffer in metres |
| **SCL cloud masking** | Prevents cloud shadows from being misclassified as new construction |
| **Prithvi Sen1Floods11 fine-tune** | Only publicly available Prithvi checkpoint for water detection; no building footprint checkpoint exists |
| **NDBI for change detection** | No building-specific model available; NDBI correlates with roofing materials and paved surfaces |
| **224×224 px patch (2.24 km²)** | Matches Prithvi's training patch size; keeps CPU inference under 3 minutes |
| **KServe RawDeployment mode** | Simpler than Serverless/Knative for a single-replica demo; no Istio dependency |
| **OpenAI tool-calling loop (no framework)** | Avoids langchain version fragility; openai SDK is stable and directly supported by LiteLLM |

---

## Limitations

This is a **proof-of-concept demo** targeting a hardcoded location. The following limitations apply:

### Hardcoded Point of Interest

The POI is fixed at **58.26599°N, 11.77902°E** — a specific summer house on Orust where an additional structure was built after 2019. The agent accepts natural-language date ranges but cannot analyse arbitrary coordinates without code changes.

### Small analysis area

Each run analyses a 224×224 px tile (2.24 km²) around the POI. The full Orust coastline is approximately 200 km. Full-island scanning would require Dask-based tiling and significantly more compute time.

### No building detection model

IBM/NASA has not published a Prithvi checkpoint trained on building footprints. The pipeline uses the Sen1Floods11 water detection model to derive the strandskydd zone, and NDBI spectral change as a proxy for new construction. NDBI can produce false positives from bare rock, seasonal vegetation changes, cleared land, and shadow edges.

### Single scene comparison

The analysis compares one before-scene to one after-scene. Atmospheric artefacts, seasonal reflectance differences (snow, deciduous foliage), and tidal variation can mimic a construction signal. A temporal consensus approach (requiring change to appear in multiple independent after-scenes) would reduce false positives.

### Water mask accuracy

Prithvi was fine-tuned on flood detection data. Coastal Swedish conditions — archipelago shorelines, sea rocks, shallow tidal bays — may not be well represented in the training distribution. Shoreline position can vary by 1–2 pixels (10–20 m) depending on tidal state and scene acquisition time.

### CPU-only inference

The cluster's GPU quota was unavailable at deployment time. Prithvi runs on CPU, taking 1–3 minutes per patch. Production use would require GPU nodes for acceptable throughput.

### No multi-tenancy

The tool module uses a module-level session dict. Concurrent requests from different users will overwrite each other's state. Not suitable for multi-user production use without per-request isolation.

### No authentication

The Gradio UI is publicly accessible to anyone with the route URL. The LiteLLM API key is consumed on every request with no rate limiting or user attribution at the application layer.

### Not legal evidence

A positive violation signal means *"this location is worth a human reviewing"*, not *"a violation occurred"*. Enforcement under strandskydd requires ground survey, official cadastral data, and a decision by a *handläggare* (case officer) at the county authority.

---

## Model information

| Model | Source | Task |
|---|---|---|
| `Prithvi-EO-2.0-300M-TL-Sen1Floods11` | [HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) | Water / flood segmentation |
| `Qwen3-14B` | LiteLLM MaaS | Agent reasoning and natural language summary |

Sentinel-2 data © ESA, accessed via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com).  
Official strandskydd boundaries via [Naturvårdsverket WMS](https://nvpub.vic-metria.nu/arcgis/services/Strandskydd/MapServer/WmsServer).
