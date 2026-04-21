# Stage 3 — Production Deployment

This folder contains everything needed to run Regio-AI as two containerised services on Red Hat OpenShift AI. The experimental notebooks from Stage 1 and Stage 2 become an automated pipeline served to end users through a natural-language chat interface.

---

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│  OpenShift AI cluster                                           │
│                                                                 │
│  ┌─────────────────────┐        ┌──────────────────────────┐   │
│  │   regio-agent       │        │   prithvi-water          │   │
│  │                     │        │   (KServe InferenceService│   │
│  │  Gradio UI          │─HTTP──▶│   RawDeployment)         │   │
│  │  Agent loop         │ :8080  │                          │   │
│  │  Tool functions     │        │  Prithvi-EO-2.0-300M     │   │
│  │                     │        │  TerraTorch / FastAPI    │   │
│  └──────────┬──────────┘        └──────────────────────────┘   │
│             │                                                   │
│             │ HTTPS                                             │
└─────────────┼───────────────────────────────────────────────────┘
              │
    ┌─────────▼──────────┐        ┌──────────────────────────┐
    │  End user          │        │  LiteLLM MaaS            │
    │  (browser)         │        │  Qwen3-14B               │
    └────────────────────┘        │  (external service)      │
                                  └──────────────────────────┘
                    regio-agent ──HTTPS──▶ LiteLLM MaaS
```

Two services run inside the cluster. All communication between them is in-cluster HTTP. Only two connections cross the cluster boundary: the user's browser (inbound, via an OpenShift Route) and the LiteLLM MaaS endpoint (outbound, for LLM inference).

---

## Services

### `prithvi-server` — KServe InferenceService

A FastAPI application that speaks the **KServe v2 REST inference protocol**. It downloads Prithvi-EO-2.0-300M-TL-Sen1Floods11 from HuggingFace Hub at first startup (cached to a PVC) and exposes:

```
POST /v2/models/prithvi-water/infer   — run inference
GET  /v2/health/ready                 — readiness probe
GET  /v2/health/live                  — liveness probe
```

The model runs on CPU (1–3 minutes per 224×224 patch). GPU would reduce this to under 5 seconds but is not required to run the demo.

Source: [`prithvi-server/`](prithvi-server/)

### `regio-agent` — Gradio chat application

A Gradio web UI backed by an OpenAI tool-calling agent loop using the `openai` SDK against the LiteLLM MaaS endpoint (Qwen3-14B). The agent orchestrates four tools in sequence, calling the Prithvi endpoint for water detection and performing the remaining steps locally.

Source: [`regio-agent/`](regio-agent/)

---

## Kubernetes and OpenShift objects

### `00-secret.yaml` — API credentials

A Kubernetes `Secret` holding the LiteLLM MaaS API key. This file contains a placeholder value and must be populated manually before applying. **Never commit the populated version to git.**

```bash
vi stage3/manifests/00-secret.yaml   # replace the placeholder
oc apply -f stage3/manifests/00-secret.yaml
```

The secret is mounted into the `regio-agent` pod as the environment variable `LITELLM_API_KEY`.

---

### `01-pvc-models.yaml` — Model weight cache

A `PersistentVolumeClaim` (10 Gi, `ReadWriteOnce`) that stores the HuggingFace model cache for Prithvi-EO-2.0-300M-TL-Sen1Floods11 (~1.2 GB). The PVC is mounted at `/mnt/models` inside the Prithvi pod.

`snapshot_download` in `predictor.py` sets `HF_HOME=/mnt/models/hf-cache`. On first startup the model is downloaded and written to the PVC. On subsequent startups `snapshot_download` contacts HuggingFace to check the revision hash, finds the files already present, and loads from the PVC — no re-download occurs.

```
/mnt/models/hf-cache/
  models--ibm-nasa-geospatial--Prithvi-EO-2.0-300M-TL-Sen1Floods11/
    snapshots/
      918b9f.../
        Prithvi-EO-V2-300M-TL-Sen1Floods11.pt
```

**Important:** The PVC uses `accessModes: ReadWriteOnce`. Only one node can mount it at a time. If a rolling update schedules the new pod on a different node before the old pod is terminated, a multi-attach error will occur. Scale the old ReplicaSet to zero before rolling out a new version if this happens.

---

### `02-serving-runtime.yaml` — KServe ServingRuntime

Defines the container template for the Prithvi model server as a KServe `ServingRuntime`. This object tells KServe which container image to use and which model format it supports.

> **Note:** The `InferenceService` in `03-inference-service.yaml` uses `RawDeployment` mode with a direct `containers` spec, bypassing the ServingRuntime format-matching mechanism. The ServingRuntime is included for completeness and future use, but is not actively used by the current deployment.

---

### `03-inference-service.yaml` — KServe InferenceService

The central object for the Prithvi model service.

**RawDeployment mode** is used instead of the default Serverless (Knative) mode:

```yaml
annotations:
  serving.kserve.io/deploymentMode: RawDeployment
```

RawDeployment creates a standard Kubernetes `Deployment` and `Service` rather than a Knative `Service`. This avoids an Istio/Knative dependency and is simpler for a single-replica demo.

**Headless service:** KServe creates the internal `Service` for the predictor with `clusterIP: None`. DNS resolution returns the pod IP directly rather than going through kube-proxy. As a result, the `regio-agent` must use **port 8080** (the container port) in its `PRITHVI_URL`, not port 80:

```
http://prithvi-water-predictor.jwesterl.svc.cluster.local:8080/v2/models/prithvi-water/infer
```

**Readiness probe** is configured with a generous timeout to accommodate CPU model loading (~2 minutes on first start after a cache hit, longer on a cold PVC):

```yaml
initialDelaySeconds: 30
periodSeconds: 15
failureThreshold: 40   # 10 minutes total
```

---

### `04-agent-deployment.yaml` — Regio-agent Deployment and Service

A standard Kubernetes `Deployment` and `Service` for the Gradio application.

Key environment variables:

| Variable | Purpose |
|---|---|
| `LITELLM_ENDPOINT` | Base URL of the LiteLLM MaaS OpenAI-compatible API |
| `LLM_MODEL` | Model name passed to LiteLLM (e.g. `qwen3-14b`) |
| `PRITHVI_URL` | In-cluster URL for the Prithvi KServe endpoint (port 8080) |
| `LITELLM_API_KEY` | Injected from the `regio-ai-secrets` Secret |

`imagePullPolicy: Always` is set on both containers. Combined with the `:latest` tag, this guarantees that a new image pushed to the registry is pulled on every pod restart without needing to change the manifest.

---

### `05-agent-route.yaml` — OpenShift Route

An OpenShift `Route` that exposes the Gradio service publicly over HTTPS. TLS is terminated at the router (edge termination) and HTTP traffic is redirected to HTTPS.

OpenShift Routes are not a standard Kubernetes resource — they are an OpenShift-specific alternative to `Ingress`. The route hostname is automatically assigned by the cluster's ingress controller based on the namespace and cluster domain.

---

## Deployment

```bash
# 1. Fill in the LiteLLM API key
vi stage3/manifests/00-secret.yaml
oc apply -f stage3/manifests/00-secret.yaml

# 2. Apply remaining manifests in order
oc apply -f stage3/manifests/01-pvc-models.yaml
oc apply -f stage3/manifests/02-serving-runtime.yaml
oc apply -f stage3/manifests/03-inference-service.yaml
oc apply -f stage3/manifests/04-agent-deployment.yaml
oc apply -f stage3/manifests/05-agent-route.yaml

# 3. Watch status
oc get inferenceservice prithvi-water
oc get deployment regio-agent
oc get route regio-agent
```

Or use the Makefile targets from the `stage3/` directory:

```bash
make deploy    # applies manifests 01–05
make status    # watch pod and route status
make teardown  # delete all objects
```

---

## Logs

```bash
# Prithvi model server — model loading, inference requests
oc logs -l serving.kserve.io/inferenceservice=prithvi-water -f

# Regio-agent — Gradio startup, agent tool calls, errors
oc logs -l app=regio-agent -f
```
