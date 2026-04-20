"""
Prithvi-EO-2.0-300M KServe predictor — speaks the KServe v2 REST inference protocol.

POST /v2/models/prithvi-water/infer
  Input : {"inputs": [{"name":"patch","shape":[1,6,224,224],"datatype":"FP32","data":[...]}]}
  Output: {"outputs": [{"name":"preds","shape":[224,224],...}, {"name":"probs","shape":[2,224,224],...}]}

The model is downloaded from HuggingFace Hub at startup and cached to HF_HOME (mounted PVC).
"""

import os
import logging
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_ID   = os.environ.get("MODEL_ID", "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11")
HF_HOME    = os.environ.get("HF_HOME",  "/mnt/models/hf-cache")
MODEL_NAME = "prithvi-water"

_model  = None
_device = None


def _find_checkpoint(model_dir: str) -> str:
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith((".ckpt", ".pt", ".pth")):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def _extract_logits(outputs):
    """Handle every output type terratorch / HuggingFace may return."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if hasattr(outputs, "output"):
        return outputs.output
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    try:
        return next(iter(outputs.values()))
    except (AttributeError, StopIteration):
        raise TypeError(
            f"Cannot extract logits from {type(outputs).__name__}. "
            f"Available attrs: {[k for k in dir(outputs) if not k.startswith('_')]}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    log.info("Downloading / loading %s ...", MODEL_ID)
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(MODEL_ID, cache_dir=HF_HOME)
    ckpt_path = _find_checkpoint(model_dir)
    log.info("Checkpoint: %s", ckpt_path)

    from terratorch.tasks import SemanticSegmentationTask
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    task    = SemanticSegmentationTask.load_from_checkpoint(
        ckpt_path, map_location=_device, strict=False
    )
    _model = task.model.eval().to(_device)
    n_params = sum(p.numel() for p in _model.parameters()) / 1e6
    log.info("Ready — %.0fM params on %s", n_params, _device)
    yield  # app runs


app = FastAPI(title="Prithvi Water Predictor", version="1.0.0", lifespan=lifespan)


@app.get("/v2/health/ready")
def health_ready():
    return {"status": "ready" if _model is not None else "loading"}


@app.get("/v2/health/live")
def health_live():
    return {"status": "live"}


@app.get("/v2/models/{model_name}")
def model_metadata(model_name: str):
    return {
        "name":     MODEL_NAME,
        "versions": ["1"],
        "platform": "terratorch",
        "inputs":   [{"name": "patch", "datatype": "FP32",  "shape": [1, 6, 224, 224]}],
        "outputs":  [
            {"name": "preds", "datatype": "INT64", "shape": [224, 224]},
            {"name": "probs", "datatype": "FP32",  "shape": [2, 224, 224]},
        ],
    }


@app.post("/v2/models/{model_name}/infer")
async def infer(model_name: str, request: dict):
    if _model is None:
        raise HTTPException(503, "Model not loaded yet — check /v2/health/ready")

    inputs = {i["name"]: i for i in request.get("inputs", [])}
    raw    = inputs.get("patch")
    if raw is None:
        raise HTTPException(400, "Expected an input named 'patch'")

    patch  = np.array(raw["data"], dtype=np.float32).reshape(raw["shape"])
    tensor = torch.from_numpy(patch).to(_device)

    with torch.no_grad():
        outputs = _model(tensor)

    logits = _extract_logits(outputs)
    preds  = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)

    return {
        "model_name": MODEL_NAME,
        "outputs": [
            {"name": "preds", "shape": list(preds.shape), "datatype": "INT64", "data": preds.flatten().tolist()},
            {"name": "probs", "shape": list(probs.shape), "datatype": "FP32",  "data": probs.flatten().tolist()},
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
