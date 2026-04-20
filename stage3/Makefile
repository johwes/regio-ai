REGISTRY   := quay.io/jwesterl
NAMESPACE  := jwesterl

PRITHVI_IMG := $(REGISTRY)/prithvi-server:latest
AGENT_IMG   := $(REGISTRY)/regio-agent:latest

.PHONY: all build build-prithvi build-agent push push-prithvi push-agent \
        deploy teardown redeploy status logs-prithvi logs-agent

all: build push deploy

# ── Build ─────────────────────────────────────────────────────────────────────
build-prithvi:
	podman build -t $(PRITHVI_IMG) stage3/prithvi-server/

build-agent:
	podman build -t $(AGENT_IMG) stage3/regio-agent/

build: build-prithvi build-agent

# ── Push ──────────────────────────────────────────────────────────────────────
push-prithvi: build-prithvi
	podman push $(PRITHVI_IMG)

push-agent: build-agent
	podman push $(AGENT_IMG)

push: push-prithvi push-agent

# ── Deploy ────────────────────────────────────────────────────────────────────
# NOTE: Apply 00-secret.yaml manually with real values before running deploy.
#       It is excluded here to prevent accidental credential commits.
deploy:
	oc apply -f stage3/manifests/01-pvc-models.yaml     -n $(NAMESPACE)
	oc apply -f stage3/manifests/02-serving-runtime.yaml -n $(NAMESPACE)
	oc apply -f stage3/manifests/03-inference-service.yaml -n $(NAMESPACE)
	oc apply -f stage3/manifests/04-agent-deployment.yaml -n $(NAMESPACE)
	oc apply -f stage3/manifests/05-agent-route.yaml     -n $(NAMESPACE)

teardown:
	oc delete -f stage3/manifests/05-agent-route.yaml      -n $(NAMESPACE) --ignore-not-found=true
	oc delete -f stage3/manifests/04-agent-deployment.yaml -n $(NAMESPACE) --ignore-not-found=true
	oc delete -f stage3/manifests/03-inference-service.yaml -n $(NAMESPACE) --ignore-not-found=true
	oc delete -f stage3/manifests/02-serving-runtime.yaml  -n $(NAMESPACE) --ignore-not-found=true
	oc delete -f stage3/manifests/01-pvc-models.yaml       -n $(NAMESPACE) --ignore-not-found=true

redeploy: push deploy

# ── Observe ───────────────────────────────────────────────────────────────────
status:
	@echo "\n=== InferenceService ==="
	oc get inferenceservice -n $(NAMESPACE)
	@echo "\n=== Agent Deployment ==="
	oc get deployment regio-agent -n $(NAMESPACE)
	@echo "\n=== Pods ==="
	oc get pods -n $(NAMESPACE)
	@echo "\n=== Route ==="
	oc get route regio-agent -n $(NAMESPACE)

logs-prithvi:
	oc logs -l serving.kserve.io/inferenceservice=prithvi-water \
	        -n $(NAMESPACE) -f --tail=100

logs-agent:
	oc logs -l app=regio-agent -n $(NAMESPACE) -f --tail=100
