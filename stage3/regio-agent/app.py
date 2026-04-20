"""
Regio-AI — Stage 3 Agentic Application

Gradio chat interface with an OpenAI tool-calling agent loop.
Uses the openai SDK directly against LiteLLM MaaS — no langchain dependency.
Compatible with Gradio 5 and 6.
"""
import os
import json
import gradio as gr
from openai import OpenAI

import tools as _tools_module

# ── Config ────────────────────────────────────────────────────────────────────
LITELLM_ENDPOINT = os.environ.get(
    "LITELLM_ENDPOINT", "https://litellm-prod.apps.maas.redhatworkshops.io/v1"
)
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
LLM_MODEL       = os.environ.get("LLM_MODEL", "qwen3-14b")
PRITHVI_URL     = os.environ.get(
    "PRITHVI_URL",
    "http://prithvi-water-predictor.jwesterl.svc.cluster.local"
    "/v2/models/prithvi-water/infer",
)

client = OpenAI(base_url=LITELLM_ENDPOINT, api_key=LITELLM_API_KEY)

# ── Tool specs ────────────────────────────────────────────────────────────────
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_and_fetch_scenes",
            "description": (
                "Search Planetary Computer for the best cloud-free Sentinel-2 scenes "
                "for Orust island in two date ranges, download them, and extract a "
                "224x224 px patch centred on the Point of Interest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date_before": {
                        "type": "string",
                        "description": "ISO date range for the before epoch, e.g. '2017-01-01/2018-12-31'",
                    },
                    "date_after": {
                        "type": "string",
                        "description": "ISO date range for the after epoch, e.g. '2022-06-01/2023-09-30'",
                    },
                },
                "required": ["date_before", "date_after"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_prithvi_water_detection",
            "description": (
                "Call the Prithvi-EO-2.0-300M KServe endpoint to segment water bodies "
                "and derive the strandskydd 100 m buffer. Call after search_and_fetch_scenes."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_ndbi_change",
            "description": (
                "Compute NDBI change detection to find new built-up surfaces. "
                "Intersects with strandskydd zone to flag potential violations. "
                "Call after run_prithvi_water_detection."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_violation_map",
            "description": (
                "Generate an interactive Folium map showing water bodies, "
                "strandskydd zone, and potential violations. "
                "Call after compute_ndbi_change."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

SYSTEM_PROMPT = """\
You are Regio-AI, an assistant that detects potential strandskydd \
(Swedish shoreline protection law) violations using satellite imagery and AI.

The fixed area of interest is Orust island, Bohuslan, Sweden. The specific \
Point of Interest is 58.26599N, 11.77902E, a summer house suspected of \
having a new structure added after 2019.

Swedish strandskydd prohibits construction within 100 metres of any water body.

You have four tools that MUST be called in this exact order:
  1. search_and_fetch_scenes  - download Sentinel-2 imagery
  2. run_prithvi_water_detection - run IBM/NASA Prithvi foundation model
  3. compute_ndbi_change - detect new built-up surfaces
  4. generate_violation_map - build the interactive map

Date range guidance: if the user says "from 2018 to 2024", use \
date_before='2017-01-01/2018-12-31' and date_after='2022-06-01/2023-09-30'.

After all four tools complete, summarise the findings clearly in plain language.\
"""

TOOL_FN = {
    "search_and_fetch_scenes":     lambda **kw: _tools_module.search_and_fetch_scenes(**kw),
    "run_prithvi_water_detection": lambda **kw: _tools_module.run_prithvi_water_detection(),
    "compute_ndbi_change":         lambda **kw: _tools_module.compute_ndbi_change(),
    "generate_violation_map":      lambda **kw: _tools_module.generate_violation_map(),
}


def run_agent(user_message: str, session: dict) -> str:
    _tools_module._SESSION     = session
    _tools_module._PRITHVI_URL = PRITHVI_URL

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    for _ in range(12):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.1,
        )
        choice = response.choices[0]
        msg    = choice.message

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn   = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                try:
                    result = TOOL_FN[fn](**args)
                except Exception as exc:
                    result = f"Tool error ({fn}): {exc}"
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      str(result),
                })
        else:
            return msg.content or "Analysis complete — see the map panel."

    return "Analysis complete — see the map panel."


# ── Gradio UI (compatible with Gradio 5 and 6) ───────────────────────────────
PLACEHOLDER_MAP = """
<div style="display:flex;align-items:center;justify-content:center;
            height:480px;background:#f0f4f8;border-radius:8px;
            border:1px dashed #b0bec5;">
  <div style="text-align:center;color:#607d8b;">
    <div style="font-size:52px;margin-bottom:14px;">&#128752;</div>
    <div style="font-size:16px;font-weight:600;">Violation map will appear here</div>
    <div style="font-size:13px;margin-top:10px;color:#90a4ae;">
      Try: <em>"analyze Orust from 2018 to 2024"</em>
    </div>
  </div>
</div>
"""

THINKING = "Analysing... retrieving satellite data, running Prithvi, computing change detection. This takes a few minutes."


def respond(message, history, session_state):
    if not message.strip():
        yield history, PLACEHOLDER_MAP, session_state
        return

    history = history + [[message, THINKING]]
    yield history, PLACEHOLDER_MAP, session_state

    try:
        answer = run_agent(message, session_state)
    except Exception as exc:
        answer = f"Error during analysis: {exc}"

    history[-1][1] = answer
    map_html = session_state.get("map_html", PLACEHOLDER_MAP)
    yield history, map_html, session_state


with gr.Blocks(title="Regio-AI — Strandskydd Violation Detector") as demo:
    gr.Markdown(
        "# Regio-AI — Strandskydd Violation Detector\n"
        "**IBM/NASA Prithvi-EO-2.0 · Sentinel-2 · Qwen3-14B via LiteLLM MaaS**\n\n"
        "Describe what you want to analyse in natural language."
    )

    session_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Regio-AI Agent", height=480)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder='e.g. "analyze Orust from 2018 to 2024"',
                    label="", scale=5,
                )
                send_btn = gr.Button("Analyse", variant="primary", scale=1)
            gr.Examples(
                examples=[
                    ["analyze Orust from 2018 to 2024"],
                    ["check for strandskydd violations between 2017 and 2023"],
                    ["run a full analysis comparing 2016 and 2022 scenes"],
                ],
                inputs=msg_box,
            )
        with gr.Column(scale=3):
            map_display = gr.HTML(value=PLACEHOLDER_MAP, label="Violation Map")

    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, session_state],
        outputs=[chatbot, map_display, session_state],
    )
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot, session_state],
        outputs=[chatbot, map_display, session_state],
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )
