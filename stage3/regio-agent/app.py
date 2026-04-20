"""
Regio-AI — Stage 3 Agentic Application

Gradio chat interface backed by a LangChain tool-calling agent.
The agent uses Qwen3-14B (via LiteLLM MaaS) to orchestrate four tools
that run a full strandskydd violation detection pipeline over Orust island.
"""
import os
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import make_tools

# ── Configuration (set via environment variables) ─────────────────────────────
LITELLM_ENDPOINT = os.environ.get(
    "LITELLM_ENDPOINT", "https://litellm-prod.apps.maas.redhatworkshops.io/v1"
)
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "")
LLM_MODEL       = os.environ.get("LLM_MODEL", "qwen3-14b")
PRITHVI_URL     = os.environ.get(
    "PRITHVI_URL",
    "http://prithvi-water-predictor.jwesterl.svc.cluster.local:8080"
    "/v2/models/prithvi-water/infer",
)

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LITELLM_ENDPOINT,
    api_key=LITELLM_API_KEY,
    temperature=0.1,
    max_tokens=2048,
)

SYSTEM_PROMPT = """\
You are Regio-AI, an assistant that detects potential strandskydd \
(Swedish shoreline protection law) violations using satellite imagery and AI.

The fixed area of interest is Orust island, Bohuslän, Sweden (58.18–58.35°N, \
11.65–11.91°E). The specific Point of Interest is 58.26599°N, 11.77902°E — \
a summer house suspected of having a new structure added after 2019.

Swedish strandskydd (MB 7 kap. 15 §) prohibits construction within 100 metres \
of any water body (lakes, sea, rivers, streams).

You have four tools that MUST be called in this exact order:
  1. search_and_fetch_scenes  — download Sentinel-2 imagery for the date range
  2. run_prithvi_water_detection — run IBM/NASA Prithvi to detect water bodies
  3. compute_ndbi_change — detect new built-up surfaces via spectral change
  4. generate_violation_map — build the interactive violation map

When interpreting a date range from the user:
  - "before" epoch: the earliest period, e.g. '2017-01-01/2018-12-31'
  - "after" epoch:  the latest  period, e.g. '2022-06-01/2023-09-30'
  If the user says "from 2018 to 2024", use before='2017-01-01/2018-12-31' \
and after='2022-06-01/2023-09-30'.

After all four tools complete, summarise the findings in plain language. \
Explain what strandskydd means, what the Prithvi model detected, and what \
the NDBI change and violation count implies for the site. Be concise.\
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])


def make_agent(session: dict) -> AgentExecutor:
    tools = make_tools(session, PRITHVI_URL)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
PLACEHOLDER_MAP = """
<div style="display:flex;align-items:center;justify-content:center;
            height:480px;background:#f0f4f8;border-radius:8px;
            border:1px dashed #b0bec5;">
  <div style="text-align:center;color:#607d8b;">
    <div style="font-size:52px;margin-bottom:14px;">🛰️</div>
    <div style="font-size:16px;font-weight:600;">Violation map will appear here</div>
    <div style="font-size:13px;margin-top:10px;color:#90a4ae;">
      Try: <em>"analyze Orust from 2018 to 2024"</em>
    </div>
  </div>
</div>
"""

THINKING_MSG = (
    "Analysing… running satellite retrieval, Prithvi inference, and change "
    "detection. This takes a few minutes on the first run while the model loads."
)


def respond(message: str, history: list, session_state: dict):
    if not message.strip():
        yield history, PLACEHOLDER_MAP, session_state
        return

    # Show thinking indicator immediately
    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": THINKING_MSG},
    ]
    yield history, PLACEHOLDER_MAP, session_state

    try:
        executor = make_agent(session_state)
        result   = executor.invoke({"input": message})
        answer   = result.get("output", "Analysis complete — see the map panel.")
    except Exception as exc:
        answer = f"Error during analysis: {exc}"

    history[-1]["content"] = answer
    map_html = session_state.get("map_html", PLACEHOLDER_MAP)
    yield history, map_html, session_state


# ── Layout ────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Regio-AI — Strandskydd Violation Detector",
    theme=gr.themes.Soft(),
    css=".chatbot { min-height: 460px; }",
) as demo:

    gr.Markdown(
        "# Regio-AI — Strandskydd Violation Detector\n"
        "**IBM/NASA Prithvi-EO-2.0 · Sentinel-2 · Qwen3-14B via LiteLLM MaaS**\n\n"
        "Describe what you want to analyse in natural language. "
        "The agent will retrieve satellite imagery, run the Prithvi foundation "
        "model for water detection, compute built-up change, and map potential "
        "strandskydd violations on Orust island, Bohuslän."
    )

    session_state = gr.State({})

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Regio-AI Agent",
                type="messages",
                height=480,
                show_copy_button=True,
                avatar_images=(None, "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/NASA_IBM_logo.png"),
            )
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder='e.g. "analyze Orust from 2018 to 2024"',
                    label="",
                    scale=5,
                    autofocus=True,
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

    # Wire events
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
