"""Start-up file for the Hugging Face Space (Gradio SDK, ZeroGPU hardware).

Hugging Face's free tier only runs Gradio apps, so this file wraps the
existing FastAPI service in one: the API keeps its routes (/api/health,
/api/chips, /api/query) and a small Gradio status page is served at "/".

Nothing here is needed to run the backend anywhere else. Locally, and on any
Docker host, start it with `uvicorn app.main:app` as before.
"""
from __future__ import annotations

import os

# Defaults for the hosted copy; a Space variable or secret with the same name
# wins. LLM_API_KEY is never set here: it is a Space secret.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("FRONTEND_ORIGIN", "https://study-maze-frontend.vercel.app")
# ZeroGPU makes torch report a GPU at all times but only attaches a real one
# inside @spaces.GPU functions, so the pipeline is pinned to the CPU.
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("PRELOAD_MODELS", "true")

try:  # present on Hugging Face only, and must be imported before torch
    import spaces
except ImportError:  # running this file somewhere else
    spaces = None

import gradio as gr
import uvicorn

from app.main import app as api

WEBSITE = os.environ["FRONTEND_ORIGIN"].split(",")[0].strip()


def gpu_self_test() -> str:
    """Report the GPU that ZeroGPU attaches for the duration of this call."""
    import torch

    if not torch.cuda.is_available():
        return "No GPU is attached to this call."
    return f"GPU attached: {torch.cuda.get_device_name(0)}"


if spaces is not None:
    gpu_self_test = spaces.GPU(duration=10)(gpu_self_test)


with gr.Blocks(title="Study Maze backend") as status_page:
    gr.Markdown(
        "# Study Maze backend\n"
        "This Space is the API behind **Study Maze**, a study buddy that answers questions about "
        "microcontroller chips from their official documentation.\n\n"
        f"Use the website: [{WEBSITE}]({WEBSITE})\n\n"
        "API status: [/api/health](/api/health)"
    )
    with gr.Accordion("GPU self-test", open=False):
        result = gr.Textbox(label="Result", interactive=False)
        gr.Button("Check the GPU").click(gpu_self_test, outputs=result)

# The API routes were registered first, so they take precedence over the page at "/".
app = gr.mount_gradio_app(api, status_page, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
