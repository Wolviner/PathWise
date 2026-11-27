"""Entry point for a free Hugging Face *Gradio* Space.

Docker Spaces are paid; Gradio Spaces are free and simply run this file with Python.
It serves the same FastAPI API as the Docker image (POST /ask, /reindex/...), plus a
small Gradio status page at /ui. Hugging Face routes all traffic to port 7860.
"""

import os

import uvicorn

from app.main import app as api

try:
    import gradio as gr

    with gr.Blocks(title="PathWise RAG") as status_page:
        gr.Markdown(
            "## PathWise RAG service\n"
            "This Space runs the API behind **PathWise Copilot**. "
            "It is called by the PathWise server only; every endpoint except the health check "
            "needs the shared secret."
        )
    api = gr.mount_gradio_app(api, status_page, path="/ui")
except ImportError:  # running outside a Gradio Space: plain API is fine
    pass

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("PORT") or 7860)
    uvicorn.run(api, host="0.0.0.0", port=port)
