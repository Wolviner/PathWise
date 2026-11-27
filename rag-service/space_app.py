"""Entry point for a free Hugging Face *Gradio* Space.

Docker Spaces are paid; Gradio Spaces are free and run this file with Python. On a Space,
port 7860 must be served by Gradio's own launcher (running a separate uvicorn on 7860
fails with "address already in use"), so Gradio starts the server and the PathWise API
routes (/ask, /reindex/..., health at /) are added to Gradio's FastAPI app.

Locally you can still run the API on its own: `uvicorn app.main:app --port 7860`.
"""

import os

import gradio as gr

from app.main import app as api

with gr.Blocks(title="PathWise RAG") as demo:
    gr.Markdown(
        "## PathWise RAG service\n"
        "This Space runs the API behind **PathWise Copilot**. It is called by the PathWise "
        "server only; every endpoint except the health check needs the shared secret."
    )


def add_api_routes(gradio_app) -> None:
    # Put the API routes first so they win over Gradio's own routes (e.g. "/").
    gradio_app.router.routes[0:0] = list(api.router.routes)


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        prevent_thread_lock=True,
        show_error=True,
    )
    add_api_routes(demo.server_app if getattr(demo, "server_app", None) else demo.app)
    print("PathWise RAG API routes added to the Gradio server", flush=True)
    demo.block_thread()
