"""Entry point for a free Hugging Face *Gradio* Space.

Docker Spaces are paid; Gradio Spaces are free and run this file with Python. Gradio's
own launcher starts the server (with server-side rendering off, so Python serves port
7860 directly), and the PathWise API routes (/ask, /reindex/..., health at /) are added
to Gradio's FastAPI app.

Locally you can still run the API on its own: `uvicorn app.main:app --port 7860`.
"""

import os

import gradio as gr

from app.main import app as api

# Free Gradio Spaces now run on ZeroGPU hardware, which refuses to start unless the app
# declares at least one @spaces.GPU function. This service never needs a GPU (all model
# calls go to the Gemini API), so the function below is never called and uses no GPU time.
try:
    import spaces

    @spaces.GPU(duration=1)
    def _zerogpu_placeholder():
        return None
except ImportError:  # not on ZeroGPU (local runs, CPU hardware)
    pass

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
        # Gradio's server-side rendering puts a Node server on 7860 that only forwards
        # Gradio's own paths, which would hide /ask and /reindex. Serve Python directly.
        ssr_mode=False,
    )
    add_api_routes(demo.server_app if getattr(demo, "server_app", None) else demo.app)
    print("PathWise RAG API routes added to the Gradio server", flush=True)
    demo.block_thread()
