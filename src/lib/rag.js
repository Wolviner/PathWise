import { inngest } from "./inngest/client";

// Server-only helpers for talking to the PathWise RAG service (rag-service/, on Hugging Face).
// The browser never calls the service directly: user ids always come from Clerk on the server.

export class RagError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export async function ragFetch(path, { method = "POST", body, timeoutMs = 55_000 } = {}) {
  const baseUrl = process.env.RAG_SERVICE_URL;
  const secret = process.env.RAG_SHARED_SECRET;
  if (!baseUrl || !secret) {
    throw new RagError("Copilot is not configured (RAG_SERVICE_URL / RAG_SHARED_SECRET).", 503);
  }

  let res;
  try {
    res = await fetch(`${baseUrl.replace(/\/$/, "")}${path}`, {
      method,
      headers: { "Content-Type": "application/json", "X-RAG-Secret": secret },
      body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeoutMs),
      cache: "no-store",
    });
  } catch (error) {
    // Free Hugging Face Spaces sleep when idle; the first request after a nap can time out.
    const timedOut = error?.name === "TimeoutError" || error?.name === "AbortError";
    throw new RagError(
      timedOut
        ? "Copilot is waking up. Please try again in a few seconds."
        : "Copilot is unreachable right now.",
      503,
    );
  }

  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json())?.detail ?? "";
    } catch {
      /* not JSON */
    }
    throw new RagError(typeof detail === "string" && detail ? detail : `Copilot error (${res.status})`, res.status);
  }
  return res.json();
}

// Ask Inngest to rebuild a user's knowledge base. Debounced per user in the Inngest function,
// so calling this after every change is cheap. If Inngest isn't reachable (local dev without
// the Inngest dev server, or an outage), ask the RAG service directly; it queues the work and
// coalesces repeated requests. Never let it break the calling action.
export async function requestReindex(userId) {
  try {
    await inngest.send({ name: "rag/reindex.requested", data: { userId } });
    return;
  } catch (error) {
    console.warn("Inngest unavailable, asking the RAG service directly:", error.message);
  }
  try {
    await ragFetch(`/reindex/${encodeURIComponent(userId)}`, { timeoutMs: 15_000 });
  } catch (error) {
    console.error("Could not queue Copilot reindex:", error.message);
  }
}
