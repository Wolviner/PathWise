"use server";

import { request } from "@arcjet/next";
import { auth } from "@clerk/nextjs/server";
import { z } from "zod";
import { db } from "../lib/prisma";
import { copilotAj } from "../lib/arcjet";
import { RagError, ragFetch, requestReindex } from "../lib/rag";

const askSchema = z.object({
  question: z.string().trim().min(1, "Ask a question").max(1000, "Keep questions under 1000 characters"),
  history: z
    .array(
      z.object({
        role: z.enum(["user", "assistant"]),
        content: z.string().max(4000),
      }),
    )
    .max(20)
    .default([]),
});

async function currentDbUser() {
  const { userId } = await auth();
  if (!userId) throw new Error("Unauthorized");
  const user = await db.user.findUnique({ where: { clerkUserId: userId } });
  if (!user) throw new Error("User not found");
  return user;
}

export async function askCopilot(question, history = []) {
  try {
    const parsed = askSchema.safeParse({ question, history });
    if (!parsed.success) {
      return { success: false, error: parsed.error.issues[0]?.message ?? "Invalid question" };
    }

    const user = await currentDbUser();

    const decision = await copilotAj.protect(await request(), { userId: user.clerkUserId });
    if (decision.isDenied()) {
      return { success: false, error: "You're asking a bit fast. Please wait a minute and try again." };
    }

    // user_id comes from Clerk on the server, never from the browser.
    const result = await ragFetch("/ask", {
      body: { user_id: user.id, question: parsed.data.question, history: parsed.data.history.slice(-6) },
    });
    return { success: true, answer: result.answer, sources: result.sources ?? [] };
  } catch (error) {
    if (error instanceof RagError) return { success: false, error: error.message };
    console.error("askCopilot failed:", error);
    return { success: false, error: error.message === "Unauthorized" ? "Please sign in again." : "Something went wrong." };
  }
}

// "Refresh my data" button on the Copilot page.
export async function refreshCopilotKnowledge() {
  try {
    const user = await currentDbUser();
    await requestReindex(user.id);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
}
