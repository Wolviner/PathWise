import CopilotChat from "./_components/copilot-chat";

// Copilot answers take several LLM calls (rewrite, rerank, answer), and a sleeping
// free Hugging Face Space needs time to wake up.
export const maxDuration = 60;

export const metadata = { title: "Copilot | PathWise" };

export default function AssistantPage() {
  return (
    <div className="max-w-3xl mx-auto px-5">
      <h1 className="text-5xl gradient-title mb-2">Copilot</h1>
      <p className="text-muted-foreground mb-6">
        Ask about your own money. Answers come from your PathWise records and show the sources they used.
      </p>
      <CopilotChat />
    </div>
  );
}
