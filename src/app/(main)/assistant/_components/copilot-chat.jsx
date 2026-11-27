"use client";

import { useEffect, useRef, useState, useTransition } from "react";
import { Bot, FileText, RefreshCw, Send, User } from "lucide-react";
import { Button } from "../../../../components/ui/button";
import { Input } from "../../../../components/ui/input";
import { Card, CardContent } from "../../../../components/ui/card";
import { askCopilot, refreshCopilotKnowledge } from "../../../../actions/copilot";

// The model sometimes uses **bold**; render it instead of showing asterisks.
function formatAnswer(text) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : part,
  );
}

const SUGGESTIONS = [
  "How much did I spend last month?",
  "Which category did I spend the most on this month?",
  "Am I over budget this month?",
  "What are my recurring transactions?",
];

export default function CopilotChat() {
  const [messages, setMessages] = useState([]); // { role, content, sources? }
  const [input, setInput] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [pending, startTransition] = useTransition();
  const [refreshing, startRefresh] = useTransition();
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  function send(text) {
    const question = text.trim();
    if (!question || pending) return;
    setError("");
    setInput("");
    const history = messages.map(({ role, content }) => ({ role, content }));
    setMessages((m) => [...m, { role: "user", content: question }]);

    startTransition(async () => {
      const res = await askCopilot(question, history);
      if (res.success) {
        setMessages((m) => [...m, { role: "assistant", content: res.answer, sources: res.sources }]);
      } else {
        setError(res.error);
      }
    });
  }

  function refresh() {
    setNotice("");
    startRefresh(async () => {
      const res = await refreshCopilotKnowledge();
      setNotice(
        res.success
          ? "Refreshing your data. New transactions show up in answers within a few minutes."
          : res.error,
      );
    });
  }

  return (
    <Card>
      <CardContent className="p-0">
        <div className="h-[55vh] min-h-[320px] overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center gap-4">
              <Bot className="h-10 w-10 text-blue-600" />
              <p className="text-muted-foreground max-w-sm">
                Try one of these, or ask your own question:
              </p>
              <div className="flex flex-wrap justify-center gap-2">
                {SUGGESTIONS.map((s) => (
                  <Button key={s} variant="outline" size="sm" onClick={() => send(s)} disabled={pending}>
                    {s}
                  </Button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
              {m.role === "assistant" && <Bot className="h-6 w-6 shrink-0 text-blue-600 mt-1" />}
              <div className={`max-w-[85%] ${m.role === "user" ? "order-first" : ""}`}>
                <div
                  className={`rounded-2xl px-4 py-2 whitespace-pre-wrap text-sm leading-relaxed ${
                    m.role === "user" ? "bg-blue-600 text-white rounded-br-sm" : "bg-muted rounded-bl-sm"
                  }`}
                >
                  {m.role === "assistant" ? formatAnswer(m.content) : m.content}
                </div>
                {m.sources?.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {m.sources.map((s) => (
                      <span
                        key={s.title}
                        className="inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs text-muted-foreground"
                      >
                        <FileText className="h-3 w-3" />
                        {s.title}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              {m.role === "user" && <User className="h-6 w-6 shrink-0 text-muted-foreground mt-1" />}
            </div>
          ))}

          {pending && (
            <div className="flex gap-3 items-center text-muted-foreground text-sm">
              <Bot className="h-6 w-6 text-blue-600 animate-pulse" />
              Searching your records…
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {(error || notice) && (
          <p className={`px-4 pb-2 text-sm ${error ? "text-red-600" : "text-muted-foreground"}`}>
            {error || notice}
          </p>
        )}

        <form
          className="flex gap-2 border-t p-3"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your spending, budget or accounts…"
            maxLength={1000}
            disabled={pending}
            aria-label="Your question"
          />
          <Button type="submit" disabled={pending || !input.trim()} aria-label="Send">
            <Send className="h-4 w-4" />
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={refresh}
            disabled={refreshing}
            title="Refresh Copilot with your latest data"
            aria-label="Refresh my data"
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
