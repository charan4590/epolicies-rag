"use client";

import { Eraser, Loader2, RotateCcw, ShieldCheck, Square, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { AssistantMessage } from "@/components/chat/assistant-message";
import { ChatInput } from "@/components/chat/chat-input";
import { Welcome } from "@/components/chat/welcome";
import { useConversations } from "@/components/layout/conversations-context";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ApiError, api, chatStream } from "@/lib/api";
import type { ChatResponse, ConversationDetail } from "@/lib/types";

type UIMessage =
  | { id: string; role: "user"; content: string }
  | { id: string; role: "assistant"; status: "pending"; stage: string; question: string }
  | { id: string; role: "assistant"; status: "done"; response: ChatResponse }
  | { id: string; role: "assistant"; status: "error"; message: string; question: string }
  | { id: string; role: "assistant"; status: "stopped"; question: string };

const DIAGNOSTICS_KEY = "epolicies.showDiagnostics";

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : String(Math.random());
}

function fromConversation(detail: ConversationDetail): UIMessage[] {
  return detail.messages.map((m): UIMessage => {
    if (m.role === "user") return { id: m.message_id, role: "user", content: m.content };
    const payload = m.payload;
    const response: ChatResponse = {
      answer: payload.answer ?? m.content,
      citations: payload.citations ?? [],
      insufficient_evidence: payload.insufficient_evidence ?? false,
      missing_information: payload.missing_information ?? [],
      conflicts: payload.conflicts ?? [],
      verification: payload.verification ?? { status: "not_applicable", revised: false, llm_checked: false, removed_citations: [], claims: [], notes: [] },
      standalone_question: payload.standalone_question ?? "",
      conversation_id: detail.conversation_id,
      message_id: m.message_id,
      diagnostics: null,
    };
    return { id: m.message_id, role: "assistant", status: "done", response };
  });
}

export function ChatWorkspace({ conversationId: initialId }: { conversationId?: string }) {
  const router = useRouter();
  const { refresh } = useConversations();
  const [conversationId, setConversationId] = useState<string | null>(initialId ?? null);
  const [title, setTitle] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(Boolean(initialId));
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const previousPath = useRef(pathname);

  // The URL is swapped to /chat/{id} in place after the first answer; navigating back
  // to "/" (New chat) reuses this component instance, so reset its state explicitly.
  useEffect(() => {
    const before = previousPath.current;
    previousPath.current = pathname;
    if (!initialId && pathname === "/" && before !== "/") {
      controllerRef.current?.abort();
      setMessages([]);
      setConversationId(null);
      setTitle(null);
      setInput("");
    }
  }, [pathname, initialId]);

  useEffect(() => {
    try {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setShowDiagnostics(window.localStorage.getItem(DIAGNOSTICS_KEY) === "1");
    } catch {
      /* storage unavailable */
    }
  }, []);

  useEffect(() => {
    if (!initialId) return;
    let active = true;
    api
      .getConversation(initialId)
      .then((detail) => {
        if (!active) return;
        setMessages(fromConversation(detail));
        setTitle(detail.title);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setLoadError(err instanceof ApiError && err.status === 404 ? "This conversation no longer exists." : err instanceof Error ? err.message : "Could not load the conversation.");
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [initialId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => () => controllerRef.current?.abort(), []);

  const toggleDiagnostics = (value: boolean) => {
    setShowDiagnostics(value);
    try {
      window.localStorage.setItem(DIAGNOSTICS_KEY, value ? "1" : "0");
    } catch {
      /* storage unavailable */
    }
  };

  const updateAssistant = (id: string, next: UIMessage) =>
    setMessages((items) => items.map((m) => (m.id === id ? next : m)));

  const ask = useCallback(
    async (rawQuestion: string) => {
      const question = rawQuestion.trim();
      if (!question || busy) return;
      const assistantId = newId();
      setMessages((items) => [
        ...items,
        { id: newId(), role: "user", content: question },
        { id: assistantId, role: "assistant", status: "pending", stage: "Understanding the question", question },
      ]);
      setInput("");
      setBusy(true);
      const controller = new AbortController();
      controllerRef.current = controller;
      try {
        const response = await chatStream(
          { question, conversation_id: conversationId, include_diagnostics: showDiagnostics },
          (event) => {
            if (event.type === "stage") {
              updateAssistant(assistantId, { id: assistantId, role: "assistant", status: "pending", stage: event.message, question });
            }
          },
          controller.signal,
        );
        updateAssistant(assistantId, { id: assistantId, role: "assistant", status: "done", response });
        if (response.conversation_id && response.conversation_id !== conversationId) {
          setConversationId(response.conversation_id);
          setTitle((t) => t ?? question);
          // Update the URL without remounting this component.
          window.history.replaceState(null, "", `/chat/${response.conversation_id}`);
        }
        void refresh();
      } catch (err) {
        if (controller.signal.aborted) {
          updateAssistant(assistantId, { id: assistantId, role: "assistant", status: "stopped", question });
        } else {
          const message = err instanceof Error ? err.message : "Something went wrong.";
          updateAssistant(assistantId, { id: assistantId, role: "assistant", status: "error", message, question });
        }
      } finally {
        controllerRef.current = null;
        setBusy(false);
      }
    },
    [busy, conversationId, showDiagnostics, refresh],
  );

  const stop = () => controllerRef.current?.abort();

  const clearChat = () => {
    controllerRef.current?.abort();
    setMessages([]);
    setConversationId(null);
    setTitle(null);
    setInput("");
    router.push("/");
  };

  const empty = messages.length === 0;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex h-12 shrink-0 items-center gap-3 border-b px-4">
        <h2 className="min-w-0 flex-1 truncate text-sm font-medium">{title ?? "New conversation"}</h2>
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch checked={showDiagnostics} onCheckedChange={toggleDiagnostics} aria-label="Show retrieval diagnostics" />
          <span className="hidden sm:inline">Diagnostics</span>
        </label>
        <Button variant="ghost" size="sm" onClick={clearChat} disabled={empty && !conversationId} aria-label="Clear chat and start a new conversation">
          <Eraser /> <span className="hidden sm:inline">Clear</span>
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto" aria-live="polite">
        {loading ? (
          <div className="mx-auto w-full max-w-3xl space-y-4 px-4 py-8">
            <Skeleton className="ml-auto h-10 w-2/3" />
            <Skeleton className="h-28 w-full" />
          </div>
        ) : loadError ? (
          <div className="mx-auto max-w-md px-4 py-16 text-center">
            <TriangleAlert className="mx-auto mb-3 size-6 text-muted-foreground" aria-hidden />
            <p className="text-sm">{loadError}</p>
            <Button asChild variant="outline" size="sm" className="mt-4">
              <Link href="/">Start a new chat</Link>
            </Button>
          </div>
        ) : empty ? (
          <Welcome />
        ) : (
          <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-6">
            {messages.map((message) =>
              message.role === "user" ? (
                <div key={message.id} className="flex justify-end">
                  <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm whitespace-pre-wrap text-primary-foreground">
                    {message.content}
                  </div>
                </div>
              ) : (
                <div key={message.id} className="flex gap-3">
                  <div className="mt-0.5 hidden size-7 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary sm:flex">
                    <ShieldCheck className="size-4" aria-hidden />
                  </div>
                  <div className="min-w-0 flex-1 text-sm">
                    {message.status === "pending" && (
                      <div className="flex items-center gap-2 py-1 text-muted-foreground" role="status">
                        <Loader2 className="size-4 animate-spin" aria-hidden />
                        <span>{message.stage}…</span>
                      </div>
                    )}
                    {message.status === "done" && <AssistantMessage response={message.response} />}
                    {message.status === "stopped" && (
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <Square className="size-3.5" aria-hidden /> Generation stopped.
                        <Button variant="link" size="sm" className="h-auto p-0" onClick={() => void ask(message.question)} disabled={busy}>
                          Ask again
                        </Button>
                      </div>
                    )}
                    {message.status === "error" && (
                      <div className="rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2.5">
                        <p className="flex items-center gap-2 font-medium text-destructive">
                          <TriangleAlert className="size-4" aria-hidden /> The answer could not be generated
                        </p>
                        <p className="mt-1 text-muted-foreground">{message.message}</p>
                        <Button variant="outline" size="sm" className="mt-2" onClick={() => void ask(message.question)} disabled={busy}>
                          <RotateCcw /> Retry
                        </Button>
                      </div>
                    )}
                  </div>
                </div>
              ),
            )}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      <div className="shrink-0 border-t bg-background/95 px-4 pt-3 pb-4">
        <div className="mx-auto w-full max-w-3xl">
          <ChatInput value={input} onChange={setInput} onSubmit={() => void ask(input)} onStop={stop} busy={busy} disabled={loading || Boolean(loadError)} />
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Answers are generated from indexed policy documents and may be incomplete. Always check the cited source.
          </p>
        </div>
      </div>
    </div>
  );
}
