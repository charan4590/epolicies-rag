"use client";

import { useEffect, useState } from "react";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

type State = { kind: "loading" } | { kind: "offline"; message: string } | { kind: "health"; health: HealthResponse };

const POLL_MS = 30_000;

export function ConnectionStatus({ className }: { className?: string }) {
  const [state, setState] = useState<State>({ kind: "loading" });

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const check = async () => {
      try {
        const health = await api.health(controller.signal);
        if (active) setState({ kind: "health", health });
      } catch (err) {
        if (active) setState({ kind: "offline", message: err instanceof Error ? err.message : "Unavailable" });
      }
    };
    void check();
    const timer = setInterval(check, POLL_MS);
    return () => {
      active = false;
      controller.abort();
      clearInterval(timer);
    };
  }, []);

  let tone = "bg-muted-foreground";
  let label = "Checking…";
  let details: string[] = [];
  if (state.kind === "offline") {
    tone = "bg-red-500";
    label = "API offline";
    details = [state.message];
  } else if (state.kind === "health") {
    const { health } = state;
    if (health.status === "error") {
      tone = "bg-red-500";
      label = "Database unavailable";
      details = [health.database.error ?? "The backend cannot reach PostgreSQL."];
    } else {
      const docs = health.index?.indexed_documents ?? 0;
      tone = health.status === "ok" && docs > 0 ? "bg-emerald-500" : "bg-amber-500";
      label = health.status === "ok" ? (docs > 0 ? "Connected" : "Connected · index empty") : "Degraded";
      details = [
        `${docs} indexed document${docs === 1 ? "" : "s"}, ${health.index?.chunks ?? 0} chunks`,
        `Generation: ${health.gemini.generation_model ?? "not configured"}`,
        `Embeddings: ${health.gemini.embedding_model} (${health.gemini.embedding_dimensions}d)`,
      ];
      if (!health.gemini.configured) details.push("Gemini API key is not configured on the server.");
      if (health.database.schema_initialized === false) details.push("Database schema not initialised.");
    }
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          className={cn("flex items-center gap-2 text-xs text-muted-foreground", className)}
          role="status"
          aria-live="polite"
          tabIndex={0}
        >
          <span className={cn("size-2 shrink-0 rounded-full", tone)} aria-hidden />
          <span className="truncate">{label}</span>
        </div>
      </TooltipTrigger>
      {details.length > 0 && (
        <TooltipContent side="top" align="start" className="max-w-72">
          <ul className="space-y-0.5">
            {details.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </TooltipContent>
      )}
    </Tooltip>
  );
}
