"use client";

import { CircleAlert, Copy, Check, GitCompareArrows, Info, ShieldCheck, ShieldQuestion, TriangleAlert } from "lucide-react";
import { useMemo, useRef, useState } from "react";

import { DiagnosticsPanel } from "@/components/chat/diagnostics-panel";
import { SafeMarkdown } from "@/components/chat/markdown";
import { SourceCard } from "@/components/citations/source-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { ChatResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

const STATUS_TEXT: Record<ChatResponse["verification"]["status"], { label: string; tone: string; hint: string }> = {
  verified: { label: "Citations verified", tone: "text-emerald-700 dark:text-emerald-400", hint: "Every statement was checked against the cited passages." },
  revised: { label: "Verified after revision", tone: "text-emerald-700 dark:text-emerald-400", hint: "Unsupported statements were removed or corrected after citation checks." },
  partially_verified: { label: "Partially verified", tone: "text-amber-700 dark:text-amber-400", hint: "Some statements could not be fully verified; review the sources." },
  not_applicable: { label: "No sources cited", tone: "text-muted-foreground", hint: "No policy passage supports this reply." },
};

export function AssistantMessage({ response }: { response: ChatResponse }) {
  const labels = useMemo(() => new Set(response.citations.map((c) => c.label)), [response.citations]);
  const [openSources, setOpenSources] = useState<Set<string>>(new Set());
  const [highlight, setHighlight] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const refs = useRef<Record<string, HTMLDivElement | null>>({});
  const status = STATUS_TEXT[response.verification.status];
  // An answer whose citations all checked out is shown with a quiet note instead of the warning banner,
  // even when the evidence check said a part of the question was not covered.
  const verifiedPartial =
    response.citations.length > 0 && ["verified", "revised"].includes(response.verification.status);

  const focusSource = (label: string) => {
    setOpenSources((prev) => new Set(prev).add(label));
    setHighlight(label);
    requestAnimationFrame(() => refs.current[label]?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    window.setTimeout(() => setHighlight((h) => (h === label ? null : h)), 1600);
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(response.answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="space-y-3">
      {response.insufficient_evidence && verifiedPartial && (
        <div className="flex gap-2 rounded-md border px-3 py-2 text-xs text-muted-foreground">
          <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          <div className="space-y-0.5">
            <p className="font-medium text-foreground">Partial answer – the cited passages are verified</p>
            {response.missing_information.length > 0 && (
              <p>Not covered by the passages found: {response.missing_information.join("; ")}</p>
            )}
          </div>
        </div>
      )}
      {response.insufficient_evidence && !verifiedPartial && (
        <Alert className="border-amber-300 bg-amber-50 text-amber-950 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-100">
          <TriangleAlert aria-hidden />
          <AlertTitle>Insufficient evidence in the knowledge base</AlertTitle>
          {response.missing_information.length > 0 && (
            <AlertDescription className="text-amber-900/80 dark:text-amber-100/80">
              Not found: {response.missing_information.join("; ")}
            </AlertDescription>
          )}
        </Alert>
      )}

      <SafeMarkdown citationLabels={labels} onCitationClick={focusSource}>
        {response.answer}
      </SafeMarkdown>

      {response.conflicts.length > 0 && (
        <Alert>
          <GitCompareArrows aria-hidden />
          <AlertTitle>Conflicting policy statements</AlertTitle>
          <AlertDescription>
            <ul className="list-disc space-y-1 pl-4">
              {response.conflicts.map((conflict) => (
                <li key={conflict.description}>
                  {conflict.description}
                  {conflict.labels.map((label) => (
                    <button
                      key={label}
                      type="button"
                      onClick={() => focusSource(label)}
                      className="ml-1 font-mono text-[11px] font-semibold text-primary hover:underline"
                    >
                      [{label}]
                    </button>
                  ))}
                </li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {response.verification.notes.length > 0 && (
        <div className="flex gap-2 rounded-md bg-muted/60 px-3 py-2 text-xs text-muted-foreground">
          <CircleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden />
          <div className="space-y-0.5">
            {response.verification.notes.map((note) => <p key={note}>{note}</p>)}
          </div>
        </div>
      )}

      <div className="flex items-center gap-2 text-xs">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className={cn("inline-flex items-center gap-1", status.tone)} tabIndex={0}>
              {response.verification.status === "partially_verified" || response.verification.status === "not_applicable" ? (
                <ShieldQuestion className="size-3.5" aria-hidden />
              ) : (
                <ShieldCheck className="size-3.5" aria-hidden />
              )}
              {status.label}
            </span>
          </TooltipTrigger>
          <TooltipContent>{status.hint}</TooltipContent>
        </Tooltip>
        <Button variant="ghost" size="icon-xs" onClick={() => void copy()} aria-label="Copy answer" className="ml-auto">
          {copied ? <Check /> : <Copy />}
        </Button>
      </div>

      {response.citations.length > 0 && (
        <section aria-label="Sources" className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">
            Sources ({response.citations.length})
          </p>
          {response.citations.map((citation) => (
            <SourceCard
              key={citation.label}
              ref={(el) => {
                refs.current[citation.label] = el;
              }}
              citation={citation}
              open={openSources.has(citation.label)}
              highlighted={highlight === citation.label}
              onOpenChange={(open) =>
                setOpenSources((prev) => {
                  const next = new Set(prev);
                  if (open) next.add(citation.label);
                  else next.delete(citation.label);
                  return next;
                })
              }
            />
          ))}
        </section>
      )}

      {response.diagnostics && <DiagnosticsPanel diagnostics={response.diagnostics} />}
    </div>
  );
}
