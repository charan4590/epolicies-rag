"use client";

import { Activity, ChevronDown } from "lucide-react";
import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { ChatDiagnostics, RankedItem } from "@/lib/types";
import { cn } from "@/lib/utils";

function RankList({ title, items }: { title: string; items: RankedItem[] }) {
  return (
    <div className="min-w-0">
      <p className="mb-1 truncate text-[11px] font-medium text-muted-foreground" title={title}>
        {title} · {items.length}
      </p>
      <ol className="space-y-0.5 text-xs">
        {items.slice(0, 5).map((item) => (
          <li key={item.chunk_id} className="flex gap-1.5">
            <span className="w-4 shrink-0 text-right font-mono text-muted-foreground">{item.rank}</span>
            <span className="truncate" title={`${item.title} › ${item.heading_path.join(" › ")}`}>
              {item.title}
              {item.heading_path.length > 1 && <span className="text-muted-foreground"> › {item.heading_path.at(-1)}</span>}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Retrieval diagnostics (rankings, rounds, models). Never contains model reasoning. */
export function DiagnosticsPanel({ diagnostics }: { diagnostics: ChatDiagnostics }) {
  const [open, setOpen] = useState(false);
  const { plan, rounds, llm_usage: usage, models, timings_ms: timings } = diagnostics;
  const relevant = new Set(diagnostics.relevant_chunk_ids ?? []);
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-lg border border-dashed">
      <CollapsibleTrigger className="flex w-full items-center gap-2 px-3 py-2 text-xs text-muted-foreground hover:text-foreground">
        <Activity className="size-3.5" aria-hidden />
        <span className="font-medium">Retrieval diagnostics</span>
        <span className="truncate">
          · {diagnostics.retrieval_rounds} round{diagnostics.retrieval_rounds === 1 ? "" : "s"} · {diagnostics.retrieved_chunks} chunks · {usage.calls} LLM calls · {Math.round((timings.total ?? 0) / 100) / 10}s
        </span>
        <ChevronDown className={cn("ml-auto size-3.5 shrink-0 transition-transform", open && "rotate-180")} aria-hidden />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-4 border-t border-dashed px-3 py-3 text-xs">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
            <dt className="text-muted-foreground">Standalone question</dt>
            <dd>{plan.standalone_question}</dd>
            <dt className="text-muted-foreground">Planner</dt>
            <dd>
              {plan.planner} · emphasis {plan.retrieval_emphasis}
              {plan.needs_multiple_documents && " · multi-document"}
            </dd>
            {plan.identifiers.length > 0 && (
              <>
                <dt className="text-muted-foreground">Exact terms</dt>
                <dd className="flex flex-wrap gap-1">
                  {plan.identifiers.map((id) => (
                    <Badge key={id} variant="secondary" className="font-mono">{id}</Badge>
                  ))}
                </dd>
              </>
            )}
            <dt className="text-muted-foreground">Stop reason</dt>
            <dd>{diagnostics.stop_reason.replaceAll("_", " ")}</dd>
            {plan.notes.length > 0 && (
              <>
                <dt className="text-muted-foreground">Notes</dt>
                <dd className="space-y-0.5">
                  {plan.notes.map((note) => <p key={note}>{note}</p>)}
                </dd>
              </>
            )}
            <dt className="text-muted-foreground">Models</dt>
            <dd>
              {models.generation} · {models.embedding} ({models.embedding_dimensions}d)
            </dd>
            <dt className="text-muted-foreground">Tokens</dt>
            <dd>
              {usage.prompt_tokens.toLocaleString()} in · {usage.output_tokens.toLocaleString()} out ·{" "}
              {Object.entries(usage.by_stage).map(([stage, n]) => `${stage}×${n}`).join(", ") || "no LLM calls"}
            </dd>
          </dl>

          {rounds.map((round) => (
            <div key={round.round} className="space-y-2 rounded-md bg-muted/40 p-2.5">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">Round {round.round}</span>
                <span className="text-muted-foreground">{round.new_chunks} new of {round.retrieved} fused</span>
                {round.assessment && (
                  <Badge variant={round.assessment.sufficient ? "secondary" : "outline"}>
                    {round.assessment.sufficient ? "evidence sufficient" : "needs more evidence"} · {round.assessment.confidence}
                  </Badge>
                )}
              </div>
              <ul className="list-disc space-y-0.5 pl-4 text-muted-foreground">
                {round.queries.map((q) => <li key={q}>{q}</li>)}
              </ul>
              {round.assessment && round.assessment.missing_information.length > 0 && (
                <p className="text-muted-foreground">Missing: {round.assessment.missing_information.join("; ")}</p>
              )}
              {round.referenced_policies && round.referenced_policies.length > 0 && (
                <p className="text-muted-foreground">
                  Next round searches referenced: {round.referenced_policies.join("; ")}
                </p>
              )}
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {Object.entries(round.search.lists).map(([name, items]) => (
                  <RankList key={name} title={name} items={items} />
                ))}
              </div>
            </div>
          ))}

          <div>
            <p className="mb-1 font-medium">Context passed to the answer</p>
            <p className="mb-1 text-muted-foreground">
              Passages the evidence check marked relevant (✓) come first, then the rest by RRF score.
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-left">
                <thead className="text-muted-foreground">
                  <tr>
                    <th className="py-1 pr-2 font-normal">#</th>
                    <th className="py-1 pr-2 font-normal">Source</th>
                    <th className="py-1 pr-2 font-normal">Vector</th>
                    <th className="py-1 pr-2 font-normal">Keyword</th>
                    <th className="py-1 pr-2 font-normal">Exact</th>
                    <th className="py-1 pr-2 font-normal">RRF</th>
                    <th className="py-1 font-normal">Relevant</th>
                  </tr>
                </thead>
                <tbody className="font-mono">
                  {diagnostics.context_chunks.map((c, i) => (
                    <tr key={c.chunk_id} className="border-t">
                      <td className="py-1 pr-2">{i + 1}</td>
                      <td className="max-w-64 truncate py-1 pr-2 font-sans" title={`${c.relative_path} › ${c.heading_path.join(" › ")}`}>
                        {c.relative_path}
                      </td>
                      <td className="py-1 pr-2">{c.vector_rank ?? "–"}</td>
                      <td className="py-1 pr-2">{c.keyword_rank ?? "–"}</td>
                      <td className="py-1 pr-2">{c.exact_rank ?? "–"}</td>
                      <td className="py-1 pr-2">{c.rrf_score.toFixed(4)}</td>
                      <td className="py-1">{relevant.has(c.chunk_id) ? "✓" : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
