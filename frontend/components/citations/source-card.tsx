"use client";

import { ChevronDown, CircleAlert, CircleCheck, CircleDashed, FileText } from "lucide-react";
import Link from "next/link";
import { forwardRef } from "react";

import { SafeLink, SafeMarkdown } from "@/components/chat/markdown";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { Citation } from "@/lib/types";
import { cn } from "@/lib/utils";

const VERIFICATION = {
  supported: { icon: CircleCheck, label: "Verified", className: "text-emerald-700 dark:text-emerald-400" },
  weak: { icon: CircleAlert, label: "Check source", className: "text-amber-700 dark:text-amber-400" },
  unverified: { icon: CircleDashed, label: "Not checked", className: "text-muted-foreground" },
} as const;

interface SourceCardProps {
  citation: Citation;
  open: boolean;
  highlighted: boolean;
  onOpenChange: (open: boolean) => void;
}

export const SourceCard = forwardRef<HTMLDivElement, SourceCardProps>(function SourceCard(
  { citation, open, highlighted, onOpenChange },
  ref,
) {
  const status = VERIFICATION[citation.verification];
  const StatusIcon = status.icon;
  const extraSections = citation.section_paths
    .map((p) => p.join(" › "))
    .filter((p) => p && p !== citation.heading_path.join(" › "));
  const urls = Array.from(
    new Set([...(citation.document_source_url ? [citation.document_source_url] : []), ...citation.source_urls]),
  );

  return (
    <Collapsible open={open} onOpenChange={onOpenChange}>
      <div
        ref={ref}
        id={`source-${citation.label}`}
        className={cn(
          "rounded-lg border bg-card transition-shadow",
          highlighted && "ring-2 ring-primary/40",
        )}
      >
        <CollapsibleTrigger className="flex w-full items-start gap-3 rounded-lg p-3 text-left hover:bg-muted/40">
          <span className="mt-0.5 inline-flex h-5 min-w-7 items-center justify-center rounded-md bg-primary/10 px-1 font-mono text-[11px] font-semibold text-primary">
            {citation.label}
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{citation.document_title}</p>
            <p className="truncate text-xs text-muted-foreground">
              {citation.heading_path.length ? citation.heading_path.join(" › ") : "Top of document"}
              {extraSections.length > 0 && ` (+${extraSections.length} more section${extraSections.length > 1 ? "s" : ""})`}
            </p>
          </div>
          <span className={cn("hidden shrink-0 items-center gap-1 text-xs sm:flex", status.className)}>
            <StatusIcon className="size-3.5" aria-hidden />
            {status.label}
          </span>
          <ChevronDown className={cn("mt-0.5 size-4 shrink-0 text-muted-foreground transition-transform", open && "rotate-180")} aria-hidden />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="space-y-3 border-t px-3 pt-3 pb-3">
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Link href={`/documents/${citation.document_id}`} className="inline-flex items-center gap-1 hover:text-foreground hover:underline">
                <FileText className="size-3.5" aria-hidden />
                {citation.relative_path}
              </Link>
              <Badge variant="outline" className={cn("sm:hidden", status.className)}>
                {status.label}
              </Badge>
            </div>
            {extraSections.length > 0 && (
              <p className="text-xs text-muted-foreground">Also covers: {extraSections.join("; ")}</p>
            )}
            <div className="max-h-80 overflow-y-auto rounded-md bg-muted/40 p-3">
              <SafeMarkdown compact>{citation.passage}</SafeMarkdown>
            </div>
            {urls.length > 0 && (
              <ul className="space-y-1 text-xs">
                {urls.map((url) => (
                  <li key={url}>
                    <SafeLink url={url} />
                  </li>
                ))}
              </ul>
            )}
          </div>
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
});
