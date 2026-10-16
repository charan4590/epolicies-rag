"use client";

import { ExternalLink } from "lucide-react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { displayUrl, safeExternalUrl } from "@/lib/safe-url";
import { cn } from "@/lib/utils";

const CITE_PREFIX = "#cite-";
const MARKER_RE = /\[(S\d+(?:\s*[,;]\s*S\d+)*)\]/g;

/** Convert `[S1]` / `[S1, S2]` markers into internal anchors rendered as chips.
 * Markers without a matching citation are dropped (never rendered as fake sources). */
export function linkCitationMarkers(text: string, validLabels: Set<string>): string {
  return text.replace(MARKER_RE, (_, group: string) =>
    group
      .split(/\s*[,;]\s*/)
      .filter((label) => validLabels.has(label))
      .map((label) => `[${label}](${CITE_PREFIX}${label})`)
      .join(""),
  );
}

interface SafeMarkdownProps {
  children: string;
  className?: string;
  /** Labels that may be rendered as citation chips. */
  citationLabels?: Set<string>;
  onCitationClick?: (label: string) => void;
  compact?: boolean;
}

/**
 * Renders Markdown without raw HTML (react-markdown escapes/ignores HTML and we
 * add no rehype-raw), with GFM tables, and with links restricted to http(s)/mailto.
 */
export function SafeMarkdown({ children, className, citationLabels, onCitationClick, compact }: SafeMarkdownProps) {
  const text = citationLabels ? linkCitationMarkers(children, citationLabels) : children;

  const components: Components = {
    a: ({ href, children: linkChildren }) => {
      if (href?.startsWith(CITE_PREFIX)) {
        const label = href.slice(CITE_PREFIX.length);
        return (
          <button
            type="button"
            onClick={() => onCitationClick?.(label)}
            className="mx-0.5 inline-flex h-4.5 min-w-6 translate-y-[-1px] items-center justify-center rounded-md border border-primary/25 bg-primary/8 px-1 align-middle font-mono text-[10px] font-semibold text-primary transition-colors hover:bg-primary/15"
            aria-label={`Show source ${label}`}
          >
            {label}
          </button>
        );
      }
      const safe = safeExternalUrl(href);
      if (!safe) return <span>{linkChildren}</span>;
      return (
        <a href={safe} target="_blank" rel="noopener noreferrer nofollow" className="inline-flex items-baseline gap-0.5 font-medium text-primary underline underline-offset-2">
          {linkChildren}
          <ExternalLink className="size-3 self-center" aria-hidden />
        </a>
      );
    },
    table: ({ children: tableChildren }) => (
      <div className="my-3 w-full overflow-x-auto rounded-lg border">
        <table className="w-full border-collapse text-sm">{tableChildren}</table>
      </div>
    ),
    thead: ({ children: c }) => <thead className="bg-muted/60">{c}</thead>,
    th: ({ children: c }) => <th className="border-b px-3 py-2 text-left font-semibold whitespace-nowrap">{c}</th>,
    td: ({ children: c }) => <td className="border-b px-3 py-2 align-top last:border-r-0">{c}</td>,
    img: () => null, // documents contain no images; never load remote images from model output
    code: ({ children: c, className: codeClass }) => (
      <code className={cn("rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]", codeClass)}>{c}</code>
    ),
    pre: ({ children: c }) => <pre className="my-3 overflow-x-auto rounded-lg bg-muted p-3 text-sm [&_code]:bg-transparent [&_code]:p-0">{c}</pre>,
  };

  return (
    <div className={cn("markdown-body", compact && "markdown-compact", className)}>
      <Markdown remarkPlugins={[remarkGfm]} components={components} skipHtml>
        {text}
      </Markdown>
    </div>
  );
}

export function SafeLink({ url, className }: { url: string; className?: string }) {
  const safe = safeExternalUrl(url);
  if (!safe) return <span className={cn("break-all text-muted-foreground", className)}>{url}</span>;
  return (
    <a href={safe} target="_blank" rel="noopener noreferrer nofollow" className={cn("inline-flex items-center gap-1 break-all text-primary hover:underline", className)}>
      <ExternalLink className="size-3 shrink-0" aria-hidden />
      {displayUrl(safe)}
    </a>
  );
}
