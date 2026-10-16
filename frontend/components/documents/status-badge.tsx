import { Badge } from "@/components/ui/badge";
import type { DocumentStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

const STYLES: Record<DocumentStatus, { label: string; className: string }> = {
  indexed: { label: "Indexed", className: "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300" },
  failed: { label: "Failed", className: "border-red-200 bg-red-50 text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300" },
  pending: { label: "Pending", className: "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-300" },
  deleted: { label: "Deleted", className: "text-muted-foreground" },
};

export function StatusBadge({ status, empty }: { status: DocumentStatus; empty?: boolean | null }) {
  const style = STYLES[status];
  return (
    <span className="inline-flex items-center gap-1">
      <Badge variant="outline" className={cn(style.className)}>{style.label}</Badge>
      {empty && status === "indexed" && <Badge variant="outline" className="text-muted-foreground">Empty</Badge>}
    </span>
  );
}

export function formatDate(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "—"
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
