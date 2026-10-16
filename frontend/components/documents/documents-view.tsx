"use client";

import { ChevronLeft, ChevronRight, FileWarning, FolderSearch, RefreshCw, Search } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { formatDate, StatusBadge } from "@/components/documents/status-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api } from "@/lib/api";
import type { DocumentList, DocumentStatus, SourceStatus } from "@/lib/types";

const PAGE_SIZE = 25;
type StatusFilter = DocumentStatus | "all";

function useDebounced<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

function PendingFiles({ status }: { status: SourceStatus }) {
  const groups = [
    { label: "New (not yet indexed)", items: status.new },
    { label: "Modified since last indexing", items: status.modified },
    { label: "Deleted from disk (still indexed)", items: status.deleted },
    { label: "Unreadable", items: status.unreadable.map((u) => `${u.path} — ${u.error}`) },
  ].filter((g) => g.items.length > 0);
  if (!groups.length) return null;
  return (
    <Alert>
      <FolderSearch aria-hidden />
      <AlertTitle>Files on disk that are not reflected in the index</AlertTitle>
      <AlertDescription>
        <div className="mt-1 space-y-2">
          {groups.map((group) => (
            <div key={group.label}>
              <p className="font-medium text-foreground">{group.label} ({group.items.length})</p>
              <ul className="mt-0.5 max-h-32 overflow-y-auto font-mono text-xs">
                {group.items.slice(0, 50).map((item) => <li key={item} className="truncate">{item}</li>)}
              </ul>
            </div>
          ))}
          <p className="text-xs">
            Run <code className="rounded bg-muted px-1">python -m app.cli ingest</code> in <code className="rounded bg-muted px-1">backend/</code> to process them.
          </p>
        </div>
      </AlertDescription>
    </Alert>
  );
}

export function DocumentsView() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<DocumentList | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [source, setSource] = useState<SourceStatus | null>(null);
  const [sourceError, setSourceError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const debouncedQuery = useDebounced(query);

  useEffect(() => {
    let active = true;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setLoading(true);
    api
      .listDocuments({ page, pageSize: PAGE_SIZE, q: debouncedQuery.trim(), status: status === "all" ? "" : status })
      .then((result) => {
        if (!active) return;
        setData(result);
        setError(null);
      })
      .catch((err: unknown) => active && setError(err instanceof Error ? err.message : "Could not load documents."))
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [page, debouncedQuery, status, reloadKey]);

  useEffect(() => {
    let active = true;
    api
      .sourceStatus()
      .then((result) => active && (setSource(result), setSourceError(null)))
      .catch((err: unknown) => active && setSourceError(err instanceof Error ? err.message : "Unavailable"));
    return () => {
      active = false;
    };
  }, [reloadKey]);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div className="mx-auto w-full max-w-6xl space-y-5 px-4 py-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Knowledge base</h1>
          <p className="text-sm text-muted-foreground">Documents indexed in PostgreSQL and their ingestion status.</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => setReloadKey((k) => k + 1)}>
          <RefreshCw /> Refresh
        </Button>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <Card size="sm">
          <CardHeader>
            <CardDescription>Documents shown</CardDescription>
            <CardTitle className="text-2xl">{data ? data.total : "—"}</CardTitle>
          </CardHeader>
        </Card>
        <Card size="sm">
          <CardHeader>
            <CardDescription>Files on disk</CardDescription>
            <CardTitle className="text-2xl">{source?.source_dir_exists ? source.discovered : "—"}</CardTitle>
          </CardHeader>
        </Card>
        <Card size="sm">
          <CardHeader>
            <CardDescription>Awaiting ingestion</CardDescription>
            <CardTitle className="text-2xl">
              {source?.source_dir_exists ? source.new.length + source.modified.length + source.deleted.length : "—"}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      {source && !source.source_dir_exists && (
        <Alert variant="destructive">
          <FileWarning aria-hidden />
          <AlertTitle>Markdown directory not found</AlertTitle>
          <AlertDescription>{source.error} Set MARKDOWN_DIR in the backend .env file.</AlertDescription>
        </Alert>
      )}
      {sourceError && <p className="text-xs text-muted-foreground">Source directory status unavailable: {sourceError}</p>}
      {source?.source_dir_exists && <PendingFiles status={source} />}

      <Card>
        <CardContent className="space-y-4">
          <div className="flex flex-col gap-2 sm:flex-row">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
              <Input
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setPage(1);
                }}
                placeholder="Filter by title or path"
                className="pl-8"
                aria-label="Filter documents by title or path"
              />
            </div>
            <Select
              value={status}
              onValueChange={(value) => {
                setStatus(value as StatusFilter);
                setPage(1);
              }}
            >
              <SelectTrigger className="w-full sm:w-44" aria-label="Filter by status">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All active</SelectItem>
                <SelectItem value="indexed">Indexed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="deleted">Deleted</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {error ? (
            <Alert variant="destructive">
              <AlertTitle>Could not load documents</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : loading && !data ? (
            <div className="space-y-2">
              {Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-9 w-full" />)}
            </div>
          ) : data && data.items.length === 0 ? (
            <div className="py-10 text-center text-sm text-muted-foreground">
              {debouncedQuery || status !== "all"
                ? "No documents match these filters."
                : "No documents have been indexed yet. Add Markdown files to MARKDOWN_DIR and run ingestion."}
            </div>
          ) : data ? (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Document</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Chunks</TableHead>
                    <TableHead className="hidden md:table-cell">Last indexed</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody className={loading ? "opacity-60" : undefined}>
                  {data.items.map((doc) => (
                    <TableRow key={doc.document_id}>
                      <TableCell className="max-w-0 min-w-56">
                        <Link href={`/documents/${doc.document_id}`} className="block truncate font-medium hover:underline">
                          {doc.title}
                        </Link>
                        <span className="block truncate font-mono text-xs text-muted-foreground">{doc.relative_path}</span>
                        {doc.status === "failed" && doc.last_error && (
                          <span className="block truncate text-xs text-destructive" title={doc.last_error}>{doc.last_error}</span>
                        )}
                      </TableCell>
                      <TableCell><StatusBadge status={doc.status} empty={doc.is_empty} /></TableCell>
                      <TableCell className="text-right tabular-nums">{doc.chunk_count}</TableCell>
                      <TableCell className="hidden text-muted-foreground md:table-cell">{formatDate(doc.last_indexed_at)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : null}

          {data && data.total > PAGE_SIZE && (
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <span>
                Page {page} of {totalPages}
              </span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                  <ChevronLeft /> Previous
                </Button>
                <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                  Next <ChevronRight />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
