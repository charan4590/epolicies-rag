"use client";

import { ArrowLeft, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { SafeLink } from "@/components/chat/markdown";
import { formatBytes, formatDate, StatusBadge } from "@/components/documents/status-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, api } from "@/lib/api";
import type { DocumentDetail } from "@/lib/types";

export function DocumentDetailView({ documentId }: { documentId: string }) {
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    api
      .getDocument(documentId)
      .then((d) => active && setDoc(d))
      .catch((err: unknown) =>
        active && setError(err instanceof ApiError && err.status === 404 ? "Document not found." : err instanceof Error ? err.message : "Could not load the document."),
      );
    return () => {
      active = false;
    };
  }, [documentId]);

  const urls = doc ? Array.from(new Set([...(doc.source_url ? [doc.source_url] : []), ...(doc.metadata.urls ?? [])])) : [];

  return (
    <div className="mx-auto w-full max-w-4xl space-y-5 px-4 py-6">
      <Button asChild variant="ghost" size="sm" className="-ml-2">
        <Link href="/documents"><ArrowLeft /> Knowledge base</Link>
      </Button>

      {error ? (
        <Alert variant="destructive">
          <TriangleAlert aria-hidden />
          <AlertTitle>{error}</AlertTitle>
        </Alert>
      ) : !doc ? (
        <div className="space-y-3">
          <Skeleton className="h-7 w-2/3" />
          <Skeleton className="h-32 w-full" />
        </div>
      ) : (
        <>
          <div>
            <h1 className="text-xl font-semibold tracking-tight">{doc.title}</h1>
            <p className="mt-1 font-mono text-xs text-muted-foreground">{doc.relative_path}</p>
          </div>

          {doc.status === "failed" && (
            <Alert variant="destructive">
              <TriangleAlert aria-hidden />
              <AlertTitle>The latest version failed to index{doc.chunk_count > 0 ? " — the previous version is still searchable" : ""}</AlertTitle>
              {doc.last_error && <AlertDescription className="break-words">{doc.last_error}</AlertDescription>}
            </Alert>
          )}

          <Card>
            <CardContent>
              <dl className="grid grid-cols-1 gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
                <div><dt className="text-xs text-muted-foreground">Status</dt><dd className="mt-0.5"><StatusBadge status={doc.status} empty={doc.is_empty} /></dd></div>
                <div><dt className="text-xs text-muted-foreground">Chunks</dt><dd className="mt-0.5 tabular-nums">{doc.chunk_count}</dd></div>
                <div><dt className="text-xs text-muted-foreground">Last indexed</dt><dd className="mt-0.5">{formatDate(doc.last_indexed_at)}</dd></div>
                <div><dt className="text-xs text-muted-foreground">File modified</dt><dd className="mt-0.5">{formatDate(doc.file_modified_at)}</dd></div>
                <div><dt className="text-xs text-muted-foreground">Size / encoding</dt><dd className="mt-0.5">{formatBytes(doc.size_bytes)} · {doc.encoding ?? "—"}</dd></div>
                <div className="min-w-0"><dt className="text-xs text-muted-foreground">Embedding configuration</dt><dd className="mt-0.5 truncate font-mono text-xs" title={doc.embedding_config ?? ""}>{doc.embedding_config ?? "—"}</dd></div>
                {urls.length > 0 && (
                  <div className="sm:col-span-2">
                    <dt className="text-xs text-muted-foreground">Links in document</dt>
                    <dd className="mt-1 space-y-1 text-xs">{urls.slice(0, 20).map((u) => <div key={u}><SafeLink url={u} /></div>)}</dd>
                  </div>
                )}
                {(doc.metadata.warnings ?? []).length > 0 && (
                  <div className="sm:col-span-2">
                    <dt className="text-xs text-muted-foreground">Warnings</dt>
                    <dd className="mt-0.5 text-xs">{(doc.metadata.warnings ?? []).join("; ")}</dd>
                  </div>
                )}
              </dl>
            </CardContent>
          </Card>

          <Card>
            <CardHeader><CardTitle className="text-base">Chunks</CardTitle></CardHeader>
            <CardContent>
              {doc.chunks.length === 0 ? (
                <p className="text-sm text-muted-foreground">This document has no indexed chunks{doc.is_empty ? " (the file is empty)" : ""}.</p>
              ) : (
                <ol className="space-y-3">
                  {doc.chunks.map((chunk) => (
                    <li key={chunk.chunk_id} className="rounded-md border p-3">
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                        <span className="font-mono">#{chunk.chunk_index}</span>
                        <span className="font-medium text-foreground">{chunk.heading_path.join(" › ") || "Top of document"}</span>
                        <span>~{chunk.token_count} tokens</span>
                      </div>
                      <p className="mt-1.5 line-clamp-3 text-sm whitespace-pre-line text-muted-foreground">{chunk.preview}</p>
                    </li>
                  ))}
                </ol>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
