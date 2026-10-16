import type { Metadata } from "next";
import { Suspense } from "react";

import { DocumentDetailView } from "@/components/documents/document-detail-view";
import { Skeleton } from "@/components/ui/skeleton";

export const metadata: Metadata = { title: "Document" };

export default function DocumentPage({ params }: PageProps<"/documents/[documentId]">) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <Suspense fallback={<div className="mx-auto max-w-4xl px-4 py-6"><Skeleton className="h-32 w-full" /></div>}>
        {params.then(({ documentId }) => <DocumentDetailView key={documentId} documentId={documentId} />)}
      </Suspense>
    </div>
  );
}
