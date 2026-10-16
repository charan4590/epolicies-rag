import type { Metadata } from "next";

import { DocumentsView } from "@/components/documents/documents-view";

export const metadata: Metadata = { title: "Knowledge base" };

export default function DocumentsPage() {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <DocumentsView />
    </div>
  );
}
