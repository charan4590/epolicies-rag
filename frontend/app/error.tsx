"use client";

import { Button } from "@/components/ui/button";

export default function ErrorBoundary({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div className="mx-auto max-w-md px-4 py-20 text-center">
      <h1 className="text-lg font-semibold">Something went wrong</h1>
      <p className="mt-2 text-sm text-muted-foreground">An unexpected error occurred while rendering this page.</p>
      <Button variant="outline" size="sm" className="mt-5" onClick={reset}>
        Try again
      </Button>
    </div>
  );
}
