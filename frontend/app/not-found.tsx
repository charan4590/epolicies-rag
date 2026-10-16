import Link from "next/link";

import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-md px-4 py-20 text-center">
      <h1 className="text-lg font-semibold">Page not found</h1>
      <p className="mt-2 text-sm text-muted-foreground">The page you are looking for does not exist.</p>
      <Button asChild variant="outline" size="sm" className="mt-5">
        <Link href="/">Back to chat</Link>
      </Button>
    </div>
  );
}
