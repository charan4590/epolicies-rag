import { Skeleton } from "@/components/ui/skeleton";

export function ChatSkeleton() {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="h-12 shrink-0 border-b" />
      <div className="mx-auto w-full max-w-3xl flex-1 space-y-4 px-4 py-8">
        <Skeleton className="ml-auto h-10 w-2/3" />
        <Skeleton className="h-28 w-full" />
      </div>
    </div>
  );
}
