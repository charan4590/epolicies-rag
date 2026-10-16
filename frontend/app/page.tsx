import { Suspense } from "react";

import { ChatSkeleton } from "@/components/chat/chat-skeleton";
import { ChatWorkspace } from "@/components/chat/chat-workspace";

export default function HomePage() {
  return (
    <Suspense fallback={<ChatSkeleton />}>
      <ChatWorkspace />
    </Suspense>
  );
}
