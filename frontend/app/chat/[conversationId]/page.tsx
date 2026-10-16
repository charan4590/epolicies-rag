import type { Metadata } from "next";
import { Suspense } from "react";

import { ChatSkeleton } from "@/components/chat/chat-skeleton";
import { ChatWorkspace } from "@/components/chat/chat-workspace";

export const metadata: Metadata = { title: "Conversation" };

export default function ConversationPage({ params }: PageProps<"/chat/[conversationId]">) {
  return (
    <Suspense fallback={<ChatSkeleton />}>
      {params.then(({ conversationId }) => (
        <ChatWorkspace key={conversationId} conversationId={conversationId} />
      ))}
    </Suspense>
  );
}
