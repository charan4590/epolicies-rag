"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { api } from "@/lib/api";
import type { ConversationSummary } from "@/lib/types";

interface ConversationsState {
  conversations: ConversationSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  remove: (id: string) => Promise<void>;
}

const ConversationsContext = createContext<ConversationsState | null>(null);

export function ConversationsProvider({ children }: { children: ReactNode }) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setConversations(await api.listConversations(30));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load conversations.");
    } finally {
      setLoading(false);
    }
  }, []);

  const remove = useCallback(async (id: string) => {
    await api.deleteConversation(id);
    setConversations((items) => items.filter((c) => c.conversation_id !== id));
  }, []);

  useEffect(() => {
    // Initial load of recent conversations from the API.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const value = useMemo(
    () => ({ conversations, loading, error, refresh, remove }),
    [conversations, loading, error, refresh, remove],
  );
  return <ConversationsContext.Provider value={value}>{children}</ConversationsContext.Provider>;
}

export function useConversations(): ConversationsState {
  const context = useContext(ConversationsContext);
  if (!context) throw new Error("useConversations must be used inside ConversationsProvider");
  return context;
}
