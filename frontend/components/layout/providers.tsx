"use client";

import { ThemeProvider } from "next-themes";
import type { ReactNode } from "react";

import { ConversationsProvider } from "@/components/layout/conversations-context";
import { TooltipProvider } from "@/components/ui/tooltip";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem disableTransitionOnChange>
      <TooltipProvider delayDuration={300}>
        <ConversationsProvider>{children}</ConversationsProvider>
      </TooltipProvider>
    </ThemeProvider>
  );
}
