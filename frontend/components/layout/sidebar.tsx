"use client";

import { FileText, MessageSquare, MessageSquarePlus, Trash2 } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";

import { ConnectionStatus } from "@/components/layout/connection-status";
import { useConversations } from "@/components/layout/conversations-context";
import { ThemeToggle } from "@/components/layout/theme-toggle";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { BrandMark, CompanyLogo } from "@/components/layout/brand-mark";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Chat", icon: MessageSquare, match: (p: string) => p === "/" || p.startsWith("/chat") },
  { href: "/documents", label: "Knowledge base", icon: FileText, match: (p: string) => p.startsWith("/documents") },
];

function Brand() {
  return (
    <div className="space-y-3 px-4 pt-4 pb-3">
      <CompanyLogo className="h-6" />
      <div className="flex items-center gap-2.5">
        <BrandMark className="size-8" iconClassName="size-4.5" />
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold leading-tight">ePolicies AI Assistant</p>
          <p className="truncate text-xs text-muted-foreground">Policy knowledge base</p>
        </div>
      </div>
    </div>
  );
}

/** Static placeholder rendered while the URL-dependent sidebar streams in. */
export function SidebarFallback() {
  return (
    <div className="flex h-full flex-col bg-sidebar">
      <Brand />
      <div className="space-y-2 px-3">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-6 w-4/5" />
        <Skeleton className="h-6 w-3/5" />
      </div>
    </div>
  );
}

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  const router = useRouter();
  const { conversations, loading, error, remove } = useConversations();
  const [deleting, setDeleting] = useState<string | null>(null);

  const handleDelete = async (id: string) => {
    setDeleting(id);
    try {
      await remove(id);
      if (pathname === `/chat/${id}`) router.push("/");
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="flex h-full flex-col bg-sidebar text-sidebar-foreground">
      <Brand />

      <div className="px-3">
        <Button asChild className="w-full justify-start" variant="outline">
          <Link href="/" onClick={onNavigate}>
            <MessageSquarePlus /> New chat
          </Link>
        </Button>
      </div>

      <nav className="mt-3 space-y-0.5 px-3" aria-label="Main">
        {NAV.map(({ href, label, icon: Icon, match }) => (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            className={cn(
              "flex items-center gap-2 rounded-md px-2.5 py-1.5 text-sm transition-colors",
              match(pathname)
                ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
            )}
            aria-current={match(pathname) ? "page" : undefined}
          >
            <Icon className="size-4" aria-hidden />
            {label}
          </Link>
        ))}
      </nav>

      <Separator className="my-3" />
      <p className="px-5 pb-1.5 text-[11px] font-medium tracking-wide text-muted-foreground uppercase">Recent</p>
      <ScrollArea className="min-h-0 flex-1 px-3">
        {loading ? (
          <div className="space-y-2 px-2 py-1">
            <Skeleton className="h-4 w-4/5" />
            <Skeleton className="h-4 w-3/5" />
          </div>
        ) : error ? (
          <p className="px-2 text-xs text-muted-foreground">History unavailable.</p>
        ) : conversations.length === 0 ? (
          <p className="px-2 text-xs text-muted-foreground">No conversations yet.</p>
        ) : (
          <ul className="space-y-0.5 pb-2">
            {conversations.map((c) => {
              const active = pathname === `/chat/${c.conversation_id}`;
              return (
                <li key={c.conversation_id} className="group relative">
                  <Link
                    href={`/chat/${c.conversation_id}`}
                    onClick={onNavigate}
                    title={c.title}
                    className={cn(
                      "block truncate rounded-md py-1.5 pr-8 pl-2.5 text-sm transition-colors",
                      active
                        ? "bg-sidebar-accent text-sidebar-accent-foreground"
                        : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
                    )}
                  >
                    {c.title}
                  </Link>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    className="absolute top-1/2 right-1 -translate-y-1/2 opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                    aria-label={`Delete conversation ${c.title}`}
                    disabled={deleting === c.conversation_id}
                    onClick={() => void handleDelete(c.conversation_id)}
                  >
                    <Trash2 />
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
      </ScrollArea>

      <div className="flex items-center justify-between gap-2 border-t px-4 py-3">
        <ConnectionStatus className="min-w-0" />
        <ThemeToggle />
      </div>
    </div>
  );
}
