"use client";

import { Menu } from "lucide-react";
import { BrandMark } from "@/components/layout/brand-mark";
import { Suspense, useState, type ReactNode } from "react";

import { Sidebar, SidebarFallback } from "@/components/layout/sidebar";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex h-dvh overflow-hidden">
      <aside className="hidden w-64 shrink-0 border-r md:block lg:w-72">
        {/* Sidebar reads the URL (usePathname), which is request-time data under Cache Components. */}
        <Suspense fallback={<SidebarFallback />}>
          <Sidebar />
        </Suspense>
      </aside>

      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="left" className="w-72 p-0" showCloseButton={false}>
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <SheetDescription className="sr-only">Chat history and knowledge base navigation</SheetDescription>
          <Suspense fallback={<SidebarFallback />}>
            <Sidebar onNavigate={() => setOpen(false)} />
          </Suspense>
        </SheetContent>
      </Sheet>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b px-3 md:hidden">
          <Button variant="ghost" size="icon-sm" onClick={() => setOpen(true)} aria-label="Open navigation">
            <Menu />
          </Button>
          <BrandMark className="size-6" iconClassName="size-3.5" />
          <span className="text-sm font-semibold">ePolicies AI Assistant</span>
        </header>
        <main className="flex min-h-0 flex-1 flex-col">{children}</main>
      </div>
    </div>
  );
}
