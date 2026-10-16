import type { Metadata, Viewport } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { Providers } from "@/components/layout/providers";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "ePolicies AI Assistant", template: "%s · ePolicies AI Assistant" },
  description: "Ask questions about company policies and get answers with verified citations.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0a0a" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className="h-full antialiased">
      <body className="h-full">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
