import type { Metadata, Viewport } from "next";

import { AppShell } from "@/components/layout/app-shell";
import { Providers } from "@/components/layout/providers";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "ePolicies AI Assistant", template: "%s · ePolicies AI Assistant" },
  description: "Ask questions about company policies and get answers with verified citations.",
  robots: { index: false, follow: false },
  // Browser-tab icon: the assistant logo when configured (frontend/.env.local), else app/favicon.ico
  ...(process.env.NEXT_PUBLIC_APP_LOGO ? { icons: { icon: process.env.NEXT_PUBLIC_APP_LOGO } } : {}),
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
