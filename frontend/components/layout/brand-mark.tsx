/* eslint-disable @next/next/no-img-element -- small local logo files; next/image adds nothing here */
import { ShieldCheck } from "lucide-react";

import { cn } from "@/lib/utils";

// Optional logo files in frontend/public/branding/ (not committed), set in frontend/.env.local:
//   NEXT_PUBLIC_APP_LOGO=/branding/app-logo.png          the assistant's icon (sidebar, home, answers, tab)
//   NEXT_PUBLIC_COMPANY_LOGO=/branding/company-logo.png  the company logo (sidebar and home page)
export const APP_LOGO = process.env.NEXT_PUBLIC_APP_LOGO || "";
export const COMPANY_LOGO = process.env.NEXT_PUBLIC_COMPANY_LOGO || "";

/** The assistant's icon: the configured logo image, or the default shield icon. */
export function BrandMark({ className, iconClassName }: { className?: string; iconClassName?: string }) {
  if (APP_LOGO) {
    return <img src={APP_LOGO} alt="" aria-hidden className={cn("shrink-0 rounded-lg object-contain", className)} />;
  }
  return (
    <div className={cn("flex shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground", className)}>
      <ShieldCheck className={iconClassName} aria-hidden />
    </div>
  );
}

/** The company logo, when configured (nothing otherwise). */
export function CompanyLogo({ className }: { className?: string }) {
  if (!COMPANY_LOGO) return null;
  return <img src={COMPANY_LOGO} alt="Company logo" className={cn("w-auto object-contain", className)} />;
}
