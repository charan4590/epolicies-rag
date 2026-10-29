import { BookOpenCheck, Quote, Search, ShieldCheck } from "lucide-react";

import { APP_LOGO, CompanyLogo } from "@/components/layout/brand-mark";

const FEATURES = [
  { icon: Search, title: "Hybrid search", text: "Semantic and keyword retrieval over every indexed policy." },
  { icon: Quote, title: "Cited answers", text: "Each statement links to the passage it came from." },
  { icon: ShieldCheck, title: "Verified", text: "Citations are checked before the answer is shown." },
];

export function Welcome() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col items-center px-4 py-10 text-center sm:py-16">
      <CompanyLogo className="mb-6 h-9" />
      {APP_LOGO ? (
        // eslint-disable-next-line @next/next/no-img-element -- small local logo file
        <img src={APP_LOGO} alt="" aria-hidden className="mb-4 size-20 rounded-2xl object-contain" />
      ) : (
        <div className="mb-4 flex size-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
          <BookOpenCheck className="size-5.5" aria-hidden />
        </div>
      )}
      <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">ePolicies AI Assistant</h1>
      <p className="mt-2 max-w-xl text-sm text-balance text-muted-foreground sm:text-base">
        Ask questions about company policies. Answers come only from the indexed policy documents, with sources
        you can check — and the assistant tells you when the documents don&apos;t cover something.
      </p>

      <div className="mt-10 grid w-full gap-4 text-left sm:grid-cols-3">
        {FEATURES.map(({ icon: Icon, title, text }) => (
          <div key={title} className="flex gap-3 sm:block">
            <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground sm:mb-2" aria-hidden />
            <div>
              <p className="text-sm font-medium">{title}</p>
              <p className="text-xs text-muted-foreground">{text}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
