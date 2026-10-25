# ePolicies AI Assistant — frontend

Next.js (App Router) + TypeScript + Tailwind CSS + shadcn/ui client for the ePolicies FastAPI backend.
See `../README.md` for full setup. Quick start:

```bash
cp .env.example .env.local   # NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
npm ci
npm run dev                  # http://localhost:3000
npm run check                # typecheck + lint
npm run build
```

Only `NEXT_PUBLIC_API_BASE_URL` belongs in frontend env
files — never API keys or database credentials.
