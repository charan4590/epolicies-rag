# Logos (local only)

Put your logo files here – this folder's images are not committed:

- `app-logo.png` – the assistant's icon (sidebar, home page, answers, browser tab). Square works best.
- `company-logo.png` – the company logo (sidebar and home page). Wide works best.

Then add to `frontend/.env.local`:

```
NEXT_PUBLIC_APP_LOGO=/branding/app-logo.png
NEXT_PUBLIC_COMPANY_LOGO=/branding/company-logo.png
```

Restart `npm run dev` (environment values are read at start). Without these lines the default icon is used.
