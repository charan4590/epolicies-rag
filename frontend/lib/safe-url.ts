const ALLOWED_PROTOCOLS = new Set(["http:", "https:", "mailto:"]);

/** Returns a normalised URL string when it is safe to render as a link, otherwise null. */
export function safeExternalUrl(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const value = raw.trim();
  if (!value || /[\u0000-\u001f\s]/.test(value)) return null;
  try {
    const url = new URL(value);
    return ALLOWED_PROTOCOLS.has(url.protocol) ? url.toString() : null;
  } catch {
    return null;
  }
}

/** Short display form of a URL (host + path). */
export function displayUrl(url: string): string {
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "mailto:") return parsed.pathname;
    const path = parsed.pathname === "/" ? "" : parsed.pathname;
    return `${parsed.host}${path}`;
  } catch {
    return url;
  }
}
