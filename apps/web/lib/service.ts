export function serviceUrl(path: string): URL {
  const base = process.env.RE_SERVICE_API_URL;
  if (!base) throw new Error("RE_SERVICE_API_URL is not configured");
  return new URL(path, base);
}

export function serviceHeaders(contentType = false): HeadersInit {
  const token = process.env.RE_SERVICE_API_TOKEN;
  if (!token) throw new Error("RE_SERVICE_API_TOKEN is not configured");
  return {
    Authorization: `Bearer ${token}`,
    ...(contentType ? { "Content-Type": "application/json" } : {})
  };
}

export function unavailableResponse(error: unknown): Response {
  const detail = error instanceof Error ? error.message : "service is not configured";
  return Response.json({ detail }, { status: 503 });
}
