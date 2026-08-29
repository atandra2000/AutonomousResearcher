import { auth } from "@/auth";
import { serviceHeaders, serviceUrl, unavailableResponse } from "@/lib/service";

export async function POST(request: Request) {
  const session = await auth();
  if (!session?.user) return Response.json({ detail: "unauthorized" }, { status: 401 });
  const body = await request.json().catch(() => null);
  if (!body || typeof body.goal !== "string" || !body.goal.trim() || body.goal.length > 10_000) {
    return Response.json({ detail: "goal must be 1-10000 characters" }, { status: 422 });
  }
  try {
    const upstream = await fetch(serviceUrl("/runs"), {
      method: "POST",
      headers: serviceHeaders(true),
      body: JSON.stringify({
        goal: body.goal.trim(),
        metadata: {
          source: "web-console",
          operator: session.user.email ?? session.user.name ?? "unknown"
        }
      }),
      cache: "no-store"
    });
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return unavailableResponse(error);
  }
}
