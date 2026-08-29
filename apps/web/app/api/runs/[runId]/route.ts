import { auth } from "@/auth";
import { serviceHeaders, serviceUrl, unavailableResponse } from "@/lib/service";

export async function GET(_request: Request, context: { params: Promise<{ runId: string }> }) {
  const session = await auth();
  if (!session?.user) return Response.json({ detail: "unauthorized" }, { status: 401 });
  const { runId } = await context.params;
  try {
    const upstream = await fetch(serviceUrl(`/runs/${encodeURIComponent(runId)}`), {
      headers: serviceHeaders(),
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
