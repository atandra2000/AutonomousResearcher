import { auth } from "@/auth";
import { serviceHeaders, serviceUrl, unavailableResponse } from "@/lib/service";

const actions = new Set(["cancel", "resume", "result"]);

export async function POST(
  _request: Request,
  context: { params: Promise<{ action: string; runId: string }> }
) {
  const session = await auth();
  if (!session?.user) return Response.json({ detail: "unauthorized" }, { status: 401 });
  const { action, runId } = await context.params;
  if (!actions.has(action) || action === "result") {
    return Response.json({ detail: "unknown run action" }, { status: 404 });
  }
  try {
    const upstream = await fetch(
      serviceUrl(`/runs/${encodeURIComponent(runId)}/${action}`),
      { method: "POST", headers: serviceHeaders(), cache: "no-store" }
    );
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return unavailableResponse(error);
  }
}

export async function GET(
  _request: Request,
  context: { params: Promise<{ action: string; runId: string }> }
) {
  const session = await auth();
  if (!session?.user) return Response.json({ detail: "unauthorized" }, { status: 401 });
  const { action, runId } = await context.params;
  if (action !== "result") {
    return Response.json({ detail: "unknown run action" }, { status: 404 });
  }
  try {
    const upstream = await fetch(
      serviceUrl(`/runs/${encodeURIComponent(runId)}/result`),
      { headers: serviceHeaders(), cache: "no-store" }
    );
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers: { "Content-Type": "application/json" }
    });
  } catch (error) {
    return unavailableResponse(error);
  }
}
