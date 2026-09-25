/** Proxy vers le backoffice-api : le navigateur ne parle qu'à ce serveur (pas de CORS,
 *  adresse interne du backoffice résolue à l'exécution via BACKOFFICE_URL). */
import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";
const TARGET = process.env.BACKOFFICE_URL ?? "http://localhost:8003";

async function proxy(req: NextRequest, { params }: { params: { path: string[] } }) {
  const url = `${TARGET}/${params.path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;
  const headers = new Headers();
  for (const h of ["authorization", "content-type", "accept"]) {
    const v = req.headers.get(h);
    if (v) headers.set(h, v);
  }
  const init: RequestInit = { method: req.method, headers, cache: "no-store" };
  if (!["GET", "HEAD"].includes(req.method)) init.body = await req.arrayBuffer();
  try {
    const res = await fetch(url, init);
    return new Response(res.body, { status: res.status, headers: { "content-type": res.headers.get("content-type") ?? "application/json" } });
  } catch {
    return Response.json({ detail: "back-office injoignable" }, { status: 502 });
  }
}

export { proxy as GET, proxy as POST, proxy as PATCH, proxy as PUT, proxy as DELETE };
