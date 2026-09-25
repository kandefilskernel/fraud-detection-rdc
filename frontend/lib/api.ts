"use client";

/** Client de l'API back-office (via le proxy Next.js /api -> backoffice-api). */
const TOKEN_KEY = "fraud-rdc-token";
const USER_KEY = "fraud-rdc-user";

export type User = { id: number; email: string; full_name: string; role: "ANALYSTE" | "SUPERVISEUR" | "ADMIN" };

export function getToken(): string | null {
  try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
}
export function getUser(): User | null {
  try { const raw = localStorage.getItem(USER_KEY); return raw ? JSON.parse(raw) : null; } catch { return null; }
}
export function logout() {
  try { localStorage.removeItem(TOKEN_KEY); localStorage.removeItem(USER_KEY); } catch { /* navigation privée */ }
  window.location.href = "/login";
}

export class ApiError extends Error {
  constructor(public status: number, message: string) { super(message); }
}

export async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}), ...init.headers },
    cache: "no-store",
  });
  if (res.status === 401 && !path.startsWith("/auth/login")) { logout(); throw new ApiError(401, "session expirée"); }
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = typeof body === "object" && body?.detail ? body.detail : body;
    throw new ApiError(res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body as T;
}

export async function login(email: string, password: string): Promise<User> {
  const form = new URLSearchParams({ username: email, password });
  const res = await fetch("/api/auth/login", {
    method: "POST", body: form, headers: { "Content-Type": "application/x-www-form-urlencoded" },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, data.detail ?? "connexion impossible");
  localStorage.setItem(TOKEN_KEY, data.access_token);
  localStorage.setItem(USER_KEY, JSON.stringify(data.user));
  return data.user;
}
