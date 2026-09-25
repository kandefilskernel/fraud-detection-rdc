"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";

/** Charge `path` puis le recharge toutes les `intervalMs` (tableau de bord temps réel). */
export function usePolling<T>(path: string | null, intervalMs = 3000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const pathRef = useRef(path);
  pathRef.current = path;

  const load = useCallback(async () => {
    if (!pathRef.current) return;
    try {
      const d = await api<T>(pathRef.current);
      setData(d); setError(null);
    } catch (e: any) {
      setError(e.message ?? "erreur");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setLoading(true);
    load();
    if (!intervalMs) return;
    const id = setInterval(() => { if (document.visibilityState === "visible") load(); }, intervalMs);
    return () => clearInterval(id);
  }, [path, intervalMs, load]);

  return { data, error, loading, reload: load };
}
