"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ShieldCheck } from "lucide-react";
import { login } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/select";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setError(null);
    try { await login(email, password); router.replace("/"); }
    catch (err: any) { setError(err.message); }
    finally { setBusy(false); }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-sidebar p-4">
      <form onSubmit={submit} className="w-full max-w-sm rounded-xl bg-card p-7 shadow-xl">
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-lg bg-primary/10 p-2 text-primary"><ShieldCheck className="h-6 w-6" /></div>
          <div>
            <h1 className="font-semibold">Détection de fraude RDC</h1>
            <p className="text-xs text-muted-foreground">Mobile Money &amp; cartes Visa virtuelles</p>
          </div>
        </div>
        <label className="mb-1 block text-sm font-medium" htmlFor="email">E-mail</label>
        <Input id="email" type="email" autoComplete="username" required value={email} onChange={(e) => setEmail(e.target.value)} />
        <label className="mb-1 mt-4 block text-sm font-medium" htmlFor="password">Mot de passe</label>
        <Input id="password" type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
        {error && <p className="mt-3 rounded-md bg-risk-critique/10 px-3 py-2 text-sm text-risk-critique">{error}</p>}
        <Button type="submit" className="mt-6 w-full" disabled={busy}>{busy ? "Connexion…" : "Se connecter"}</Button>
      </form>
    </div>
  );
}
