"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { AlertCircle, ArrowRight, FileSearch, Gauge, Lock, Mail, ShieldCheck } from "lucide-react";
import { Logo, LogoMark } from "@/components/logo";
import { ThemeToggle } from "@/components/theme";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/select";
import { login } from "@/lib/api";

const POINTS = [
  { icon: Gauge, title: "Décision en une vingtaine de millisecondes", text: "Approuver, vérifier ou bloquer avant que l'opérateur n'exécute la transaction." },
  { icon: ShieldCheck, title: "Chaque décision est expliquée", text: "Variables décisives, part de chaque modèle, journal d'audit inaltérable." },
  { icon: FileSearch, title: "Assistant d'enquête", text: "Cas passés semblables et procédures internes pour instruire chaque alerte." },
];

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
    <div className="grid min-h-screen lg:grid-cols-[1.05fr_1fr]">
      {/* Panneau de marque : toujours sombre, quel que soit le thème */}
      <aside className="relative hidden overflow-hidden bg-[#0b1322] p-10 text-white lg:flex lg:flex-col">
        <div aria-hidden className="absolute inset-0 opacity-[0.35]"
             style={{ backgroundImage: "radial-gradient(rgba(125,170,255,.18) 1px, transparent 1px)", backgroundSize: "22px 22px" }} />
        <div aria-hidden className="absolute -right-32 -top-32 h-[420px] w-[420px] rounded-full bg-[#0a6cd6] opacity-20 blur-[120px]" />
        <Logo inverted className="relative" />
        <div className="relative mt-auto max-w-md">
          <p className="label-caps !text-[#7fb4ff]">Mobile Money · Visa virtuelle</p>
          <h1 className="mt-3 text-[32px] font-semibold leading-[1.15] tracking-tight">
            Supervision temps réel de la fraude Mobile Money en RDC
          </h1>
          <ul className="mt-8 space-y-5">
            {POINTS.map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex gap-3.5">
                <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-white/[0.07] ring-1 ring-white/10">
                  <Icon className="h-4 w-4 text-[#7fb4ff]" />
                </div>
                <div>
                  <p className="text-sm font-medium">{title}</p>
                  <p className="mt-0.5 text-[13px] leading-relaxed text-white/60">{text}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
        <p className="relative mt-12 text-2xs text-white/40">Prototype de recherche · données synthétiques · Vodacom, Airtel, Orange et Visa sont des formats d'API simulés.</p>
      </aside>

      <main className="relative flex items-center justify-center bg-background px-5 py-12">
        <ThemeToggle className="absolute right-5 top-5" />
        <div className="w-full max-w-[380px]">
          <div className="mb-8 lg:hidden"><Logo /></div>
          <LogoMark className="hidden h-10 w-10 lg:block" />
          <h2 className="mt-5 text-2xl font-semibold tracking-tight">Connexion</h2>
          <p className="mt-1.5 text-[13px] text-muted-foreground">Espace réservé aux analystes, superviseurs et administrateurs.</p>

          <form onSubmit={submit} className="mt-7 space-y-4" noValidate={false}>
            <div>
              <Label htmlFor="email">Adresse e-mail</Label>
              <Input id="email" icon={Mail} type="email" autoComplete="username" required placeholder="prenom.nom@operateur.cd"
                     value={email} onChange={(e) => setEmail(e.target.value)} />
            </div>
            <div>
              <Label htmlFor="password">Mot de passe</Label>
              <Input id="password" icon={Lock} type="password" autoComplete="current-password" required
                     value={password} onChange={(e) => setPassword(e.target.value)} />
            </div>
            {error && (
              <p role="alert" className="flex items-start gap-2 rounded-md border border-risk-critique/25 bg-risk-critique/10 px-3 py-2.5 text-[13px] text-risk-critique">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{error}
              </p>
            )}
            <Button type="submit" size="lg" className="w-full" disabled={busy}>
              {busy ? "Connexion…" : <>Se connecter <ArrowRight className="h-4 w-4" /></>}
            </Button>
          </form>
          <p className="mt-8 text-center text-2xs text-subtle">Session de 8 heures · toute action est inscrite au journal d'audit.</p>
        </div>
      </main>
    </div>
  );
}
