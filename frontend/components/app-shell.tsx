"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Activity, ArrowLeftRight, LogOut, ShieldAlert, ShieldCheck, Users } from "lucide-react";
import { getToken, getUser, logout, type User } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Tableau de bord", icon: Activity, role: "ANALYSTE" },
  { href: "/transactions", label: "Transactions", icon: ArrowLeftRight, role: "ANALYSTE" },
  { href: "/alertes", label: "Alertes & dossiers", icon: ShieldAlert, role: "ANALYSTE" },
  { href: "/audit", label: "Audit", icon: ShieldCheck, role: "SUPERVISEUR" },
  { href: "/utilisateurs", label: "Utilisateurs", icon: Users, role: "ADMIN" },
] as const;
const RANK = { ANALYSTE: 0, SUPERVISEUR: 1, ADMIN: 2 };
const ROLE_LABEL = { ANALYSTE: "Analyste", SUPERVISEUR: "Superviseur", ADMIN: "Administrateur" };

export function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const path = usePathname();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    if (!getToken()) router.replace("/login");
    else setUser(getUser());
  }, [router]);

  if (!user) return null;
  const nav = NAV.filter((n) => RANK[user.role] >= RANK[n.role]);

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col bg-sidebar text-sidebar-foreground md:flex">
        <div className="px-5 py-5">
          <div className="text-xs font-semibold uppercase tracking-wider text-white/50">Détection de fraude</div>
          <div className="text-lg font-semibold text-white">RDC · temps réel</div>
        </div>
        <nav className="flex-1 space-y-0.5 px-3">
          {nav.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link key={href} href={href}
                className={cn("flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                  active ? "bg-white/10 font-medium text-white" : "hover:bg-white/5 hover:text-white")}>
                <Icon className="h-4 w-4" /> {label}
              </Link>
            );
          })}
        </nav>
        <div className="border-t border-white/10 p-4 text-sm">
          <div className="truncate font-medium text-white">{user.full_name}</div>
          <div className="truncate text-xs text-white/60">{ROLE_LABEL[user.role]} · {user.email}</div>
          <button onClick={logout} className="mt-3 flex items-center gap-2 text-xs text-white/70 hover:text-white">
            <LogOut className="h-3.5 w-3.5" /> Se déconnecter
          </button>
        </div>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b bg-card px-4 py-3 md:hidden">
          <span className="font-semibold">Fraude RDC</span>
          <button onClick={logout} className="text-sm text-muted-foreground">Déconnexion</button>
        </header>
        <nav className="flex gap-1 overflow-x-auto border-b bg-card px-2 py-2 md:hidden">
          {nav.map(({ href, label }) => (
            <Link key={href} href={href} className={cn("whitespace-nowrap rounded-md px-3 py-1.5 text-sm",
              (href === "/" ? path === "/" : path.startsWith(href)) ? "bg-muted font-medium" : "text-muted-foreground")}>{label}</Link>
          ))}
        </nav>
        <main className="mx-auto w-full max-w-[1400px] flex-1 p-4 md:p-6">{children}</main>
      </div>
    </div>
  );
}

export function PageHeader({ title, subtitle, children }: { title: string; subtitle?: string; children?: React.ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-muted-foreground">{subtitle}</p>}
      </div>
      <div className="flex flex-wrap items-center gap-2">{children}</div>
    </div>
  );
}

export function LiveDot({ error }: { error?: string | null }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-muted-foreground">
      <span className={cn("h-2 w-2 rounded-full", error ? "bg-risk-critique" : "animate-pulse bg-risk-faible")} />
      {error ? `Hors ligne : ${error}` : "Temps réel"}
    </span>
  );
}
