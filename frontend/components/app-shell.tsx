"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { Activity, ArrowLeftRight, FlaskConical, LogOut, MapPinned, Menu, ShieldAlert, ShieldCheck, Users, X } from "lucide-react";
import { Logo } from "@/components/logo";
import { ThemeToggle } from "@/components/theme";
import { getToken, getUser, logout, type User } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";
import { cn } from "@/lib/utils";

type Role = User["role"];
const NAV: { group: string; items: { href: string; label: string; icon: typeof Activity; role: Role; badge?: "cases" }[] }[] = [
  { group: "Supervision", items: [
    { href: "/", label: "Tableau de bord", icon: Activity, role: "ANALYSTE" },
    { href: "/transactions", label: "Transactions", icon: ArrowLeftRight, role: "ANALYSTE" },
    { href: "/alertes", label: "Alertes & dossiers", icon: ShieldAlert, role: "ANALYSTE", badge: "cases" },
    { href: "/carte", label: "Carte des fraudes", icon: MapPinned, role: "ANALYSTE" },
  ] },
  { group: "Conformité", items: [{ href: "/audit", label: "Journal d'audit", icon: ShieldCheck, role: "SUPERVISEUR" }] },
  { group: "Administration", items: [{ href: "/utilisateurs", label: "Utilisateurs", icon: Users, role: "ADMIN" }] },
];
const RANK: Record<Role, number> = { ANALYSTE: 0, SUPERVISEUR: 1, ADMIN: 2 };
const ROLE_LABEL: Record<Role, string> = { ANALYSTE: "Analyste", SUPERVISEUR: "Superviseur", ADMIN: "Administrateur" };

const isActive = (path: string, href: string) => (href === "/" ? path === "/" : path.startsWith(href));
const initials = (name: string) => name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]?.toUpperCase()).join("");

function SidebarContent({ user, path, openCases, onNavigate }:
  { user: User; path: string; openCases?: number; onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-14 items-center px-4"><Logo /></div>
      <nav className="scrollbar-thin flex-1 space-y-5 overflow-y-auto px-3 pt-3">
        {NAV.map(({ group, items }) => {
          const visible = items.filter((i) => RANK[user.role] >= RANK[i.role]);
          if (!visible.length) return null;
          return (
            <div key={group}>
              <div className="label-caps px-2.5 pb-1.5">{group}</div>
              <div className="space-y-0.5">
                {visible.map(({ href, label, icon: Icon, badge }) => {
                  const on = isActive(path, href);
                  return (
                    <Link key={href} href={href} onClick={onNavigate} aria-current={on ? "page" : undefined}
                      className={cn("group flex h-9 items-center gap-2.5 rounded-md px-2.5 text-[13px] transition-colors",
                        on ? "bg-primary-soft font-medium text-primary" : "text-muted-foreground hover:bg-surface-2 hover:text-foreground")}>
                      <Icon className={cn("h-4 w-4", on ? "text-primary" : "text-subtle group-hover:text-foreground")} />
                      <span className="flex-1">{label}</span>
                      {badge === "cases" && !!openCases && (
                        <span className="tabular rounded-full bg-risk-critique/12 px-1.5 text-2xs font-semibold text-risk-critique">
                          {openCases > 999 ? "999+" : openCases}
                        </span>
                      )}
                    </Link>
                  );
                })}
              </div>
            </div>
          );
        })}
      </nav>
      <div className="space-y-3 p-3">
        <div className="flex items-start gap-2 rounded-lg border border-dashed bg-surface-2/60 p-2.5 text-2xs text-muted-foreground">
          <FlaskConical className="mt-px h-3.5 w-3.5 shrink-0 text-subtle" />
          <span>Prototype de recherche · données synthétiques, aucun client réel.</span>
        </div>
        <div className="flex items-center gap-2.5 rounded-lg p-1.5">
          <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-primary/12 text-xs font-semibold text-primary">{initials(user.full_name)}</div>
          <div className="min-w-0 flex-1 leading-tight">
            <div className="truncate text-[13px] font-medium">{user.full_name}</div>
            <div className="truncate text-2xs text-muted-foreground">{ROLE_LABEL[user.role]}</div>
          </div>
          <button onClick={logout} title="Se déconnecter" aria-label="Se déconnecter"
            className="grid h-8 w-8 place-items-center rounded-md text-subtle transition-colors hover:bg-surface-2 hover:text-foreground">
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

function SystemStatus() {
  const { data, error } = usePolling<{ status: string }>("/health", 30000);
  const ok = !error && data?.status === "ok";
  return (
    <span className="hidden items-center gap-2 rounded-full border bg-surface px-2.5 py-1 text-2xs text-muted-foreground sm:inline-flex"
      title={ok ? "Back-office et base de données joignables" : "Back-office ou base de données injoignable"}>
      <span className={cn("relative flex h-2 w-2")}>
        {ok && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-risk-faible opacity-50" />}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", ok ? "bg-risk-faible" : data || error ? "bg-risk-critique" : "bg-subtle")} />
      </span>
      {ok ? "Services opérationnels" : data || error ? "Service dégradé" : "Vérification…"}
    </span>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const path = usePathname();
  const [user, setUser] = useState<User | null>(null);
  const [drawer, setDrawer] = useState(false);
  const cases = usePolling<any>(user ? "/cases?status=OUVERT&size=1" : null, 15000);

  useEffect(() => {
    if (!getToken()) router.replace("/login");
    else setUser(getUser());
  }, [router]);
  useEffect(() => setDrawer(false), [path]);

  if (!user) return null;
  const openCases = cases.data?.counts?.OUVERT;
  const section = NAV.flatMap((g) => g.items).find((i) => isActive(path, i.href));

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 border-r bg-surface lg:block">
        <SidebarContent user={user} path={path} openCases={openCases} />
      </aside>

      {drawer && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
          <button aria-label="Fermer le menu" className="absolute inset-0 bg-black/40 backdrop-blur-[1px]" onClick={() => setDrawer(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 max-w-[85%] animate-fade-in border-r bg-surface shadow-pop">
            <button onClick={() => setDrawer(false)} aria-label="Fermer le menu"
              className="absolute right-3 top-3.5 grid h-7 w-7 place-items-center rounded-md text-subtle hover:bg-surface-2"><X className="h-4 w-4" /></button>
            <SidebarContent user={user} path={path} openCases={openCases} onNavigate={() => setDrawer(false)} />
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b bg-background/85 px-4 backdrop-blur md:px-6">
          <button onClick={() => setDrawer(true)} aria-label="Ouvrir le menu"
            className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground hover:bg-surface-2 lg:hidden"><Menu className="h-4 w-4" /></button>
          <nav aria-label="Fil d'Ariane" className="flex min-w-0 items-center gap-1.5 text-[13px]">
            <span className="hidden text-muted-foreground sm:inline">Nuru</span>
            <span className="hidden text-subtle sm:inline">/</span>
            <span className="truncate font-medium">{section?.label ?? "Détail"}</span>
          </nav>
          <div className="ml-auto flex items-center gap-2.5">
            <span className="hidden rounded-full bg-star/15 px-2.5 py-1 text-2xs font-medium text-foreground/80 md:inline">Démo</span>
            <SystemStatus />
            <ThemeToggle />
          </div>
        </header>
        <main className="mx-auto w-full max-w-[1440px] flex-1 animate-fade-in px-4 py-5 md:px-6 md:py-6">{children}</main>
      </div>
    </div>
  );
}

export function PageHeader({ title, subtitle, children }: { title: React.ReactNode; subtitle?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-x-4 gap-y-3">
      <div className="min-w-0">
        <h1 className="text-[22px] font-semibold leading-7 tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 text-[13px] text-muted-foreground">{subtitle}</p>}
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

export function LiveDot({ error }: { error?: string | null }) {
  return (
    <span className="inline-flex h-9 items-center gap-2 px-1 text-xs text-muted-foreground" title={error ?? "Mise à jour automatique"}>
      <span className={cn("h-2 w-2 rounded-full", error ? "bg-risk-critique" : "animate-pulse bg-risk-faible")} />
      {error ? "Hors ligne" : "Temps réel"}
    </span>
  );
}
