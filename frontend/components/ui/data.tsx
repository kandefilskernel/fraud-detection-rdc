"use client";

/** Briques d'affichage de données : tableau, contrôle segmenté, squelettes, état vide, info-bulle de graphique. */
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------- tableau
export function Table({ className, children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className={cn("scrollbar-thin overflow-x-auto", className)}>
      <table className="w-full border-separate border-spacing-0 text-[13px]">{children}</table>
    </div>
  );
}
export function Th({ className, children, align = "left" }:
  { className?: string; children?: React.ReactNode; align?: "left" | "right" | "center" }) {
  return (
    <th className={cn("sticky top-0 z-[1] whitespace-nowrap border-b bg-surface-2/80 px-3 py-2 text-2xs font-medium uppercase tracking-[0.06em] text-muted-foreground backdrop-blur first:pl-5 last:pr-5",
      align === "right" && "text-right", align === "center" && "text-center", className)}>{children}</th>
  );
}
export function Td({ className, children, align = "left", title, colSpan }:
  { className?: string; children?: React.ReactNode; align?: "left" | "right" | "center"; title?: string; colSpan?: number }) {
  return (
    <td title={title} colSpan={colSpan} className={cn("relative whitespace-nowrap border-b px-3 py-2.5 align-middle first:pl-5 last:pr-5 group-last:border-b-0",
      align === "right" && "text-right tabular", align === "center" && "text-center", className)}>{children}</td>
  );
}
export function Tr({ className, children, onClick, selected }:
  { className?: string; children: React.ReactNode; onClick?: () => void; selected?: boolean }) {
  return (
    <tr onClick={onClick} className={cn("group transition-colors hover:bg-surface-2/70", onClick && "cursor-pointer",
      selected && "bg-primary-soft/70 hover:bg-primary-soft", className)}>{children}</tr>
  );
}

// ---------------------------------------------------------------------- contrôle segmenté
export function Segmented<T extends string>({ value, onChange, options, className }:
  { value: T; onChange: (v: T) => void; options: { value: T; label: React.ReactNode; count?: number }[]; className?: string }) {
  return (
    <div role="tablist" className={cn("inline-flex rounded-lg border bg-surface-2 p-0.5", className)}>
      {options.map((o) => {
        const on = o.value === value;
        return (
          <button key={o.value} role="tab" aria-selected={on} onClick={() => onChange(o.value)}
            className={cn("inline-flex h-7 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium transition-colors",
              on ? "bg-surface text-foreground shadow-card" : "text-muted-foreground hover:text-foreground")}>
            {o.label}
            {o.count !== undefined && (
              <span className={cn("tabular rounded px-1 text-2xs", on ? "bg-surface-2 text-foreground" : "text-subtle")}>{o.count.toLocaleString("fr-FR")}</span>
            )}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------- chargement et vide
export function Skeleton({ className }: { className?: string }) {
  return (
    <div className={cn("relative overflow-hidden rounded-md bg-surface-2", className)}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-surface-3/60 to-transparent" />
    </div>
  );
}
export function SkeletonRows({ cols, rows = 6 }: { cols: number; rows?: number }) {
  return (
    <>
      {Array.from({ length: rows }).map((_, i) => (
        <tr key={i}>{Array.from({ length: cols }).map((__, j) => (
          <td key={j} className="border-b px-3 py-3 first:pl-5 last:pr-5"><Skeleton className="h-3.5 w-full max-w-[120px]" /></td>
        ))}</tr>
      ))}
    </>
  );
}
export function EmptyState({ icon: Icon, title, hint, className }:
  { icon?: React.ComponentType<{ className?: string }>; title: string; hint?: string; className?: string }) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 px-6 py-12 text-center", className)}>
      {Icon && <div className="rounded-full bg-surface-2 p-3 text-subtle"><Icon className="h-5 w-5" /></div>}
      <p className="text-sm font-medium">{title}</p>
      {hint && <p className="max-w-sm text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------- graphiques
export function ChartTooltip({ active, payload, label, format }:
  { active?: boolean; payload?: any[]; label?: string; format?: (v: number) => string }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="min-w-[150px] rounded-lg border bg-surface px-3 py-2 text-xs shadow-pop">
      {label && <div className="mb-1.5 font-medium text-foreground">{label}</div>}
      <div className="space-y-1">
        {payload.map((p) => (
          <div key={p.dataKey} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <span className="h-2 w-2 rounded-sm" style={{ background: p.color ?? p.payload?.fill }} />{p.name}
            </span>
            <span className="tabular font-medium text-foreground">{format ? format(p.value) : Number(p.value).toLocaleString("fr-FR")}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Variable CSS du thème -> couleur utilisable par Recharts. */
export const cssColor = (name: string, alpha?: number) => (alpha === undefined ? `hsl(var(--${name}))` : `hsl(var(--${name}) / ${alpha})`);
