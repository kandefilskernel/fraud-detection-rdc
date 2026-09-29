"use client";

import { Area, AreaChart, ResponsiveContainer, YAxis } from "recharts";
import { Skeleton } from "@/components/ui/data";
import { cn } from "@/lib/utils";

type Tone = "default" | "critique" | "moyen" | "faible" | "primary";
const TONE: Record<Tone, { text: string; color: string }> = {
  default: { text: "text-foreground", color: "primary" },
  primary: { text: "text-foreground", color: "primary" },
  critique: { text: "text-risk-critique", color: "risk-critique" },
  moyen: { text: "text-foreground", color: "risk-moyen" },
  faible: { text: "text-foreground", color: "risk-faible" },
};

/** Indicateur clé : libellé, valeur, précision, et mini-courbe de la période. */
export function Stat({ label, value, hint, tone = "default", spark, icon: Icon, loading }:
  { label: string; value: React.ReactNode; hint?: React.ReactNode; tone?: Tone; spark?: number[];
    icon?: React.ComponentType<{ className?: string }>; loading?: boolean }) {
  const t = TONE[tone];
  const data = (spark ?? []).map((v, i) => ({ i, v }));
  const gid = `spark-${label.replace(/\W/g, "")}`;
  return (
    <div className="relative flex min-w-0 flex-col overflow-hidden rounded-lg border bg-surface p-4 shadow-card">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs font-medium text-muted-foreground">{label}</span>
        {Icon && <Icon className="h-4 w-4 shrink-0 text-subtle" />}
      </div>
      {loading ? <Skeleton className="mt-2.5 h-7 w-24" /> : (
        <div className={cn("tabular mt-1.5 text-[26px] font-semibold leading-8 tracking-tight", t.text)}>{value}</div>
      )}
      <div className="mt-1 min-h-[16px] truncate text-2xs text-muted-foreground">{hint}</div>
      {data.length > 1 && (
        <div className="pointer-events-none -mx-4 -mb-4 mt-2 h-10">
          <ResponsiveContainer>
            <AreaChart data={data} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={`hsl(var(--${t.color}))`} stopOpacity={0.22} />
                  <stop offset="100%" stopColor={`hsl(var(--${t.color}))`} stopOpacity={0} />
                </linearGradient>
              </defs>
              <YAxis hide domain={["dataMin", "dataMax"]} />
              <Area type="monotone" dataKey="v" stroke={`hsl(var(--${t.color}))`} strokeWidth={1.5}
                    fill={`url(#${gid})`} isAnimationActive={false} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
