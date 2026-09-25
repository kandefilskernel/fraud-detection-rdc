"use client";

import Link from "next/link";
import { useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { LiveDot, PageHeader } from "@/components/app-shell";
import { ActionBadge, RiskBadge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { usePolling } from "@/lib/use-polling";
import { fmtDate, fmtInt, fmtPct, fmtUsd } from "@/lib/utils";

const WINDOWS: [string, string][] = [["15", "15 dernières minutes"], ["60", "Dernière heure"], ["1440", "24 heures"], ["10080", "7 jours"]];
const BUCKET: Record<string, number> = { "15": 15, "60": 60, "1440": 900, "10080": 3600 };

function Kpi({ label, value, hint, tone }: { label: string; value: string; hint?: string; tone?: "danger" | "warn" }) {
  return (
    <Card>
      <CardContent className="pt-4">
        <div className="text-xs font-medium text-muted-foreground">{label}</div>
        <div className={`tabular mt-1 text-2xl font-semibold ${tone === "danger" ? "text-risk-critique" : tone === "warn" ? "text-[hsl(38_90%_36%)]" : ""}`}>{value}</div>
        {hint && <div className="mt-0.5 text-xs text-muted-foreground">{hint}</div>}
      </CardContent>
    </Card>
  );
}

export default function Dashboard() {
  const [minutes, setMinutes] = useState("60");
  const kpis = usePolling<any>(`/reports/kpis?minutes=${minutes}`);
  const series = usePolling<any[]>(`/reports/timeseries?minutes=${minutes}&bucket_seconds=${BUCKET[minutes]}`, 5000);
  const byOperator = usePolling<any[]>(`/reports/breakdown?dimension=operator&minutes=${minutes}`, 10000);
  const byType = usePolling<any[]>(`/reports/breakdown?dimension=tx_type&minutes=${minutes}`, 10000);
  const alerts = usePolling<any>(`/transactions?alerts_only=true&size=8`);
  const k = kpis.data;
  const chart = (series.data ?? []).map((r) => ({
    t: new Date(r.bucket).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: minutes === "15" ? "2-digit" : undefined }),
    Approuvées: r.volume - r.verified - r.blocked, "À vérifier": r.verified, Bloquées: r.blocked,
  }));

  return (
    <>
      <PageHeader title="Tableau de bord" subtitle="Transactions scorées par le modèle hybride, décisions et alertes">
        <LiveDot error={kpis.error} />
        <Select value={minutes} onChange={(e) => setMinutes(e.target.value)} options={WINDOWS} />
      </PageHeader>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-6">
        <Kpi label="Transactions scorées" value={fmtInt(k?.volume)} hint={`${fmtUsd(k?.amount_usd)} traités`} />
        <Kpi label="Taux d'alerte" value={fmtPct(k?.alert_rate, 2)} hint={`${fmtInt(k?.verified)} à vérifier · ${fmtInt(k?.blocked)} bloquées`} tone="warn" />
        <Kpi label="Montant bloqué" value={fmtUsd(k?.amount_blocked_usd)} tone="danger" />
        <Kpi label="Montant en vérification" value={fmtUsd(k?.amount_verified_usd)} />
        <Kpi label="Latence du scoring" value={k ? `${k.latency_avg_ms.toFixed(1)} ms` : "—"} hint={k ? `p95 : ${k.latency_p95_ms.toFixed(1)} ms` : undefined} />
        <Kpi label="Dossiers ouverts" value={fmtInt((k?.cases?.OUVERT ?? 0) + (k?.cases?.EN_COURS ?? 0))}
             hint={`${fmtInt(k?.cases?.FRAUDE_CONFIRMEE ?? 0)} fraudes confirmées`} />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader><CardTitle>Flux de transactions et décisions</CardTitle></CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer>
              <AreaChart data={chart} margin={{ left: -10, right: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                <XAxis dataKey="t" tick={{ fontSize: 11 }} minTickGap={24} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area type="monotone" dataKey="Approuvées" stackId="1" stroke="hsl(var(--risk-faible))" fill="hsl(var(--risk-faible))" fillOpacity={0.25} />
                <Area type="monotone" dataKey="À vérifier" stackId="1" stroke="hsl(var(--risk-moyen))" fill="hsl(var(--risk-moyen))" fillOpacity={0.6} />
                <Area type="monotone" dataKey="Bloquées" stackId="1" stroke="hsl(var(--risk-critique))" fill="hsl(var(--risk-critique))" fillOpacity={0.7} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Alertes par opérateur</CardTitle></CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer>
              <BarChart data={byOperator.data ?? []} layout="vertical" margin={{ left: 10, right: 12 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="hsl(var(--border))" />
                <XAxis type="number" tick={{ fontSize: 11 }} allowDecimals={false} />
                <YAxis type="category" dataKey="key" tick={{ fontSize: 11 }} width={70} />
                <Tooltip />
                <Bar dataKey="alerts" name="Alertes" fill="hsl(var(--risk-eleve))" radius={[0, 4, 4, 0]} />
                <Bar dataKey="blocked" name="Bloquées" fill="hsl(var(--risk-critique))" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle>Dernières alertes</CardTitle>
            <Link href="/alertes" className="text-xs font-medium text-primary hover:underline">Voir les dossiers →</Link>
          </CardHeader>
          <CardContent className="overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead className="border-y bg-muted/50 text-left text-xs text-muted-foreground">
                <tr><th className="px-5 py-2">Scorée à</th><th>Client</th><th>Type</th><th className="text-right">Montant</th><th className="px-3 text-right">Probabilité</th><th>Risque</th><th className="pr-5">Décision</th></tr>
              </thead>
              <tbody>
                {(alerts.data?.items ?? []).map((t: any) => (
                  <tr key={t.transaction_id} className="border-b last:border-0 hover:bg-muted/40">
                    <td className="px-5 py-2 text-xs text-muted-foreground">{fmtDate(t.scored_at)}</td>
                    <td><Link className="font-medium text-primary hover:underline" href={`/transactions/${t.transaction_id}`}>{t.user_id}</Link></td>
                    <td className="text-xs">{t.tx_type}<div className="text-muted-foreground">{t.operator ?? "VISA"}</div></td>
                    <td className="tabular text-right">{fmtUsd(t.amount_usd)}</td>
                    <td className="tabular px-3 text-right">{fmtPct(t.fraud_probability)}</td>
                    <td><RiskBadge level={t.risk_level} /></td>
                    <td className="pr-5"><ActionBadge action={t.action} /></td>
                  </tr>
                ))}
                {alerts.data && alerts.data.items.length === 0 && (
                  <tr><td colSpan={7} className="px-5 py-8 text-center text-muted-foreground">Aucune alerte. Lancez le simulateur pour générer du trafic.</td></tr>
                )}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Taux d'alerte par type d'opération</CardTitle></CardHeader>
          <CardContent className="space-y-2.5">
            {(byType.data ?? []).map((r) => {
              const rate = r.volume ? r.alerts / r.volume : 0;
              return (
                <div key={r.key}>
                  <div className="flex justify-between text-xs"><span>{r.key}</span><span className="tabular text-muted-foreground">{fmtPct(rate, 2)} · {fmtInt(r.volume)}</span></div>
                  <div className="mt-1 h-1.5 rounded-full bg-muted"><div className="h-1.5 rounded-full bg-risk-eleve" style={{ width: `${Math.min(100, rate * 100 * 10)}%` }} /></div>
                </div>
              );
            })}
            <p className="pt-1 text-[11px] text-muted-foreground">Barre à l'échelle ×10 pour rendre visibles les taux faibles.</p>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
