"use client";

import Link from "next/link";
import { useState } from "react";
import { Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowRight, Ban, Banknote, FolderOpen, Gauge, Inbox, ShieldQuestion, Users } from "lucide-react";
import { LiveDot, PageHeader } from "@/components/app-shell";
import { ActionBadge, RiskBadge, RiskStripe } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ChartTooltip, EmptyState, Segmented, Skeleton, SkeletonRows, Table, Td, Th, Tr, cssColor } from "@/components/ui/data";
import { Stat } from "@/components/ui/stat";
import { usePolling } from "@/lib/use-polling";
import { cn, fmtInt, fmtPct, fmtRelative, fmtUsd, fmtUsdCompact, TX_TYPE_LABEL } from "@/lib/utils";

type Win = "15" | "60" | "1440" | "10080";
const WINDOWS: { value: Win; label: string }[] = [
  { value: "15", label: "15 min" }, { value: "60", label: "1 h" }, { value: "1440", label: "24 h" }, { value: "10080", label: "7 j" },
];
const BUCKET: Record<Win, number> = { "15": 15, "60": 60, "1440": 900, "10080": 3600 };
const DECISIONS = [
  { key: "approved", label: "Approuvées", color: "risk-faible" },
  { key: "verified", label: "À vérifier", color: "risk-moyen" },
  { key: "blocked", label: "Bloquées", color: "risk-critique" },
] as const;

/** Barre horizontale proportionnelle (HTML : plus nette qu'un graphique pour quelques lignes). */
function BarRow({ label, value, max, right, tone = "bg-primary", sub }:
  { label: React.ReactNode; value: number; max: number; right: React.ReactNode; tone?: string; sub?: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-[13px]">
        <span className="truncate">{label}</span>
        <span className="tabular shrink-0 text-xs text-muted-foreground">{right}</span>
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-surface-2">
        <div className={cn("h-full rounded-full transition-[width] duration-500", tone)} style={{ width: `${max ? Math.max(2, (value / max) * 100) : 0}%` }} />
      </div>
      {sub && <div className="mt-1 text-2xs text-subtle">{sub}</div>}
    </div>
  );
}

export default function Dashboard() {
  const [win, setWin] = useState<Win>("60");
  const kpis = usePolling<any>(`/reports/kpis?minutes=${win}`);
  const series = usePolling<any[]>(`/reports/timeseries?minutes=${win}&bucket_seconds=${BUCKET[win]}`, 5000);
  const byOperator = usePolling<any[]>(`/reports/breakdown?dimension=operator&minutes=${win}`, 10000);
  const byType = usePolling<any[]>(`/reports/breakdown?dimension=tx_type&minutes=${win}`, 10000);
  const byProvince = usePolling<any[]>(`/reports/breakdown?dimension=province&minutes=${win}`, 15000);
  const risky = usePolling<any[]>(`/reports/top-risky-users?minutes=${win}&limit=6`, 15000);
  const alerts = usePolling<any>(`/transactions?alerts_only=true&size=8`);
  const k = kpis.data;
  const s = series.data ?? [];

  const chart = s.map((r) => ({
    t: new Date(r.bucket).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", second: win === "15" ? "2-digit" : undefined }),
    Approuvées: r.volume - r.verified - r.blocked, "À vérifier": r.verified, Bloquées: r.blocked,
  }));
  const donut = k ? DECISIONS.map((d) => ({ name: d.label, value: k[d.key] ?? 0, color: cssColor(d.color) })) : [];
  const cases = k?.cases ?? {};
  const openCases = (cases.OUVERT ?? 0) + (cases.EN_COURS ?? 0);
  const caseTotal = Object.values(cases).reduce((a: number, b: any) => a + Number(b), 0) as number;
  const typeRows = (byType.data ?? []).map((r) => ({ ...r, rate: r.volume ? r.alerts / r.volume : 0 })).sort((a, b) => b.rate - a.rate);
  const maxTypeRate = Math.max(0.0001, ...typeRows.map((r) => r.rate));
  const opRows = byOperator.data ?? [];
  const maxOp = Math.max(1, ...opRows.map((r) => r.alerts));
  const provRows = (byProvince.data ?? []).filter((r) => r.alerts > 0).sort((a, b) => b.alerts - a.alerts).slice(0, 6);
  const maxProv = Math.max(1, ...provRows.map((r) => r.alerts));

  return (
    <>
      <PageHeader title="Tableau de bord" subtitle="Transactions scorées, décisions du modèle et alertes à instruire">
        <LiveDot error={kpis.error} />
        <Segmented value={win} onChange={setWin} options={WINDOWS} />
      </PageHeader>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="Transactions scorées" icon={Inbox} loading={!k} value={fmtInt(k?.volume)} hint={`${fmtUsdCompact(k?.amount_usd)} traités`}
              spark={s.map((r) => r.volume)} />
        <Stat label="Taux d'alerte" icon={ShieldQuestion} loading={!k} tone="moyen" value={fmtPct(k?.alert_rate, 2)}
              hint={`${fmtInt(k?.verified)} à vérifier · ${fmtInt(k?.blocked)} bloquées`}
              spark={s.map((r) => (r.volume ? (r.verified + r.blocked) / r.volume : 0))} />
        <Stat label="Montant bloqué" icon={Ban} loading={!k} tone="critique" value={fmtUsdCompact(k?.amount_blocked_usd)}
              hint={`${fmtInt(k?.blocked)} opérations refusées`} spark={s.map((r) => r.blocked)} />
        <Stat label="En vérification" icon={Banknote} loading={!k} value={fmtUsdCompact(k?.amount_verified_usd)}
              hint="confirmation PIN, 3-D Secure ou agence" spark={s.map((r) => r.verified)} />
        <Stat label="Latence du scoring" icon={Gauge} loading={!k} tone="faible"
              value={k ? <>{k.latency_avg_ms.toFixed(1)}<span className="ml-0.5 text-sm font-medium text-muted-foreground">ms</span></> : "—"}
              hint={k ? `p95 ${k.latency_p95_ms.toFixed(1)} ms · budget 100 ms` : undefined} spark={s.map((r) => r.latency_avg_ms ?? 0)} />
        <div className="relative flex min-w-0 flex-col overflow-hidden rounded-lg border bg-surface p-4 shadow-card">
          <div className="flex items-center justify-between"><span className="text-xs font-medium text-muted-foreground">Dossiers ouverts</span><FolderOpen className="h-4 w-4 text-subtle" /></div>
          {k ? <div className="tabular mt-1.5 text-[26px] font-semibold leading-8 tracking-tight">{fmtInt(openCases)}</div> : <Skeleton className="mt-2.5 h-7 w-16" />}
          <div className="mt-1 text-2xs text-muted-foreground">{fmtInt(cases.FRAUDE_CONFIRMEE ?? 0)} fraudes confirmées</div>
          <div className="mt-auto flex h-1.5 gap-0.5 overflow-hidden rounded-full pt-0" title="Ouverts · en cours · fraudes confirmées · faux positifs">
            {[["OUVERT", "bg-primary"], ["EN_COURS", "bg-risk-moyen"], ["FRAUDE_CONFIRMEE", "bg-risk-critique"], ["FAUX_POSITIF", "bg-subtle"]].map(([st, cls]) => (
              <div key={st} className={cn("h-full", cls)} style={{ width: `${caseTotal ? ((cases[st] ?? 0) / caseTotal) * 100 : 0}%` }} />
            ))}
          </div>
          <Link href="/alertes" className="mt-2 inline-flex items-center gap-1 text-2xs font-medium text-primary hover:underline">Traiter les dossiers <ArrowRight className="h-3 w-3" /></Link>
        </div>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader title="Flux de transactions et décisions" description="Nombre de transactions par intervalle, empilées par décision"
            actions={<div className="hidden items-center gap-3 sm:flex">{DECISIONS.map((d) => (
              <span key={d.key} className="flex items-center gap-1.5 text-2xs text-muted-foreground">
                <span className="h-2 w-2 rounded-sm" style={{ background: cssColor(d.color) }} />{d.label}
              </span>))}</div>} />
          <CardContent className="h-[280px] pl-2">
            {series.loading && !s.length ? <Skeleton className="h-full w-full" /> : !s.length ? (
              <EmptyState icon={Inbox} title="Aucune transaction sur la période" hint="Lancez le simulateur pour générer du trafic des quatre opérateurs." />
            ) : (
              <ResponsiveContainer>
                <AreaChart data={chart} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
                  <defs>
                    {DECISIONS.map((d) => (
                      <linearGradient key={d.key} id={`g-${d.key}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={cssColor(d.color)} stopOpacity={d.key === "approved" ? 0.28 : 0.75} />
                        <stop offset="100%" stopColor={cssColor(d.color)} stopOpacity={d.key === "approved" ? 0.04 : 0.35} />
                      </linearGradient>
                    ))}
                  </defs>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="t" tickLine={false} axisLine={false} minTickGap={28} />
                  <YAxis tickLine={false} axisLine={false} allowDecimals={false} width={44} />
                  <Tooltip content={<ChartTooltip />} cursor={{ stroke: cssColor("border"), strokeWidth: 1 }} />
                  <Area type="monotone" dataKey="Approuvées" stackId="1" stroke={cssColor("risk-faible")} strokeWidth={1.5} fill="url(#g-approved)" />
                  <Area type="monotone" dataKey="À vérifier" stackId="1" stroke={cssColor("risk-moyen")} strokeWidth={1.5} fill="url(#g-verified)" />
                  <Area type="monotone" dataKey="Bloquées" stackId="1" stroke={cssColor("risk-critique")} strokeWidth={1.5} fill="url(#g-blocked)" />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader title="Répartition des décisions" description="Sur la période sélectionnée" />
          <CardContent>
            <div className="relative mx-auto h-[172px] w-[172px]">
              {k && k.volume > 0 ? (
                <ResponsiveContainer>
                  <PieChart>
                    <Pie data={donut} dataKey="value" nameKey="name" innerRadius={62} outerRadius={82} paddingAngle={1.5} stroke="none" startAngle={90} endAngle={-270}>
                      {donut.map((d) => <Cell key={d.name} fill={d.color} />)}
                    </Pie>
                    <Tooltip content={<ChartTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              ) : <Skeleton className="h-full w-full rounded-full" />}
              <div className="pointer-events-none absolute inset-0 grid place-items-center text-center">
                <div>
                  <div className="tabular text-xl font-semibold">{fmtPct(k?.alert_rate, 1)}</div>
                  <div className="text-2xs text-muted-foreground">d'alertes</div>
                </div>
              </div>
            </div>
            <div className="mt-4 divide-y">
              {DECISIONS.map((d) => (
                <div key={d.key} className="flex items-center justify-between py-2 text-[13px]">
                  <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-sm" style={{ background: cssColor(d.color) }} />{d.label}</span>
                  <span className="tabular text-muted-foreground">
                    <span className="font-medium text-foreground">{fmtInt(k?.[d.key])}</span>
                    <span className="ml-2 inline-block w-14 text-right">{k?.volume ? fmtPct((k[d.key] ?? 0) / k.volume, 1) : "—"}</span>
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card className="overflow-hidden xl:col-span-2">
          <CardHeader title="Dernières alertes" description="Transactions à vérifier ou bloquées, les plus récentes d'abord"
            actions={<Link href="/alertes" className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline">Tous les dossiers <ArrowRight className="h-3.5 w-3.5" /></Link>} />
          <Table>
            <thead><tr><Th>Transaction</Th><Th>Opération</Th><Th align="right">Montant</Th><Th align="right">P(fraude)</Th><Th>Risque</Th><Th>Décision</Th></tr></thead>
            <tbody>
              {alerts.loading && !alerts.data && <SkeletonRows cols={6} rows={5} />}
              {(alerts.data?.items ?? []).map((t: any) => (
                <Tr key={t.transaction_id}>
                  <Td>
                    <RiskStripe level={t.risk_level} />
                    <Link href={`/transactions/${t.transaction_id}`} className="font-mono text-xs font-medium text-foreground hover:text-primary">{t.transaction_id}</Link>
                    <div className="text-2xs text-muted-foreground">{t.user_id} · {fmtRelative(t.scored_at)}</div>
                  </Td>
                  <Td><div className="text-[13px]">{TX_TYPE_LABEL[t.tx_type] ?? t.tx_type}</div><div className="text-2xs text-muted-foreground">{t.channel === "VISA_VIRTUAL" ? "Visa virtuelle" : t.operator}</div></Td>
                  <Td align="right" className="font-medium">{fmtUsd(t.amount_usd)}</Td>
                  <Td align="right">{fmtPct(t.fraud_probability)}</Td>
                  <Td><RiskBadge level={t.risk_level} /></Td>
                  <Td><ActionBadge action={t.action} /></Td>
                </Tr>
              ))}
              {alerts.data && alerts.data.items.length === 0 && (
                <tr><td colSpan={6}><EmptyState icon={Inbox} title="Aucune alerte" hint="Les transactions à vérifier ou bloquées apparaîtront ici." /></td></tr>
              )}
            </tbody>
          </Table>
        </Card>

        <Card>
          <CardHeader title="Alertes par opérateur" description="Alertes (vérification ou blocage) et part du volume" />
          <CardContent className="space-y-4">
            {!byOperator.data && [0, 1, 2].map((i) => <Skeleton key={i} className="h-8 w-full" />)}
            {opRows.map((r) => (
              <BarRow key={r.key} label={r.key === "N/A" ? "Visa virtuelle" : r.key} value={r.alerts} max={maxOp} tone="bg-risk-eleve"
                right={<><span className="font-medium text-foreground">{fmtInt(r.alerts)}</span> · {fmtPct(r.volume ? r.alerts / r.volume : 0, 1)}</>}
                sub={`${fmtInt(r.blocked)} bloquées · ${fmtUsdCompact(r.amount_blocked_usd)} retenus · ${fmtInt(r.volume)} transactions`} />
            ))}
            {byOperator.data && !opRows.length && <EmptyState title="Pas encore de données" />}
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader title="Taux d'alerte par type d'opération" description="Les types les plus alertés en premier" />
          <CardContent className="space-y-3.5">
            {!byType.data && [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-7 w-full" />)}
            {typeRows.slice(0, 7).map((r) => (
              <BarRow key={r.key} label={TX_TYPE_LABEL[r.key] ?? r.key} value={r.rate} max={maxTypeRate}
                tone={r.rate >= 0.1 ? "bg-risk-critique" : r.rate >= 0.03 ? "bg-risk-eleve" : "bg-primary/70"}
                right={<><span className="font-medium text-foreground">{fmtPct(r.rate, 1)}</span> · {fmtInt(r.volume)}</>} />
            ))}
          </CardContent>
        </Card>

        <Card className="overflow-hidden">
          <CardHeader title="Clients à surveiller" description="Le plus d'alertes sur la période" actions={<Users className="h-4 w-4 text-subtle" />} />
          <div className="divide-y border-t">
            {!risky.data && [0, 1, 2, 3].map((i) => <div key={i} className="px-5 py-3"><Skeleton className="h-4 w-full" /></div>)}
            {(risky.data ?? []).map((u) => (
              <Link key={u.user_id} href={`/transactions?search=${encodeURIComponent(u.user_id)}`}
                className="flex items-center justify-between gap-3 px-5 py-2.5 transition-colors hover:bg-surface-2/70">
                <div className="min-w-0">
                  <div className="font-mono text-xs font-medium">{u.user_id}</div>
                  <div className="text-2xs text-muted-foreground">{fmtInt(u.alerts)} alerte{u.alerts > 1 ? "s" : ""} · {fmtUsd(u.amount_usd)}</div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-2">
                    <div className={cn("h-full rounded-full", u.max_probability >= 0.9 ? "bg-risk-critique" : u.max_probability >= 0.5 ? "bg-risk-eleve" : "bg-risk-moyen")}
                         style={{ width: `${u.max_probability * 100}%` }} />
                  </div>
                  <span className="tabular w-11 text-right text-xs text-muted-foreground">{fmtPct(u.max_probability, 0)}</span>
                </div>
              </Link>
            ))}
            {risky.data && !risky.data.length && <EmptyState title="Aucun client alerté" />}
          </div>
        </Card>

        <Card>
          <CardHeader title="Alertes par province" description="Là où se concentrent les alertes" />
          <CardContent className="space-y-3.5">
            {!byProvince.data && [0, 1, 2, 3].map((i) => <Skeleton key={i} className="h-7 w-full" />)}
            {provRows.map((r) => (
              <BarRow key={r.key} label={r.key} value={r.alerts} max={maxProv} tone="bg-primary"
                right={<><span className="font-medium text-foreground">{fmtInt(r.alerts)}</span> · {fmtPct(r.volume ? r.alerts / r.volume : 0, 1)}</>} />
            ))}
            {byProvince.data && !provRows.length && <EmptyState title="Aucune alerte sur la période" />}
          </CardContent>
        </Card>
      </div>
    </>
  );
}
