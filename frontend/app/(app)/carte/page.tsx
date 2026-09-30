"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { Ban, MapPinned, Navigation, ShieldQuestion, X } from "lucide-react";
import { LiveDot, PageHeader } from "@/components/app-shell";
import { DrcMap, provinceKey, type MapDatum } from "@/components/drc-map";
import { ActionBadge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { EmptyState, Segmented, Skeleton } from "@/components/ui/data";
import { Select } from "@/components/ui/select";
import { Stat } from "@/components/ui/stat";
import { usePolling } from "@/lib/use-polling";
import { cn, fmtInt, fmtPct, fmtRelative, fmtUsd, fmtUsdCompact, TX_TYPE_LABEL } from "@/lib/utils";

type Win = "60" | "1440" | "10080" | "43200";
const WINDOWS: { value: Win; label: string }[] = [
  { value: "60", label: "1 h" }, { value: "1440", label: "24 h" }, { value: "10080", label: "7 j" }, { value: "43200", label: "30 j" },
];
type Metric = "alerts" | "alert_rate" | "amount_blocked_usd" | "volume";
const METRICS: { value: Metric; label: string; legend: string; color: string; fmt: (v: number) => string }[] = [
  { value: "alerts", label: "Alertes", legend: "Alertes (à vérifier + bloquées)", color: "risk-critique", fmt: (v) => fmtInt(v) },
  { value: "alert_rate", label: "Taux d'alerte", legend: "Part des transactions en alerte", color: "risk-eleve", fmt: (v) => fmtPct(v, 1) },
  { value: "amount_blocked_usd", label: "Montant bloqué", legend: "Montant bloqué (USD)", color: "risk-critique", fmt: (v) => fmtUsdCompact(v) },
  { value: "volume", label: "Volume", legend: "Transactions scorées", color: "primary", fmt: (v) => fmtInt(v) },
];
const OPERATORS: [string, string][] = [
  ["ALL", "Tous les opérateurs"], ["VODACOM", "Vodacom M-Pesa"], ["AIRTEL", "Airtel Money"],
  ["ORANGE", "Orange Money"], ["VISA", "Cartes Visa virtuelles"],
];
const OP_LABEL: Record<string, string> = { VODACOM: "Vodacom", AIRTEL: "Airtel", ORANGE: "Orange", VISA: "Visa" };
const RULE_LABEL = (r: string) => r.toLowerCase().replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

type GeoRow = { province: string; volume: number; alerts: number; verified: number; blocked: number; users_alerted: number;
  amount_usd: number; amount_blocked_usd: number; avg_probability: number; alerts_away_from_home: number;
  confirmed_fraud: number; alert_rate: number };

function Line({ k, v, strong }: { k: string; v: React.ReactNode; strong?: boolean }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-muted-foreground">{k}</span>
      <span className={cn("tabular", strong ? "font-semibold text-foreground" : "text-foreground")}>{v}</span>
    </div>
  );
}

function Bar({ label, value, max, right, tone }: { label: string; value: number; max: number; right: React.ReactNode; tone: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-[13px]">
        <span className="truncate">{label}</span><span className="tabular shrink-0 text-xs text-muted-foreground">{right}</span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2">
        <div className={cn("h-full rounded-full transition-[width] duration-500", tone)} style={{ width: `${max ? Math.max(2, (value / max) * 100) : 0}%` }} />
      </div>
    </div>
  );
}

function ProvinceDetail({ row, win, op, onClose }: { row: GeoRow; win: Win; op: string; onClose: () => void }) {
  const { data } = usePolling<any>(`/reports/geo/${encodeURIComponent(row.province)}?minutes=${win}&operator=${op}`, 10000);
  const ops: any[] = data?.by_operator ?? [];
  const types: any[] = (data?.by_tx_type ?? []).filter((t: any) => t.alerts > 0);
  const maxOp = Math.max(1, ...ops.map((o) => o.alerts));
  const maxType = Math.max(1, ...types.map((t) => t.alerts));
  return (
    <Card className="animate-fade-in">
      <CardHeader title={row.province} description="Transactions déclarées dans la province"
        actions={<button onClick={onClose} aria-label="Fermer le détail"
          className="grid h-7 w-7 place-items-center rounded-md text-subtle hover:bg-surface-2 hover:text-foreground"><X className="h-4 w-4" /></button>} />
      <CardContent className="space-y-5">
        <div className="grid grid-cols-2 gap-2 text-center">
          {[["Alertes", fmtInt(row.alerts), "text-risk-critique"], ["Taux d'alerte", fmtPct(row.alert_rate, 2), ""],
            ["Bloquées", fmtInt(row.blocked), ""], ["Montant bloqué", fmtUsdCompact(row.amount_blocked_usd), ""]].map(([k, v, c]) => (
            <div key={k} className="rounded-md bg-surface-2 px-2 py-2">
              <div className={cn("tabular text-lg font-semibold", c)}>{v}</div><div className="text-2xs text-muted-foreground">{k}</div>
            </div>
          ))}
        </div>
        <div className="space-y-1 text-xs">
          <Line k="Transactions scorées" v={fmtInt(row.volume)} />
          <Line k="Clients alertés" v={fmtInt(row.users_alerted)} />
          <Line k="Alertes hors province d'origine" v={`${fmtInt(row.alerts_away_from_home)} (${fmtPct(row.alerts ? row.alerts_away_from_home / row.alerts : 0, 0)})`} />
          <Line k="Fraudes confirmées" v={fmtInt(row.confirmed_fraud)} />
        </div>

        <div>
          <div className="label-caps mb-2">Alertes par opérateur</div>
          <div className="space-y-2.5">
            {!data && <Skeleton className="h-16 w-full" />}
            {ops.map((o) => <Bar key={o.key} label={OP_LABEL[o.key] ?? o.key} value={o.alerts} max={maxOp} tone="bg-primary"
                                 right={<><span className="font-medium text-foreground">{fmtInt(o.alerts)}</span> / {fmtInt(o.volume)}</>} />)}
          </div>
        </div>
        {!!types.length && (
          <div>
            <div className="label-caps mb-2">Types d'opération en alerte</div>
            <div className="space-y-2.5">
              {types.slice(0, 5).map((t) => <Bar key={t.key} label={TX_TYPE_LABEL[t.key] ?? t.key} value={t.alerts} max={maxType} tone="bg-risk-eleve"
                                                  right={<span className="font-medium text-foreground">{fmtInt(t.alerts)}</span>} />)}
            </div>
          </div>
        )}
        {!!data?.top_rules?.length && (
          <div>
            <div className="label-caps mb-2">Règles déclenchées</div>
            <ul className="space-y-1 text-xs">
              {data.top_rules.map((r: any) => (
                <li key={r.key} className="flex justify-between gap-3"><span className="truncate" title={r.key}>{RULE_LABEL(r.key)}</span>
                  <span className="tabular text-muted-foreground">{fmtInt(r.n)}</span></li>
              ))}
            </ul>
          </div>
        )}
        <div>
          <div className="label-caps mb-2">Dernières alertes</div>
          <div className="-mx-2 divide-y">
            {(data?.recent_alerts ?? []).map((t: any) => (
              <Link key={t.transaction_id} href={`/transactions/${t.transaction_id}`}
                className="flex items-center gap-2 rounded-md px-2 py-2 text-xs hover:bg-surface-2">
                <ActionBadge action={t.action} />
                <span className="min-w-0 flex-1 truncate">{TX_TYPE_LABEL[t.tx_type] ?? t.tx_type} · {OP_LABEL[t.operator] ?? t.operator ?? "—"}</span>
                <span className="tabular font-medium">{fmtUsd(t.amount_usd)}</span>
                <span className="w-16 text-right text-2xs text-subtle">{fmtRelative(t.scored_at)}</span>
              </Link>
            ))}
            {data && !data.recent_alerts?.length && <p className="px-2 py-2 text-xs text-muted-foreground">Aucune alerte sur la période.</p>}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default function CartePage() {
  const [win, setWin] = useState<Win>("1440");
  const [op, setOp] = useState("ALL");
  const [metric, setMetric] = useState<Metric>("alerts");
  const [selected, setSelected] = useState<string | null>(null);
  const geo = usePolling<{ provinces: GeoRow[] }>(`/reports/geo?minutes=${win}&operator=${op}`, 10000);
  const rows = geo.data?.provinces ?? [];
  const m = METRICS.find((x) => x.value === metric)!;

  const mapData = useMemo(() => {
    const out: Record<string, MapDatum> = {};
    for (const r of rows) {
      out[provinceKey(r.province)] = {
        value: Number(r[metric]) || 0,
        tooltip: (
          <div className="space-y-0.5">
            <Line k="Transactions" v={fmtInt(r.volume)} />
            <Line k="Alertes" v={fmtInt(r.alerts)} strong />
            <Line k="Taux d'alerte" v={fmtPct(r.alert_rate, 2)} />
            <Line k="Bloquées" v={fmtInt(r.blocked)} />
            <Line k="Montant bloqué" v={fmtUsdCompact(r.amount_blocked_usd)} />
          </div>
        ),
      };
    }
    return out;
  }, [rows, metric]);

  const tot = rows.reduce((a, r) => ({ alerts: a.alerts + r.alerts, away: a.away + r.alerts_away_from_home,
    blocked: a.blocked + r.amount_blocked_usd, touched: a.touched + (r.alerts > 0 ? 1 : 0) }), { alerts: 0, away: 0, blocked: 0, touched: 0 });
  const ranked = [...rows].sort((a, b) => Number(b[metric]) - Number(a[metric]));
  const maxRank = Math.max(1e-9, ...ranked.map((r) => Number(r[metric])));
  const selRow = rows.find((r) => provinceKey(r.province) === provinceKey(selected ?? ""));

  return (
    <>
      <PageHeader title="Carte des fraudes" subtitle="Géolocalisation des alertes par province de la RDC">
        <LiveDot error={geo.error} />
        <Select value={op} onChange={(e) => setOp(e.target.value)} options={OPERATORS} className="w-52" aria-label="Opérateur" />
        <Segmented value={win} onChange={setWin} options={WINDOWS} />
      </PageHeader>

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <Stat label="Provinces avec alertes" icon={MapPinned} loading={!geo.data} value={`${tot.touched} / 26`} hint={`${rows.length} province(s) active(s)`} />
        <Stat label="Alertes" icon={ShieldQuestion} tone="moyen" loading={!geo.data} value={fmtInt(tot.alerts)} hint="à vérifier + bloquées" />
        <Stat label="Hors province d'origine" icon={Navigation} loading={!geo.data} value={fmtPct(tot.alerts ? tot.away / tot.alerts : 0, 0)}
              hint={`${fmtInt(tot.away)} alertes loin du domicile habituel`} />
        <Stat label="Montant bloqué" icon={Ban} tone="critique" loading={!geo.data} value={fmtUsdCompact(tot.blocked)} hint="fraude évitée (estimation)" />
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card>
          <CardHeader title="République démocratique du Congo — 26 provinces"
            description="Survolez une province pour ses chiffres, cliquez pour le détail"
            actions={<Segmented value={metric} onChange={(v) => setMetric(v)} options={METRICS.map(({ value, label }) => ({ value, label }))} />} />
          <CardContent>
            {!geo.data ? <Skeleton className="aspect-square w-full" /> : (
              <div className="mx-auto max-w-[760px]">
                <DrcMap data={mapData} color={m.color} selected={selected} onSelect={setSelected} formatValue={m.fmt} legendTitle={m.legend} />
              </div>
            )}
            <p className="mt-3 text-2xs text-subtle">
              Province déclarée de la transaction. Contours : geoBoundaries (CC BY 4.0). Données synthétiques.
            </p>
          </CardContent>
        </Card>

        {selRow ? <ProvinceDetail row={selRow} win={win} op={op} onClose={() => setSelected(null)} /> : (
          <Card>
            <CardHeader title="Classement des provinces" description={m.legend} />
            <CardContent className="space-y-1">
              {!geo.data && [0, 1, 2, 3, 4].map((i) => <Skeleton key={i} className="h-9 w-full" />)}
              {ranked.map((r, i) => (
                <button key={r.province} onClick={() => setSelected(r.province)}
                  className="block w-full rounded-md px-2 py-1.5 text-left hover:bg-surface-2">
                  <div className="flex items-baseline justify-between gap-3 text-[13px]">
                    <span className="truncate"><span className="tabular mr-2 text-2xs text-subtle">{i + 1}</span>{r.province}</span>
                    <span className="tabular shrink-0 text-xs font-medium">{m.fmt(Number(r[metric]))}</span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-2">
                    <div className="h-full rounded-full transition-[width] duration-500"
                         style={{ width: `${Math.max(2, (Number(r[metric]) / maxRank) * 100)}%`, background: `hsl(var(--${m.color}))` }} />
                  </div>
                </button>
              ))}
              {geo.data && !rows.length && <EmptyState icon={MapPinned} title="Aucune transaction sur la période"
                                                       hint="Lancez le simulateur ou élargissez la fenêtre de temps." />}
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}
