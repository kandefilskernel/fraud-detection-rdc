"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PageHeader } from "@/components/app-shell";
import { ActionBadge, RiskBadge, Tag } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { usePolling } from "@/lib/use-polling";
import { FEATURE_LABEL, fmtDate, fmtPct, fmtUsd } from "@/lib/utils";

const BRANCH_LABEL: Record<string, string> = {
  xgboost: "XGBoost (profil instantané)", lstm_attention: "LSTM + attention (séquence)", autoencoder: "Autoencodeur (anomalie)",
};

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return <div className="flex justify-between gap-4 border-b py-1.5 text-sm last:border-0"><span className="text-muted-foreground">{k}</span><span className="text-right font-medium">{v}</span></div>;
}

export default function TransactionDetail({ params }: { params: { id: string } }) {
  const { data: t, error } = usePolling<any>(`/transactions/${encodeURIComponent(params.id)}`, 0);
  const history = usePolling<any[]>(t ? `/transactions/by-user/${encodeURIComponent(t.user_id)}/history?limit=12` : null, 0);

  if (error) return <p className="text-risk-critique">{error}</p>;
  if (!t) return <p className="text-muted-foreground">Chargement…</p>;

  const shap = (t.explanation?.top_features ?? []).map((f: any) => ({
    name: FEATURE_LABEL[f.feature] ?? f.feature, shap: f.shap, value: f.value,
  }));
  const branches = Object.entries(t.explanation?.branch_contributions ?? {}).map(([k, v]) => ({ name: BRANCH_LABEL[k] ?? k, v: v as number }));
  const attention: number[] = t.explanation?.attention_on_history ?? [];

  return (
    <>
      <Link href="/transactions" className="mb-3 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"><ArrowLeft className="h-4 w-4" /> Transactions</Link>
      <PageHeader title={`Transaction ${t.transaction_id}`} subtitle={`Client ${t.user_id} · ${fmtDate(t.tx_time)} (heure RDC)`}>
        <RiskBadge level={t.risk_level} /><ActionBadge action={t.action} />
      </PageHeader>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader><CardTitle>Décision</CardTitle></CardHeader>
          <CardContent>
            <div className="tabular text-4xl font-semibold">{fmtPct(t.fraud_probability)}</div>
            <div className="mb-3 text-xs text-muted-foreground">probabilité de fraude (seuil F1 du modèle : voir métadonnées)</div>
            <Row k="Motif" v={t.reason} />
            <Row k="Règles déclenchées" v={t.rules_triggered.length ? t.rules_triggered.join(", ") : "aucune"} />
            <Row k="Version du modèle" v={t.model_version} />
            <Row k="Latence du scoring" v={`${t.latency_ms.toFixed(1)} ms${t.degraded ? " (mode dégradé)" : ""}`} />
            <Row k="Verdict analyste" v={t.label == null ? "non traité" : t.label === 1 ? "fraude confirmée" : "faux positif"} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Opération</CardTitle></CardHeader>
          <CardContent>
            <Row k="Canal" v={t.channel === "VISA_VIRTUAL" ? "Carte Visa virtuelle" : `Mobile Money · ${t.operator}`} />
            <Row k="Type" v={t.tx_type} />
            <Row k="Accès" v={<Tag>{t.access_channel}</Tag>} />
            <Row k="Montant" v={`${t.amount.toLocaleString("fr-FR")} ${t.currency} (${fmtUsd(t.amount_usd)})`} />
            <Row k="Province" v={t.province} />
            <Row k="Appareil" v={t.device_id} />
            {t.counterparty_id && <Row k="Destinataire" v={t.counterparty_id} />}
            {t.agent_id && <Row k="Agent" v={t.agent_id} />}
            {t.merchant_id && <Row k="Marchand" v={t.merchant_id} />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Part de chaque branche du modèle hybride</CardTitle></CardHeader>
          <CardContent className="h-56">
            <ResponsiveContainer>
              <BarChart data={branches} layout="vertical" margin={{ left: 10, right: 16 }}>
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} />
                <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" />
                <Tooltip formatter={(v: number) => v.toFixed(3)} />
                <Bar dataKey="v" name="Contribution (log-odds)">
                  {branches.map((b, i) => <Cell key={i} fill={b.v > 0 ? "hsl(var(--risk-critique))" : "hsl(var(--risk-faible))"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-[11px] text-muted-foreground">Rouge : pousse vers la fraude · vert : vers une transaction normale.</p>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Pourquoi cette décision ? (SHAP, branche XGBoost)</CardTitle></CardHeader>
          <CardContent className="h-72">
            {shap.length ? (
              <ResponsiveContainer>
                <BarChart data={shap} layout="vertical" margin={{ left: 10, right: 16 }}>
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" width={230} tick={{ fontSize: 11 }} />
                  <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip formatter={(v: number, _n, p: any) => [`${v.toFixed(3)} (valeur : ${p.payload.value})`, "Contribution"]} />
                  <Bar dataKey="shap">
                    {shap.map((s: any, i: number) => <Cell key={i} fill={s.shap > 0 ? "hsl(var(--risk-critique))" : "hsl(var(--risk-faible))"} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <p className="pt-10 text-center text-sm text-muted-foreground">Risque faible : l'explication détaillée n'est calculée qu'à partir du risque moyen.</p>}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Attention du LSTM sur l'historique</CardTitle></CardHeader>
          <CardContent>
            {attention.length ? (
              <div className="flex h-40 items-end gap-1.5">
                {attention.map((w, i) => (
                  <div key={i} className="flex h-full flex-1 flex-col items-center gap-1">
                    <div className="flex w-full flex-1 items-end">
                      <div className="w-full rounded-t bg-primary/80" style={{ height: `${Math.max(3, w * 100)}%` }} title={w.toFixed(3)} />
                    </div>
                    <span className="text-[10px] text-muted-foreground">{i === attention.length - 1 ? "act." : `-${attention.length - 1 - i}`}</span>
                  </div>
                ))}
              </div>
            ) : <p className="text-sm text-muted-foreground">Pas d'historique.</p>}
            <p className="mt-2 text-[11px] text-muted-foreground">Poids donnés aux transactions précédentes du client (-1 = la précédente, act. = celle-ci).</p>
          </CardContent>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader><CardTitle>Historique récent du client</CardTitle></CardHeader>
          <CardContent className="overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead className="border-y bg-muted/50 text-left text-xs text-muted-foreground">
                <tr><th className="px-5 py-2">Heure</th><th>Type</th><th>Accès</th><th className="text-right">Montant</th><th className="px-3 text-right">P(fraude)</th><th className="pr-5">Décision</th></tr>
              </thead>
              <tbody>
                {(history.data ?? []).map((h) => (
                  <tr key={h.transaction_id} className={`border-b last:border-0 ${h.transaction_id === t.transaction_id ? "bg-primary/5" : ""}`}>
                    <td className="px-5 py-1.5 text-xs">{fmtDate(h.tx_time)}</td><td className="text-xs">{h.tx_type}</td><td><Tag>{h.access_channel}</Tag></td>
                    <td className="tabular text-right">{fmtUsd(h.amount_usd)}</td><td className="tabular px-3 text-right">{fmtPct(h.fraud_probability)}</td>
                    <td className="pr-5"><ActionBadge action={h.action} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
