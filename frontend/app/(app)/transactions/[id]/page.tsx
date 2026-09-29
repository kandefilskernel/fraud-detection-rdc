"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowLeft, Check, Copy, History, Info } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { InvestigationAssistant } from "@/components/investigation-assistant";
import { ActionBadge, RiskBadge, RiskStripe, RISK_TONE, Tag } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { EmptyState, Skeleton, Table, Td, Th, Tr } from "@/components/ui/data";
import { usePolling } from "@/lib/use-polling";
import { cn, FEATURE_LABEL, fmtDate, fmtPct, fmtUsd, TX_TYPE_LABEL } from "@/lib/utils";

const BRANCH_LABEL: Record<string, string> = {
  random_forest: "Forêt aléatoire", xgboost: "XGBoost", lstm_attention: "LSTM + attention", autoencoder: "Autoencodeur",
};
const BRANCH_HINT: Record<string, string> = {
  random_forest: "profil instantané", xgboost: "profil instantané", lstm_attention: "séquence des 10 dernières opérations",
  autoencoder: "écart au comportement normal",
};

function CopyId({ value }: { value: string }) {
  const [ok, setOk] = useState(false);
  return (
    <button title="Copier l'identifiant" aria-label="Copier l'identifiant"
      onClick={() => { navigator.clipboard?.writeText(value).then(() => { setOk(true); setTimeout(() => setOk(false), 1200); }).catch(() => {}); }}
      className="grid h-7 w-7 place-items-center rounded-md text-subtle transition-colors hover:bg-surface-2 hover:text-foreground">
      {ok ? <Check className="h-3.5 w-3.5 text-risk-faible" /> : <Copy className="h-3.5 w-3.5" />}
    </button>
  );
}

function Field({ k, v, mono }: { k: string; v: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-2 text-[13px]">
      <dt className="shrink-0 text-muted-foreground">{k}</dt>
      <dd className={cn("min-w-0 truncate text-right font-medium", mono && "font-mono text-xs")}>{v}</dd>
    </div>
  );
}

/** Barres divergentes : à droite pousse vers la fraude, à gauche vers une transaction normale. */
function Diverging({ rows, unit }: { rows: { name: string; v: number; hint?: string }[]; unit?: string }) {
  const max = Math.max(1e-9, ...rows.map((r) => Math.abs(r.v)));
  return (
    <div className="space-y-2.5">
      {rows.map((r) => (
        <div key={r.name} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)_56px] items-center gap-3">
          <div className="min-w-0">
            <div className="truncate text-[13px]">{r.name}</div>
            {r.hint && <div className="truncate text-2xs text-subtle">{r.hint}</div>}
          </div>
          <div className="relative h-2 rounded-full bg-surface-2">
            <span className="absolute left-1/2 top-[-3px] h-[14px] w-px bg-border" />
            <div className={cn("absolute top-0 h-2 rounded-full", r.v >= 0 ? "left-1/2 bg-risk-critique" : "right-1/2 bg-risk-faible")}
                 style={{ width: `${(Math.abs(r.v) / max) * 50}%` }} />
          </div>
          <div className={cn("tabular text-right text-xs font-medium", r.v >= 0 ? "text-risk-critique" : "text-risk-faible")}>
            {r.v >= 0 ? "+" : "−"}{Math.abs(r.v).toFixed(unit === "pt" ? 3 : 2)}
          </div>
        </div>
      ))}
    </div>
  );
}


export default function TransactionDetail({ params }: { params: { id: string } }) {
  const { data: t, error } = usePolling<any>(`/transactions/${encodeURIComponent(params.id)}`, 0);
  const history = usePolling<any[]>(t ? `/transactions/by-user/${encodeURIComponent(t.user_id)}/history?limit=12` : null, 0);

  if (error) return <EmptyState icon={Info} title="Transaction introuvable" hint={error} />;
  if (!t) return (
    <div className="space-y-4"><Skeleton className="h-8 w-72" /><Skeleton className="h-28 w-full" />
      <div className="grid gap-4 lg:grid-cols-3"><Skeleton className="h-72 lg:col-span-2" /><Skeleton className="h-72" /></div></div>
  );

  const contributions = (t.explanation?.top_features ?? []).map((f: any) => ({
    name: FEATURE_LABEL[f.feature] ?? f.feature, v: f.shap, hint: `valeur : ${f.value}`,
  }));
  const branches = Object.entries(t.explanation?.branch_contributions ?? {}).map(([k, v]) => ({
    name: BRANCH_LABEL[k] ?? k, v: v as number, hint: BRANCH_HINT[k],
  }));
  const attention: number[] = t.explanation?.attention_on_history ?? [];
  const maxAtt = Math.max(1e-9, ...attention);
  const attribution: string | undefined = t.explanation?.attribution;
  const tone = RISK_TONE[t.risk_level];
  const verdict = t.label == null ? "Non traité" : t.label === 1 ? "Fraude confirmée" : "Faux positif";

  return (
    <>
      <Link href="/transactions" className="mb-3 inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-3.5 w-3.5" /> Transactions
      </Link>
      <PageHeader
        title={<span className="flex items-center gap-1.5"><span className="font-mono text-xl">{t.transaction_id}</span><CopyId value={t.transaction_id} /></span>}
        subtitle={<>Client <span className="font-mono">{t.user_id}</span> · {fmtDate(t.tx_time)} (heure RDC) · {t.channel === "VISA_VIRTUAL" ? "Visa virtuelle" : `Mobile Money · ${t.operator}`}</>}>
        <RiskBadge level={t.risk_level} /><ActionBadge action={t.action} />
      </PageHeader>

      <Card className="mb-4 grid divide-y md:grid-cols-4 md:divide-x md:divide-y-0">
        <div className="p-5">
          <div className="label-caps">Probabilité de fraude</div>
          <div className={cn("tabular mt-1 text-[34px] font-semibold leading-10 tracking-tight", tone?.text)}>{fmtPct(t.fraud_probability)}</div>
          <div className="relative mt-3 h-1.5 rounded-full bg-surface-2">
            <div className={cn("h-full rounded-full", tone?.bar ?? "bg-subtle")} style={{ width: `${Math.max(1.5, t.fraud_probability * 100)}%` }} />
          </div>
        </div>
        <div className="p-5">
          <div className="label-caps">Décision</div>
          <div className="mt-2"><ActionBadge action={t.action} /></div>
          <p className="mt-2 text-[13px] text-muted-foreground">{t.reason}</p>
        </div>
        <div className="p-5">
          <div className="label-caps">Règles déclenchées</div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {t.rules_triggered.length ? t.rules_triggered.map((r: string) => <Tag key={r} className="h-auto py-0.5">{r}</Tag>)
              : <span className="text-[13px] text-muted-foreground">Aucune : décision du modèle seul</span>}
          </div>
        </div>
        <div className="p-5">
          <div className="label-caps">Traitement</div>
          <dl className="mt-1">
            <Field k="Verdict analyste" v={<span className={cn(t.label === 1 && "text-risk-critique", t.label === 0 && "text-risk-faible")}>{verdict}</span>} />
            <Field k="Latence" v={`${t.latency_ms.toFixed(1)} ms${t.degraded ? " · dégradé" : ""}`} />
            <Field k="Modèle" v={t.model_version} mono />
          </dl>
        </div>
      </Card>

      <div className="grid items-start gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Pourquoi cette décision ?"
            description={attribution ? `Contributions des variables · ${attribution}` : "Contributions des variables du profil comportemental"}
            actions={<div className="hidden items-center gap-3 text-2xs text-muted-foreground sm:flex">
              <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-risk-critique" />vers la fraude</span>
              <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-sm bg-risk-faible" />vers la normale</span></div>} />
          <CardContent>
            {contributions.length ? <Diverging rows={contributions} unit={attribution?.startsWith("Saabas") ? "pt" : undefined} />
              : <EmptyState title="Explication non calculée" hint="Le détail n'est calculé qu'à partir du risque moyen, pour tenir le budget temps réel." className="py-8" />}
          </CardContent>
        </Card>
        <Card>
          <CardHeader title="Opération" />
          <CardContent className="pb-3">
            <dl className="divide-y">
              <Field k="Type" v={TX_TYPE_LABEL[t.tx_type] ?? t.tx_type} />
              <Field k="Accès" v={<Tag>{t.access_channel}</Tag>} />
              <Field k="Montant" v={<span className="tabular">{t.amount.toLocaleString("fr-FR")} {t.currency} <span className="text-muted-foreground">· {fmtUsd(t.amount_usd)}</span></span>} />
              <Field k="Province" v={t.province} />
              <Field k="Appareil" v={t.device_id} mono />
              {t.counterparty_id && <Field k="Destinataire" v={t.counterparty_id} mono />}
              {t.agent_id && <Field k="Agent" v={t.agent_id} mono />}
              {t.merchant_id && <Field k="Marchand" v={t.merchant_id} mono />}
            </dl>
          </CardContent>
        </Card>
      </div>

      {(t.action !== "APPROVE" || t.label != null) && <div className="mt-4"><InvestigationAssistant transactionId={t.transaction_id} /></div>}

      <div className="mt-4 grid items-start gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader title="Modèle hybride" description="Part de chaque branche dans la décision (log-odds)" />
          <CardContent>{branches.length ? <Diverging rows={branches} /> : <p className="text-[13px] text-muted-foreground">Non disponible.</p>}</CardContent>
        </Card>
        <Card>
          <CardHeader title="Attention du LSTM" description="Poids donnés aux opérations précédentes du client" />
          <CardContent>
            {attention.length ? (
              <div className="flex h-28 items-end gap-1">
                {attention.map((w, i) => (
                  <div key={i} className="flex h-full flex-1 flex-col items-center gap-1" title={`${(w * 100).toFixed(1)} %`}>
                    <div className="flex w-full flex-1 items-end">
                      <div className={cn("w-full rounded-t-sm", i === attention.length - 1 ? "bg-primary" : "bg-primary/45")}
                           style={{ height: `${Math.max(4, (w / maxAtt) * 100)}%` }} />
                    </div>
                    <span className="text-[10px] text-subtle">{i === attention.length - 1 ? "act." : `−${attention.length - 1 - i}`}</span>
                  </div>
                ))}
              </div>
            ) : <p className="text-[13px] text-muted-foreground">Pas d'historique pour ce client.</p>}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-4 overflow-hidden">
        <CardHeader title="Historique récent du client" description={`12 dernières opérations de ${t.user_id}`} actions={<History className="h-4 w-4 text-subtle" />} />
        <Table>
          <thead><tr><Th>Heure (RDC)</Th><Th>Opération</Th><Th>Accès</Th><Th align="right">Montant</Th><Th align="right">P(fraude)</Th><Th>Décision</Th></tr></thead>
          <tbody>
            {(history.data ?? []).map((h) => (
              <Tr key={h.transaction_id} selected={h.transaction_id === t.transaction_id}>
                <Td className="text-xs text-muted-foreground"><RiskStripe level={h.risk_level} />{fmtDate(h.tx_time)}</Td>
                <Td>{h.transaction_id === t.transaction_id ? <span className="font-medium">{TX_TYPE_LABEL[h.tx_type] ?? h.tx_type} · cette opération</span>
                  : <Link href={`/transactions/${h.transaction_id}`} className="hover:text-primary">{TX_TYPE_LABEL[h.tx_type] ?? h.tx_type}</Link>}</Td>
                <Td><Tag>{h.access_channel}</Tag></Td>
                <Td align="right" className="font-medium">{fmtUsd(h.amount_usd)}</Td>
                <Td align="right">{fmtPct(h.fraud_probability)}</Td>
                <Td><ActionBadge action={h.action} /></Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </>
  );
}
