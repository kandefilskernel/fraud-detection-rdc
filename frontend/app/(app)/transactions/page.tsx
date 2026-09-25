"use client";

import Link from "next/link";
import { useState } from "react";
import { LiveDot, PageHeader } from "@/components/app-shell";
import { ActionBadge, RiskBadge, Tag } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input, Select } from "@/components/ui/select";
import { usePolling } from "@/lib/use-polling";
import { fmtDate, fmtInt, fmtPct, fmtUsd } from "@/lib/utils";

const SIZE = 50;

export default function TransactionsPage() {
  const [f, setF] = useState({ channel: "", operator: "", action: "", risk_level: "", search: "" });
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const qs = new URLSearchParams({ page: String(page), size: String(SIZE),
    ...Object.fromEntries(Object.entries(f).filter(([, v]) => v)) }).toString();
  const { data, error } = usePolling<any>(`/transactions?${qs}`, 4000);
  const set = (k: string) => (e: React.ChangeEvent<HTMLSelectElement>) => { setPage(1); setF({ ...f, [k]: e.target.value }); };
  const pages = data ? Math.max(1, Math.ceil(data.total / SIZE)) : 1;

  return (
    <>
      <PageHeader title="Transactions" subtitle={data ? `${fmtInt(data.total)} transactions scorées` : "Chargement…"}>
        <LiveDot error={error} />
      </PageHeader>

      <Card className="mb-4">
        <CardContent className="flex flex-wrap items-center gap-2 pt-4">
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); setPage(1); setF({ ...f, search: search.trim() }); }}>
            <Input className="w-56" placeholder="ID transaction ou client…" value={search} onChange={(e) => setSearch(e.target.value)} />
            <Button type="submit" variant="outline">Rechercher</Button>
          </form>
          <Select value={f.channel} onChange={set("channel")} options={[["", "Tous canaux"], ["MOBILE_MONEY", "Mobile Money"], ["VISA_VIRTUAL", "Visa virtuelle"]]} />
          <Select value={f.operator} onChange={set("operator")} options={[["", "Tous opérateurs"], ["VODACOM", "Vodacom"], ["AIRTEL", "Airtel"], ["ORANGE", "Orange"]]} />
          <Select value={f.action} onChange={set("action")} options={[["", "Toutes décisions"], ["APPROVE", "Approuvées"], ["VERIFY", "À vérifier"], ["BLOCK", "Bloquées"]]} />
          <Select value={f.risk_level} onChange={set("risk_level")} options={[["", "Tous risques"], ["CRITIQUE", "Critique"], ["ELEVE", "Élevé"], ["MOYEN", "Moyen"], ["FAIBLE", "Faible"]]} />
        </CardContent>
      </Card>

      <Card>
        <CardContent className="overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
              <tr>
                <th className="px-4 py-2.5">Heure transaction</th><th>Client</th><th>Canal</th><th>Type</th><th>Accès</th>
                <th className="text-right">Montant</th><th className="px-3 text-right">P(fraude)</th><th>Risque</th><th>Décision</th>
                <th className="px-4 text-right">Latence</th>
              </tr>
            </thead>
            <tbody>
              {(data?.items ?? []).map((t: any) => (
                <tr key={t.transaction_id} className="border-b last:border-0 hover:bg-muted/40">
                  <td className="px-4 py-2 text-xs text-muted-foreground">{fmtDate(t.tx_time)}</td>
                  <td><Link href={`/transactions/${t.transaction_id}`} className="font-medium text-primary hover:underline">{t.user_id}</Link></td>
                  <td className="text-xs">{t.channel === "VISA_VIRTUAL" ? "Visa virtuelle" : t.operator}</td>
                  <td className="text-xs">{t.tx_type}</td>
                  <td><Tag>{t.access_channel}</Tag></td>
                  <td className="tabular text-right">{fmtUsd(t.amount_usd)}</td>
                  <td className="tabular px-3 text-right">{fmtPct(t.fraud_probability)}</td>
                  <td><RiskBadge level={t.risk_level} /></td>
                  <td><ActionBadge action={t.action} /></td>
                  <td className="tabular px-4 text-right text-xs text-muted-foreground">{t.latency_ms.toFixed(1)} ms</td>
                </tr>
              ))}
              {data && data.items.length === 0 && <tr><td colSpan={10} className="py-10 text-center text-muted-foreground">Aucune transaction pour ces filtres.</td></tr>}
            </tbody>
          </table>
        </CardContent>
      </Card>
      <div className="mt-3 flex items-center justify-end gap-2 text-sm">
        <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>Précédent</Button>
        <span className="text-muted-foreground">Page {page} / {pages}</span>
        <Button variant="outline" size="sm" disabled={page >= pages} onClick={() => setPage(page + 1)}>Suivant</Button>
      </div>
    </>
  );
}
