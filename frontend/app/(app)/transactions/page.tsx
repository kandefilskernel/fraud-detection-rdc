"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, Inbox, Search, X } from "lucide-react";
import { LiveDot, PageHeader } from "@/components/app-shell";
import { ActionBadge, RiskBadge, RiskStripe, Tag } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState, SkeletonRows, Table, Td, Th, Tr } from "@/components/ui/data";
import { Input, Select } from "@/components/ui/select";
import { usePolling } from "@/lib/use-polling";
import { cn, fmtDate, fmtInt, fmtPct, fmtUsd, TX_TYPE_LABEL } from "@/lib/utils";

const SIZE = 50;
const EMPTY = { channel: "", operator: "", action: "", risk_level: "", search: "" };

function ProbabilityCell({ p }: { p: number }) {
  const tone = p >= 0.9 ? "bg-risk-critique" : p >= 0.5 ? "bg-risk-eleve" : p >= 0.05 ? "bg-risk-moyen" : "bg-risk-faible";
  return (
    <div className="flex items-center justify-end gap-2">
      <div className="hidden h-1 w-12 overflow-hidden rounded-full bg-surface-2 md:block">
        <div className={cn("h-full rounded-full", tone)} style={{ width: `${Math.max(3, p * 100)}%` }} />
      </div>
      <span className="w-12 text-right">{fmtPct(p)}</span>
    </div>
  );
}

export default function TransactionsPage() {
  const [f, setF] = useState(EMPTY);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);

  // lien depuis le tableau de bord : /transactions?search=U001234
  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get("search");
    if (q) { setSearch(q); setF((prev) => ({ ...prev, search: q })); }
  }, []);

  const qs = new URLSearchParams({ page: String(page), size: String(SIZE),
    ...Object.fromEntries(Object.entries(f).filter(([, v]) => v)) }).toString();
  const { data, error, loading } = usePolling<any>(`/transactions?${qs}`, 4000);
  const set = (k: keyof typeof EMPTY) => (e: React.ChangeEvent<HTMLSelectElement>) => { setPage(1); setF({ ...f, [k]: e.target.value }); };
  const pages = data ? Math.max(1, Math.ceil(data.total / SIZE)) : 1;
  const filtered = Object.values(f).some(Boolean);
  const from = data?.total ? (page - 1) * SIZE + 1 : 0;
  const to = data ? Math.min(page * SIZE, data.total) : 0;

  return (
    <>
      <PageHeader title="Transactions" subtitle={data ? `${fmtInt(data.total)} transactions scorées${filtered ? " pour ces filtres" : ""}` : "Chargement…"}>
        <LiveDot error={error} />
      </PageHeader>

      <Card className="overflow-hidden">
        <div className="flex flex-wrap items-center gap-2 border-b bg-surface p-3">
          <form className="flex min-w-[240px] flex-1 gap-2" onSubmit={(e) => { e.preventDefault(); setPage(1); setF({ ...f, search: search.trim() }); }}>
            <Input icon={Search} className="max-w-sm flex-1" placeholder="Identifiant de transaction ou de client" value={search}
                   onChange={(e) => setSearch(e.target.value)} aria-label="Rechercher" />
            <Button type="submit" variant="outline">Rechercher</Button>
          </form>
          <div className="flex flex-wrap items-center gap-2">
            <Select value={f.channel} onChange={set("channel")} aria-label="Canal" options={[["", "Tous les canaux"], ["MOBILE_MONEY", "Mobile Money"], ["VISA_VIRTUAL", "Visa virtuelle"]]} />
            <Select value={f.operator} onChange={set("operator")} aria-label="Opérateur" options={[["", "Tous les opérateurs"], ["VODACOM", "Vodacom"], ["AIRTEL", "Airtel"], ["ORANGE", "Orange"]]} />
            <Select value={f.action} onChange={set("action")} aria-label="Décision" options={[["", "Toutes les décisions"], ["APPROVE", "Approuvées"], ["VERIFY", "À vérifier"], ["BLOCK", "Bloquées"]]} />
            <Select value={f.risk_level} onChange={set("risk_level")} aria-label="Risque" options={[["", "Tous les risques"], ["CRITIQUE", "Critique"], ["ELEVE", "Élevé"], ["MOYEN", "Moyen"], ["FAIBLE", "Faible"]]} />
            {filtered && (
              <Button variant="ghost" size="sm" onClick={() => { setF(EMPTY); setSearch(""); setPage(1); }}>
                <X className="h-3.5 w-3.5" /> Réinitialiser
              </Button>
            )}
          </div>
        </div>

        <Table className="max-h-[calc(100vh-260px)]">
          <thead>
            <tr>
              <Th>Transaction</Th><Th>Heure (RDC)</Th><Th>Canal</Th><Th>Opération</Th><Th>Accès</Th>
              <Th align="right">Montant</Th><Th align="right">P(fraude)</Th><Th>Risque</Th><Th>Décision</Th><Th align="right">Latence</Th>
            </tr>
          </thead>
          <tbody>
            {loading && !data && <SkeletonRows cols={10} rows={10} />}
            {(data?.items ?? []).map((t: any) => (
              <Tr key={t.transaction_id}>
                <Td>
                  <RiskStripe level={t.risk_level} />
                  <Link href={`/transactions/${t.transaction_id}`} className="font-mono text-xs font-medium hover:text-primary">{t.transaction_id}</Link>
                  <div className="text-2xs text-muted-foreground">{t.user_id}</div>
                </Td>
                <Td className="text-xs text-muted-foreground">{fmtDate(t.tx_time)}</Td>
                <Td className="text-xs">{t.channel === "VISA_VIRTUAL" ? "Visa virtuelle" : t.operator}</Td>
                <Td className="text-[13px]">{TX_TYPE_LABEL[t.tx_type] ?? t.tx_type}</Td>
                <Td><Tag>{t.access_channel}</Tag></Td>
                <Td align="right" className="font-medium">{fmtUsd(t.amount_usd)}</Td>
                <Td align="right"><ProbabilityCell p={t.fraud_probability} /></Td>
                <Td><RiskBadge level={t.risk_level} /></Td>
                <Td><ActionBadge action={t.action} /></Td>
                <Td align="right" className="text-xs text-muted-foreground">{t.latency_ms.toFixed(1)} ms</Td>
              </Tr>
            ))}
            {data && data.items.length === 0 && (
              <tr><td colSpan={10}><EmptyState icon={Inbox} title="Aucune transaction" hint="Aucun résultat pour ces filtres. Modifiez-les ou réinitialisez la recherche." /></td></tr>
            )}
          </tbody>
        </Table>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t px-5 py-3 text-xs text-muted-foreground">
          <span className="tabular">{data ? `${fmtInt(from)}–${fmtInt(to)} sur ${fmtInt(data.total)}` : "—"}</span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}><ChevronLeft className="h-3.5 w-3.5" /> Précédent</Button>
            <span className="tabular px-1">Page {page} / {pages}</span>
            <Button variant="outline" size="sm" disabled={page >= pages} onClick={() => setPage(page + 1)}>Suivant <ChevronRight className="h-3.5 w-3.5" /></Button>
          </div>
        </div>
      </Card>
    </>
  );
}
