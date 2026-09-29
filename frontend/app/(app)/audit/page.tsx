"use client";

import { useState } from "react";
import { ChevronLeft, ChevronRight, Link2, ScrollText, ShieldCheck, ShieldX } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState, SkeletonRows, Table, Td, Th, Tr } from "@/components/ui/data";
import { api } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";
import { cn, fmtDate, fmtInt } from "@/lib/utils";

const ACTION_TONE: Record<string, string> = {
  PLAINTE_CLIENT: "text-risk-critique", DOSSIER_MODIFIE: "text-primary", ASSISTANT_ENQUETE: "text-foreground",
  RETOUR_OPERATEUR: "text-risk-eleve",
};

export default function AuditPage() {
  const [page, setPage] = useState(1);
  const { data, loading } = usePolling<any>(`/audit?page=${page}&size=50`, 5000);
  const [check, setCheck] = useState<any | null>(null);
  const [busy, setBusy] = useState(false);

  return (
    <>
      <PageHeader title="Journal d'audit" subtitle="Ajout seul, chaque entrée scellée par l'empreinte SHA-256 de la précédente : toute modification est détectable.">
        <Button variant="outline" disabled={busy} onClick={async () => { setBusy(true); try { setCheck(await api("/audit/verify")); } finally { setBusy(false); } }}>
          <Link2 className="h-4 w-4" /> {busy ? "Vérification…" : "Vérifier l'intégrité"}
        </Button>
      </PageHeader>

      {check && (
        <div className={cn("mb-4 flex animate-fade-in items-center gap-3 rounded-lg border p-3.5 text-[13px]",
          check.valid ? "border-risk-faible/30 bg-risk-faible/10" : "border-risk-critique/30 bg-risk-critique/10")}>
          {check.valid ? <ShieldCheck className="h-5 w-5 shrink-0 text-risk-faible" /> : <ShieldX className="h-5 w-5 shrink-0 text-risk-critique" />}
          <div>
            <div className="font-medium">{check.valid ? "Chaîne intègre" : "Rupture de la chaîne détectée"}</div>
            <div className="text-xs text-muted-foreground">
              {check.valid ? `${fmtInt(check.checked)} entrées vérifiées, aucune modification ni suppression.`
                : `À l'entrée #${check.broken_at_id} (${fmtInt(check.checked)} entrées valides avant).`}
            </div>
          </div>
        </div>
      )}

      <Card className="overflow-hidden">
        <Table className="max-h-[calc(100vh-240px)]">
          <thead><tr><Th>#</Th><Th>Date</Th><Th>Acteur</Th><Th>Action</Th><Th>Objet</Th><Th>Détails</Th><Th>Empreinte</Th></tr></thead>
          <tbody>
            {loading && !data && <SkeletonRows cols={7} rows={10} />}
            {(data?.items ?? []).map((a: any) => {
              const { event_id, ...details } = a.details ?? {};
              const text = JSON.stringify(details);
              return (
                <Tr key={a.id}>
                  <Td className="tabular text-xs text-muted-foreground">{a.id}</Td>
                  <Td className="text-xs">{fmtDate(a.ts)}</Td>
                  <Td className="text-xs">{a.actor}</Td>
                  <Td className={cn("font-mono text-2xs font-medium", ACTION_TONE[a.action] ?? "text-foreground")}>{a.action}</Td>
                  <Td className="text-xs"><span className="text-muted-foreground">{a.entity}</span> <span className="font-mono">{a.entity_id}</span></Td>
                  <Td className="max-w-[320px] truncate font-mono text-2xs text-muted-foreground" title={text}>{text}</Td>
                  <Td className="font-mono text-2xs text-subtle" title={a.hash}>{a.hash.slice(0, 12)}…</Td>
                </Tr>
              );
            })}
            {data && data.items.length === 0 && <tr><td colSpan={7}><EmptyState icon={ScrollText} title="Journal vide" /></td></tr>}
          </tbody>
        </Table>
        <div className="flex items-center justify-between border-t px-5 py-3 text-xs text-muted-foreground">
          <span className="tabular">{data ? `${fmtInt(data.total)} entrées` : "—"}</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}><ChevronLeft className="h-3.5 w-3.5" /> Plus récentes</Button>
            <Button variant="outline" size="sm" disabled={!data || page * 50 >= data.total} onClick={() => setPage(page + 1)}>Plus anciennes <ChevronRight className="h-3.5 w-3.5" /></Button>
          </div>
        </div>
      </Card>
    </>
  );
}
