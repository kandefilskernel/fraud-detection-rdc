"use client";

import { useState } from "react";
import { ShieldCheck, ShieldX } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { api } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";
import { fmtDate, fmtInt } from "@/lib/utils";

export default function AuditPage() {
  const [page, setPage] = useState(1);
  const { data } = usePolling<any>(`/audit?page=${page}&size=50`, 5000);
  const [check, setCheck] = useState<any | null>(null);
  const [busy, setBusy] = useState(false);

  return (
    <>
      <PageHeader title="Journal d'audit" subtitle="Ajout seul, chaîné par SHA-256 : toute modification ou suppression est détectable.">
        <Button variant="outline" disabled={busy} onClick={async () => { setBusy(true); setCheck(await api("/audit/verify")); setBusy(false); }}>
          {busy ? "Vérification…" : "Vérifier l'intégrité de la chaîne"}
        </Button>
      </PageHeader>
      {check && (
        <div className={`mb-4 flex items-center gap-3 rounded-lg border p-3 text-sm ${check.valid ? "border-risk-faible/40 bg-risk-faible/10" : "border-risk-critique/40 bg-risk-critique/10"}`}>
          {check.valid ? <ShieldCheck className="h-5 w-5 text-risk-faible" /> : <ShieldX className="h-5 w-5 text-risk-critique" />}
          {check.valid ? `Chaîne intègre : ${fmtInt(check.checked)} entrées vérifiées.` : `Rupture détectée à l'entrée #${check.broken_at_id} (${fmtInt(check.checked)} entrées valides avant).`}
        </div>
      )}
      <Card>
        <CardContent className="overflow-x-auto p-0">
          <table className="w-full text-sm">
            <thead className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
              <tr><th className="px-4 py-2.5">#</th><th>Date</th><th>Acteur</th><th>Action</th><th>Objet</th><th>Détails</th><th className="px-4">Empreinte</th></tr>
            </thead>
            <tbody>
              {(data?.items ?? []).map((a: any) => {
                const { event_id, ...details } = a.details ?? {};
                return (
                  <tr key={a.id} className="border-b last:border-0 align-top">
                    <td className="px-4 py-2 text-xs text-muted-foreground">{a.id}</td>
                    <td className="whitespace-nowrap text-xs">{fmtDate(a.ts)}</td>
                    <td className="text-xs">{a.actor}</td>
                    <td className="text-xs font-medium">{a.action}</td>
                    <td className="text-xs">{a.entity} {a.entity_id}</td>
                    <td className="max-w-xs truncate font-mono text-[11px] text-muted-foreground" title={JSON.stringify(details)}>{JSON.stringify(details)}</td>
                    <td className="px-4 font-mono text-[11px] text-muted-foreground">{a.hash.slice(0, 12)}…</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </CardContent>
      </Card>
      <div className="mt-3 flex items-center justify-end gap-2 text-sm">
        <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>Plus récents</Button>
        <Button variant="outline" size="sm" disabled={!data || page * 50 >= data.total} onClick={() => setPage(page + 1)}>Plus anciens</Button>
      </div>
    </>
  );
}
