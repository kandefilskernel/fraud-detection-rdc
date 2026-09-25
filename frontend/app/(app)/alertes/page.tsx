"use client";

import Link from "next/link";
import { useState } from "react";
import { CheckCircle2, UserCheck, XCircle } from "lucide-react";
import { LiveDot, PageHeader } from "@/components/app-shell";
import { ActionBadge, RiskBadge, StatusBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { api, getUser } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";
import { cn, fmtDate, fmtInt, fmtPct, fmtUsd, RISK_LABEL } from "@/lib/utils";

const LEVELS = ["", "CRITIQUE", "ELEVE", "MOYEN", "FAIBLE"];

export default function AlertesPage() {
  const me = getUser();
  const [level, setLevel] = useState("");
  const [status, setStatus] = useState("OUVERT");
  const [selected, setSelected] = useState<any | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const qs = new URLSearchParams({ size: "100", ...(level && { risk_level: level }), ...(status && { status }) }).toString();
  const { data, error, reload } = usePolling<any>(`/cases?${qs}`, 4000);
  const counts = data?.counts ?? {};

  async function act(body: Record<string, unknown>) {
    if (!selected) return;
    setBusy(true); setMsg(null);
    try {
      const updated = await api(`/cases/${selected.id}`, { method: "PATCH", body: JSON.stringify(body) });
      setSelected(updated); setNote(""); reload();
    } catch (e: any) { setMsg(e.message); } finally { setBusy(false); }
  }

  return (
    <>
      <PageHeader title="Alertes & dossiers" subtitle="Chaque alerte VERIFY ou BLOCK ouvre un dossier ; votre verdict entraîne le modèle suivant.">
        <LiveDot error={error} />
        <Select value={status} onChange={(e) => setStatus(e.target.value)}
          options={[["OUVERT", `Ouverts (${fmtInt(counts.OUVERT ?? 0)})`], ["EN_COURS", `En cours (${fmtInt(counts.EN_COURS ?? 0)})`],
                    ["FRAUDE_CONFIRMEE", `Fraudes confirmées (${fmtInt(counts.FRAUDE_CONFIRMEE ?? 0)})`],
                    ["FAUX_POSITIF", `Faux positifs (${fmtInt(counts.FAUX_POSITIF ?? 0)})`], ["", "Tous"]]} />
      </PageHeader>

      <div className="mb-4 flex flex-wrap gap-2">
        {LEVELS.map((l) => (
          <button key={l || "all"} onClick={() => setLevel(l)}
            className={cn("rounded-full border px-3 py-1 text-sm transition-colors", level === l ? "border-primary bg-primary text-white" : "bg-card hover:bg-muted")}>
            {l ? RISK_LABEL[l] : "Tous niveaux"}
          </button>
        ))}
      </div>

      <div className="grid gap-4 xl:grid-cols-5">
        <Card className="xl:col-span-3">
          <CardContent className="overflow-x-auto p-0">
            <table className="w-full text-sm">
              <thead className="border-b bg-muted/50 text-left text-xs text-muted-foreground">
                <tr><th className="px-4 py-2.5">#</th><th>Client</th><th>Type</th><th className="text-right">Montant</th><th className="px-3 text-right">P(fraude)</th><th>Risque</th><th>Statut</th></tr>
              </thead>
              <tbody>
                {(data?.items ?? []).map((c: any) => (
                  <tr key={c.id} onClick={() => { setSelected(c); setMsg(null); }}
                      className={cn("cursor-pointer border-b last:border-0 hover:bg-muted/40", selected?.id === c.id && "bg-primary/5")}>
                    <td className="px-4 py-2 text-xs text-muted-foreground">{c.id}</td>
                    <td className="font-medium">{c.user_id}</td>
                    <td className="text-xs">{c.tx_type}<div className="text-muted-foreground">{c.channel === "VISA_VIRTUAL" ? "Visa" : "Mobile Money"}</div></td>
                    <td className="tabular text-right">{fmtUsd(c.amount_usd)}</td>
                    <td className="tabular px-3 text-right">{fmtPct(c.fraud_probability)}</td>
                    <td><RiskBadge level={c.risk_level} /></td>
                    <td><StatusBadge status={c.status} /></td>
                  </tr>
                ))}
                {data && data.items.length === 0 && <tr><td colSpan={7} className="py-10 text-center text-muted-foreground">Aucun dossier.</td></tr>}
              </tbody>
            </table>
          </CardContent>
        </Card>

        <Card className="xl:col-span-2 xl:sticky xl:top-4 xl:self-start">
          {!selected ? (
            <CardContent className="py-16 text-center text-sm text-muted-foreground">Sélectionnez un dossier pour le traiter.</CardContent>
          ) : (
            <>
              <CardHeader><CardTitle>Dossier #{selected.id}</CardTitle><StatusBadge status={selected.status} /></CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex flex-wrap gap-2"><RiskBadge level={selected.risk_level} /><ActionBadge action={selected.action} /></div>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                  <dt className="text-muted-foreground">Client</dt><dd className="font-medium">{selected.user_id}</dd>
                  <dt className="text-muted-foreground">Opération</dt><dd>{selected.tx_type}</dd>
                  <dt className="text-muted-foreground">Montant</dt><dd className="tabular">{fmtUsd(selected.amount_usd)}</dd>
                  <dt className="text-muted-foreground">Probabilité</dt><dd className="tabular">{fmtPct(selected.fraud_probability)}</dd>
                  <dt className="text-muted-foreground">Heure</dt><dd>{fmtDate(selected.tx_time)}</dd>
                  <dt className="text-muted-foreground">Ouvert le</dt><dd>{fmtDate(selected.created_at)}</dd>
                </dl>
                <Link href={`/transactions/${selected.transaction_id}`} className="inline-block text-xs font-medium text-primary hover:underline">
                  Voir l'explication du modèle et l'historique du client →
                </Link>
                {selected.resolution_note && <p className="rounded-md bg-muted p-2.5 text-xs">{selected.resolution_note}</p>}
                {["OUVERT", "EN_COURS"].includes(selected.status) && (
                  <>
                    {selected.status === "OUVERT" && (
                      <Button variant="outline" className="w-full" disabled={busy} onClick={() => act({ status: "EN_COURS", assigned_to: me?.id })}>
                        <UserCheck className="h-4 w-4" /> Prendre en charge
                      </Button>
                    )}
                    <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={3}
                      placeholder="Note de résolution (obligatoire) : appel client, vérification agent…"
                      className="w-full rounded-md border bg-card p-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40" />
                    <div className="grid grid-cols-2 gap-2">
                      <Button variant="danger" disabled={busy || !note.trim()} onClick={() => act({ status: "FRAUDE_CONFIRMEE", resolution_note: note })}>
                        <XCircle className="h-4 w-4" /> Fraude confirmée
                      </Button>
                      <Button variant="success" disabled={busy || !note.trim()} onClick={() => act({ status: "FAUX_POSITIF", resolution_note: note })}>
                        <CheckCircle2 className="h-4 w-4" /> Faux positif
                      </Button>
                    </div>
                  </>
                )}
                {msg && <p className="text-xs text-risk-critique">{msg}</p>}
              </CardContent>
            </>
          )}
        </Card>
      </div>
    </>
  );
}
