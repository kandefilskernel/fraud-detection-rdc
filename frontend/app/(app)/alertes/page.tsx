"use client";

import Link from "next/link";
import { useState } from "react";
import { ArrowUpRight, CheckCircle2, FolderSearch, Inbox, UserCheck, XCircle } from "lucide-react";
import { LiveDot, PageHeader } from "@/components/app-shell";
import { ActionBadge, RiskBadge, RiskStripe, StatusBadge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { EmptyState, Segmented, SkeletonRows, Table, Td, Th, Tr } from "@/components/ui/data";
import { Textarea } from "@/components/ui/select";
import { api, getUser } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";
import { cn, fmtDate, fmtPct, fmtRelative, fmtUsd, TX_TYPE_LABEL } from "@/lib/utils";

type Status = "OUVERT" | "EN_COURS" | "FRAUDE_CONFIRMEE" | "FAUX_POSITIF" | "";
type Level = "" | "CRITIQUE" | "ELEVE" | "MOYEN" | "FAIBLE";
const NOTES = [
  { label: "Opération confirmée", text: "Client joint par un canal sûr : il confirme l'opération." },
  { label: "Opération non reconnue", text: "Client joint par un canal sûr : il ne reconnaît pas l'opération." },
  { label: "Bénéficiaire suspect", text: "Bénéficiaire lié à d'autres fraudes confirmées." },
  { label: "SIM changée", text: "Changement de SIM récent confirmé par l'opérateur." },
];

export default function AlertesPage() {
  const me = getUser();
  const [level, setLevel] = useState<Level>("");
  const [status, setStatus] = useState<Status>("OUVERT");
  const [selected, setSelected] = useState<any | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const qs = new URLSearchParams({ size: "100", ...(level && { risk_level: level }), ...(status && { status }) }).toString();
  const { data, error, loading, reload } = usePolling<any>(`/cases?${qs}`, 4000);
  const counts = data?.counts ?? {};
  const total = Object.values(counts).reduce((a: number, b: any) => a + Number(b), 0) as number;

  async function act(body: Record<string, unknown>) {
    if (!selected) return;
    setBusy(true); setMsg(null);
    try {
      const updated = await api(`/cases/${selected.id}`, { method: "PATCH", body: JSON.stringify(body) });
      setSelected(updated); setNote(""); reload();
    } catch (e: any) { setMsg(e.message); } finally { setBusy(false); }
  }
  const open = selected && ["OUVERT", "EN_COURS"].includes(selected.status);

  return (
    <>
      <PageHeader title="Alertes & dossiers" subtitle="Chaque vérification ou blocage ouvre un dossier. Votre verdict devient une étiquette pour le prochain modèle.">
        <LiveDot error={error} />
      </PageHeader>

      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <Segmented<Status> value={status} onChange={(v) => { setStatus(v); setSelected(null); }} options={[
          { value: "OUVERT", label: "Ouverts", count: counts.OUVERT ?? 0 },
          { value: "EN_COURS", label: "En cours", count: counts.EN_COURS ?? 0 },
          { value: "FRAUDE_CONFIRMEE", label: "Fraudes confirmées", count: counts.FRAUDE_CONFIRMEE ?? 0 },
          { value: "FAUX_POSITIF", label: "Faux positifs", count: counts.FAUX_POSITIF ?? 0 },
          { value: "", label: "Tous", count: total },
        ]} />
        <Segmented<Level> value={level} onChange={setLevel} options={[
          { value: "", label: "Tous risques" }, { value: "CRITIQUE", label: "Critique" }, { value: "ELEVE", label: "Élevé" },
          { value: "MOYEN", label: "Moyen" }, { value: "FAIBLE", label: "Faible" },
        ]} />
      </div>

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
        <Card className="overflow-hidden">
          <Table className="max-h-[calc(100vh-240px)]">
            <thead><tr><Th>Dossier</Th><Th>Opération</Th><Th align="right">Montant</Th><Th align="right">P(fraude)</Th><Th>Risque</Th><Th>Statut</Th><Th align="right">Ouvert</Th></tr></thead>
            <tbody>
              {loading && !data && <SkeletonRows cols={7} rows={10} />}
              {(data?.items ?? []).map((c: any) => (
                <Tr key={c.id} onClick={() => { setSelected(c); setMsg(null); setNote(""); }} selected={selected?.id === c.id}>
                  <Td>
                    <RiskStripe level={c.risk_level} />
                    <div className="text-[13px] font-medium">#{c.id}</div>
                    <div className="font-mono text-2xs text-muted-foreground">{c.user_id}</div>
                  </Td>
                  <Td><div className="text-[13px]">{TX_TYPE_LABEL[c.tx_type] ?? c.tx_type}</div><div className="text-2xs text-muted-foreground">{c.channel === "VISA_VIRTUAL" ? "Visa virtuelle" : "Mobile Money"}</div></Td>
                  <Td align="right" className="font-medium">{fmtUsd(c.amount_usd)}</Td>
                  <Td align="right">{fmtPct(c.fraud_probability)}</Td>
                  <Td><RiskBadge level={c.risk_level} /></Td>
                  <Td><StatusBadge status={c.status} /></Td>
                  <Td align="right" className="text-xs text-muted-foreground">{fmtRelative(c.created_at)}</Td>
                </Tr>
              ))}
              {data && data.items.length === 0 && (
                <tr><td colSpan={7}><EmptyState icon={Inbox} title="Aucun dossier" hint="Aucun dossier ne correspond à ces filtres." /></td></tr>
              )}
            </tbody>
          </Table>
        </Card>

        <Card className="xl:sticky xl:top-20">
          {!selected ? (
            <EmptyState icon={FolderSearch} title="Sélectionnez un dossier" hint="Cliquez sur une ligne pour voir le détail, prendre le dossier en charge et rendre votre verdict." className="py-16" />
          ) : (
            <div className="animate-fade-in">
              <div className="flex items-start justify-between gap-3 border-b px-5 py-4">
                <div>
                  <div className="label-caps">Dossier</div>
                  <div className="mt-0.5 text-lg font-semibold">#{selected.id}</div>
                </div>
                <StatusBadge status={selected.status} />
              </div>
              <CardContent className="space-y-4 pt-4">
                <div className="flex flex-wrap gap-1.5"><RiskBadge level={selected.risk_level} /><ActionBadge action={selected.action} /></div>
                <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border bg-border">
                  {[
                    ["Montant", fmtUsd(selected.amount_usd)], ["Probabilité", fmtPct(selected.fraud_probability)],
                    ["Opération", TX_TYPE_LABEL[selected.tx_type] ?? selected.tx_type], ["Client", selected.user_id],
                    ["Heure de l'opération", fmtDate(selected.tx_time)], ["Dossier ouvert", fmtDate(selected.created_at)],
                  ].map(([k, v]) => (
                    <div key={k} className="bg-surface px-3 py-2.5">
                      <div className="text-2xs text-muted-foreground">{k}</div>
                      <div className={cn("tabular mt-0.5 truncate text-[13px] font-medium", k === "Client" && "font-mono text-xs")}>{v}</div>
                    </div>
                  ))}
                </div>
                <Link href={`/transactions/${selected.transaction_id}`}
                  className="flex items-center justify-between rounded-lg border px-3 py-2.5 text-[13px] font-medium transition-colors hover:border-primary/40 hover:bg-primary-soft/50">
                  Explication du modèle, assistant d'enquête et historique <ArrowUpRight className="h-4 w-4 text-primary" />
                </Link>

                {selected.resolution_note && (
                  <div className="rounded-lg bg-surface-2 p-3 text-[13px]">
                    <div className="label-caps mb-1">Note de résolution</div>{selected.resolution_note}
                  </div>
                )}

                {open && (
                  <div className="space-y-3 border-t pt-4">
                    {selected.status === "OUVERT" && (
                      <Button variant="outline" className="w-full" disabled={busy} onClick={() => act({ status: "EN_COURS", assigned_to: me?.id })}>
                        <UserCheck className="h-4 w-4" /> Prendre en charge
                      </Button>
                    )}
                    <div>
                      <label htmlFor="note" className="mb-1.5 block text-xs font-medium">Note de résolution <span className="font-normal text-muted-foreground">(obligatoire)</span></label>
                      <Textarea id="note" value={note} onChange={(e) => setNote(e.target.value)} rows={3}
                        placeholder="Appel au client, vérification auprès de l'agent ou de l'opérateur…" />
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {NOTES.map((n) => (
                          <button key={n.label} title={n.text} onClick={() => setNote((prev) => (prev ? `${prev} ${n.text}` : n.text))}
                            className="rounded-full border bg-surface px-2 py-1 text-2xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground">
                            + {n.label}
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <Button variant="danger" disabled={busy || !note.trim()} onClick={() => act({ status: "FRAUDE_CONFIRMEE", resolution_note: note })}>
                        <XCircle className="h-4 w-4" /> Fraude confirmée
                      </Button>
                      <Button variant="success" disabled={busy || !note.trim()} onClick={() => act({ status: "FAUX_POSITIF", resolution_note: note })}>
                        <CheckCircle2 className="h-4 w-4" /> Faux positif
                      </Button>
                    </div>
                    <p className="text-2xs text-subtle">Une fraude confirmée marque aussi l'appareil, le bénéficiaire et l'agent dans le profil de réputation.</p>
                  </div>
                )}
                {msg && <p className="rounded-md bg-risk-critique/10 px-3 py-2 text-xs text-risk-critique">{msg}</p>}
              </CardContent>
            </div>
          )}
        </Card>
      </div>
    </>
  );
}
