"use client";

/** Assistant d'enquête (RAG) : cas passés similaires, procédures internes et note d'instruction.
 *  Avis préparatoire à la demande de l'analyste ; la décision reste la sienne. */
import { useState } from "react";
import Link from "next/link";
import { BookOpen, FileSearch, Loader2, RefreshCw, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { api } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";
import { cn, fmtPct, fmtUsd } from "@/lib/utils";

type Case = {
  ref: string; id?: string; source: string; date: string; hour: number; tx_type: string; amount_usd: number;
  province?: string; outcome: "FRAUDE_CONFIRMEE" | "FAUX_POSITIF"; typology?: string; similarity: number;
  facts?: string[]; mitigating?: string[]; report_delay_days?: number; transaction_id?: string;
};
type Procedure = { ref: string; id: string; title: string; section: string; text: string };
type Result = {
  mode: "llm" | "extractif"; model: string | null; fallback_used: boolean; truncated: boolean; notice: string | null;
  disclaimer: string; synthesis: string; cached: boolean; generated_at: string;
  typology_hypotheses: { typology: string; label: string; share: number }[];
  precedents: { n: number; n_confirmed?: number; n_cleared?: number; fraud_share: number | null };
  entity_links: { entity: string; confirmed_frauds: number; alerts: number; other_clients: number }[];
  similar_cases: Case[]; procedures: Procedure[];
};

const SOURCE_LABEL: Record<string, string> = {
  ALERTE_INSTRUITE: "archive · alerte instruite", PLAINTE_CLIENT: "archive · plainte client",
  VERDICT_PLATEFORME: "verdict récent d'un analyste",
};

/** Rendu Markdown minimal (titres, listes, gras) ; les références [C1] / [P2] deviennent des pastilles. */
function Inline({ text, titles }: { text: string; titles: Record<string, string> }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\[[CP]\d+\])/g);
  return <>{parts.map((p, i) => {
    if (/^\*\*[^*]+\*\*$/.test(p)) return <strong key={i}>{p.slice(2, -2)}</strong>;
    const ref = p.match(/^\[([CP]\d+)\]$/)?.[1];
    if (ref) return (
      <a key={i} href={`#ref-${ref}`} title={titles[ref]}
         className={cn("mx-0.5 rounded px-1 py-px align-baseline text-[11px] font-semibold no-underline ring-1 ring-inset",
           ref.startsWith("C") ? "bg-primary-soft text-primary ring-primary/30" : "bg-surface-2 text-foreground ring-border")}>{ref}</a>
    );
    return <span key={i}>{p}</span>;
  })}</>;
}

function Markdown({ source, titles }: { source: string; titles: Record<string, string> }) {
  const blocks: React.ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  const flush = () => {
    if (!list) return;
    const Tag = list.ordered ? "ol" : "ul";
    blocks.push(<Tag key={blocks.length} className={cn("my-1.5 space-y-1 pl-5 text-sm", list.ordered ? "list-decimal" : "list-disc")}>
      {list.items.map((it, i) => <li key={i}><Inline text={it} titles={titles} /></li>)}</Tag>);
    list = null;
  };
  for (const raw of source.split("\n")) {
    const line = raw.trimEnd();
    const h = line.match(/^#{1,4}\s+(.*)$/);
    const ul = line.match(/^\s*[-*]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (h) { flush(); blocks.push(<h4 key={blocks.length} className="mb-1 mt-4 text-[13px] font-semibold text-foreground first:mt-0">{h[1]}</h4>); }
    else if (ul || ol) {
      const ordered = !!ol;
      if (!list || list.ordered !== ordered) { flush(); list = { ordered, items: [] }; }
      list.items.push((ul ?? ol)![1]);
    } else if (!line.trim()) flush();
    else { flush(); blocks.push(<p key={blocks.length} className="my-1 text-sm leading-relaxed"><Inline text={line} titles={titles} /></p>); }
  }
  flush();
  return <div>{blocks}</div>;
}

export function InvestigationAssistant({ transactionId }: { transactionId: string }) {
  const status = usePolling<{ llm_enabled: boolean; model: string | null; archive_cases: number; procedure_sections: number }>("/assistant/status", 0);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run(refresh = false) {
    setLoading(true); setError(null);
    try {
      setResult(await api<Result>(`/transactions/${encodeURIComponent(transactionId)}/assistant${refresh ? "?refresh=true" : ""}`, { method: "POST" }));
    } catch (e: any) {
      setError(e.message ?? "erreur");
    } finally {
      setLoading(false);
    }
  }

  const titles: Record<string, string> = {};
  result?.similar_cases.forEach((c) => { titles[c.ref] = `${c.date} · ${c.tx_type} · ${fmtUsd(c.amount_usd)} · ${c.outcome === "FRAUDE_CONFIRMEE" ? "fraude confirmée" : "faux positif"}`; });
  result?.procedures.forEach((p) => { titles[p.ref] = `${p.title} — ${p.section}`; });
  const s = status.data;

  return (
    <Card className="overflow-hidden">
      <CardHeader
        title={<span className="flex items-center gap-2"><FileSearch className="h-4 w-4 text-primary" /> Assistant d'enquête</span>}
        description={<>Cas passés semblables{s ? ` (${s.archive_cases.toLocaleString("fr-FR")} dossiers)` : ""}, procédures internes et note d'instruction.{" "}
          {s && (s.llm_enabled ? <>Rédaction par <b className="font-medium text-foreground">{s.model}</b>, sans aucun identifiant client.</> : <>Rédaction sans modèle de langage (aucune clé d'API configurée).</>)}</>}
        actions={result ? (
          <Button variant="outline" size="sm" onClick={() => run(true)} disabled={loading}><RefreshCw className="h-3.5 w-3.5" /> Régénérer</Button>
        ) : (
          <Button size="sm" onClick={() => run()} disabled={loading}>
            {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />} Préparer la note d'instruction
          </Button>
        )} />
      <CardContent>
        {loading && (
          <div className="space-y-2.5 py-2">
            <p className="flex items-center gap-2 text-[13px] text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" /> Recherche des cas similaires et rédaction de la note…</p>
            <div className="h-2 w-2/3 animate-pulse rounded bg-surface-2" /><div className="h-2 w-1/2 animate-pulse rounded bg-surface-2" />
          </div>
        )}
        {error && <p className="rounded-md bg-risk-critique/10 px-3 py-2 text-[13px] text-risk-critique">{error}</p>}
        {!result && !loading && !error && (
          <p className="rounded-lg border border-dashed bg-surface-2/50 px-4 py-3 text-[13px] text-muted-foreground">
            Avis préparatoire à la demande : l'assistant ne modifie ni la décision ni le dossier, et chaque consultation est inscrite au journal d'audit.
          </p>
        )}
        {result && !loading && (
          <div className="grid animate-fade-in gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <div className="min-w-0">
              {result.notice && <p className="mb-3 rounded-md border border-risk-moyen/30 bg-risk-moyen/10 px-3 py-2 text-xs">{result.notice}</p>}
              <div className="rounded-lg border bg-surface-2/40 p-4"><Markdown source={result.synthesis} titles={titles} /></div>
              <p className="mt-3 text-2xs text-subtle">
                {result.disclaimer} · {result.mode === "llm" ? `Rédigé par ${result.model}${result.fallback_used ? " (modèle de repli)" : ""}` : "Note assemblée sans modèle de langage"}
                {result.truncated && " · note tronquée"}{result.cached && " · résultat récent réutilisé"}
              </p>
            </div>

            <div className="min-w-0 space-y-5 text-[13px]">
              <section>
                <h4 className="label-caps mb-2">Ce que disent les précédents</h4>
                {result.precedents.n ? <p className="text-xs text-muted-foreground">
                  <b className="font-medium text-foreground">{result.precedents.n_confirmed} fraudes confirmées</b> et {result.precedents.n_cleared} alertes classées parmi les {result.precedents.n} cas les plus proches
                  (part de fraude pondérée : {fmtPct(result.precedents.fraud_share ?? 0, 0)}).</p> : <p className="text-xs text-muted-foreground">Aucun cas proche.</p>}
                <div className="mt-2.5 space-y-2">
                  {result.typology_hypotheses.map((t) => (
                    <div key={t.typology}>
                      <div className="flex justify-between gap-3 text-xs"><span className="truncate">{t.label}</span><span className="tabular font-medium">{fmtPct(t.share, 0)}</span></div>
                      <div className="mt-1 h-1.5 rounded-full bg-surface-2"><div className="h-1.5 rounded-full bg-primary" style={{ width: `${t.share * 100}%` }} /></div>
                    </div>
                  ))}
                </div>
              </section>

              {result.entity_links.length > 0 && (
                <section>
                  <h4 className="label-caps mb-2">Liens sur la plateforme · 30 jours</h4>
                  <ul className="space-y-1 text-xs">{result.entity_links.map((e) => (
                    <li key={e.entity} className="flex justify-between gap-3"><span className="capitalize">{e.entity}</span>
                      <span className="text-muted-foreground"><b className={cn("font-medium", e.confirmed_frauds ? "text-risk-critique" : "text-foreground")}>{e.confirmed_frauds} fraude(s)</b> · {e.alerts} alerte(s) · {e.other_clients} client(s)</span></li>
                  ))}</ul>
                </section>
              )}

              <section>
                <h4 className="label-caps mb-2">Cas similaires</h4>
                <ul className="space-y-1.5">{result.similar_cases.map((c) => (
                  <li key={c.ref} id={`ref-${c.ref}`} className={cn("rounded-md border border-l-[3px] bg-surface px-2.5 py-2 text-xs",
                    c.outcome === "FRAUDE_CONFIRMEE" ? "border-l-risk-critique" : "border-l-risk-faible")}>
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate"><b className="font-semibold">{c.ref}</b> · {c.date} · {c.tx_type} · {fmtUsd(c.amount_usd)}</span>
                      <span className="tabular shrink-0 text-2xs text-muted-foreground">sim. {c.similarity.toFixed(2)}</span>
                    </div>
                    <div className={cn("mt-0.5", c.outcome === "FRAUDE_CONFIRMEE" ? "text-risk-critique" : "text-risk-faible")}>
                      {c.outcome === "FRAUDE_CONFIRMEE" ? `Fraude confirmée${c.typology ? ` · ${c.typology}` : ""}` : "Classée sans suite (faux positif)"}
                    </div>
                    <div className="text-2xs text-muted-foreground">{SOURCE_LABEL[c.source] ?? c.source}
                      {c.transaction_id && <> · <Link className="text-primary hover:underline" href={`/transactions/${encodeURIComponent(c.transaction_id)}`}>ouvrir</Link></>}</div>
                    {c.facts && c.facts.length > 0 && <details className="mt-1"><summary className="cursor-pointer text-2xs text-muted-foreground hover:text-foreground">faits</summary>
                      <ul className="mt-1 list-disc space-y-0.5 pl-4 text-muted-foreground">{c.facts.map((f, i) => <li key={i}>{f}</li>)}</ul></details>}
                  </li>
                ))}</ul>
              </section>

              <section>
                <h4 className="label-caps mb-2 flex items-center gap-1.5"><BookOpen className="h-3.5 w-3.5" /> Procédures citées</h4>
                <ul className="space-y-1.5">{result.procedures.map((p) => (
                  <li key={p.ref} id={`ref-${p.ref}`} className="rounded-md border bg-surface px-2.5 py-2 text-xs">
                    <details><summary className="cursor-pointer"><b className="font-semibold">{p.ref}</b> · {p.title} <span className="text-muted-foreground">· {p.section}</span></summary>
                      <div className="mt-1.5 whitespace-pre-line leading-relaxed text-muted-foreground">{p.text}</div></details>
                  </li>
                ))}</ul>
              </section>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
