import { Ban, CheckCircle2, CircleDot, Clock3, ShieldAlert, ShieldCheck, ShieldQuestion, XCircle } from "lucide-react";
import { cn, ACTION_LABEL, RISK_LABEL, STATUS_LABEL } from "@/lib/utils";

/** Couleur par niveau de risque : réutilisée par les bandes de tableau et les jauges. */
export const RISK_TONE: Record<string, { text: string; bg: string; dot: string; bar: string }> = {
  CRITIQUE: { text: "text-risk-critique", bg: "bg-risk-critique/10 ring-risk-critique/25", dot: "bg-risk-critique", bar: "bg-risk-critique" },
  ELEVE: { text: "text-risk-eleve", bg: "bg-risk-eleve/10 ring-risk-eleve/25", dot: "bg-risk-eleve", bar: "bg-risk-eleve" },
  MOYEN: { text: "text-risk-moyen", bg: "bg-risk-moyen/12 ring-risk-moyen/30", dot: "bg-risk-moyen", bar: "bg-risk-moyen" },
  FAIBLE: { text: "text-risk-faible", bg: "bg-risk-faible/10 ring-risk-faible/25", dot: "bg-risk-faible", bar: "bg-risk-faible" },
};
const NEUTRAL = { text: "text-muted-foreground", bg: "bg-surface-2 ring-border", dot: "bg-subtle", bar: "bg-subtle" };

function Pill({ className, children, title }: { className?: string; children: React.ReactNode; title?: string }) {
  return (
    <span title={title} className={cn("inline-flex h-[22px] items-center gap-1.5 whitespace-nowrap rounded-full px-2 text-2xs font-medium ring-1 ring-inset", className)}>
      {children}
    </span>
  );
}

export function RiskBadge({ level }: { level: string }) {
  const t = RISK_TONE[level] ?? NEUTRAL;
  return <Pill className={cn(t.bg, t.text)}><span className={cn("h-1.5 w-1.5 rounded-full", t.dot)} />{RISK_LABEL[level] ?? level}</Pill>;
}

const ACTION_STYLE: Record<string, { cls: string; icon: typeof ShieldCheck }> = {
  APPROVE: { cls: "bg-risk-faible/10 text-risk-faible ring-risk-faible/25", icon: ShieldCheck },
  VERIFY: { cls: "bg-risk-moyen/12 text-risk-moyen ring-risk-moyen/30", icon: ShieldQuestion },
  BLOCK: { cls: "bg-risk-critique/10 text-risk-critique ring-risk-critique/25", icon: Ban },
};
export function ActionBadge({ action }: { action: string }) {
  const s = ACTION_STYLE[action];
  const Icon = s?.icon ?? ShieldAlert;
  return <Pill className={s?.cls ?? cn(NEUTRAL.bg, NEUTRAL.text)}><Icon className="h-3 w-3" />{ACTION_LABEL[action] ?? action}</Pill>;
}

const STATUS_STYLE: Record<string, { cls: string; icon: typeof CircleDot }> = {
  OUVERT: { cls: "bg-primary-soft text-primary ring-primary/25", icon: CircleDot },
  EN_COURS: { cls: "bg-risk-moyen/12 text-risk-moyen ring-risk-moyen/30", icon: Clock3 },
  FRAUDE_CONFIRMEE: { cls: "bg-risk-critique/10 text-risk-critique ring-risk-critique/25", icon: XCircle },
  FAUX_POSITIF: { cls: "bg-surface-2 text-muted-foreground ring-border", icon: CheckCircle2 },
};
export function StatusBadge({ status }: { status: string }) {
  const s = STATUS_STYLE[status];
  const Icon = s?.icon ?? CircleDot;
  return <Pill className={s?.cls ?? cn(NEUTRAL.bg, NEUTRAL.text)}><Icon className="h-3 w-3" />{STATUS_LABEL[status] ?? status}</Pill>;
}

export function Tag({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cn("inline-flex h-5 items-center rounded px-1.5 font-mono text-2xs text-muted-foreground ring-1 ring-inset ring-border", className)}>{children}</span>;
}

/** Bande verticale de couleur en début de ligne : la gravité se lit sans lire le texte. */
export function RiskStripe({ level }: { level?: string }) {
  return <span aria-hidden className={cn("absolute inset-y-1.5 left-0 w-[3px] rounded-r", (RISK_TONE[level ?? ""] ?? NEUTRAL).bar, level === "FAIBLE" && "opacity-30")} />;
}
