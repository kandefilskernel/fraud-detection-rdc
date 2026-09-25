import { cn, ACTION_LABEL, RISK_LABEL, STATUS_LABEL } from "@/lib/utils";

const RISK_STYLE: Record<string, string> = {
  CRITIQUE: "bg-risk-critique/12 text-risk-critique ring-risk-critique/30",
  ELEVE: "bg-risk-eleve/12 text-risk-eleve ring-risk-eleve/30",
  MOYEN: "bg-risk-moyen/15 text-[hsl(38_90%_32%)] ring-risk-moyen/35",
  FAIBLE: "bg-risk-faible/12 text-risk-faible ring-risk-faible/30",
};
const ACTION_STYLE: Record<string, string> = {
  BLOCK: RISK_STYLE.CRITIQUE, VERIFY: RISK_STYLE.MOYEN, APPROVE: RISK_STYLE.FAIBLE,
};
const STATUS_STYLE: Record<string, string> = {
  OUVERT: "bg-primary/10 text-primary ring-primary/30", EN_COURS: RISK_STYLE.MOYEN,
  FRAUDE_CONFIRMEE: RISK_STYLE.CRITIQUE, FAUX_POSITIF: "bg-muted text-muted-foreground ring-border",
};

function Pill({ className, children }: { className?: string; children: React.ReactNode }) {
  return <span className={cn("inline-flex items-center whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset", className)}>{children}</span>;
}
export const RiskBadge = ({ level }: { level: string }) => <Pill className={RISK_STYLE[level]}>{RISK_LABEL[level] ?? level}</Pill>;
export const ActionBadge = ({ action }: { action: string }) => <Pill className={ACTION_STYLE[action]}>{ACTION_LABEL[action] ?? action}</Pill>;
export const StatusBadge = ({ status }: { status: string }) => <Pill className={STATUS_STYLE[status]}>{STATUS_LABEL[status] ?? status}</Pill>;
export const Tag = ({ children }: { children: React.ReactNode }) => <Pill className="bg-muted text-muted-foreground ring-border">{children}</Pill>;
