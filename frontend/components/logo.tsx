import { cn } from "@/lib/utils";

/** Logo Nuru (« lumière » en swahili) : un bouclier bleu ciel portant l'étoile jaune du drapeau de la RDC. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden className={cn("h-8 w-8", className)}>
      <path d="M16 2.5 5 6.6v8.2c0 7.1 4.7 12.3 11 14.7 6.3-2.4 11-7.6 11-14.7V6.6L16 2.5Z" fill="hsl(var(--primary))" />
      <path d="M16 2.5 27 6.6v8.2c0 7.1-4.7 12.3-11 14.7V2.5Z" fill="#000" opacity="0.14" />
      <path d="m16 9.2 1.9 3.9 4.3.6-3.1 3 .7 4.2-3.8-2-3.8 2 .7-4.2-3.1-3 4.3-.6L16 9.2Z" fill="hsl(var(--star))" />
    </svg>
  );
}

export function Logo({ className, inverted }: { className?: string; inverted?: boolean }) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <LogoMark />
      <div className="leading-tight">
        <div className={cn("text-[15px] font-semibold tracking-tight", inverted ? "text-white" : "text-foreground")}>Nuru</div>
        <div className={cn("text-2xs", inverted ? "text-white/60" : "text-muted-foreground")}>Détection de fraude · RDC</div>
      </div>
    </div>
  );
}
