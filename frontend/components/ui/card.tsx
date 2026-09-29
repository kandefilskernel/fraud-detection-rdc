import { cn } from "@/lib/utils";

export function Card({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("min-w-0 rounded-lg border bg-surface text-foreground shadow-card", className)} {...p} />;
}

/** En-tête de carte : titre, description facultative et actions alignées à droite. */
export function CardHeader({ title, description, actions, className, children }:
  { title?: React.ReactNode; description?: React.ReactNode; actions?: React.ReactNode; className?: string;
    children?: React.ReactNode }) {
  return (
    <div className={cn("flex items-start justify-between gap-3 px-5 pb-3 pt-4", className)}>
      {children ?? (
        <div className="min-w-0">
          {title && <h3 className="text-[13px] font-semibold leading-5 text-foreground">{title}</h3>}
          {description && <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>}
        </div>
      )}
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

/** Compatibilité : ancien titre de carte. */
export function CardTitle({ className, ...p }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("text-[13px] font-semibold leading-5", className)} {...p} />;
}

export function CardContent({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 pb-5", className)} {...p} />;
}

export function CardFooter({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("flex items-center justify-between gap-3 border-t px-5 py-3 text-xs text-muted-foreground", className)} {...p} />;
}
