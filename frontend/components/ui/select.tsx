import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

const field = "h-9 rounded-md border border-input bg-surface text-sm text-foreground shadow-sm transition-colors " +
  "placeholder:text-subtle hover:border-subtle/60 focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring/25";

export function Select({ className, options, ...p }:
  React.SelectHTMLAttributes<HTMLSelectElement> & { options: [string, string][] }) {
  return (
    <div className={cn("relative inline-flex", className)}>
      <select className={cn(field, "w-full appearance-none pl-3 pr-8")} {...p}>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
      <ChevronDown aria-hidden className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-subtle" />
    </div>
  );
}

export function Input({ className, icon: Icon, ...p }:
  React.InputHTMLAttributes<HTMLInputElement> & { icon?: React.ComponentType<{ className?: string }> }) {
  if (!Icon) return <input className={cn(field, "w-full px-3", className)} {...p} />;
  return (
    <div className={cn("relative", className)}>
      <Icon aria-hidden className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-subtle" />
      <input className={cn(field, "w-full pl-9 pr-3")} {...p} />
    </div>
  );
}

export function Textarea({ className, ...p }: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cn(field, "h-auto w-full px-3 py-2 leading-relaxed", className)} {...p} />;
}

export function Label({ className, ...p }: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn("mb-1.5 block text-xs font-medium text-foreground", className)} {...p} />;
}
