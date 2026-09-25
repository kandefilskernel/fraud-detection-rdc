import { cn } from "@/lib/utils";

export function Select({ className, options, ...p }:
  React.SelectHTMLAttributes<HTMLSelectElement> & { options: [string, string][] }) {
  return (
    <select className={cn("h-9 rounded-md border bg-card px-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40", className)} {...p}>
      {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
    </select>
  );
}
export function Input({ className, ...p }: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn("h-9 w-full rounded-md border bg-card px-3 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40", className)} {...p} />;
}
