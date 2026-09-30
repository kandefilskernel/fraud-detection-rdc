"use client";

import { useMemo, useRef, useState } from "react";
import { DRC_PROVINCES, DRC_VIEWBOX, type Province } from "@/lib/drc-provinces";
import { cn } from "@/lib/utils";

/** Clé de rapprochement : « Équateur » (carte) = « Equateur » (données). */
export const provinceKey = (s: string) => s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();

export type MapDatum = { value: number; tooltip: React.ReactNode };

const CLASSES = 5;
const ALPHAS = [0.14, 0.3, 0.48, 0.68, 0.9];

/** Carte choroplèthe des 26 provinces de la RDC (SVG hors ligne, suit le thème clair/sombre). */
export function DrcMap({ data, color = "risk-critique", selected, onSelect, formatValue, legendTitle }:
  { data: Record<string, MapDatum>; color?: string; selected?: string | null;
    onSelect?: (name: string | null) => void; formatValue: (v: number) => string; legendTitle: string }) {
  const box = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ p: Province; x: number; y: number } | null>(null);

  const max = useMemo(() => Math.max(0, ...Object.values(data).map((d) => d.value)), [data]);
  const cls = (v: number) => (max > 0 && v > 0 ? Math.min(CLASSES - 1, Math.floor((v / max) * CLASSES - 1e-9)) : -1);
  const fill = (v: number | undefined) => {
    if (v === undefined) return "hsl(var(--surface-2))";
    const c = cls(v);
    return c < 0 ? `hsl(var(--${color}) / 0.06)` : `hsl(var(--${color}) / ${ALPHAS[c]})`;
  };

  const move = (p: Province, e: React.MouseEvent) => {
    const r = box.current?.getBoundingClientRect();
    if (r) setHover({ p, x: e.clientX - r.left, y: e.clientY - r.top });
  };
  const hoverDatum = hover ? data[provinceKey(hover.p.name)] : undefined;
  const sel = selected ? provinceKey(selected) : null;
  // province sélectionnée dessinée en dernier : son contour n'est pas masqué par les voisines
  const ordered = [...DRC_PROVINCES].sort((a, b) => Number(provinceKey(a.name) === sel) - Number(provinceKey(b.name) === sel));

  return (
    <div ref={box} className="relative">
      <svg viewBox={DRC_VIEWBOX} className="h-auto w-full select-none" role="img"
           aria-label="Carte des provinces de la République démocratique du Congo"
           onClick={(e) => { if (e.target === e.currentTarget) onSelect?.(null); }}>
        {ordered.map((p) => {
          const k = provinceKey(p.name);
          const d = data[k];
          const on = k === sel;
          return (
            <path key={p.iso} d={p.d} fill={fill(d?.value)}
              stroke={on ? "hsl(var(--primary))" : "hsl(var(--border))"} strokeWidth={on ? 3 : 1}
              strokeLinejoin="round"
              className={cn("outline-none transition-[fill] duration-500", d && "cursor-pointer hover:brightness-95 focus-visible:stroke-[hsl(var(--primary))]")}
              tabIndex={d ? 0 : -1} role={d ? "button" : undefined}
              aria-label={d ? `${p.name} : ${formatValue(d.value)}` : `${p.name} : aucune donnée`}
              aria-pressed={d ? on : undefined}
              onMouseMove={(e) => move(p, e)} onMouseLeave={() => setHover(null)}
              onClick={() => d && onSelect?.(on ? null : p.name)}
              onKeyDown={(e) => { if (d && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); onSelect?.(on ? null : p.name); } }} />
          );
        })}
        {DRC_PROVINCES.filter((p) => data[provinceKey(p.name)]).map((p) => (
          <g key={`c-${p.iso}`} className="pointer-events-none">
            <circle cx={p.city.xy[0]} cy={p.city.xy[1]} r={4.5} fill="hsl(var(--surface))" stroke="hsl(var(--foreground))" strokeWidth={1.5} />
            <text x={p.city.xy[0] + 8} y={p.city.xy[1] + 4} fontSize={15} fontWeight={600}
                  fill="hsl(var(--foreground))" stroke="hsl(var(--surface))" strokeWidth={3.5} paintOrder="stroke">
              {p.city.name}
            </text>
          </g>
        ))}
      </svg>

      {hover && (
        <div className="pointer-events-none absolute z-10 min-w-[180px] rounded-lg border bg-surface px-3 py-2 text-xs shadow-pop"
             style={{ left: Math.min(hover.x + 14, (box.current?.clientWidth ?? 0) - 200), top: hover.y + 14 }}>
          <div className="font-medium text-foreground">{hover.p.name}</div>
          <div className="mb-1.5 text-2xs text-muted-foreground">Chef-lieu : {hover.p.city.name}</div>
          {hoverDatum ? hoverDatum.tooltip : <div className="text-muted-foreground">Aucune transaction sur la période</div>}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-2xs text-muted-foreground">
        <span className="font-medium text-foreground">{legendTitle}</span>
        <div className="flex items-center gap-1">
          {ALPHAS.map((a, i) => (
            <div key={a} className="flex flex-col items-center gap-0.5">
              <span className="h-2.5 w-9 rounded-sm" style={{ background: `hsl(var(--${color}) / ${a})` }} />
              <span className="tabular">{max > 0 ? formatValue((max * (i + 1)) / CLASSES) : "—"}</span>
            </div>
          ))}
        </div>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-4 rounded-sm border bg-surface-2" />Aucune donnée</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full border-[1.5px] border-foreground bg-surface" />Chef-lieu</span>
      </div>
    </div>
  );
}
