"use client";

import { useEffect, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";

type Theme = "light" | "dark" | "system";
const KEY = "nuru-theme";

function apply(t: Theme) {
  const dark = t === "dark" || (t === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
}

/** Thème clair / sombre / système, mémorisé pour ce navigateur. */
export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    let saved: Theme = "system";
    try { saved = (localStorage.getItem(KEY) as Theme) || "system"; } catch { /* navigation privée */ }
    setTheme(saved);
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => { if ((localStorage.getItem(KEY) ?? "system") === "system") apply("system"); };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  function choose(t: Theme) {
    setTheme(t);
    try { localStorage.setItem(KEY, t); } catch { /* navigation privée */ }
    apply(t);
  }

  const items: { t: Theme; icon: typeof Sun; label: string }[] = [
    { t: "light", icon: Sun, label: "Thème clair" }, { t: "system", icon: Monitor, label: "Thème du système" },
    { t: "dark", icon: Moon, label: "Thème sombre" },
  ];
  return (
    <div className={cn("inline-flex rounded-lg border bg-surface-2 p-0.5", className)} role="radiogroup" aria-label="Thème">
      {items.map(({ t, icon: Icon, label }) => (
        <button key={t} role="radio" aria-checked={theme === t} title={label} aria-label={label} onClick={() => choose(t)}
          className={cn("grid h-6 w-7 place-items-center rounded-md transition-colors",
            theme === t ? "bg-surface text-foreground shadow-card" : "text-subtle hover:text-foreground")}>
          <Icon className="h-3.5 w-3.5" />
        </button>
      ))}
    </div>
  );
}
