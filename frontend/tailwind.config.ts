import type { Config } from "tailwindcss";

/** Système de design « Nuru » : jetons définis dans app/globals.css (thèmes clair et sombre). */
const token = (name: string) => `hsl(var(--${name}) / <alpha-value>)`;

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["IBM Plex Sans", "Segoe UI", "system-ui", "-apple-system", "Roboto", "Arial", "sans-serif"],
        mono: ["IBM Plex Mono", "Cascadia Mono", "Consolas", "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      colors: {
        border: token("border"),
        input: token("input"),
        ring: token("ring"),
        background: token("background"),
        foreground: token("foreground"),
        surface: { DEFAULT: token("surface"), 2: token("surface-2"), 3: token("surface-3") },
        muted: { DEFAULT: token("surface-2"), foreground: token("muted-foreground") },
        subtle: token("subtle"),
        card: { DEFAULT: token("surface"), foreground: token("foreground") },
        primary: { DEFAULT: token("primary"), foreground: token("primary-foreground"), soft: token("primary-soft") },
        star: token("star"),
        risk: {
          critique: token("risk-critique"),
          eleve: token("risk-eleve"),
          moyen: token("risk-moyen"),
          faible: token("risk-faible"),
        },
      },
      borderRadius: { xl: "0.875rem", lg: "0.625rem", md: "0.5rem", sm: "0.375rem" },
      boxShadow: {
        card: "0 1px 2px 0 hsl(var(--shadow) / 0.05), 0 1px 3px 0 hsl(var(--shadow) / 0.04)",
        pop: "0 10px 30px -10px hsl(var(--shadow) / 0.25), 0 2px 6px hsl(var(--shadow) / 0.08)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0", transform: "translateY(2px)" }, to: { opacity: "1", transform: "none" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: { "fade-in": "fade-in .18s ease-out", shimmer: "shimmer 1.4s infinite" },
    },
  },
  plugins: [],
};
export default config;
