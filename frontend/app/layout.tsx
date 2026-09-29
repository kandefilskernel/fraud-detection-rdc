import type { Metadata, Viewport } from "next";
// Polices servies par l'application elle-même (aucun appel à un service tiers depuis le back-office)
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Nuru — Détection de fraude RDC", template: "%s · Nuru" },
  description: "Supervision temps réel de la fraude Mobile Money et cartes Visa virtuelles",
};

export const viewport: Viewport = {
  themeColor: [{ media: "(prefers-color-scheme: light)", color: "#f5f7fa" },
               { media: "(prefers-color-scheme: dark)", color: "#0c111a" }],
};

// Applique le thème avant le premier rendu (évite l'éclair blanc en mode sombre)
const themeScript = `(function(){try{var t=localStorage.getItem("nuru-theme")||"system";
var d=t==="dark"||(t==="system"&&matchMedia("(prefers-color-scheme: dark)").matches);
document.documentElement.classList.toggle("dark",d);}catch(e){}})();`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
