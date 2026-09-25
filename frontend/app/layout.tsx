import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Détection de fraude — RDC",
  description: "Supervision temps réel de la fraude Mobile Money et cartes Visa virtuelles",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
