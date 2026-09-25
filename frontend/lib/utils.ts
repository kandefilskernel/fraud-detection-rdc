import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const fmtUsd = (v: number | null | undefined) =>
  v == null ? "—" : v.toLocaleString("fr-FR", { style: "currency", currency: "USD", maximumFractionDigits: 2 });

export const fmtInt = (v: number | null | undefined) => (v == null ? "—" : Math.round(v).toLocaleString("fr-FR"));

export const fmtPct = (v: number | null | undefined, digits = 1) =>
  v == null ? "—" : `${(v * 100).toLocaleString("fr-FR", { maximumFractionDigits: digits })} %`;

export const fmtDate = (iso: string) =>
  new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "medium" });

export const RISK_LABEL: Record<string, string> = {
  CRITIQUE: "Critique", ELEVE: "Élevé", MOYEN: "Moyen", FAIBLE: "Faible", INCONNU: "Inconnu",
};
export const ACTION_LABEL: Record<string, string> = { APPROVE: "Approuvée", VERIFY: "À vérifier", BLOCK: "Bloquée" };
export const STATUS_LABEL: Record<string, string> = {
  OUVERT: "Ouvert", EN_COURS: "En cours", FRAUDE_CONFIRMEE: "Fraude confirmée", FAUX_POSITIF: "Faux positif",
};

/** Libellés lisibles des variables du modèle (explicabilité). */
export const FEATURE_LABEL: Record<string, string> = {
  log_amount_usd: "Montant", amount_zscore_user: "Montant inhabituel pour ce client",
  amount_ratio_user_mean: "Montant / moyenne du client", amount_to_balance: "Part du solde débitée",
  is_near_full_drain: "Vidage quasi total du compte", amount_to_kyc_limit: "Montant / plafond KYC",
  log_balance_before: "Solde avant opération", tx_count_1h: "Transactions sur 1 h",
  tx_count_24h: "Transactions sur 24 h", tx_count_7d: "Transactions sur 7 j", debit_count_1h: "Débits sur 1 h",
  log_debit_sum_24h: "Total débité sur 24 h", log_secs_since_last_tx: "Délai depuis la dernière opération",
  failed_count_24h: "Échecs sur 24 h", is_new_device: "Nouvel appareil", log_device_prior_uses: "Usages de l'appareil",
  n_devices_user: "Appareils du client", device_n_users: "Autres clients sur cet appareil",
  is_smartphone: "Smartphone", access_app: "Via application", access_ussd: "Via USSD", access_agent: "Via agent",
  is_foreign_ip: "IP étrangère", is_new_province: "Nouvelle province", is_away_from_home: "Hors province d'origine",
  is_new_counterparty: "Nouveau destinataire", counterparty_n_users: "Destinataire lié à d'autres clients (mule ?)",
  is_new_agent: "Nouvel agent", agent_n_users_24h: "Clients de l'agent sur 24 h", is_new_merchant: "Nouveau marchand",
  is_high_risk_merchant: "Marchand à risque", is_foreign_merchant: "Marchand étranger", hour_sin: "Heure (sin)",
  hour_cos: "Heure (cos)", is_night: "Nuit (0 h – 5 h)", user_night_ratio: "Habitude nocturne du client",
  night_unusual: "Nuit inhabituelle pour ce client", hour_deviation_user: "Heure inhabituelle pour ce client",
  is_weekend: "Week-end", is_month_end: "Fin/début de mois", log_history_days: "Ancienneté de l'historique",
  log_user_tx_count: "Nombre d'opérations du client", is_card: "Carte Visa virtuelle",
  log_mins_since_topup: "Délai depuis la recharge carte", purchase_to_topup_ratio: "Achat / recharge carte",
  kyc_level: "Niveau KYC", log_account_age_days: "Ancienneté du compte", log_monthly_income: "Revenu mensuel déclaré",
  has_visa_virtual: "Détient une carte virtuelle",
};
