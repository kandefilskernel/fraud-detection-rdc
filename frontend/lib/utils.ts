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

/** Montant compact pour les indicateurs : 12,4 k$US, 1,2 M$US. */
export const fmtUsdCompact = (v: number | null | undefined) =>
  v == null ? "—" : v.toLocaleString("fr-FR", { style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1 });

/** « il y a 12 s », « il y a 5 min », « il y a 3 h », sinon la date courte. */
export function fmtRelative(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 0) return "à l'instant";
  if (s < 60) return `il y a ${Math.floor(s)} s`;
  if (s < 3600) return `il y a ${Math.floor(s / 60)} min`;
  if (s < 86400) return `il y a ${Math.floor(s / 3600)} h`;
  return new Date(iso).toLocaleDateString("fr-FR", { day: "2-digit", month: "short" });
}

export const TX_TYPE_LABEL: Record<string, string> = {
  P2P_SEND: "Envoi", P2P_RECEIVE: "Réception", CASH_IN: "Dépôt", CASH_OUT: "Retrait",
  MERCHANT_PAYMENT: "Paiement marchand", AIRTIME: "Crédit téléphonique", BILL_PAYMENT: "Facture",
  CARD_TOPUP: "Recharge carte", CARD_PURCHASE: "Achat carte",
};

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
  // réseau du bénéficiaire (C1)
  cp_is_customer: "Bénéficiaire client de l'opérateur", cp_log_age_days: "Ancienneté du bénéficiaire",
  cp_in_senders_7d: "Expéditeurs du bénéficiaire (7 j)", cp_in_new_ratio_7d: "Part de premiers contacts du bénéficiaire",
  cp_in_operators_7d: "Opérateurs qui paient le bénéficiaire", cp_out_receivers_7d: "Destinataires du bénéficiaire (7 j)",
  cp_log_in_amount_7d: "Montant reçu par le bénéficiaire (7 j)", refund_ratio_to_cp: "Renvoi / montant reçu de ce numéro",
  log_mins_since_received_from_cp: "Délai depuis la réception de ce numéro", in_senders_24h: "Expéditeurs du client (24 h)",
  in_new_senders_24h: "Nouveaux expéditeurs du client (24 h)", log_inflow_24h: "Montant reçu (24 h)",
  passthrough_ratio: "Part du montant reçue juste avant",
  // réputation (fraudes déjà signalées)
  rep_device_frauds: "Appareil lié à des fraudes", rep_cp_frauds: "Bénéficiaire lié à des fraudes",
  rep_agent_frauds_7d: "Agent lié à des fraudes (7 j)", rep_merchant_frauds_30d: "Marchand lié à des fraudes (30 j)",
  rep_user_frauds_30d: "Client déjà victime (30 j)",
  // type d'opération
  type_P2P_SEND: "Type : envoi", type_P2P_RECEIVE: "Type : réception", type_CASH_IN: "Type : dépôt",
  type_CASH_OUT: "Type : retrait", type_MERCHANT_PAYMENT: "Type : paiement marchand", type_AIRTIME: "Type : crédit téléphonique",
  type_BILL_PAYMENT: "Type : facture", type_CARD_TOPUP: "Type : recharge carte", type_CARD_PURCHASE: "Type : achat carte",
};
