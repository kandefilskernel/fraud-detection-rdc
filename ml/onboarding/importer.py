"""Import d'un export opérateur vers le format pivot (ml/data/raw/transactions.csv).

Principes :
    - liste blanche : seules les colonnes déclarées dans le fichier de correspondance sont lues ;
      noms, pièces d'identité, adresses... ne sont jamais copiés ;
    - pseudonymisation DÈS LA LECTURE (clé PSEUDONYMIZATION_KEY, la même qu'en temps réel) :
      aucun numéro de téléphone en clair n'est écrit sur le disque ;
    - rien n'est inventé en silence : chaque ligne écartée et chaque valeur par défaut est
      comptée dans import_log.json, repris par le rapport qualité.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features.feature_engineering import DEFAULT_PROFILE
from ml.onboarding.mapping import TX_TYPES, FileFormat, OperatorMapping, SubMapping
from shared.privacy.pseudonymize import P2P_TYPES, Pseudonymizer

RAW_COLUMNS = ["transaction_id", "timestamp", "user_id", "wallet_id", "card_id", "channel", "operator",
               "tx_type", "access_channel", "amount", "currency", "amount_usd", "balance_before_usd",
               "balance_after_usd", "status", "counterparty_id", "merchant_id", "merchant_category",
               "merchant_country", "agent_id", "device_id", "device_type", "ip_country", "location_province",
               "is_fraud", "fraud_type", "fraud_episode_id", "fraud_reported_at"]
USER_COLUMNS = ["user_id", "wallet_id", "operator", "province", "kyc_level", "kyc_tx_limit_usd",
                "account_age_days", "monthly_income_usd", "has_visa_virtual"]

# 26 provinces (orthographe des données du système) ; la comparaison ignore accents et casse
PROVINCES = ["Bas-Uele", "Équateur", "Haut-Katanga", "Haut-Lomami", "Haut-Uele", "Ituri", "Kasai",
             "Kasai-Central", "Kasai-Oriental", "Kinshasa", "Kongo-Central", "Kwango", "Kwilu", "Lomami",
             "Lualaba", "Mai-Ndombe", "Maniema", "Mongala", "Nord-Kivu", "Nord-Ubangi", "Sankuru",
             "Sud-Kivu", "Sud-Ubangi", "Tanganyika", "Tshopo", "Tshuapa"]
BALANCE_UNKNOWN_USD = 1e9    # solde non fourni par l'opérateur (voir build_workspace)
CURRENCY_ALIASES = {"CDF": "CDF", "FC": "CDF", "FRANC CONGOLAIS": "CDF", "USD": "USD", "$": "USD", "US$": "USD"}


class ImportError_(ValueError):
    """Export inutilisable avec ce fichier de correspondance (message destiné à l'utilisateur)."""


def _key(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


_PROVINCE_BY_KEY = {_key(p): p for p in PROVINCES}
_PROVINCE_BY_KEY.update({_key("Kasaï"): "Kasai", _key("Kasaï-Central"): "Kasai-Central",
                         _key("Kasaï-Oriental"): "Kasai-Oriental", _key("Maï-Ndombe"): "Mai-Ndombe"})


def read_table(path: str | Path, fmt: FileFormat) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise ImportError_(f"fichier introuvable : {path}")
    if fmt.type == "csv":
        return pd.read_csv(path, sep=fmt.sep, encoding=fmt.encoding, dtype=str, keep_default_na=False,
                           na_values=[""], low_memory=False)
    if fmt.type == "parquet":
        return pd.read_parquet(path).astype("string").astype(object)
    try:
        return pd.read_excel(path, sheet_name=fmt.sheet, dtype=str)
    except ImportError as e:   # openpyxl absent
        raise ImportError_("lecture Excel : installer openpyxl (pip install openpyxl) ou exporter en CSV") from e


def _select(df: pd.DataFrame, columns: dict, where: str) -> pd.DataFrame:
    missing = {f: c for f, c in columns.items() if c not in df.columns}
    if missing:
        raise ImportError_(f"{where} : colonnes absentes de l'export {missing}. "
                           f"Colonnes présentes : {list(df.columns)[:40]}")
    out = pd.DataFrame({f: df[c] for f, c in columns.items()})
    for c in out.columns:
        out[c] = out[c].map(lambda v: v.strip() if isinstance(v, str) else v).replace({"": None})
    return out


def normalize_msisdn(v):
    """+243 81 234 5678 / 00243812345678 / 0812345678 -> 243812345678 (identifiants non numériques inchangés)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"[\s\-\.\(\)]", "", s)
    if digits.startswith("+"):
        digits = digits[1:]
    if not digits.isdigit():
        return s
    if digits.startswith("00"):
        digits = digits[2:]
    if len(digits) == 10 and digits.startswith("0"):
        digits = "243" + digits[1:]
    elif len(digits) == 9:
        digits = "243" + digits
    return digits


def parse_number(s: pd.Series, decimal: str = ".") -> pd.Series:
    t = s.astype(str).str.replace(r"[\s  ']", "", regex=True)
    if decimal == ",":
        t = t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        t = t.str.replace(",", "", regex=False)
    t = t.str.replace(r"[^0-9.\-eE]", "", regex=True)
    return pd.to_numeric(t, errors="coerce")


def parse_timestamps(s: pd.Series, fmt: str | None, source_tz: str, target_tz: str) -> pd.Series:
    ts = pd.to_datetime(s, format=fmt, errors="coerce") if fmt else pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
    # heure locale « naïve », convention du système (variables horaires : nuit, heure habituelle)
    return ts.dt.tz_convert(target_tz).dt.tz_localize(None)


def map_values(s: pd.Series, mapping: dict | None, allowed: tuple | None = None) -> tuple[pd.Series, dict]:
    """Codes de l'opérateur -> codes du système. Renvoie (série, {valeur non reconnue: nombre})."""
    mapping = {str(k).strip().upper(): v for k, v in (mapping or {}).items()}
    up = s.map(lambda v: None if v is None or v != v else str(v).strip().upper())
    out = up.map(lambda v: mapping.get(v, v if (allowed and v in allowed) else None) if v is not None else None)
    unknown = up[out.isna() & up.notna()].value_counts().head(20).to_dict()
    return out, {str(k): int(v) for k, v in unknown.items()}


def _fraud_code(v) -> str:
    """« SIM swap » -> SIM_SWAP ; vide -> NON_PRECISE."""
    if not isinstance(v, str) or not v.strip():
        return "NON_PRECISE"
    s = unicodedata.normalize("NFD", v).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9]+", "_", s).strip("_") or "NON_PRECISE"


def normalize_province(v):
    if v is None or v != v:
        return None
    return _PROVINCE_BY_KEY.get(_key(v))


def _to_usd(amount: pd.Series, currency: pd.Series, rate: float) -> pd.Series:
    return np.where(currency == "USD", amount, amount / rate)


def normalize_transactions(df: pd.DataFrame, m: OperatorMapping) -> tuple[pd.DataFrame, dict]:
    log: dict = {"rows_read": int(len(df)), "dropped": {}, "defaults": {}, "unknown_values": {}}
    t = _select(df, m.columns, "transactions")
    n0 = len(t)

    def drop(mask: pd.Series, reason: str):
        nonlocal t
        k = int(mask.sum())
        if k:
            log["dropped"][reason] = log["dropped"].get(reason, 0) + k
            t = t[~mask]

    t["timestamp"] = parse_timestamps(t["timestamp"], m.timestamp_format, m.source_timezone, m.target_timezone)
    drop(t["timestamp"].isna(), "horodatage illisible")
    drop(t["transaction_id"].isna() | t["user_id"].isna(), "identifiant de transaction ou de client vide")

    t["tx_type"], unknown = map_values(t["tx_type"], m.values.get("tx_type"), TX_TYPES)
    if unknown:
        log["unknown_values"]["tx_type"] = unknown
    drop(t["tx_type"].isna(), "type d'opération non reconnu (compléter values.tx_type)")

    if "channel" in t:
        t["channel"], unknown = map_values(t["channel"], m.values.get("channel"), ("MOBILE_MONEY", "VISA_VIRTUAL"))
        t["channel"] = t["channel"].fillna(m.channel)
    else:
        t["channel"] = m.channel
    t["operator"] = np.where(t["channel"] == "MOBILE_MONEY", m.operator if m.operator != "VISA" else None, None)

    if "status" in t:
        t["status"], unknown = map_values(t["status"], m.values.get("status"), ("SUCCESS", "FAILED"))
        if unknown:
            log["unknown_values"]["status"] = unknown
        n = int(t["status"].isna().sum())
        if n:
            log["defaults"]["status=SUCCESS (valeur absente ou non reconnue)"] = n
        t["status"] = t["status"].fillna("SUCCESS")
    else:
        t["status"] = "SUCCESS"
        log["defaults"]["status=SUCCESS (colonne absente)"] = len(t)

    # montants et devise
    t["amount"] = parse_number(t["amount"], m.format.decimal) * m.amount_scale
    if "currency" in t:
        t["currency"] = t["currency"].map(lambda v: CURRENCY_ALIASES.get(str(v).strip().upper()) if v else None)
        n = int(t["currency"].isna().sum())
        if n:
            log["defaults"][f"devise={m.default_currency} (absente ou inconnue)"] = n
        t["currency"] = t["currency"].fillna(m.default_currency)
    else:
        t["currency"] = m.default_currency
    drop(t["amount"].isna() | (t["amount"] <= 0), "montant absent, nul ou négatif")
    t["amount_usd"] = _to_usd(t["amount"], t["currency"], m.usd_cdf_rate).round(4)
    for side in ("before", "after"):
        col = f"balance_{side}"
        if col in t:
            v = parse_number(t[col], m.format.decimal) * m.amount_scale
            t[f"{col}_usd"] = _to_usd(v, t["currency"], m.usd_cdf_rate)
        else:
            t[f"{col}_usd"] = np.nan

    # identifiants : numéros au format international 243..., autres identifiants en texte
    for c in ("user_id", "wallet_id", "counterparty_id"):
        if c in t:
            t[c] = t[c].map(normalize_msisdn)
    if "wallet_id" not in t:
        t["wallet_id"] = t["user_id"] if m.channel == "MOBILE_MONEY" else None
    for c in ("card_id", "merchant_id", "agent_id", "merchant_category", "counterparty_id", "device_id",
              "device_type", "ip_country", "merchant_country", "location_province", "access_channel"):
        if c not in t:
            t[c] = None
    t["access_channel"], unknown = map_values(t["access_channel"], m.values.get("access_channel"),
                                              ("APP", "USSD", "AGENT", "WEB"))
    t["access_channel"] = t["access_channel"].fillna("UNKNOWN")
    if t["device_id"].isna().any():
        # appareil inconnu : un appareil « fictif » par client, pour ne pas fusionner tous les
        # clients sur une même valeur vide (fausses alertes « appareil partagé »)
        n = int(t["device_id"].isna().sum())
        log["defaults"]["appareil inconnu -> identifiant fictif par client"] = n
        t["device_id"] = t["device_id"].fillna("UNK-" + t["user_id"].astype(str))
    t["device_type"] = t["device_type"].map(lambda v: str(v).lower() if v else "unknown")
    t["ip_country"] = t["ip_country"].map(lambda v: str(v).upper()[:2] if v else None)
    t["merchant_country"] = t["merchant_country"].map(lambda v: str(v).upper()[:2] if v else None)
    t["merchant_category"] = t["merchant_category"].map(lambda v: str(v).upper() if v else None)
    raw_prov = t["location_province"]
    t["location_province"] = raw_prov.map(normalize_province)
    bad = raw_prov[raw_prov.notna() & t["location_province"].isna()].value_counts().head(20)
    if len(bad):
        log["unknown_values"]["location_province"] = {str(k): int(v) for k, v in bad.items()}

    # étiquettes présentes dans l'export (sinon : fichier de signalements, voir attach_fraud_reports)
    if "is_fraud" in t:
        t["is_fraud"] = t["is_fraud"].map(lambda v: int(str(v).strip().upper() in m.fraud_positive_values)
                                          if v is not None else 0)
    else:
        t["is_fraud"] = 0
    for c in ("fraud_type", "fraud_reported_at"):
        if c not in t:
            t[c] = None
    t["fraud_reported_at"] = pd.to_datetime(t["fraud_reported_at"], errors="coerce")
    t["fraud_episode_id"] = None

    before = len(t)
    t = t.drop_duplicates("transaction_id", keep="first")
    if len(t) < before:
        log["dropped"]["transaction en double (même identifiant)"] = before - len(t)
    log["rows_kept"] = int(len(t))
    log["rows_dropped"] = int(n0 - len(t))
    log["columns_mapped"] = sorted(m.columns)
    return t.reset_index(drop=True), log


def attach_fraud_reports(t: pd.DataFrame, reports: pd.DataFrame, sub: SubMapping, m: OperatorMapping,
                         log: dict) -> pd.DataFrame:
    r = _select(reports, sub.columns, "fraud_reports")
    if "reported_at" in r:
        r["reported_at"] = parse_timestamps(r["reported_at"], sub.timestamp_format, m.source_timezone,
                                            m.target_timezone)
    r = r.drop_duplicates("transaction_id", keep="first").set_index("transaction_id")
    known = t["transaction_id"].isin(r.index)
    log["fraud_reports"] = {"lines": int(len(r)), "matched": int(known.sum()),
                            "unmatched": int(len(r) - known.sum())}
    t.loc[known, "is_fraud"] = 1
    if "reported_at" in r:
        t.loc[known, "fraud_reported_at"] = t.loc[known, "transaction_id"].map(r["reported_at"]).to_numpy()
    if "fraud_type" in r:
        t.loc[known, "fraud_type"] = t.loc[known, "transaction_id"].map(r["fraud_type"]).to_numpy()
    return t


def finalize_labels(t: pd.DataFrame, m: OperatorMapping, log: dict) -> pd.DataFrame:
    fraud = t["is_fraud"] == 1
    t.loc[fraud, "fraud_type"] = t.loc[fraud, "fraud_type"].map(_fraud_code)
    t.loc[~fraud, "fraud_type"] = None
    # date de signalement inconnue : l'opérateur n'a pas su la fraude instantanément
    no_date = fraud & t["fraud_reported_at"].isna()
    if no_date.any():
        log["defaults"][f"date de signalement = transaction + {m.default_report_delay_days:g} j"] = int(no_date.sum())
        t.loc[no_date, "fraud_reported_at"] = (t.loc[no_date, "timestamp"]
                                               + pd.Timedelta(days=m.default_report_delay_days))
    early = fraud & (t["fraud_reported_at"] < t["timestamp"])
    if early.any():   # signalement antérieur à la transaction : incohérent, ramené à la transaction
        log["defaults"]["date de signalement antérieure à la transaction (corrigée)"] = int(early.sum())
        t.loc[early, "fraud_reported_at"] = t.loc[early, "timestamp"]
    log["labels"] = {"frauds": int(fraud.sum()), "fraud_rate": round(float(fraud.mean()), 6) if len(t) else 0.0}
    return t


def normalize_kyc(df: pd.DataFrame, sub: SubMapping, m: OperatorMapping) -> pd.DataFrame:
    k = _select(df, sub.columns, "kyc")
    k["user_id"] = k["user_id"].map(normalize_msisdn)
    k["wallet_id"] = k["wallet_id"].map(normalize_msisdn) if "wallet_id" in k else k["user_id"]
    k["province"] = k["province"].map(normalize_province) if "province" in k else None
    out = pd.DataFrame({"user_id": k["user_id"], "wallet_id": k["wallet_id"], "province": k["province"]})
    out["kyc_level"] = pd.to_numeric(k.get("kyc_level"), errors="coerce") if "kyc_level" in k else np.nan
    if "kyc_tx_limit" in k:
        lim = parse_number(k["kyc_tx_limit"], m.format.decimal) * m.amount_scale
        out["kyc_tx_limit_usd"] = lim / (1.0 if m.default_currency == "USD" else m.usd_cdf_rate)
    else:
        out["kyc_tx_limit_usd"] = np.nan
    if "monthly_income" in k:
        inc = parse_number(k["monthly_income"], m.format.decimal) * m.amount_scale
        out["monthly_income_usd"] = inc / (1.0 if m.default_currency == "USD" else m.usd_cdf_rate)
    else:
        out["monthly_income_usd"] = np.nan
    out["account_opened_at"] = (pd.to_datetime(k["account_opened_at"], format=sub.timestamp_format, errors="coerce")
                                if "account_opened_at" in k else pd.NaT)
    out["has_visa_virtual"] = (k["has_visa_virtual"].map(lambda v: int(str(v).strip().upper() in
                                                                    ("1", "TRUE", "OUI", "YES")))
                               if "has_visa_virtual" in k else np.nan)
    out["operator"] = m.operator if m.operator != "VISA" else None
    return out.dropna(subset=["user_id"]).drop_duplicates("user_id", keep="last")


def pseudonymize_frame(t: pd.DataFrame, pseudo: Pseudonymizer | None, cols: tuple[str, ...]) -> pd.DataFrame:
    """Même règle que l'integration-layer (shared.privacy) ; calcul une fois par valeur distincte."""
    if pseudo is None:
        return t
    t = t.copy()
    for c in cols:
        if c not in t:
            continue
        if c == "counterparty_id" and "tx_type" in t:
            p2p = t["tx_type"].isin(P2P_TYPES) & t[c].notna()
            uniq = {v: pseudo(str(v)) for v in t.loc[p2p, c].unique()}
            t.loc[p2p, c] = t.loc[p2p, c].map(uniq)
        else:
            mask = t[c].notna()
            uniq = {v: pseudo(str(v)) for v in t.loc[mask, c].unique()}
            t.loc[mask, c] = t.loc[mask, c].map(uniq)
    return t


def import_operator(m: OperatorMapping, input_path: str | Path, workspace: Path,
                    frauds_path: str | Path | None = None, kyc_path: str | Path | None = None,
                    pseudo: Pseudonymizer | None = None, since: str | None = None,
                    sample_users: int | None = None, seed: int = 42) -> dict:
    """Lit un export, le normalise, le pseudonymise et l'écrit dans workspace/sources/<mapping>/."""
    t, log = normalize_transactions(read_table(input_path, m.format), m)
    log["source"] = {"file": Path(input_path).name, "mapping": m.path.name, "operator": m.operator,
                     "channel": m.channel}
    if frauds_path:
        if m.fraud_reports is None:
            raise ImportError_("fichier de signalements fourni mais section fraud_reports absente du mapping")
        t = attach_fraud_reports(t, read_table(frauds_path, m.fraud_reports.format), m.fraud_reports, m, log)
    t = finalize_labels(t, m, log)
    if since:
        before = len(t)
        t = t[t["timestamp"] >= pd.Timestamp(since)]
        log["dropped"][f"antérieure au {since} (--since)"] = before - len(t)
    if sample_users:
        users = pd.Series(t["user_id"].unique())
        keep = set(users.sample(min(sample_users, len(users)), random_state=seed))
        before = len(t)
        t = t[t["user_id"].isin(keep)]
        log["sample"] = {"users": len(keep), "rows_removed": before - len(t)}
    kyc = None
    if kyc_path:
        if m.kyc is None:
            raise ImportError_("fichier KYC fourni mais section kyc absente du mapping")
        kyc = normalize_kyc(read_table(kyc_path, m.kyc.format), m.kyc, m)
        log["kyc"] = {"users": int(len(kyc)),
                      "coverage": round(float(t["user_id"].isin(set(kyc["user_id"])).mean()), 4) if len(t) else 0}
    log["pseudonymized"] = pseudo is not None
    t = pseudonymize_frame(t, pseudo, ("user_id", "wallet_id", "card_id", "device_id", "counterparty_id"))
    out = workspace / "sources" / m.name
    out.mkdir(parents=True, exist_ok=True)
    t[RAW_COLUMNS].to_csv(out / "transactions.csv", index=False)
    if kyc is not None:
        pseudonymize_frame(kyc, pseudo, ("user_id", "wallet_id")).to_csv(out / "kyc.csv", index=False)
    log["period"] = {"start": str(t["timestamp"].min()), "end": str(t["timestamp"].max())} if len(t) else {}
    (out / "import_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str),
                                         encoding="utf-8")
    return log


def build_workspace(workspace: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fusionne toutes les sources (plusieurs opérateurs) -> workspace/raw/{transactions,users}.csv."""
    sources = sorted(p for p in (workspace / "sources").glob("*") if (p / "transactions.csv").exists())
    if not sources:
        raise ImportError_(f"aucune source importée dans {workspace / 'sources'} (lancer d'abord « import »)")
    logs = {p.name: json.loads((p / "import_log.json").read_text(encoding="utf-8")) for p in sources}
    tx = pd.concat([pd.read_csv(p / "transactions.csv", dtype={c: str for c in (
        "transaction_id", "user_id", "wallet_id", "card_id", "counterparty_id", "merchant_id", "agent_id",
        "device_id")}, low_memory=False) for p in sources], ignore_index=True)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    before = len(tx)
    tx = tx.drop_duplicates("transaction_id", keep="first")
    build_log = {"sources": [p.name for p in sources], "duplicates_across_sources": before - len(tx)}
    tx = tx.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)

    kyc_frames = [pd.read_csv(p / "kyc.csv", dtype={"user_id": str, "wallet_id": str})
                  for p in sources if (p / "kyc.csv").exists()]
    kyc = (pd.concat(kyc_frames).drop_duplicates("user_id", keep="last").set_index("user_id")
           if kyc_frames else pd.DataFrame(columns=["wallet_id", "province"]).rename_axis("user_id"))
    period_start = tx["timestamp"].min().normalize()

    # référentiel clients : KYC quand il existe, sinon profil déduit de l'activité + valeurs prudentes
    first = tx.groupby("user_id").agg(wallet_id=("wallet_id", "first"), operator=("operator", "first"))
    modal_prov = (tx.dropna(subset=["location_province"]).groupby("user_id")["location_province"]
                  .agg(lambda s: s.mode().iat[0]))
    users = first.copy()
    users["province"] = modal_prov.reindex(users.index)
    for col, default in (("kyc_level", DEFAULT_PROFILE["kyc_level"]),
                         ("kyc_tx_limit_usd", DEFAULT_PROFILE["kyc_tx_limit_usd"]),
                         ("monthly_income_usd", DEFAULT_PROFILE["monthly_income_usd"]),
                         ("has_visa_virtual", DEFAULT_PROFILE["has_visa_virtual"])):
        users[col] = kyc[col].reindex(users.index) if col in kyc else np.nan
        users[col] = users[col].fillna(default)
    if "province" in kyc:
        users["province"] = kyc["province"].reindex(users.index).fillna(users["province"])
    users["province"] = users["province"].fillna(DEFAULT_PROFILE["province"])
    if "wallet_id" in kyc:
        users["wallet_id"] = kyc["wallet_id"].reindex(users.index).fillna(users["wallet_id"])
    opened = (pd.to_datetime(kyc["account_opened_at"], errors="coerce").reindex(users.index)
              if "account_opened_at" in kyc else pd.Series(pd.NaT, index=users.index))
    users["account_age_days"] = ((period_start - opened).dt.days.clip(lower=0)
                                 .fillna(DEFAULT_PROFILE["account_age_days"]))
    users = users.reset_index()[USER_COLUMNS]
    build_log["users"] = {"total": int(len(users)),
                          "with_kyc": int(users["user_id"].isin(set(kyc.index)).sum()),
                          "with_account_age": int(opened.notna().sum())}

    # province de la transaction inconnue : province habituelle du client (pas de fausse alerte « hors zone »)
    miss = tx["location_province"].isna()
    build_log["province_filled_from_home"] = int(miss.sum())
    tx.loc[miss, "location_province"] = tx.loc[miss, "user_id"].map(users.set_index("user_id")["province"])
    # solde absent : valeur conventionnelle très élevée -> « montant / solde » ~ 0 et « vidage du
    # compte » jamais vrai : les variables de solde deviennent neutres au lieu d'inventer un solde.
    # Le flux temps réel de cet opérateur doit appliquer la même convention (docs/DONNEES_REELLES.md).
    no_bal = tx["balance_before_usd"].isna()
    build_log["balance_missing"] = int(no_bal.sum())
    tx["balance_before_usd"] = tx["balance_before_usd"].fillna(BALANCE_UNKNOWN_USD)

    raw = workspace / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    tx[RAW_COLUMNS].to_csv(raw / "transactions.csv", index=False)
    users.to_csv(raw / "users.csv", index=False)
    (workspace / "build_log.json").write_text(json.dumps({"build": build_log, "imports": logs}, indent=2,
                                                         ensure_ascii=False, default=str), encoding="utf-8")
    return tx, users, {"build": build_log, "imports": logs}
