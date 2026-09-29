"""
Personnages de la démonstration : de VRAIS clients du jeu de données synthétique, pris avec
leur historique (téléphone habituel, contacts, agent, solde) tel qu'il était au début de la
période de test. La plateforme les connaît donc déjà : leurs opérations habituelles passent,
les scénarios de fraude se voient.

Les noms sont fictifs et ajoutés pour la démonstration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

OPERATORS = ("VODACOM", "AIRTEL", "ORANGE")
DISPLAY_NAMES = {"VODACOM": "Maman Nsimba", "AIRTEL": "Papa Kabeya", "ORANGE": "Grâce Mbuyi"}
CONTACT_NAMES = ["Frère Jonas", "Maman Chantal", "Boutique Lulu", "Oncle Didier"]
ID_COLS = {c: str for c in ("user_id", "wallet_id", "counterparty_id", "agent_id", "device_id")}


@dataclass
class Contact:
    name: str
    wallet: str


@dataclass
class Persona:
    id: str
    name: str
    operator: str
    user_id: str
    wallet_id: str
    province: str
    device_id: str
    device_type: str
    balance_usd: float
    currency: str
    kyc_limit_usd: float
    agent_id: str | None
    contacts: list[Contact] = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("device_id")
        return d


def _canon(v) -> str | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v)
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def build_personas(data_dir: Path, cutoff: pd.Timestamp) -> tuple[list[Persona], str | None]:
    """Un client exemplaire par opérateur, et le portefeuille d'une mule déjà signalée
    avant `cutoff` (donc présente dans le profil de réputation de la plateforme)."""
    users = pd.read_csv(data_dir / "users.csv", dtype=ID_COLS)
    tx = pd.read_csv(data_dir / "transactions.csv", dtype=ID_COLS, low_memory=False,
                     usecols=["timestamp", "user_id", "channel", "operator", "tx_type", "counterparty_id",
                              "agent_id", "device_id", "balance_after_usd", "is_fraud", "fraud_reported_at"])
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx[tx["timestamp"] < cutoff].sort_values("timestamp", kind="mergesort")
    for c in ("counterparty_id", "agent_id", "device_id", "user_id"):
        tx[c] = tx[c].map(_canon)
    mm = tx[tx.channel == "MOBILE_MONEY"]

    stats = mm.groupby("user_id").agg(n=("tx_type", "size"), frauds=("is_fraud", "sum"))
    good = users[(~users.is_mule_account.astype(bool)) & (users.device_type == "smartphone")
                 & (users.kyc_level >= 2)].merge(stats, left_on="user_id", right_index=True)
    good = good[good.frauds == 0].sort_values(["n", "user_id"], ascending=[False, True])

    personas = []
    for op in OPERATORS:
        cand = good[good.operator == op]
        if cand.empty:
            continue
        u = cand.iloc[0]
        own = mm[mm.user_id == u.user_id]
        # téléphone habituel : hors dépôts (initiés depuis la ligne de l'agent)
        device = own.loc[own.tx_type != "CASH_IN", "device_id"].mode().iloc[0]
        sends = own.loc[own.tx_type == "P2P_SEND", "counterparty_id"].value_counts().index[:len(CONTACT_NAMES)]
        agents = own.loc[own.tx_type == "CASH_OUT", "agent_id"].dropna()
        personas.append(Persona(
            id=op.lower(), name=DISPLAY_NAMES[op], operator=op, user_id=u.user_id, wallet_id=_canon(u.wallet_id),
            province=u.province, device_id=device, device_type=u.device_type,
            balance_usd=round(float(own["balance_after_usd"].iloc[-1]), 2), currency=u.preferred_currency,
            kyc_limit_usd=float(u.kyc_tx_limit_usd), agent_id=agents.mode().iloc[0] if len(agents) else None,
            contacts=[Contact(CONTACT_NAMES[i], w) for i, w in enumerate(sends)]))

    # mule déjà signalée avant le début de la démonstration (réputation connue de la plateforme)
    reported = tx[(tx.is_fraud == 1) & (tx.tx_type == "P2P_SEND") & tx.fraud_reported_at.notna()]
    reported = reported[pd.to_datetime(reported.fraud_reported_at) < cutoff]
    mule = None
    if len(reported):
        mule = reported.groupby("counterparty_id").user_id.nunique().sort_values(ascending=False).index[0]
    return personas, mule
