"""
Feature store temps réel (Redis) pour le profil comportemental.

Le calcul des variables N'EST PAS réécrit : on réutilise `BehavioralFeatureExtractor`
(le code exact de l'entraînement). Pour chaque transaction :

    1. on lit dans Redis l'état des seules entités concernées (titulaire, appareil,
       contrepartie, agent) et on le place dans un extracteur « de travail » ;
    2. on appelle extract() puis update(), comme lors du rejeu d'entraînement ;
    3. on réécrit l'état modifié dans Redis.

Les variables produites en production sont donc identiques à celles vues par le modèle
(vérifié par tests/test_serving_parity.py) : pas d'écart entraînement/production.

Clés Redis (préfixe fs:) :
    fs:profile:{uid}  JSON  profil KYC (province, niveau KYC, plafond, revenu...)
    fs:user:{uid}     JSON  état comportemental (_UserState sérialisé)
    fs:dev:{device}   SET   titulaires ayant utilisé l'appareil
    fs:cp:{wallet}    SET   titulaires en relation avec ce portefeuille (réseau de mules)
    fs:agent:{agent}  ZSET  titulaire -> dernier passage (fenêtre glissante 24 h)
    fs:seq:{uid}      LIST  vecteurs normalisés des SEQ_LEN-1 dernières transactions (LSTM)
    fs:lock:{uid}     verrou court : deux transactions d'un même client sont traitées en série
"""
from __future__ import annotations

import json
import time
import uuid
from collections import deque
from contextlib import contextmanager

import numpy as np

from ml.features.feature_engineering import DAY, BehavioralFeatureExtractor, _UserState

PREFIX = "fs"
DEFAULT_PROFILE = {  # client inconnu du référentiel KYC : profil prudent
    "province": "Kinshasa", "kyc_level": 1, "kyc_tx_limit_usd": 100.0,
    "account_age_days": 0.0, "monthly_income_usd": 100.0, "has_visa_virtual": 0,
}


# ---------------------------------------------------------------------- sérialisation
def user_state_to_json(st: _UserState) -> str:
    return json.dumps({
        "n": st.n, "log_amt_sum": st.log_amt_sum, "log_amt_sq": st.log_amt_sq, "amt_sum": st.amt_sum,
        "first_ts": st.first_ts, "last_ts": st.last_ts,
        "recent": [list(r) for r in st.recent], "failures": list(st.failures),
        "devices": st.devices, "provinces": sorted(st.provinces),
        "counterparties": sorted(st.counterparties), "agents": sorted(st.agents),
        "merchants": sorted(st.merchants), "night_count": st.night_count,
        "hour_sin_sum": st.hour_sin_sum, "hour_cos_sum": st.hour_cos_sum,
        "last_topup_ts": st.last_topup_ts, "last_topup_amt": st.last_topup_amt,
    }, separators=(",", ":"))


def user_state_from_json(raw: str | bytes) -> _UserState:
    d = json.loads(raw)
    return _UserState(
        n=d["n"], log_amt_sum=d["log_amt_sum"], log_amt_sq=d["log_amt_sq"], amt_sum=d["amt_sum"],
        first_ts=d["first_ts"], last_ts=d["last_ts"],
        recent=deque((t, a, bool(b)) for t, a, b in d["recent"]), failures=deque(d["failures"]),
        devices=d["devices"], provinces=set(d["provinces"]),
        counterparties=set(d["counterparties"]), agents=set(d["agents"]),
        merchants=set(d["merchants"]), night_count=d["night_count"],
        hour_sin_sum=d["hour_sin_sum"], hour_cos_sum=d["hour_cos_sum"],
        last_topup_ts=d["last_topup_ts"], last_topup_amt=d["last_topup_amt"],
    )


_RELEASE_LUA = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) end return 0"


def _s(v) -> str:
    return v.decode() if isinstance(v, bytes) else v


class RedisFeatureStore:
    def __init__(self, redis_client, period_start_s: float, seq_len: int = 10, n_features: int = 58):
        self.r = redis_client
        self.period_start_s = period_start_s
        self.seq_len = seq_len
        self.n_features = n_features

    # ------------------------------------------------------------------ verrou par client
    @contextmanager
    def user_lock(self, uid: str, ttl_ms: int = 2000, wait_s: float = 1.0):
        key, token = f"{PREFIX}:lock:{uid}", uuid.uuid4().hex
        deadline = time.monotonic() + wait_s
        while not self.r.set(key, token, nx=True, px=ttl_ms):
            if time.monotonic() > deadline:
                raise TimeoutError(f"profil {uid} verrouillé")
            time.sleep(0.002)
        try:
            yield
        finally:
            # libère le verrou seulement s'il nous appartient encore (1 aller-retour, atomique)
            self.r.eval(_RELEASE_LUA, 1, key, token)

    # ------------------------------------------------------------------ lecture / écriture
    def _load(self, tx: dict) -> tuple[BehavioralFeatureExtractor, list[np.ndarray], bool]:
        uid, dev = tx["user_id"], tx["device_id"]
        cp = tx["counterparty_id"] if tx["tx_type"] in ("P2P_SEND", "P2P_RECEIVE") else None
        agent = tx["agent_id"] if isinstance(tx["agent_id"], str) else None

        p = self.r.pipeline(transaction=False)
        p.get(f"{PREFIX}:profile:{uid}")
        p.get(f"{PREFIX}:user:{uid}")
        p.smembers(f"{PREFIX}:dev:{dev}")
        p.smembers(f"{PREFIX}:cp:{cp}") if cp else p.echo("")
        p.zrange(f"{PREFIX}:agent:{agent}", 0, -1, withscores=True) if agent else p.echo("")
        p.lrange(f"{PREFIX}:seq:{uid}", 0, -1)
        prof_raw, user_raw, dev_users, cp_users, agent_rows, seq_raw = p.execute()

        ext = BehavioralFeatureExtractor.__new__(BehavioralFeatureExtractor)
        ext.period_start_s = self.period_start_s
        known = prof_raw is not None
        ext.profiles = {uid: json.loads(prof_raw) if known else DEFAULT_PROFILE}
        ext.users = {uid: user_state_from_json(user_raw)} if user_raw else {}
        ext.device_users = {dev: {_s(u) for u in dev_users}}
        ext.counterparty_users = {cp: {_s(u) for u in cp_users}} if cp else {}
        ext.agent_recent = ({agent: deque(sorted(((s, _s(u)) for u, s in agent_rows)))}
                            if agent else {})
        seq = [np.frombuffer(b, dtype=np.float32) for b in seq_raw]
        return ext, seq, known

    def _save(self, ext: BehavioralFeatureExtractor, tx: dict, x_scaled: np.ndarray) -> None:
        uid, dev, ts = tx["user_id"], tx["device_id"], tx["ts"]
        p = self.r.pipeline(transaction=True)
        p.set(f"{PREFIX}:user:{uid}", user_state_to_json(ext.users[uid]))
        p.sadd(f"{PREFIX}:dev:{dev}", uid)
        if tx["tx_type"] in ("P2P_SEND", "P2P_RECEIVE") and tx["counterparty_id"]:
            p.sadd(f"{PREFIX}:cp:{tx['counterparty_id']}", uid)
        if isinstance(tx["agent_id"], str):
            key = f"{PREFIX}:agent:{tx['agent_id']}"
            p.zadd(key, {uid: ts})
            p.zremrangebyscore(key, "-inf", f"({ts - DAY}")
        seq_key = f"{PREFIX}:seq:{uid}"
        p.rpush(seq_key, x_scaled.astype(np.float32).tobytes())
        p.ltrim(seq_key, -(self.seq_len - 1), -1)
        p.execute()

    # ------------------------------------------------------------------ API
    def compute(self, tx: dict, scale_fn, decide_fn=None) -> dict:
        """Calcule les variables de `tx`, (optionnellement) décide, puis met le profil à jour.

        decide_fn(features, x, history) -> (résultat, statut) est appelé AVANT la mise à
        jour, sous le verrou du client. Si tx["status"] est None (temps réel : transaction
        pas encore exécutée), le statut renvoyé par decide_fn est enregistré dans le
        profil — un blocage compte alors comme une tentative échouée."""
        with self.user_lock(tx["user_id"]):
            ext, history, known = self._load(tx)
            feats = ext.extract(tx)
            x = scale_fn(feats)
            decided, status = decide_fn(feats, x, history) if decide_fn else (None, "SUCCESS")
            if tx.get("status") is None:
                tx = {**tx, "status": status}
            ext.update(tx)
            self._save(ext, tx, x)
        return {"features": feats, "x": x, "history": history, "known_user": known, "decided": decided}

    # ------------------------------------------------------------------ amorçage
    def bulk_load(self, extractor: BehavioralFeatureExtractor, sequences: dict[str, list[np.ndarray]],
                  batch: int = 2000) -> dict:
        """Copie dans Redis l'état complet d'un extracteur rejoué hors ligne (amorçage)."""
        r, n = self.r, 0

        def chunks(items):
            items = list(items)
            for i in range(0, len(items), batch):
                yield items[i:i + batch]

        for part in chunks(extractor.profiles.items()):
            p = r.pipeline(transaction=False)
            for uid, prof in part:
                p.set(f"{PREFIX}:profile:{uid}", json.dumps({k: (v.item() if hasattr(v, "item") else v)
                                                             for k, v in prof.items()}))
            p.execute()
        for part in chunks(extractor.users.items()):
            p = r.pipeline(transaction=False)
            for uid, st in part:
                p.set(f"{PREFIX}:user:{uid}", user_state_to_json(st))
                n += 1
            p.execute()
        for name, mapping in (("dev", extractor.device_users), ("cp", extractor.counterparty_users)):
            for part in chunks(mapping.items()):
                p = r.pipeline(transaction=False)
                for key, users in part:
                    p.delete(f"{PREFIX}:{name}:{key}")
                    if users:
                        p.sadd(f"{PREFIX}:{name}:{key}", *users)
                p.execute()
        for part in chunks(extractor.agent_recent.items()):
            p = r.pipeline(transaction=False)
            for agent, q in part:
                key = f"{PREFIX}:agent:{agent}"
                p.delete(key)
                latest: dict[str, float] = {}
                for t, u in q:
                    latest[u] = max(t, latest.get(u, t))
                if latest:
                    p.zadd(key, latest)
            p.execute()
        for part in chunks(sequences.items()):
            p = r.pipeline(transaction=False)
            for uid, vecs in part:
                key = f"{PREFIX}:seq:{uid}"
                p.delete(key)
                if vecs:
                    p.rpush(key, *[v.astype(np.float32).tobytes() for v in vecs[-(self.seq_len - 1):]])
            p.execute()
        return {"users": n, "devices": len(extractor.device_users),
                "counterparties": len(extractor.counterparty_users), "agents": len(extractor.agent_recent)}
