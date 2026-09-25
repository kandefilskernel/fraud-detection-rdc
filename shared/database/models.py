"""
Modèle de données PostgreSQL / TimescaleDB, partagé par le backoffice-api et les workers.

    bo_users             utilisateurs du back-office (analystes, superviseurs, admins)
    scored_transactions  toutes les transactions scorées + décision + explication
                         (hypertable TimescaleDB partitionnée sur tx_time) ; sert aussi de
                         journal des prédictions pour l'apprentissage continu (label)
    cases                dossiers ouverts pour les alertes (VERIFY / BLOCK)
    notifications        alertes envoyées (e-mail, SMS, webhook opérateur)
    audit_log            journal d'audit inaltérable, chaîné par hachage SHA-256
                         (UPDATE / DELETE interdits par trigger) — traçabilité BCC
    model_registry       versions du modèle, métriques, statut champion/challenger
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (BigInteger, Boolean, DateTime, Float, ForeignKey, Index, Integer, SmallInteger,
                        String, Text, func)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class BackofficeUser(Base):
    __tablename__ = "bo_users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20))            # ANALYSTE | SUPERVISEUR | ADMIN
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ScoredTransaction(Base):
    __tablename__ = "scored_transactions"
    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tx_time: Mapped[datetime] = mapped_column(DateTime, primary_key=True)   # heure locale RDC
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    channel: Mapped[str] = mapped_column(String(20))
    operator: Mapped[str | None] = mapped_column(String(20))
    tx_type: Mapped[str] = mapped_column(String(20))
    access_channel: Mapped[str] = mapped_column(String(10))
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3))
    amount_usd: Mapped[float] = mapped_column(Float)
    province: Mapped[str] = mapped_column(String(40))
    device_id: Mapped[str] = mapped_column(String(64))
    counterparty_id: Mapped[str | None] = mapped_column(String(64))
    agent_id: Mapped[str | None] = mapped_column(String(64))
    merchant_id: Mapped[str | None] = mapped_column(String(64))
    fraud_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(10))
    action: Mapped[str] = mapped_column(String(10))
    reason: Mapped[str] = mapped_column(String(255))
    rules_triggered: Mapped[list] = mapped_column(JSONB, default=list)
    model_version: Mapped[str] = mapped_column(String(40))
    latency_ms: Mapped[float] = mapped_column(Float)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    explanation: Mapped[dict] = mapped_column(JSONB)
    features: Mapped[dict] = mapped_column(JSONB)
    # étiquette issue du retour des analystes : 1 fraude confirmée, 0 faux positif
    label: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    labeled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_scored_action_time", "action", "tx_time"),
        Index("ix_scored_risk_time", "risk_level", "tx_time"),
        Index("ix_scored_scored_at", "scored_at"),   # KPIs temps réel (horloge réelle)
    )


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tx_time: Mapped[datetime] = mapped_column(DateTime)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    channel: Mapped[str] = mapped_column(String(20))
    tx_type: Mapped[str] = mapped_column(String(20))
    amount_usd: Mapped[float] = mapped_column(Float)
    fraud_probability: Mapped[float] = mapped_column(Float)
    risk_level: Mapped[str] = mapped_column(String(10), index=True)
    action: Mapped[str] = mapped_column(String(10))
    # OUVERT -> EN_COURS -> FRAUDE_CONFIRMEE | FAUX_POSITIF
    status: Mapped[str] = mapped_column(String(20), default="OUVERT", index=True)
    assigned_to: Mapped[int | None] = mapped_column(ForeignKey("bo_users.id"), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_by: Mapped[int | None] = mapped_column(ForeignKey("bo_users.id"), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 onupdate=func.now())


class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int | None] = mapped_column(ForeignKey("cases.id"), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(String(10))       # EMAIL | SMS | WEBHOOK
    recipient: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(10))        # ENVOYE | ECHEC
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    actor: Mapped[str] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(60))
    entity: Mapped[str] = mapped_column(String(40))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict] = mapped_column(JSONB)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64), unique=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(40), unique=True)
    status: Mapped[str] = mapped_column(String(12))        # CHAMPION | CHALLENGER | ARCHIVE | REJETE
    metrics: Mapped[dict] = mapped_column(JSONB)
    trained_on: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
