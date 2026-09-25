"""Topics Kafka (Redpanda). Le scoring est synchrone ; ces topics transportent les
événements APRÈS la décision (persistance, alertes, audit, retour des analystes)."""

TOPIC_TRANSACTIONS_SCORED = "transactions.scored"   # toute transaction scorée + décision
TOPIC_FRAUD_ALERTS = "fraud.alerts"                 # décisions VERIFY / BLOCK
TOPIC_ANALYST_FEEDBACK = "analyst.feedback"         # verdict des analystes (étiquettes)
TOPIC_AUDIT_LOGS = "audit.logs"                     # actions des utilisateurs du back-office

ALL_TOPICS = [TOPIC_TRANSACTIONS_SCORED, TOPIC_FRAUD_ALERTS, TOPIC_ANALYST_FEEDBACK, TOPIC_AUDIT_LOGS]
