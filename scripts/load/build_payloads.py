"""Prépare les messages opérateurs (formats propriétaires) utilisés par le test de charge k6."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "services" / "integration-layer")]
from simulate_transactions import ADAPTERS, HMAC_SECRETS, KEYS, load, provider_of  # noqa: E402

rows = load(n=5000, skip=20000, only_fraud=False)
# le secret HMAC sert à k6 pour signer chaque requête au moment de l'envoi (horodatage)
out = [{"provider": provider_of(r), "key": KEYS[provider_of(r)], "secret": HMAC_SECRETS[provider_of(r)],
        "body": ADAPTERS[provider_of(r)].from_unified(r)} for r in rows]
dest = Path(__file__).parent / "payloads.json"
dest.write_text(json.dumps(out, default=str), encoding="utf-8")
print(f"{len(out)} messages -> {dest}")
