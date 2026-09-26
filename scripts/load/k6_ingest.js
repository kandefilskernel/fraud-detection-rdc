// Test de charge : trafic opérateurs vers /ingest (Nginx -> integration-layer -> scoring).
// docker run --rm --network fraud-detection-rdc_default -v ./scripts/load:/load grafana/k6 run /load/k6_ingest.js
import http from "k6/http";
import { check } from "k6";
import { SharedArray } from "k6/data";
import { Trend, Counter } from "k6/metrics";

const payloads = new SharedArray("payloads", () => JSON.parse(open("./payloads.json")));
const BASE = __ENV.BASE_URL || "http://nginx/ingest";
const scoring = new Trend("scoring_ms", true);
const decisions = new Counter("decisions");

export const options = {
  scenarios: {
    montee: {
      executor: "ramping-arrival-rate",
      startRate: 10, timeUnit: "1s", preAllocatedVUs: 50, maxVUs: 200,
      stages: [
        { target: 50, duration: "30s" },    // trafic normal
        { target: 100, duration: "30s" },   // pointe (fin de mois, paie)
        { target: 100, duration: "60s" },
        { target: 0, duration: "10s" },
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"],          // < 1 % d'erreurs
    http_req_duration: ["p(95)<250"],        // bout-en-bout, depuis le réseau Docker
    scoring_ms: ["p(95)<100"],               // temps de scoring mesuré par le service
  },
};

export default function () {
  // chaque itération rejoue un message différent ; les ID sont rendus uniques
  const p = payloads[(__ITER * 7 + __VU * 131) % payloads.length];
  const body = JSON.parse(JSON.stringify(p.body));
  const suffix = `-${__VU}-${__ITER}`;
  if (body.TransID) body.TransID += suffix;
  else if (body.transaction) body.transaction.id += suffix;
  else if (body.id_transaction) body.id_transaction += suffix;
  else if (body.stan) body.stan += suffix;
  const res = http.post(`${BASE}/v1/transactions/${p.provider}`, JSON.stringify(body),
    { headers: { "Content-Type": "application/json", "X-API-Key": p.key } });
  const ok = check(res, { "statut 200": (r) => r.status === 200 });
  if (ok) {
    const d = res.json();
    if (d.latency_ms) scoring.add(d.latency_ms);
    decisions.add(1, { action: d.action });
  }
}
