// Portefeuille démo : application client simulée (aucune dépendance, aucun secret côté téléphone).
"use strict";

const $ = (id) => document.getElementById(id);
const state = { sid: null, persona: null, balance: 0, mule: null, pin: "1234", rate: 2850, simOn: false };

// ------------------------------------------------------------------ utilitaires
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `erreur ${res.status}`);
  return data;
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function money(usd, currency) {
  if (currency === "CDF") return `${Math.round(usd * state.rate).toLocaleString("fr-FR")} FC`;
  return `${usd.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} $`;
}
function toast(msg, ms = 3200) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => t.classList.add("hidden"), ms);
}
function openSheet(html) {
  $("sheet-body").innerHTML = html;
  $("sheet").classList.remove("hidden"); $("sheet-backdrop").classList.remove("hidden");
}
function closeSheet() { $("sheet").classList.add("hidden"); $("sheet-backdrop").classList.add("hidden"); }
$("sheet-backdrop").onclick = closeSheet;
const cur = () => state.persona.currency;

// ------------------------------------------------------------------ profils
async function loadProfiles() {
  const data = await api("api/personas");
  state.mule = data.known_mule; state.pin = data.demo_pin; state.rate = data.usd_cdf;
  $("profiles").innerHTML = data.personas.map((p) => `
    <button class="profile" data-id="${esc(p.id)}">
      <div class="avatar">${esc(p.name.split(" ").map((w) => w[0]).join("").slice(0, 2))}</div>
      <div class="meta"><div class="name">${esc(p.name)}</div>
        <div class="sub">${esc(p.operator)} · ${esc(p.province)} · ${esc(p.wallet_id)}</div></div>
      <div class="sub">${money(p.balance_usd, p.currency)}</div>
    </button>`).join("") || `<div class="empty">Aucun profil disponible.</div>`;
  document.querySelectorAll(".profile").forEach((b) => (b.onclick = () => startSession(b.dataset.id)));
}

async function startSession(id) {
  const data = await api("api/session", { persona_id: id });
  state.sid = data.session_id; state.persona = data.persona; state.simOn = false;
  $("screen-profiles").classList.add("hidden"); $("screen-home").classList.remove("hidden");
  await refresh();
}

async function refresh() {
  const s = await api(`api/session/${state.sid}`);
  state.balance = s.balance_usd; state.simOn = s.sim_swapped;
  const p = s.persona;
  $("who").textContent = p.name;
  $("who-sub").textContent = `${p.operator} · ${p.wallet_id}`;
  $("demo-clock").textContent = new Date(s.demo_time).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
  $("balance").textContent = money(s.balance_usd, p.currency);
  $("balance-alt").textContent = p.currency === "CDF" ? `≈ ${money(s.balance_usd, "USD")}` : "";
  $("sim-banner").classList.toggle("hidden", !s.sim_swapped);
  $("sc-sim").classList.toggle("on", s.sim_swapped);
  $("sc-sim-title").textContent = s.sim_swapped ? "Arrêter la simulation de SIM swap" : "Simuler un SIM swap";
  $("sc-drain").classList.toggle("hidden", !s.sim_swapped);
  renderHistory(s.history);
}

function renderHistory(items) {
  const labels = { EXECUTEE: "Exécutée", REFUSEE: "Refusée", EN_ATTENTE_PIN: "PIN demandé", EN_ATTENTE_AGENCE: "En agence",
    ANNULEE: "Annulée", ECHEC: "Échec" };
  $("history").innerHTML = items.length ? items.map((h) => `
    <div class="tx">
      <div class="tx-row"><span class="tx-label">${esc(h.label)}</span>
        <span class="tx-amount ${h.debit ? "debit" : "credit"}">${h.debit ? "−" : "+"}${money(h.amount_usd, h.currency)}</span></div>
      <div class="tx-meta"><span class="chip ${h.status}">${labels[h.status] || h.status}${h.reported ? " · signalée" : ""}</span>
        <span class="tx-actions">
          <button class="link" data-resend="${esc(h.tx_id)}" title="Coupure réseau : l'opérateur renvoie le même message">Renvoyer</button>
          ${h.debit && !h.reported ? `<button class="link danger" data-report="${esc(h.tx_id)}">Signaler une fraude</button>` : ""}
        </span></div>
    </div>`).join("") : `<div class="empty">Aucune opération pour l'instant.</div>`;
  document.querySelectorAll("[data-resend]").forEach((b) => (b.onclick = () => resend(b.dataset.resend)));
  document.querySelectorAll("[data-report]").forEach((b) => (b.onclick = () => report(b.dataset.report)));
}

// ------------------------------------------------------------------ envoyer / retirer
function amountFields(defaultAmount) {
  return `<div class="field"><label>Montant</label><div class="amount-row">
      <input id="f-amount" type="number" inputmode="decimal" min="0" step="any" value="${defaultAmount}">
      <select id="f-cur"><option value="CDF" ${cur() === "CDF" ? "selected" : ""}>FC</option>
        <option value="USD" ${cur() === "USD" ? "selected" : ""}>$</option></select></div></div>`;
}
const defaultAmount = (usd) => (cur() === "CDF" ? Math.round(usd * state.rate / 500) * 500 : usd);

function sendSheet(prefill = {}) {
  const contacts = state.persona.contacts.map((c) => `
    <button class="contact" data-w="${esc(c.wallet)}" data-n="${esc(c.name)}">${esc(c.name)}</button>`).join("");
  openSheet(`<h3>Envoyer de l'argent</h3>
    <div class="field"><label>Contacts habituels</label><div class="chips">${contacts || "<span class='sub'>aucun</span>"}</div></div>
    <div class="field"><label>Numéro du destinataire</label>
      <input id="f-to" inputmode="numeric" placeholder="243 8XX XXX XXX" value="${esc(prefill.to || "")}"></div>
    ${prefill.currency ? `<div class="field"><label>Montant</label><div class="amount-row">
      <input id="f-amount" type="number" inputmode="decimal" value="${prefill.amount}">
      <select id="f-cur"><option ${prefill.currency === "CDF" ? "selected" : ""} value="CDF">FC</option>
        <option ${prefill.currency === "USD" ? "selected" : ""} value="USD">$</option></select></div></div>`
      : amountFields(defaultAmount(10))}
    <button class="btn" id="f-go">Envoyer</button>
    <button class="btn secondary" id="f-cancel">Annuler</button>`);
  let name = prefill.name || null;
  document.querySelectorAll(".contact").forEach((b) => (b.onclick = () => {
    document.querySelectorAll(".contact").forEach((x) => x.classList.remove("active"));
    b.classList.add("active"); $("f-to").value = b.dataset.w; name = b.dataset.n;
  }));
  $("f-cancel").onclick = closeSheet;
  $("f-go").onclick = () => submit("api/transfer", {
    session_id: state.sid, to_wallet: $("f-to").value.replace(/\s/g, ""), to_name: name,
    amount: parseFloat($("f-amount").value), currency: $("f-cur").value });
}

function withdrawSheet(opts = {}) {
  openSheet(`<h3>${opts.title || "Retrait chez un agent"}</h3>
    ${amountFields(opts.amount ?? defaultAmount(20))}
    <label class="check"><input type="checkbox" id="f-other" ${opts.otherAgent ? "checked" : ""}> Chez un agent inhabituel</label>
    <button class="btn" id="f-go">Retirer</button>
    <button class="btn secondary" id="f-cancel">Annuler</button>`);
  $("f-cancel").onclick = closeSheet;
  $("f-go").onclick = () => submit("api/withdraw", {
    session_id: state.sid, amount: parseFloat($("f-amount").value), currency: $("f-cur").value,
    other_agent: $("f-other").checked });
}

async function submit(path, body) {
  const btn = $("f-go"); btn.disabled = true; btn.textContent = "Vérification…";
  try { showResult(await api(path, body)); }
  catch (e) { btn.disabled = false; btn.textContent = "Réessayer"; toast(e.message); }
}

// ------------------------------------------------------------------ résultat de la plateforme
const ICONS = { success: "✓", pin: "🔒", agency: "🏢", blocked: "⛔", error: "!" };

function backstageHtml(b) {
  if (!b) return "";
  const p = b.fraud_probability == null ? null : Math.round(b.fraud_probability * 1000) / 10;
  const feats = (b.top_features || []).map((f) => `<div class="feat"><span>${esc(f.label)}</span>
    <span class="${f.shap > 0 ? "up" : "down"}">${f.shap > 0 ? "▲" : "▼"} ${Math.abs(f.shap).toFixed(2)}</span></div>`).join("");
  return `<details class="backstage"><summary>Coulisses : ce que la plateforme a calculé</summary>
    ${p == null ? "" : `<div class="sub" style="margin-top:8px">Probabilité de fraude : <strong>${p} %</strong></div>
      <div class="bar"><div style="width:${Math.min(p, 100)}%"></div></div>`}
    <dl class="bs-grid">
      <dt>Décision</dt><dd>${esc(b.action)}${b.idempotent_replay ? " (renvoi reconnu)" : ""}</dd>
      <dt>Risque</dt><dd>${esc(b.risk_level)}</dd>
      <dt>Motif</dt><dd>${esc(b.reason)}</dd>
      <dt>Vérification</dt><dd>${esc(b.verification_method || "—")}</dd>
      <dt>Règles</dt><dd>${esc((b.rules_triggered || []).join(", ") || "aucune")}</dd>
      <dt>Latence</dt><dd>${b.latency_ms ?? "—"} ms scoring · ${b.end_to_end_ms ?? "—"} ms au total</dd>
    </dl>
    ${feats ? `<div class="sub" style="margin-top:10px">Variables décisives (▲ pousse vers la fraude)</div>${feats}` : ""}
  </details>`;
}

function showResult(r) {
  const v = r.view;
  const warnings = (v.warnings || []).map((w) => `<div class="warning">⚠ ${esc(w)}</div>`).join("");
  let actions = `<button class="btn" id="r-ok">Terminé</button>`;
  if (v.screen === "pin") {
    actions = `<div class="field"><input id="r-pin" type="password" inputmode="numeric" maxlength="4" placeholder="Code PIN"></div>
      <div class="pin-hint">PIN de démonstration : ${esc(state.pin)}</div>
      <button class="btn" id="r-confirm">Confirmer</button>
      <button class="btn secondary" id="r-cancel">Annuler l'opération</button>`;
  }
  openSheet(`<div class="result-icon ${v.screen}">${ICONS[v.screen] || "•"}</div>
    <p class="result-title">${esc(v.title)}</p>
    ${v.message ? `<p class="result-msg">${esc(v.message)}</p>` : ""}
    ${warnings}${actions}${backstageHtml(r.backstage)}`);
  if ($("r-ok")) $("r-ok").onclick = () => { closeSheet(); refresh(); };
  if ($("r-confirm")) $("r-confirm").onclick = async () => {
    try { await api("api/confirm", { session_id: state.sid, tx_id: r.tx_id, pin: $("r-pin").value });
      closeSheet(); toast("Opération confirmée et exécutée."); refresh(); }
    catch (e) { toast(e.message); }
  };
  if ($("r-cancel")) $("r-cancel").onclick = async () => {
    await api("api/confirm", { session_id: state.sid, tx_id: r.tx_id, pin: null });
    closeSheet(); toast("Opération annulée : votre argent est resté sur votre compte."); refresh();
  };
  refresh();
}

// ------------------------------------------------------------------ scénarios
async function scenarioScam() {
  const r = await api("api/scenario/unknown-sender", { session_id: state.sid, enabled: true });
  const s = r.scam;
  openSheet(`<h3>Nouveau message</h3>
    <div class="warning" style="background:var(--info-bg);border-color:var(--brand)">
      Vous avez reçu ${money(s.received / (s.currency === "CDF" ? state.rate : 1), s.currency)} de ${esc(s.from)}.</div>
    <div class="chat"><div class="bubble-from">SMS de ${esc(s.from)}</div><div class="bubble">${esc(s.message)}</div></div>
    <button class="btn" id="s-send">Renvoyer l'argent demandé</button>
    <button class="btn secondary" id="s-report">⚑ Signaler ce SMS comme arnaque</button>
    <button class="btn secondary" id="s-ignore">Ignorer</button>
    ${backstageHtml(r.backstage)}`);
  $("s-ignore").onclick = () => { closeSheet(); refresh(); };
  $("s-report").onclick = async () => {
    const btn = $("s-report"); btn.disabled = true; btn.textContent = "Analyse du SMS…";
    try {
      const r = await api("api/report-sms", { session_id: state.sid, sender: s.from, text: s.message });
      const pct = Math.round(r.scam_probability * 100);
      btn.textContent = r.is_scam ? "✓ Signalé" : "✓ Transmis";
      toast(r.is_scam
        ? `SMS classé « ${r.category} » (${pct} %). Le numéro ${s.from} est signalé : tout envoi vers lui demandera une confirmation.`
        : `SMS analysé : ${r.category} (${pct} % d'arnaque). Aucun numéro n'a été signalé.`, 6500);
    } catch (e) { btn.disabled = false; btn.textContent = "⚑ Signaler ce SMS comme arnaque"; toast(e.message); }
  };
  $("s-send").onclick = () => sendSheet({ to: s.from, name: s.from, amount: s.asked, currency: s.currency });
  refresh();
}

async function toggleSim() {
  await api("api/scenario/sim-swap", { session_id: state.sid, enabled: !state.simOn });
  await refresh();
  toast(state.simOn ? "Vous jouez maintenant le fraudeur : nouvelle SIM, autre téléphone."
                    : "Retour au vrai client, sur son téléphone habituel.");
}

async function resend(txId) {
  try {
    const r = await api("api/resend", { session_id: state.sid, tx_id: txId });
    toast(r.idempotent_replay ? "Renvoi reconnu : même décision, rien n'est débité deux fois."
                              : `Renvoyé : ${r.action}`, 4200);
  } catch (e) { toast(e.message); }
}

async function report(txId) {
  try {
    const r = await api("api/report", { session_id: state.sid, tx_id: txId });
    toast(r.http_status === 200 ? "Fraude signalée : la plateforme apprend de ce cas (étiquette + réputation)."
                                : `Signalement refusé : ${r.detail || r.http_status}`, 4500);
    refresh();
  } catch (e) { toast(e.message); }
}

// ------------------------------------------------------------------ branchements
$("btn-send").onclick = () => sendSheet();
$("btn-withdraw").onclick = () => withdrawSheet();
$("sc-scam").onclick = () => scenarioScam().catch((e) => toast(e.message));
$("sc-mule").onclick = () => state.mule ? sendSheet({ to: state.mule, name: "numéro inconnu", amount: defaultAmount(30), currency: cur() })
                                        : toast("Aucune mule signalée dans les données.");
$("sc-sim").onclick = () => toggleSim().catch((e) => toast(e.message));
$("sc-drain").onclick = () => withdrawSheet({ title: "Le fraudeur retire presque tout", otherAgent: true,
  amount: defaultAmount(Math.floor(state.balance * 0.95)) });
$("btn-switch").onclick = () => { $("screen-home").classList.add("hidden"); $("screen-profiles").classList.remove("hidden"); loadProfiles(); };

if ("serviceWorker" in navigator) navigator.serviceWorker.register("sw.js").catch(() => {});
loadProfiles().catch((e) => { $("profiles").innerHTML = `<div class="empty">Service indisponible : ${esc(e.message)}</div>`; });
