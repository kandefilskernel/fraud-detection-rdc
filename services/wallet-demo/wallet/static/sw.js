// Service worker minimal : met en cache l'interface pour l'installation sur l'écran d'accueil.
// Les appels à l'API (décisions de fraude) ne sont JAMAIS mis en cache.
const CACHE = "portefeuille-demo-v1";
const SHELL = ["./", "static/style.css", "static/app.js", "static/icon.svg", "manifest.webmanifest"];

self.addEventListener("install", (e) => e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL))));
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))));
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET" || new URL(e.request.url).pathname.includes("/api/")) return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
