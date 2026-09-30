// Offline cache: the big, versioned assets (models, WASM, fonts) come from the cache
// first; the app's own files come from the network first so updates show up at once.
const CACHE = "facial-v1";
const HEAVY = /\.(task|wasm|ttf)$|vision_bundle\.mjs$|vision_wasm_\w+\.js$|anthropic-sdk\.mjs$/;

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()),
));

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin || url.pathname.includes("/api/")) return;
  e.respondWith(caches.open(CACHE).then(async (cache) => {
    const hit = await cache.match(e.request);
    if (hit && HEAVY.test(url.pathname)) return hit;
    try {
      const res = await fetch(e.request);
      if (res.ok && res.status === 200) cache.put(e.request, res.clone());
      return res;
    } catch (err) {
      if (hit) return hit;
      throw err;
    }
  }));
});
