const APG_CACHE='apg-ui-v2';
const APG_STATIC=["/static/apg-charts.js", "/static/apg-sse.js", "/static/apg.css", "/static/htmx.min.js", "/static/sortable.min.js", "/static/uplot.min.css", "/static/uplot.min.js", "/ui"];
self.addEventListener('install',event=>{event.waitUntil(caches.open(APG_CACHE).then(cache=>cache.addAll(APG_STATIC)).then(()=>self.skipWaiting()))});
self.addEventListener('activate',event=>{event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key!==APG_CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim()))});
self.addEventListener('message',event=>{if(event.data&&event.data.type==='SKIP_WAITING')self.skipWaiting()});
self.addEventListener('fetch',event=>{const req=event.request;if(req.method!=='GET'||new URL(req.url).origin!==location.origin)return;event.respondWith(fetch(req).then(res=>{const copy=res.clone();if(res.ok){caches.open(APG_CACHE).then(cache=>cache.put(req,copy))}return res}).catch(()=>caches.match(req).then(cached=>cached||caches.match('/ui'))))});
