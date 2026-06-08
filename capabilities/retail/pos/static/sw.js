/**
 * APG Retail POS — Service Worker (Phase 5A: Offline-First)
 *
 * Implements offline-first for the POS terminal:
 *   1. INSTALL: Pre-cache POS shell assets (HTML, CSS, JS)
 *   2. FETCH: Serve cached assets offline; queue failed API requests
 *   3. SYNC: Background-sync queued transactions when connectivity resumes
 *
 * Offline transaction storage: IndexedDB (apg-pos-offline-db)
 * Sync trigger: Background Sync API or manual on reconnect
 *
 * PCI DSS note: Card data is never stored in the service worker cache or
 * IndexedDB. Only tokenized values (from TokenizationService) are queued.
 */

const CACHE_NAME = 'apg-pos-v1';
const OFFLINE_TRANSACTION_STORE = 'offline-transactions';
const DB_NAME = 'apg-pos-offline-db';
const DB_VERSION = 1;

// Assets to pre-cache on install
const SHELL_ASSETS = [
  '/pos/',
  '/pos/static/css/pos.css',
  '/pos/static/js/pos.js',
  '/pos/offline',
];

// API endpoints that support offline queuing
const QUEUEABLE_ENDPOINTS = [
  '/pos/api/transactions',
  '/pos/api/sales',
];

// ── Install: pre-cache shell ───────────────────────────────────────────────

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_ASSETS.filter(async (url) => {
        try { await fetch(url, { method: 'HEAD' }); return true; }
        catch { return false; }
      })))
      .then(() => self.skipWaiting())
  );
});

// ── Activate: clean stale caches ─────────────────────────────────────────────

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      ))
      .then(() => self.clients.claim())
  );
});

// ── Fetch: cache-first for shell, network-first with offline queue for API ──

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // API requests: try network, queue on failure
  if (QUEUEABLE_ENDPOINTS.some((ep) => url.pathname.startsWith(ep))) {
    if (event.request.method === 'POST') {
      event.respondWith(networkOrQueue(event.request));
      return;
    }
  }

  // Shell assets: cache-first
  event.respondWith(
    caches.match(event.request)
      .then((cached) => cached || fetch(event.request)
        .then((resp) => {
          // Cache successful GET responses for shell assets
          if (resp.ok && event.request.method === 'GET') {
            const clone = resp.clone();
            caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
          }
          return resp;
        })
        .catch(() => caches.match('/pos/offline'))
      )
  );
});

// ── Background Sync: replay queued transactions ───────────────────────────────

self.addEventListener('sync', (event) => {
  if (event.tag === 'apg-pos-sync') {
    event.waitUntil(syncQueuedTransactions());
  }
});

// ── Helpers ───────────────────────────────────────────────────────────────────

async function networkOrQueue(request) {
  try {
    const response = await fetch(request.clone());
    return response;
  } catch (err) {
    // Network failed — queue the transaction for later sync
    const body = await request.clone().json().catch(() => ({}));
    await enqueueTransaction({
      url: request.url,
      method: request.method,
      headers: Object.fromEntries(request.headers.entries()),
      body,
      queuedAt: new Date().toISOString(),
    });
    // Return a synthetic "queued" response so the UI can proceed
    return new Response(
      JSON.stringify({ status: 'queued', message: 'Transaction queued for sync' }),
      { status: 202, headers: { 'Content-Type': 'application/json' } }
    );
  }
}

async function enqueueTransaction(tx) {
  const db = await openDB();
  const store = db.transaction(OFFLINE_TRANSACTION_STORE, 'readwrite')
    .objectStore(OFFLINE_TRANSACTION_STORE);
  store.add({ ...tx, id: `tx-${Date.now()}-${Math.random().toString(36).slice(2)}` });
}

async function syncQueuedTransactions() {
  const db = await openDB();
  const txns = await getAllFromStore(db, OFFLINE_TRANSACTION_STORE);

  for (const tx of txns) {
    try {
      const response = await fetch(tx.url, {
        method: tx.method,
        headers: tx.headers,
        body: JSON.stringify(tx.body),
      });
      if (response.ok) {
        // Remove from queue on success
        const store = db.transaction(OFFLINE_TRANSACTION_STORE, 'readwrite')
          .objectStore(OFFLINE_TRANSACTION_STORE);
        store.delete(tx.id);
        // Notify all POS clients that a transaction was synced
        const clients = await self.clients.matchAll();
        clients.forEach((c) => c.postMessage({ type: 'TX_SYNCED', tx }));
      }
    } catch (err) {
      // Will retry on next sync event
      console.warn('APG POS sync failed for tx:', tx.id, err);
    }
  }
}

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(OFFLINE_TRANSACTION_STORE)) {
        db.createObjectStore(OFFLINE_TRANSACTION_STORE, { keyPath: 'id' });
      }
    };
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = (e) => reject(e.target.error);
  });
}

function getAllFromStore(db, storeName) {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const req = tx.objectStore(storeName).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
