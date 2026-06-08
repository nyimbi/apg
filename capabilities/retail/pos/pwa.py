"""PWA (Progressive Web App) integration for APG Retail POS.

Registers the service worker, serves the offline fallback page, and exposes
the PWA manifest. Import and register this blueprint alongside the main
retail_pos blueprint.

Phase 5A: Offline-first POS capability.
  - Pre-caches shell assets on install
  - Queues transactions to IndexedDB when network is unavailable
  - Background-syncs on reconnect via the Background Sync API
  - Sends a synthetic 202 response to unblock the cashier UI during offline

PCI DSS note: The service worker NEVER caches actual card data. Only tokens
produced by TokenizationService appear in any queued payload.
"""
from __future__ import annotations

from flask import Blueprint, Response, render_template_string, send_from_directory
from pathlib import Path

_STATIC_DIR = Path(__file__).parent / "static"

pwa_blueprint = Blueprint("pos_pwa", __name__, url_prefix="/pos")


@pwa_blueprint.route("/sw.js")
def service_worker() -> Response:
    """Serve the service worker at the POS scope root.

    The service worker must be served from the same scope it controls,
    so it lives at /pos/sw.js (not under /static/).
    """
    sw_path = _STATIC_DIR / "sw.js"
    if not sw_path.exists():
        return Response("// Service worker not found", status=404, content_type="application/javascript")

    return Response(
        sw_path.read_text(),
        status=200,
        content_type="application/javascript",
        headers={
            "Service-Worker-Allowed": "/pos/",
            "Cache-Control": "no-cache",
        },
    )


@pwa_blueprint.route("/manifest.json")
def manifest() -> Response:
    """Serve the PWA web app manifest."""
    manifest_path = _STATIC_DIR / "pwa-manifest.json"
    if not manifest_path.exists():
        return Response("{}", status=404, content_type="application/json")
    return Response(
        manifest_path.read_text(),
        status=200,
        content_type="application/manifest+json",
        headers={"Cache-Control": "max-age=86400"},
    )


@pwa_blueprint.route("/offline")
def offline_page() -> str:
    """Offline fallback page shown when network is unavailable."""
    return render_template_string(_OFFLINE_HTML)


@pwa_blueprint.route("/offline-status")
def offline_status() -> Response:
    """API endpoint to check connectivity status (cached as offline indicator)."""
    return Response(
        '{"online": true}',
        status=200,
        content_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


_OFFLINE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APG POS — Offline Mode</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: system-ui, -apple-system, sans-serif; background: #1B5E20; color: #fff; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 24px; }
        .card { background: rgba(255,255,255,0.12); border-radius: 12px; padding: 40px; max-width: 480px; width: 100%; text-align: center; }
        .icon { font-size: 64px; margin-bottom: 24px; }
        h1 { font-size: 28px; margin-bottom: 12px; }
        p { font-size: 16px; opacity: 0.85; margin-bottom: 24px; line-height: 1.5; }
        .badge { display: inline-block; background: #F57F17; color: #fff; border-radius: 20px; padding: 4px 16px; font-size: 14px; font-weight: 600; margin-bottom: 24px; }
        .queue-count { font-size: 48px; font-weight: 700; color: #A5D6A7; margin: 16px 0; }
        .status { font-size: 13px; opacity: 0.6; }
        button { background: #fff; color: #1B5E20; border: none; border-radius: 8px; padding: 14px 32px; font-size: 16px; font-weight: 600; cursor: pointer; margin-top: 8px; }
        button:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="card">
        <div class="icon">📶</div>
        <div class="badge">OFFLINE MODE</div>
        <h1>APG POS is Offline</h1>
        <p>You're working without internet connectivity. Transactions are being saved locally and will sync automatically when the connection is restored.</p>
        <div class="queue-count" id="queue-count">0</div>
        <p class="status">transactions queued for sync</p>
        <button onclick="window.location.href='/pos/'">Return to POS</button>
    </div>
    <script>
        // Count queued transactions from IndexedDB
        (async function() {
            try {
                const db = await new Promise((resolve, reject) => {
                    const req = indexedDB.open('apg-pos-offline-db', 1);
                    req.onsuccess = e => resolve(e.target.result);
                    req.onerror = e => reject(e);
                });
                const store = db.transaction('offline-transactions', 'readonly').objectStore('offline-transactions');
                const req = store.count();
                req.onsuccess = () => {
                    document.getElementById('queue-count').textContent = req.result;
                };
            } catch(e) { /* IndexedDB not available */ }
        })();

        // Listen for service worker sync events
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.addEventListener('message', (e) => {
                if (e.data.type === 'TX_SYNCED') {
                    const el = document.getElementById('queue-count');
                    const n = Math.max(0, (parseInt(el.textContent) || 0) - 1);
                    el.textContent = n;
                    if (n === 0) window.location.href = '/pos/';
                }
            });
        }
    </script>
</body>
</html>"""


def register_pwa(app: object) -> None:
    """Register the PWA blueprint and inject service worker registration.

    Call after creating the Flask app. The SW registration script is injected
    into every HTML response via an after_request hook.
    """
    import flask
    assert isinstance(app, flask.Flask)
    app.register_blueprint(pwa_blueprint)

    _SW_SNIPPET = """<script>
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/pos/sw.js', { scope: '/pos/' })
    .then(reg => {
      reg.addEventListener('updatefound', () => {
        const worker = reg.installing;
        worker.addEventListener('statechange', () => {
          if (worker.state === 'installed' && navigator.serviceWorker.controller) {
            console.log('APG POS: new service worker ready');
          }
        });
      });
    });
}
</script>"""

    @app.after_request
    def inject_sw_registration(response: flask.Response) -> flask.Response:
        if (response.content_type.startswith("text/html")
                and b"</body>" in (response.data or b"")):
            response.data = response.data.replace(
                b"</body>", _SW_SNIPPET.encode() + b"</body>"
            )
        return response
