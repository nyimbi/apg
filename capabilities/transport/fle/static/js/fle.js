/**
 * APG Fleet Management — Frontend JS
 * Zero-dependency vanilla JS. Handles:
 *   - Auto-refresh dashboard KPIs every 60s
 *   - Confirm dialogs for destructive actions
 *   - Inline status updates via fetch API
 *   - Keyboard shortcuts
 *   - Relative time display
 */

'use strict';

// ── Config ──────────────────────────────────────────────────────
const FLE = {
  apiBase: '/api/fle/v1',
  tenantId: document.cookie.match(/tenant_id=([^;]+)/)?.[1] || 'default',
  actorId:  document.cookie.match(/user_id=([^;]+)/)?.[1]  || 'ui',
  refreshInterval: 60_000,
};

// ── Helpers ─────────────────────────────────────────────────────

function apiHeaders() {
  return {
    'Content-Type': 'application/json',
    'X-Tenant-ID': FLE.tenantId,
    'X-Actor-ID':  FLE.actorId,
  };
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(FLE.apiBase + path, {
    headers: apiHeaders(),
    ...opts,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(body.error || res.statusText);
  }
  return res.json();
}

function showToast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `fle-toast fle-toast-${type}`;
  el.textContent = msg;
  el.style.cssText = `
    position:fixed;bottom:1.5rem;right:1.5rem;z-index:9999;
    padding:.75rem 1.25rem;border-radius:8px;font-size:.88rem;font-weight:600;
    max-width:340px;box-shadow:0 4px 12px rgba(0,0,0,.2);
    background:${type === 'success' ? '#16a34a' : type === 'danger' ? '#dc2626' : '#1e40af'};
    color:#fff;opacity:0;transition:opacity .2s;
  `;
  document.body.appendChild(el);
  requestAnimationFrame(() => { el.style.opacity = '1'; });
  setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 220); }, 3500);
}

// ── Dashboard auto-refresh ───────────────────────────────────────

function initDashboardRefresh() {
  if (!document.querySelector('.fle-kpi-grid')) return;

  async function refresh() {
    try {
      const data = await apiFetch('/dashboard');
      const updates = {
        '[data-kpi="total_vehicles"]':       data.total_vehicles,
        '[data-kpi="active_vehicles"]':      data.active_vehicles,
        '[data-kpi="trips_in_progress"]':    data.trips_in_progress,
        '[data-kpi="drivers_on_duty"]':      data.drivers_on_duty,
        '[data-kpi="compliance_alerts"]':    data.compliance_alerts,
        '[data-kpi="fleet_utilisation"]':    data.fleet_utilisation_pct?.toFixed(1) + '%',
      };
      for (const [sel, val] of Object.entries(updates)) {
        const el = document.querySelector(sel);
        if (el) el.textContent = val;
      }
    } catch (e) {
      console.debug('[FLE] KPI refresh failed:', e.message);
    }
  }

  setInterval(refresh, FLE.refreshInterval);
}

// ── Confirm destructive actions ──────────────────────────────────

function initConfirmActions() {
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-confirm]');
    if (!btn) return;
    const msg = btn.dataset.confirm || 'Are you sure?';
    if (!confirm(msg)) e.preventDefault();
  });
}

// ── Inline trip actions (dispatch / cancel / breakdown) ──────────

function initTripActions() {
  document.addEventListener('submit', async (e) => {
    const form = e.target;
    if (!form.matches('[data-trip-action]')) return;
    e.preventDefault();

    const action = form.dataset.tripAction;
    const tripId = form.dataset.tripId;
    const body   = form.dataset.body ? JSON.parse(form.dataset.body) : {};

    try {
      await apiFetch(`/trips/${tripId}/${action}`, {
        method: 'POST',
        body:   JSON.stringify(body),
      });
      showToast(`Trip ${action} successful`, 'success');
      setTimeout(() => location.reload(), 800);
    } catch (err) {
      showToast(`Error: ${err.message}`, 'danger');
    }
  });
}

// ── Relative time display ────────────────────────────────────────

function initRelativeTimes() {
  const els = document.querySelectorAll('[data-utc]');
  if (!els.length) return;

  function rtf(date) {
    const secs = Math.round((Date.now() - date.getTime()) / 1000);
    if (secs < 60)  return `${secs}s ago`;
    if (secs < 3600) return `${Math.floor(secs/60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs/3600)}h ago`;
    return `${Math.floor(secs/86400)}d ago`;
  }

  function update() {
    els.forEach(el => {
      const d = new Date(el.dataset.utc + 'Z');
      if (!isNaN(d)) el.textContent = rtf(d);
    });
  }

  update();
  setInterval(update, 30_000);
}

// ── Keyboard shortcuts ───────────────────────────────────────────

function initKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    switch (e.key) {
      case 'd': window.location.href = '/fle/';           break;
      case 'v': window.location.href = '/fle/vehicles';   break;
      case 'r': window.location.href = '/fle/drivers';    break;
      case 't': window.location.href = '/fle/trips';      break;
      case 'm': window.location.href = '/fle/maintenance'; break;
      case 'c': window.location.href = '/fle/compliance';  break;
    }
  });
}

// ── Odometer validation ──────────────────────────────────────────

function initOdometerValidation() {
  const odoInput = document.getElementById('odometer_km');
  if (!odoInput) return;

  odoInput.addEventListener('blur', () => {
    const val = parseFloat(odoInput.value);
    if (isNaN(val) || val < 0) {
      odoInput.style.borderColor = 'var(--fle-danger)';
    } else {
      odoInput.style.borderColor = '';
    }
  });
}

// ── Boot ─────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initDashboardRefresh();
  initConfirmActions();
  initTripActions();
  initRelativeTimes();
  initKeyboardShortcuts();
  initOdometerValidation();
  console.debug('[FLE] Fleet Management UI initialised — tenant:', FLE.tenantId);
});
