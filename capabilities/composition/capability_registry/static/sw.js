/**
 * APG Capability Registry Service Worker
 * 
 * Provides offline capabilities, background sync, and caching for the mobile app.
 * 
 * © 2025 Datacraft. All rights reserved.
 */

const CACHE_NAME = 'apg-registry-v1.0.0';
const OFFLINE_CACHE = 'apg-registry-offline';
const API_CACHE = 'apg-registry-api';

// Files to cache immediately
const STATIC_CACHE_FILES = [
	'/capability-registry/mobile',
	'/capability-registry/static/manifest.json',
	'/static/css/bootstrap.min.css',
	'/static/js/jquery.min.js',
	'/static/js/bootstrap.min.js',
	'/static/fonts/fontawesome-webfont.woff2',
	'/capability-registry/static/icons/icon-192x192.png',
	'/capability-registry/static/icons/icon-512x512.png'
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
	'/capability-registry/api/capabilities',
	'/capability-registry/api/compositions',
	'/capability-registry/api/dashboard'
];

// Runtime cache patterns
const RUNTIME_CACHE_PATTERNS = [
	/^https:\/\/cdn\.jsdelivr\.net/,
	/^https:\/\/fonts\.googleapis\.com/,
	/^https:\/\/fonts\.gstatic\.com/
];

// =============================================================================
// Service Worker Installation
// =============================================================================

self.addEventListener('install', event => {
	console.log('[SW] Installing service worker...');
	
	event.waitUntil(
		Promise.all([
			// Cache static files
			caches.open(CACHE_NAME).then(cache => {
				console.log('[SW] Caching static files');
				return cache.addAll(STATIC_CACHE_FILES);
			}),
			
			// Initialize offline cache
			caches.open(OFFLINE_CACHE).then(cache => {
				console.log('[SW] Initializing offline cache');
				return cache.put('/offline', new Response(
					getOfflinePage(),
					{ headers: { 'Content-Type': 'text/html' } }
				));
			}),
			
			// Skip waiting to activate immediately
			self.skipWaiting()
		])
	);
});

// =============================================================================
// Service Worker Activation
// =============================================================================

self.addEventListener('activate', event => {
	console.log('[SW] Activating service worker...');
	
	event.waitUntil(
		Promise.all([
			// Clean up old caches
			caches.keys().then(cacheNames => {
				return Promise.all(
					cacheNames.map(cacheName => {
						if (cacheName !== CACHE_NAME && 
							cacheName !== OFFLINE_CACHE && 
							cacheName !== API_CACHE) {
							console.log('[SW] Deleting old cache:', cacheName);
							return caches.delete(cacheName);
						}
					})
				);
			}),
			
			// Take control of all clients
			self.clients.claim()
		])
	);
});

// =============================================================================
// Fetch Event Handling
// =============================================================================

self.addEventListener('fetch', event => {
	const { request } = event;
	const url = new URL(request.url);
	
	// Skip non-GET requests
	if (request.method !== 'GET') {
		return;
	}
	
	// Handle different request types
	if (isAPIRequest(url)) {
		event.respondWith(handleAPIRequest(request));
	} else if (isStaticAsset(url)) {
		event.respondWith(handleStaticAsset(request));
	} else if (isHTMLRequest(request)) {
		event.respondWith(handleHTMLRequest(request));
	} else if (isRuntimeCacheable(url)) {
		event.respondWith(handleRuntimeCache(request));
	}
});

// =============================================================================
// Request Handlers
// =============================================================================

async function handleAPIRequest(request) {
	const url = new URL(request.url);
	const cacheKey = `${url.pathname}${url.search}`;
	
	try {
		// Try network first for API requests
		const networkResponse = await fetch(request);
		
		if (networkResponse.ok) {
			// Cache successful API responses
			const cache = await caches.open(API_CACHE);
			await cache.put(cacheKey, networkResponse.clone());
			return networkResponse;
		}
		
		throw new Error(`API request failed: ${networkResponse.status}`);
		
	} catch (error) {
		console.log('[SW] API request failed, trying cache:', error.message);
		
		// Fallback to cache
		const cache = await caches.open(API_CACHE);
		const cachedResponse = await cache.match(cacheKey);
		
		if (cachedResponse) {
			// Add offline indicator header
			const response = cachedResponse.clone();
			response.headers.set('X-Served-By', 'service-worker-cache');
			return response;
		}
		
		// Return offline data if available
		return getOfflineAPIResponse(url.pathname);
	}
}

async function handleStaticAsset(request) {
	// Cache first strategy for static assets
	const cache = await caches.open(CACHE_NAME);
	const cachedResponse = await cache.match(request);
	
	if (cachedResponse) {
		return cachedResponse;
	}
	
	try {
		const networkResponse = await fetch(request);
		if (networkResponse.ok) {
			await cache.put(request, networkResponse.clone());
		}
		return networkResponse;
	} catch (error) {
		console.log('[SW] Static asset fetch failed:', error.message);
		return new Response('Asset not available offline', { status: 404 });
	}
}

async function handleHTMLRequest(request) {
	try {
		// Network first for HTML requests
		const networkResponse = await fetch(request);
		return networkResponse;
	} catch (error) {
		console.log('[SW] HTML request failed, serving offline page:', error.message);
		
		// Serve cached version or offline page
		const cache = await caches.open(OFFLINE_CACHE);
		const offlineResponse = await cache.match('/offline');
		return offlineResponse || new Response('Offline', { status: 503 });
	}
}

async function handleRuntimeCache(request) {
	const cache = await caches.open(CACHE_NAME);
	
	try {
		const networkResponse = await fetch(request);
		if (networkResponse.ok) {
			await cache.put(request, networkResponse.clone());
		}
		return networkResponse;
	} catch (error) {
		const cachedResponse = await cache.match(request);
		return cachedResponse || new Response('Resource not available', { status: 404 });
	}
}

// =============================================================================
// Background Sync
// =============================================================================

self.addEventListener('sync', event => {
	console.log('[SW] Background sync triggered:', event.tag);
	
	if (event.tag === 'capability-registry-sync') {
		event.waitUntil(syncOfflineActions());
	} else if (event.tag === 'capability-registry-data-sync') {
		event.waitUntil(syncCachedData());
	}
});

async function syncOfflineActions() {
	console.log('[SW] Syncing offline actions...');
	
	try {
		// Get offline actions from IndexedDB
		const offlineActions = await getOfflineActions();
		
		for (const action of offlineActions) {
			try {
				await processOfflineAction(action);
				await removeOfflineAction(action.id);
				console.log('[SW] Synced offline action:', action.type);
			} catch (error) {
				console.error('[SW] Failed to sync action:', action.type, error);
			}
		}
		
		// Notify clients about sync completion
		await notifyClients('sync-completed', { 
			synced: offlineActions.length 
		});
		
	} catch (error) {
		console.error('[SW] Background sync failed:', error);
	}
}

async function syncCachedData() {
	console.log('[SW] Syncing cached data...');
	
	try {
		// Update cached API data
		const apiEndpoints = [
			'/capability-registry/api/capabilities/mobile',
			'/capability-registry/api/compositions/mobile',
			'/capability-registry/api/dashboard/mobile'
		];
		
		const cache = await caches.open(API_CACHE);
		
		for (const endpoint of apiEndpoints) {
			try {
				const response = await fetch(endpoint);
				if (response.ok) {
					await cache.put(endpoint, response.clone());
					console.log('[SW] Updated cache for:', endpoint);
				}
			} catch (error) {
				console.error('[SW] Failed to update cache for:', endpoint, error);
			}
		}
		
	} catch (error) {
		console.error('[SW] Data sync failed:', error);
	}
}

// =============================================================================
// Push Notifications
// =============================================================================

self.addEventListener('push', event => {
	console.log('[SW] Push notification received:', event);
	
	const options = {
		body: 'APG Registry has been updated',
		icon: '/capability-registry/static/icons/icon-192x192.png',
		badge: '/capability-registry/static/icons/badge-72x72.png',
		vibrate: [100, 50, 100],
		data: {
			dateOfArrival: Date.now(),
			primaryKey: 1
		},
		actions: [
			{
				action: 'open',
				title: 'Open App',
				icon: '/capability-registry/static/icons/action-open.png'
			},
			{
				action: 'close',
				title: 'Close',
				icon: '/capability-registry/static/icons/action-close.png'
			}
		]
	};
	
	if (event.data) {
		const pushData = event.data.json();
		options.body = pushData.body || options.body;
		options.title = pushData.title || 'APG Registry';
	}
	
	event.waitUntil(
		self.registration.showNotification('APG Registry', options)
	);
});

self.addEventListener('notificationclick', event => {
	console.log('[SW] Notification click:', event);
	
	event.notification.close();
	
	if (event.action === 'open') {
		event.waitUntil(
			clients.openWindow('/capability-registry/mobile')
		);
	}
});

// =============================================================================
// Helper Functions
// =============================================================================

function isAPIRequest(url) {
	return url.pathname.startsWith('/capability-registry/api/');
}

function isStaticAsset(url) {
	return url.pathname.includes('/static/') ||
		   url.pathname.endsWith('.css') ||
		   url.pathname.endsWith('.js') ||
		   url.pathname.endsWith('.png') ||
		   url.pathname.endsWith('.jpg') ||
		   url.pathname.endsWith('.svg') ||
		   url.pathname.endsWith('.woff') ||
		   url.pathname.endsWith('.woff2');
}

function isHTMLRequest(request) {
	return request.headers.get('accept')?.includes('text/html');
}

function isRuntimeCacheable(url) {
	return RUNTIME_CACHE_PATTERNS.some(pattern => pattern.test(url.href));
}

function getOfflinePage() {
	return `
<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>APG Registry - Offline</title>
	<style>
		body {
			font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
			margin: 0;
			padding: 40px 20px;
			text-align: center;
			background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
			color: white;
			min-height: 100vh;
			display: flex;
			flex-direction: column;
			justify-content: center;
			align-items: center;
		}
		.offline-icon {
			font-size: 4rem;
			margin-bottom: 1rem;
			opacity: 0.8;
		}
		.offline-title {
			font-size: 1.5rem;
			margin-bottom: 0.5rem;
			font-weight: 600;
		}
		.offline-message {
			font-size: 1rem;
			opacity: 0.9;
			margin-bottom: 2rem;
			max-width: 400px;
		}
		.retry-button {
			background: rgba(255,255,255,0.2);
			border: 2px solid rgba(255,255,255,0.3);
			color: white;
			padding: 12px 24px;
			border-radius: 25px;
			font-size: 1rem;
			font-weight: 500;
			cursor: pointer;
			transition: all 0.3s;
		}
		.retry-button:hover {
			background: rgba(255,255,255,0.3);
			border-color: rgba(255,255,255,0.5);
		}
	</style>
</head>
<body>
	<div class="offline-icon">📱</div>
	<h1 class="offline-title">You're Offline</h1>
	<p class="offline-message">
		APG Registry is currently unavailable. 
		Check your internet connection and try again.
	</p>
	<button class="retry-button" onclick="window.location.reload()">
		Try Again
	</button>
</body>
</html>
	`;
}

function getOfflineAPIResponse(pathname) {
	const offlineData = {
		'/capability-registry/api/capabilities': {
			capabilities: [],
			message: 'Offline mode - limited capabilities available'
		},
		'/capability-registry/api/compositions': {
			compositions: [],
			message: 'Offline mode - no compositions available'
		},
		'/capability-registry/api/dashboard': {
			capabilities_count: 0,
			compositions_count: 0,
			is_offline: true,
			message: 'Offline mode'
		}
	};
	
	const data = offlineData[pathname] || { error: 'Not available offline' };
	
	return new Response(JSON.stringify(data), {
		headers: {
			'Content-Type': 'application/json',
			'X-Served-By': 'service-worker-offline'
		}
	});
}

async function getOfflineActions() {
	// This would typically use IndexedDB
	// For now, return empty array
	return [];
}

async function processOfflineAction(action) {
	// Process different types of offline actions
	switch (action.type) {
		case 'create_composition':
			return fetch('/capability-registry/api/compositions', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(action.data)
			});
		
		case 'update_capability':
			return fetch(`/capability-registry/api/capabilities/${action.id}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(action.data)
			});
		
		default:
			throw new Error(`Unknown action type: ${action.type}`);
	}
}

async function removeOfflineAction(actionId) {
	// Remove from IndexedDB
	console.log('[SW] Removing offline action:', actionId);
}

async function notifyClients(type, data) {
	const clients = await self.clients.matchAll();
	
	clients.forEach(client => {
		client.postMessage({
			type,
			data,
			timestamp: Date.now()
		});
	});
}

// =============================================================================
// Debug and Monitoring
// =============================================================================

self.addEventListener('error', event => {
	console.error('[SW] Service Worker error:', event.error);
});

self.addEventListener('unhandledrejection', event => {
	console.error('[SW] Unhandled promise rejection:', event.reason);
});

// Log service worker status
console.log('[SW] APG Registry Service Worker loaded');