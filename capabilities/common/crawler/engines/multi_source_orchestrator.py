"""
APG Crawler Capability - Multi-Source Orchestration Framework
=============================================================

Advanced multi-source orchestration inspired by:
- CloudScraper: Anti-bot detection and Cloudflare bypass
- Crawlee: Request queue management and retry logic  
- Crawl4AI: AI-powered content extraction and smart crawling

Key Features:
- Intelligent request routing and queuing
- Multi-strategy anti-detection (CloudScraper, Playwright, Selenium)
- Smart retry logic with exponential backoff
- Rate limiting and respectful crawling
- Session management and cookie persistence
- Proxy rotation and geographic distribution
- AI-powered content extraction and cleaning

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from urllib.parse import urljoin, urlparse, urlencode
import hashlib
from pathlib import Path

# HTTP and crawling libraries
import httpx
import aiohttp
import cloudscraper
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PlaywrightTimeoutError
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium_stealth import stealth
import undetected_chromedriver as uc

# Smart crawling and AI
from fake_useragent import UserAgent
import trafilatura
from readability import Document

# APG imports
from .stealth_engine import CrawlRequest, CrawlResult, StealthMethod, ProtectionDetectionEngine
from .content_pipeline import ContentProcessingPipeline, ContentExtractionResult
from .content_intelligence import ContentIntelligenceEngine, ContentIntelligenceResult
from ..views import CrawlTarget, ContentCleaningConfig, RAGProcessingConfig

# =====================================================
# REQUEST QUEUE AND SESSION MANAGEMENT
# =====================================================

logger = logging.getLogger(__name__)

class RequestPriority(int, Enum):
	"""Request priority levels (inspired by Crawlee)"""
	HIGHEST = 1
	HIGH = 2  
	NORMAL = 3
	LOW = 4
	LOWEST = 5

class RequestStatus(str, Enum):
	"""Request processing status"""
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	RETRYING = "retrying"
	SKIPPED = "skipped"

@dataclass
class QueuedRequest:
	"""Request with queue metadata (inspired by Crawlee's Request class)"""
	url: str
	method: str = "GET"
	headers: Optional[Dict[str, str]] = None
	data: Optional[Dict[str, Any]] = None
	priority: RequestPriority = RequestPriority.NORMAL
	retries: int = 0
	max_retries: int = 3
	delay: float = 0.0
	timeout: int = 30
	
	# Crawlee-inspired features
	unique_key: Optional[str] = None
	user_data: Dict[str, Any] = field(default_factory=dict)
	labels: Set[str] = field(default_factory=set)
	no_retry: bool = False
	keep_alive: bool = True
	
	# APG-specific features
	tenant_id: Optional[str] = None
	target_id: Optional[str] = None
	stealth_method: Optional[StealthMethod] = None
	javascript_required: bool = False
	extract_links: bool = True
	
	# Status tracking
	status: RequestStatus = RequestStatus.PENDING
	created_at: datetime = field(default_factory=datetime.utcnow)
	started_at: Optional[datetime] = None
	completed_at: Optional[datetime] = None
	error_messages: List[str] = field(default_factory=list)
	
	def __post_init__(self):
		if not self.unique_key:
			# Generate unique key from URL and method (like Crawlee)
			key_string = f"{self.method}:{self.url}"
			if self.data:
				key_string += f":{json.dumps(self.data, sort_keys=True)}"
			self.unique_key = hashlib.sha256(key_string.encode()).hexdigest()[:16]

class RequestQueue:
	"""Async request queue with priority and deduplication (Crawlee-inspired)"""
	
	def __init__(self, max_size: int = 10000):
		self.max_size = max_size
		self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
		self._processed: Set[str] = set()
		self._in_progress: Dict[str, QueuedRequest] = {}
		self._stats = {
			'added': 0,
			'processed': 0,
			'failed': 0,
			'duplicates': 0
		}
		self._lock = asyncio.Lock()
	
	async def add_request(self, request: QueuedRequest) -> bool:
		"""Add request to queue with deduplication"""
		async with self._lock:
			# Check for duplicates
			if request.unique_key in self._processed or request.unique_key in self._in_progress:
				self._stats['duplicates'] += 1
				logger.debug(f"Duplicate request skipped: {request.url}")
				return False
			
			# Add to queue with priority
			await self._queue.put((request.priority.value, time.time(), request))
			self._stats['added'] += 1
			logger.debug(f"Request added to queue: {request.url} (priority: {request.priority.name})")
			return True
	
	async def get_request(self) -> Optional[QueuedRequest]:
		"""Get next request from queue"""
		try:
			_, _, request = await asyncio.wait_for(self._queue.get(), timeout=1.0)
			
			async with self._lock:
				self._in_progress[request.unique_key] = request
				request.status = RequestStatus.PROCESSING
				request.started_at = datetime.utcnow()
			
			return request
		except asyncio.TimeoutError:
			return None
	
	async def mark_completed(self, request: QueuedRequest, success: bool = True):
		"""Mark request as completed"""
		async with self._lock:
			if request.unique_key in self._in_progress:
				del self._in_progress[request.unique_key]
			
			self._processed.add(request.unique_key)
			request.completed_at = datetime.utcnow()
			
			if success:
				request.status = RequestStatus.COMPLETED
				self._stats['processed'] += 1
			else:
				request.status = RequestStatus.FAILED
				self._stats['failed'] += 1
			
			self._queue.task_done()
	
	async def requeue_request(self, request: QueuedRequest) -> bool:
		"""Requeue failed request with backoff"""
		if request.retries >= request.max_retries or request.no_retry:
			await self.mark_completed(request, success=False)
			return False
		
		# Exponential backoff
		request.retries += 1
		request.delay = min(2 ** request.retries + random.uniform(0, 1), 300)  # Max 5 min
		request.status = RequestStatus.RETRYING
		
		async with self._lock:
			if request.unique_key in self._in_progress:
				del self._in_progress[request.unique_key]
		
		# Re-add to queue with delay
		await asyncio.sleep(request.delay)
		await self._queue.put((request.priority.value, time.time(), request))
		return True
	
	@property
	def size(self) -> int:
		return self._queue.qsize()
	
	@property
	def is_empty(self) -> bool:
		return self._queue.empty() and not self._in_progress
	
	@property
	def stats(self) -> Dict[str, int]:
		return self._stats.copy()


# =====================================================
# SESSION MANAGEMENT (CloudScraper + Crawlee inspired)
# =====================================================

class SessionPool:
	"""Manages multiple HTTP sessions with different strategies"""
	
	def __init__(self, max_sessions: int = 10):
		self.max_sessions = max_sessions
		self._sessions: Dict[str, Any] = {}
		self._session_stats: Dict[str, Dict[str, Any]] = {}
		self._lock = asyncio.Lock()
		self.user_agent = UserAgent()
	
	async def get_session(self, strategy: StealthMethod, **kwargs) -> Any:
		"""Get or create session for strategy"""
		session_key = f"{strategy.value}_{hash(frozenset(kwargs.items()))}"
		
		async with self._lock:
			if session_key not in self._sessions:
				if len(self._sessions) >= self.max_sessions:
					# Remove oldest session
					oldest_key = min(
						self._session_stats.keys(),
						key=lambda k: self._session_stats[k]['last_used']
					)
					await self._close_session(oldest_key)
				
				# Create new session
				self._sessions[session_key] = await self._create_session(strategy, **kwargs)
				self._session_stats[session_key] = {
					'created_at': time.time(),
					'last_used': time.time(),
					'requests_count': 0,
					'success_count': 0,
					'strategy': strategy
				}
			
			# Update usage stats
			self._session_stats[session_key]['last_used'] = time.time()
			self._session_stats[session_key]['requests_count'] += 1
			
			return self._sessions[session_key]
	
	async def _create_session(self, strategy: StealthMethod, **kwargs) -> Any:
		"""Create session based on strategy"""
		if strategy == StealthMethod.CLOUDSCRAPER:
			return cloudscraper.create_scraper(
				browser={
					'browser': 'chrome',
					'platform': 'win32',
					'mobile': False
				},
				**kwargs
			)
		
		elif strategy == StealthMethod.HTTP_MIMICRY:
			headers = {
				'User-Agent': self.user_agent.random,
				'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
				'Accept-Language': 'en-US,en;q=0.5',
				'Accept-Encoding': 'gzip, deflate, br',
				'Connection': 'keep-alive',
				'Upgrade-Insecure-Requests': '1',
				'Sec-Fetch-Dest': 'document',
				'Sec-Fetch-Mode': 'navigate',
				'Sec-Fetch-Site': 'none',
			}
			
			# Create connector with connection pooling
			connector = aiohttp.TCPConnector(
				limit=100,
				limit_per_host=10,
				ttl_dns_cache=300,
				use_dns_cache=True,
			)
			
			return aiohttp.ClientSession(
				headers=headers,
				connector=connector,
				timeout=aiohttp.ClientTimeout(total=30),
				**kwargs
			)
		
		elif strategy == StealthMethod.PLAYWRIGHT:
			playwright = await async_playwright().start()
			browser = await playwright.chromium.launch(
				headless=True,
				args=[
					'--no-sandbox',
					'--disable-blink-features=AutomationControlled',
					'--disable-dev-shm-usage',
					'--disable-gpu',
					'--no-first-run',
					'--disable-extensions'
				]
			)
			
			context = await browser.new_context(
				viewport={'width': 1920, 'height': 1080},
				user_agent=self.user_agent.random,
				extra_http_headers={
					'Accept-Language': 'en-US,en;q=0.9',
				}
			)
			
			# Anti-detection scripts (crawl4ai inspired)
			await context.add_init_script("""
				// Remove webdriver property
				Object.defineProperty(navigator, 'webdriver', {
					get: () => undefined,
				});
				
				// Mock plugins
				Object.defineProperty(navigator, 'plugins', {
					get: () => [1, 2, 3, 4, 5],
				});
				
				// Mock languages
				Object.defineProperty(navigator, 'languages', {
					get: () => ['en-US', 'en'],
				});
				
				// Mock chrome object
				window.chrome = {
					runtime: {},
				};
				
				// Override permissions
				const originalQuery = window.navigator.permissions.query;
				window.navigator.permissions.query = (parameters) => (
					parameters.name === 'notifications' ?
						Promise.resolve({ state: Notification.permission }) :
						originalQuery(parameters)
				);
			""")
			
			return {
				'playwright': playwright,
				'browser': browser,
				'context': context
			}
		
		else:
			# Default to httpx
			return httpx.AsyncClient(
				headers={'User-Agent': self.user_agent.random},
				timeout=30.0,
				**kwargs
			)
	
	async def _close_session(self, session_key: str):
		"""Close and cleanup session"""
		if session_key in self._sessions:
			session = self._sessions[session_key]
			stats = self._session_stats[session_key]
			
			try:
				if stats['strategy'] == StealthMethod.PLAYWRIGHT:
					if 'context' in session:
						await session['context'].close()
					if 'browser' in session:
						await session['browser'].close()
					if 'playwright' in session:
						await session['playwright'].stop()
				elif hasattr(session, 'close'):
					await session.close()
			except Exception as e:
				logger.warning(f"Error closing session {session_key}: {e}")
			
			del self._sessions[session_key]
			del self._session_stats[session_key]
	
	async def close_all(self):
		"""Close all sessions"""
		for session_key in list(self._sessions.keys()):
			await self._close_session(session_key)


# =====================================================
# SMART CRAWLER (Crawl4AI inspired)
# =====================================================

class SmartCrawler:
	"""AI-powered smart crawler with content understanding"""
	
	def __init__(self, session_pool: SessionPool):
		self.session_pool = session_pool
		self.content_pipeline = ContentProcessingPipeline()
		self.content_intelligence = ContentIntelligenceEngine()
		self.protection_detector = ProtectionDetectionEngine()
		
		# Smart crawling patterns (crawl4ai inspired)
		self.content_indicators = [
			'article', 'main', '[role="main"]', '.content', '#content',
			'.post', '.entry', '.story', '.article-body', '.text',
			'section', 'div[class*="content"]', 'div[id*="content"]'
		]
		
		# Anti-patterns for noise
		self.noise_selectors = [
			'nav', 'header', 'footer', '.sidebar', '.menu', '.navigation',
			'.ad', '.advertisement', '.social', '.share', '.comment',
			'script', 'style', 'noscript'
		]
	
	async def smart_crawl(self, request: QueuedRequest, 
						  config: ContentCleaningConfig,
						  business_context: Dict[str, Any] = None) -> Tuple[CrawlResult, ContentExtractionResult, ContentIntelligenceResult]:
		"""Perform smart crawl with AI-powered content extraction and intelligence"""
		
		# Determine optimal strategy
		strategy = await self._select_strategy(request)
		
		# Execute crawl
		crawl_result = await self._execute_crawl(request, strategy)
		
		# Process content with AI
		if crawl_result.success and crawl_result.content:
			extraction_result = await self.content_pipeline.process_crawl_result(
				crawl_result, config
			)
			
			# Apply content intelligence if extraction was successful
			if extraction_result.success:
				intelligence_result = await self.content_intelligence.analyze_content(
					extraction_result, business_context or {}
				)
			else:
				intelligence_result = None
		else:
			extraction_result = ContentExtractionResult(
				url=request.url,
				title=None,
				main_content="",
				raw_content=crawl_result.content,
				cleaned_content="",
				markdown_content="",
				content_type="text/html",
				language=None,
				publish_date=None,
				author=None,
				description=None,
				keywords=[],
				images=[],
				links=[],
				content_fingerprint="",
				processing_stage="raw_extracted",
				metadata={},
				success=False,
				error=crawl_result.error
			)
			intelligence_result = None
		
		return crawl_result, extraction_result, intelligence_result
	
	async def _select_strategy(self, request: QueuedRequest) -> StealthMethod:
		"""Select optimal crawling strategy (crawl4ai inspired logic)"""
		
		if request.stealth_method:
			return request.stealth_method
		
		# Analyze URL for strategy hints
		url_lower = request.url.lower()
		
		# JavaScript-heavy sites
		if any(js_hint in url_lower for js_hint in ['react', 'vue', 'angular', 'spa', 'app']):
			return StealthMethod.PLAYWRIGHT
		
		# Known protected sites
		if any(protected in url_lower for protected in ['cloudflare', 'recaptcha', 'captcha']):
			return StealthMethod.CLOUDSCRAPER
		
		# Social media and complex sites
		if any(social in url_lower for social in ['twitter', 'facebook', 'linkedin', 'instagram']):
			return StealthMethod.PLAYWRIGHT
		
		# News and article sites
		if any(news in url_lower for news in ['news', 'article', 'blog', 'post']):
			return StealthMethod.HTTP_MIMICRY
		
		# Default to HTTP for simple sites
		return StealthMethod.HTTP_MIMICRY
	
	async def _execute_crawl(self, request: QueuedRequest, strategy: StealthMethod) -> CrawlResult:
		"""Execute crawl with selected strategy"""
		
		if strategy == StealthMethod.CLOUDSCRAPER:
			return await self._crawl_with_cloudscraper(request)
		elif strategy == StealthMethod.PLAYWRIGHT:
			return await self._crawl_with_playwright(request)
		elif strategy == StealthMethod.HTTP_MIMICRY:
			return await self._crawl_with_http(request)
		else:
			return await self._crawl_with_http(request)  # Default fallback
	
	async def _crawl_with_cloudscraper(self, request: QueuedRequest) -> CrawlResult:
		"""Crawl using CloudScraper (anti-Cloudflare)"""
		session = await self.session_pool.get_session(StealthMethod.CLOUDSCRAPER)
		start_time = time.time()
		
		try:
			response = session.get(
				request.url,
				headers=request.headers or {},
				data=request.data,
				timeout=request.timeout
			)
			
			response_time = time.time() - start_time
			
			# Detect protections
			protections = self.protection_detector.detect_protection(
				response.text, dict(response.headers), response.status_code
			)
			
			return CrawlResult(
				url=request.url,
				status_code=response.status_code,
				content=response.text,
				headers=dict(response.headers),
				cookies=dict(response.cookies),
				final_url=response.url,
				response_time=response_time,
				method_used=StealthMethod.CLOUDSCRAPER,
				protection_detected=protections,
				success=response.status_code == 200
			)
			
		except Exception as e:
			return CrawlResult(
				url=request.url,
				status_code=0,
				content="",
				headers={},
				cookies={},
				final_url=request.url,
				response_time=time.time() - start_time,
				method_used=StealthMethod.CLOUDSCRAPER,
				protection_detected=[],
				success=False,
				error=str(e)
			)
	
	async def _crawl_with_playwright(self, request: QueuedRequest) -> CrawlResult:
		"""Crawl using Playwright (JavaScript rendering)"""
		session_data = await self.session_pool.get_session(StealthMethod.PLAYWRIGHT)
		context = session_data['context']
		start_time = time.time()
		
		try:
			page = await context.new_page()
			
			# Set extra headers if provided
			if request.headers:
				await page.set_extra_http_headers(request.headers)
			
			# Navigate with smart waiting (crawl4ai inspired)
			try:
				response = await page.goto(
					request.url,
					wait_until='domcontentloaded',
					timeout=request.timeout * 1000
				)
			except PlaywrightTimeoutError:
				# Try with different wait strategy
				response = await page.goto(
					request.url,
					wait_until='networkidle',
					timeout=(request.timeout + 10) * 1000
				)
			
			# Smart content waiting (wait for main content to appear)
			await self._wait_for_content(page)
			
			# Extract content using smart selectors
			content = await self._extract_smart_content(page)
			
			response_time = time.time() - start_time
			
			result = CrawlResult(
				url=request.url,
				status_code=response.status if response else 200,
				content=content,
				headers=dict(response.headers) if response else {},
				cookies={},  # Could extract from context
				final_url=page.url,
				response_time=response_time,
				method_used=StealthMethod.PLAYWRIGHT,
				protection_detected=[],
				success=bool(content and len(content) > 100)
			)
			
			await page.close()
			return result
			
		except Exception as e:
			return CrawlResult(
				url=request.url,
				status_code=0,
				content="",
				headers={},
				cookies={},
				final_url=request.url,
				response_time=time.time() - start_time,
				method_used=StealthMethod.PLAYWRIGHT,
				protection_detected=[],
				success=False,
				error=str(e)
			)
	
	async def _crawl_with_http(self, request: QueuedRequest) -> CrawlResult:
		"""Crawl using async HTTP client"""
		session = await self.session_pool.get_session(StealthMethod.HTTP_MIMICRY)
		start_time = time.time()
		
		try:
			# Add random delay to mimic human behavior
			await asyncio.sleep(random.uniform(0.1, 0.5))
			
			async with session.get(
				request.url,
				headers=request.headers,
				data=request.data,
				timeout=request.timeout
			) as response:
				content = await response.text()
				response_time = time.time() - start_time
				
				# Detect protections
				protections = self.protection_detector.detect_protection(
					content, dict(response.headers), response.status
				)
				
				return CrawlResult(
					url=request.url,
					status_code=response.status,
					content=content,
					headers=dict(response.headers),
					cookies={},  # aiohttp cookies handling
					final_url=str(response.url),
					response_time=response_time,
					method_used=StealthMethod.HTTP_MIMICRY,
					protection_detected=protections,
					success=response.status == 200
				)
		
		except Exception as e:
			return CrawlResult(
				url=request.url,
				status_code=0,
				content="",
				headers={},
				cookies={},
				final_url=request.url,
				response_time=time.time() - start_time,
				method_used=StealthMethod.HTTP_MIMICRY,
				protection_detected=[],
				success=False,
				error=str(e)
			)
	
	async def _wait_for_content(self, page: Page):
		"""Smart waiting for content to load (crawl4ai inspired)"""
		try:
			# Wait for any of the main content indicators
			for selector in self.content_indicators[:3]:  # Try first 3
				try:
					await page.wait_for_selector(selector, timeout=3000)
					break
				except:
					continue
			
			# Wait for network to settle
			await page.wait_for_load_state('networkidle', timeout=5000)
			
		except Exception:
			# If smart waiting fails, just wait a bit
			await asyncio.sleep(2)
	
	async def _extract_smart_content(self, page: Page) -> str:
		"""Extract content using smart selectors (crawl4ai approach)"""
		try:
			# Try to get main content using smart selectors
			for selector in self.content_indicators:
				try:
					element = await page.query_selector(selector)
					if element:
						content = await element.inner_html()
						if content and len(content) > 200:
							return content
				except:
					continue
			
			# Fallback to full page content
			return await page.content()
			
		except Exception:
			# Final fallback
			try:
				return await page.content()
			except:
				return ""


# =====================================================
# MAIN ORCHESTRATOR
# =====================================================

class MultiSourceOrchestrator:
	"""Main orchestrator combining all crawling strategies"""
	
	def __init__(self, max_concurrent: int = 10, max_sessions: int = 20):
		self.max_concurrent = max_concurrent
		self.max_sessions = max_sessions
		
		# Core components
		self.request_queue = RequestQueue()
		self.session_pool = SessionPool(max_sessions)
		self.smart_crawler = SmartCrawler(self.session_pool)
		
		# Rate limiting (Crawlee inspired)
		self.rate_limiter = self._create_rate_limiter()
		
		# Statistics
		self.stats = {
			'requests_processed': 0,
			'requests_successful': 0,
			'requests_failed': 0,
			'start_time': None,
			'strategy_usage': {}
		}
		
		# Semaphore for concurrency control
		self._semaphore = asyncio.Semaphore(max_concurrent)
		self._running = False
		self._workers: List[asyncio.Task] = []
	
	def _create_rate_limiter(self) -> Dict[str, Any]:
		"""Create rate limiter per domain"""
		return {}
	
	async def add_requests(self, requests: List[QueuedRequest]) -> int:
		"""Add multiple requests to queue"""
		added_count = 0
		for request in requests:
			if await self.request_queue.add_request(request):
				added_count += 1
		return added_count
	
	async def add_urls(self, urls: List[str], tenant_id: str, target_id: str,
					   priority: RequestPriority = RequestPriority.NORMAL,
					   **kwargs) -> int:
		"""Add URLs as requests"""
		requests = [
			QueuedRequest(
				url=url,
				tenant_id=tenant_id,
				target_id=target_id,
				priority=priority,
				**kwargs
			)
			for url in urls
		]
		return await self.add_requests(requests)
	
	async def start_crawling(self, config: ContentCleaningConfig, 
							business_context: Dict[str, Any] = None) -> None:
		"""Start the crawling process"""
		if self._running:
			logger.warning("Crawler is already running")
			return
		
		self._running = True
		self.stats['start_time'] = datetime.utcnow()
		
		logger.info(f"Starting crawler with {self.max_concurrent} workers")
		
		# Start worker tasks
		self._workers = [
			asyncio.create_task(self._worker(i, config, business_context))
			for i in range(self.max_concurrent)
		]
		
		# Wait for all workers to complete
		await asyncio.gather(*self._workers, return_exceptions=True)
	
	async def stop_crawling(self):
		"""Stop the crawling process"""
		self._running = False
		
		# Cancel all workers
		for worker in self._workers:
			worker.cancel()
		
		# Wait for workers to finish
		await asyncio.gather(*self._workers, return_exceptions=True)
		
		# Cleanup sessions
		await self.session_pool.close_all()
		
		logger.info("Crawler stopped")
	
	async def _worker(self, worker_id: int, config: ContentCleaningConfig, 
					 business_context: Dict[str, Any] = None):
		"""Worker task for processing requests"""
		logger.info(f"Worker {worker_id} started")
		
		while self._running:
			try:
				# Get next request
				request = await self.request_queue.get_request()
				if not request:
					await asyncio.sleep(0.1)
					continue
				
				# Rate limiting
				await self._apply_rate_limit(request.url)
				
				# Process request
				async with self._semaphore:
					await self._process_request(request, config, business_context)
				
			except asyncio.CancelledError:
				logger.info(f"Worker {worker_id} cancelled")
				break
			except Exception as e:
				logger.error(f"Worker {worker_id} error: {e}")
				await asyncio.sleep(1)
		
		logger.info(f"Worker {worker_id} stopped")
	
	async def _apply_rate_limit(self, url: str):
		"""Apply rate limiting per domain"""
		domain = urlparse(url).netloc
		
		# Simple rate limiting - 1 request per second per domain
		current_time = time.time()
		
		if domain not in self.rate_limiter:
			self.rate_limiter[domain] = current_time
		else:
			time_since_last = current_time - self.rate_limiter[domain]
			if time_since_last < 1.0:
				await asyncio.sleep(1.0 - time_since_last)
		
		self.rate_limiter[domain] = time.time()
	
	async def _process_request(self, request: QueuedRequest, config: ContentCleaningConfig,
							  business_context: Dict[str, Any] = None):
		"""Process a single request"""
		try:
			logger.debug(f"Processing: {request.url}")
			
			# Execute smart crawl with content intelligence
			crawl_result, extraction_result, intelligence_result = await self.smart_crawler.smart_crawl(
				request, config, business_context
			)
			
			# Update statistics
			self.stats['requests_processed'] += 1
			
			if crawl_result.success:
				self.stats['requests_successful'] += 1
				await self.request_queue.mark_completed(request, success=True)
				
				# Track strategy usage
				strategy = crawl_result.method_used.value
				self.stats['strategy_usage'][strategy] = self.stats['strategy_usage'].get(strategy, 0) + 1
				
				# Store results (would integrate with database service)
				await self._store_results(request, crawl_result, extraction_result, intelligence_result)
				
			else:
				self.stats['requests_failed'] += 1
				
				# Try to requeue with different strategy
				if await self.request_queue.requeue_request(request):
					logger.info(f"Requeued failed request: {request.url}")
				else:
					logger.warning(f"Request permanently failed: {request.url}")
		
		except Exception as e:
			logger.error(f"Error processing request {request.url}: {e}")
			self.stats['requests_failed'] += 1
			await self.request_queue.mark_completed(request, success=False)
	
	async def _store_results(self, request: QueuedRequest, crawl_result: CrawlResult, 
							 extraction_result: ContentExtractionResult,
							 intelligence_result: Optional[ContentIntelligenceResult] = None):
		"""Store crawl, extraction, and intelligence results"""
		
		# Here we would integrate with the database service
		# For now, just log the results
		intelligence_info = ""
		if intelligence_result and intelligence_result.success:
			entity_count = len(intelligence_result.extracted_entities)
			primary_category = intelligence_result.content_classification.primary_category.value
			intelligence_info = f", entities: {entity_count}, category: {primary_category}"
		
		logger.info(
			f"Stored result: {request.url} "
			f"(status: {crawl_result.status_code}, "
			f"content_len: {len(extraction_result.markdown_content)}, "
			f"method: {crawl_result.method_used.value}"
			f"{intelligence_info})"
		)
		
		# TODO: Integrate with CrawlerDatabaseService
		# await database_service.create_data_record({
		#     'tenant_id': request.tenant_id,
		#     'dataset_id': request.target_id,  # or derive from target
		#     'source_url': request.url,
		#     'raw_content': crawl_result.content,
		#     'cleaned_content': extraction_result.cleaned_content,
		#     'markdown_content': extraction_result.markdown_content,
		#     'content_fingerprint': extraction_result.content_fingerprint,
		#     'language': extraction_result.language,
		#     'extraction_metadata': extraction_result.metadata,
		#     'intelligence_data': intelligence_result.model_dump() if intelligence_result else None
		# })
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get crawling statistics"""
		stats = self.stats.copy()
		stats['queue_size'] = self.request_queue.size
		stats['queue_stats'] = self.request_queue.stats
		
		if stats['start_time']:
			runtime_seconds = (datetime.utcnow() - stats['start_time']).total_seconds()
			stats['runtime_seconds'] = runtime_seconds
			stats['requests_per_second'] = stats['requests_processed'] / max(runtime_seconds, 1)
		
		return stats
	
	async def wait_for_completion(self):
		"""Wait for all requests to be processed"""
		while not self.request_queue.is_empty:
			await asyncio.sleep(1)
		
		logger.info("All requests completed")


# =====================================================
# EXPORTS
# =====================================================

__all__ = [
	'MultiSourceOrchestrator',
	'RequestQueue',
	'QueuedRequest',
	'RequestPriority', 
	'SmartCrawler',
	'SessionPool'
]