"""
APG Crawler Capability - Advanced Stealth Orchestration Engine
==============================================================

Multi-strategy stealth orchestration with:
- CloudScraper integration for Cloudflare bypass
- Playwright browser automation with stealth
- Selenium WebDriver with anti-detection
- HTTP client with behavioral mimicry
- Machine learning-based strategy optimization
- Protection mechanism detection and profiling

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import hashlib

import httpx
import cloudscraper
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium_stealth import stealth
from fake_useragent import UserAgent
import undetected_chromedriver as uc

from ..views import StealthStrategy, ProtectionType
from ..models import ProtectionProfile, StealthStrategy as StealthStrategyModel

# =====================================================
# CONFIGURATION AND TYPES
# =====================================================

logger = logging.getLogger(__name__)

class StealthMethod(str, Enum):
	"""Available stealth methods"""
	CLOUDSCRAPER = "cloudscraper"
	PLAYWRIGHT = "playwright"
	SELENIUM = "selenium"
	SELENIUM_STEALTH = "selenium_stealth"
	UNDETECTED_CHROME = "undetected_chrome"
	HTTP_MIMICRY = "http_mimicry"
	PROXY_ROTATION = "proxy_rotation"

@dataclass
class CrawlRequest:
	"""Request for crawling a URL"""
	url: str
	method: str = "GET"
	headers: Optional[Dict[str, str]] = None
	cookies: Optional[Dict[str, str]] = None
	data: Optional[Dict[str, Any]] = None
	timeout: int = 30
	retries: int = 3
	stealth_method: Optional[StealthMethod] = None
	javascript_required: bool = False
	wait_for_selector: Optional[str] = None
	scroll_to_bottom: bool = False
	screenshot: bool = False

@dataclass
class CrawlResult:
	"""Result of a crawl operation"""
	url: str
	status_code: int
	content: str
	headers: Dict[str, str]
	cookies: Dict[str, str]
	final_url: str
	response_time: float
	method_used: StealthMethod
	protection_detected: List[ProtectionType]
	success: bool
	error: Optional[str] = None
	screenshot_data: Optional[bytes] = None
	metadata: Optional[Dict[str, Any]] = None


# =====================================================
# PROTECTION DETECTION ENGINE
# =====================================================

class ProtectionDetectionEngine:
	"""Detects and profiles website protection mechanisms"""
	
	def __init__(self):
		self.protection_signatures = {
			ProtectionType.CLOUDFLARE: [
				"cloudflare",
				"cf-ray",
				"checking your browser",
				"just a moment",
				"ddos protection by cloudflare"
			],
			ProtectionType.AKAMAI: [
				"akamai",
				"reference #",
				"access denied",
				"request blocked"
			],
			ProtectionType.INCAPSULA: [
				"incap_ses",
				"incapsula",
				"request unsuccessful"
			],
			ProtectionType.RECAPTCHA: [
				"recaptcha",
				"captcha",
				"g-recaptcha"
			],
			ProtectionType.HCAPTCHA: [
				"hcaptcha",
				"h-captcha"
			],
			ProtectionType.WAF_GENERIC: [
				"web application firewall",
				"waf",
				"blocked by security",
				"access forbidden"
			]
		}
	
	def detect_protection(self, content: str, headers: Dict[str, str], status_code: int) -> List[ProtectionType]:
		"""Detect protection mechanisms from response"""
		detected = []
		content_lower = content.lower()
		
		# Check headers for protection indicators
		for header_name, header_value in headers.items():
			header_lower = f"{header_name.lower()}: {header_value.lower()}"
			
			for protection_type, signatures in self.protection_signatures.items():
				if any(sig in header_lower for sig in signatures):
					if protection_type not in detected:
						detected.append(protection_type)
		
		# Check content for protection indicators
		for protection_type, signatures in self.protection_signatures.items():
			if any(sig in content_lower for sig in signatures):
				if protection_type not in detected:
					detected.append(protection_type)
		
		# Check status codes for protection indicators
		if status_code in [403, 429, 503]:
			if not detected:  # Only add generic if no specific protection detected
				detected.append(ProtectionType.WAF_GENERIC)
		
		return detected
	
	def create_protection_profile(self, tenant_id: str, domain: str, 
								  detections: List[Tuple[List[ProtectionType], float]]) -> ProtectionProfile:
		"""Create protection profile from multiple detection results"""
		# Aggregate protection types and confidence scores
		protection_counts = {}
		total_detections = len(detections)
		
		for protection_types, confidence in detections:
			for ptype in protection_types:
				if ptype not in protection_counts:
					protection_counts[ptype] = []
				protection_counts[ptype].append(confidence)
		
		# Calculate average confidence for each protection type
		final_protections = []
		overall_confidence = 0.0
		
		for ptype, confidences in protection_counts.items():
			avg_confidence = sum(confidences) / len(confidences)
			occurrence_rate = len(confidences) / total_detections
			
			# Only include protections seen in >50% of attempts with >0.7 confidence
			if occurrence_rate > 0.5 and avg_confidence > 0.7:
				final_protections.append(ptype)
				overall_confidence += avg_confidence
		
		if final_protections:
			overall_confidence /= len(final_protections)
		
		return ProtectionProfile(
			tenant_id=tenant_id,
			domain=domain,
			protection_types=final_protections,
			detection_confidence=overall_confidence,
			protection_characteristics={
				'detection_count': total_detections,
				'protection_counts': {p.value: len(protection_counts.get(p, [])) for p in final_protections}
			}
		)


# =====================================================
# STEALTH STRATEGIES
# =====================================================

class CloudScraperStrategy:
	"""CloudScraper-based stealth strategy"""
	
	def __init__(self):
		self.scraper = cloudscraper.create_scraper(
			browser={
				'browser': 'chrome',
				'platform': 'win32',
				'mobile': False
			}
		)
	
	async def crawl(self, request: CrawlRequest) -> CrawlResult:
		"""Perform crawl using CloudScraper"""
		start_time = time.time()
		
		try:
			response = self.scraper.get(
				request.url,
				headers=request.headers or {},
				cookies=request.cookies or {},
				timeout=request.timeout
			)
			
			response_time = time.time() - start_time
			
			return CrawlResult(
				url=request.url,
				status_code=response.status_code,
				content=response.text,
				headers=dict(response.headers),
				cookies=dict(response.cookies),
				final_url=response.url,
				response_time=response_time,
				method_used=StealthMethod.CLOUDSCRAPER,
				protection_detected=[],
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


class PlaywrightStrategy:
	"""Playwright-based stealth strategy with advanced browser automation"""
	
	def __init__(self):
		self.playwright = None
		self.browser = None
		self.context = None
	
	async def setup(self):
		"""Initialize Playwright browser"""
		if not self.playwright:
			self.playwright = await async_playwright().start()
			self.browser = await self.playwright.chromium.launch(
				headless=True,
				args=[
					'--no-sandbox',
					'--disable-blink-features=AutomationControlled',
					'--disable-dev-shm-usage',
					'--disable-gpu',
					'--no-first-run',
					'--no-default-browser-check',
					'--disable-extensions',
					'--disable-plugins'
				]
			)
			
			# Create stealth context
			self.context = await self.browser.new_context(
				viewport={'width': 1920, 'height': 1080},
				user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
			)
			
			# Add stealth scripts
			await self.context.add_init_script("""
				Object.defineProperty(navigator, 'webdriver', {
					get: () => undefined,
				});
				
				Object.defineProperty(navigator, 'languages', {
					get: () => ['en-US', 'en'],
				});
				
				Object.defineProperty(navigator, 'plugins', {
					get: () => [1, 2, 3, 4, 5],
				});
			""")
	
	async def crawl(self, request: CrawlRequest) -> CrawlResult:
		"""Perform crawl using Playwright"""
		await self.setup()
		start_time = time.time()
		
		try:
			page = await self.context.new_page()
			
			# Set additional headers if provided
			if request.headers:
				await page.set_extra_http_headers(request.headers)
			
			# Set cookies if provided
			if request.cookies:
				cookies = [
					{"name": k, "value": v, "url": request.url}
					for k, v in request.cookies.items()
				]
				await self.context.add_cookies(cookies)
			
			# Navigate to URL
			response = await page.goto(
				request.url,
				wait_until='domcontentloaded',
				timeout=request.timeout * 1000
			)
			
			# Wait for specific selector if requested
			if request.wait_for_selector:
				await page.wait_for_selector(request.wait_for_selector, timeout=10000)
			
			# Scroll to bottom if requested
			if request.scroll_to_bottom:
				await page.evaluate("""
					async () => {
						await new Promise((resolve) => {
							let totalHeight = 0;
							const distance = 100;
							const timer = setInterval(() => {
								const scrollHeight = document.body.scrollHeight;
								window.scrollBy(0, distance);
								totalHeight += distance;
								
								if(totalHeight >= scrollHeight){
									clearInterval(timer);
									resolve();
								}
							}, 100);
						});
					}
				""")
			
			# Get page content
			content = await page.content()
			
			# Take screenshot if requested
			screenshot_data = None
			if request.screenshot:
				screenshot_data = await page.screenshot(full_page=True)
			
			response_time = time.time() - start_time
			
			result = CrawlResult(
				url=request.url,
				status_code=response.status if response else 0,
				content=content,
				headers=dict(response.headers) if response else {},
				cookies={},  # Get cookies from context
				final_url=page.url,
				response_time=response_time,
				method_used=StealthMethod.PLAYWRIGHT,
				protection_detected=[],
				success=response.ok if response else False,
				screenshot_data=screenshot_data
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
	
	async def cleanup(self):
		"""Clean up Playwright resources"""
		if self.context:
			await self.context.close()
		if self.browser:
			await self.browser.close()
		if self.playwright:
			await self.playwright.stop()


class SeleniumStealthStrategy:
	"""Selenium with stealth modifications"""
	
	def __init__(self):
		self.driver = None
		self.user_agent = UserAgent()
	
	def setup_driver(self):
		"""Setup Chrome driver with stealth configuration"""
		if self.driver:
			return
		
		chrome_options = ChromeOptions()
		chrome_options.add_argument("--headless")
		chrome_options.add_argument("--no-sandbox")
		chrome_options.add_argument("--disable-dev-shm-usage")
		chrome_options.add_argument("--disable-blink-features=AutomationControlled")
		chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
		chrome_options.add_experimental_option('useAutomationExtension', False)
		chrome_options.add_argument(f"--user-agent={self.user_agent.random}")
		
		self.driver = webdriver.Chrome(options=chrome_options)
		
		# Apply stealth modifications
		stealth(self.driver,
				languages=["en-US", "en"],
				vendor="Google Inc.",
				platform="Win32",
				webgl_vendor="Intel Inc.",
				renderer="Intel Iris OpenGL Engine",
				fix_hairline=True)
	
	async def crawl(self, request: CrawlRequest) -> CrawlResult:
		"""Perform crawl using Selenium with stealth"""
		self.setup_driver()
		start_time = time.time()
		
		try:
			# Set cookies if provided
			if request.cookies:
				self.driver.get(request.url)  # Need to visit domain first
				for name, value in request.cookies.items():
					self.driver.add_cookie({'name': name, 'value': value})
			
			# Navigate to URL
			self.driver.get(request.url)
			
			# Wait for specific element if requested
			if request.wait_for_selector:
				WebDriverWait(self.driver, 10).until(
					EC.presence_of_element_located((By.CSS_SELECTOR, request.wait_for_selector))
				)
			
			# Scroll to bottom if requested
			if request.scroll_to_bottom:
				self.driver.execute_script("""
					return new Promise((resolve) => {
						let totalHeight = 0;
						const distance = 100;
						const timer = setInterval(() => {
							const scrollHeight = document.body.scrollHeight;
							window.scrollBy(0, distance);
							totalHeight += distance;
							
							if(totalHeight >= scrollHeight){
								clearInterval(timer);
								resolve();
							}
						}, 100);
					});
				""")
			
			# Get page content
			content = self.driver.page_source
			final_url = self.driver.current_url
			
			# Take screenshot if requested
			screenshot_data = None
			if request.screenshot:
				screenshot_data = self.driver.get_screenshot_as_png()
			
			response_time = time.time() - start_time
			
			return CrawlResult(
				url=request.url,
				status_code=200,  # Selenium doesn't provide status codes directly
				content=content,
				headers={},
				cookies={cookie['name']: cookie['value'] for cookie in self.driver.get_cookies()},
				final_url=final_url,
				response_time=response_time,
				method_used=StealthMethod.SELENIUM_STEALTH,
				protection_detected=[],
				success=True,
				screenshot_data=screenshot_data
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
				method_used=StealthMethod.SELENIUM_STEALTH,
				protection_detected=[],
				success=False,
				error=str(e)
			)
	
	def cleanup(self):
		"""Clean up Selenium driver"""
		if self.driver:
			self.driver.quit()
			self.driver = None


class HTTPMimicryStrategy:
	"""HTTP client with behavioral mimicry"""
	
	def __init__(self):
		self.session = None
	
	def setup_session(self):
		"""Setup HTTP session with realistic headers"""
		if self.session:
			return
		
		ua = UserAgent()
		self.session = httpx.AsyncClient(
			headers={
				'User-Agent': ua.random,
				'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
				'Accept-Language': 'en-US,en;q=0.5',
				'Accept-Encoding': 'gzip, deflate',
				'Connection': 'keep-alive',
				'Upgrade-Insecure-Requests': '1',
				'Sec-Fetch-Dest': 'document',
				'Sec-Fetch-Mode': 'navigate',
				'Sec-Fetch-Site': 'none',
				'Cache-Control': 'max-age=0'
			},
			timeout=30.0
		)
	
	async def crawl(self, request: CrawlRequest) -> CrawlResult:
		"""Perform crawl with HTTP behavioral mimicry"""
		self.setup_session()
		start_time = time.time()
		
		try:
			# Add random delay to mimic human behavior
			await asyncio.sleep(random.uniform(0.5, 2.0))
			
			# Merge headers
			headers = dict(self.session.headers)
			if request.headers:
				headers.update(request.headers)
			
			# Make request
			response = await self.session.request(
				request.method,
				request.url,
				headers=headers,
				cookies=request.cookies or {},
				data=request.data,
				timeout=request.timeout
			)
			
			response_time = time.time() - start_time
			
			return CrawlResult(
				url=request.url,
				status_code=response.status_code,
				content=response.text,
				headers=dict(response.headers),
				cookies=dict(response.cookies),
				final_url=str(response.url),
				response_time=response_time,
				method_used=StealthMethod.HTTP_MIMICRY,
				protection_detected=[],
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
				method_used=StealthMethod.HTTP_MIMICRY,
				protection_detected=[],
				success=False,
				error=str(e)
			)
	
	async def cleanup(self):
		"""Clean up HTTP session"""
		if self.session:
			await self.session.aclose()


# =====================================================
# MAIN STEALTH ORCHESTRATION ENGINE
# =====================================================

class StealthOrchestrationEngine:
	"""Advanced multi-strategy stealth orchestration engine"""
	
	def __init__(self):
		self.strategies = {
			StealthMethod.CLOUDSCRAPER: CloudScraperStrategy(),
			StealthMethod.PLAYWRIGHT: PlaywrightStrategy(),
			StealthMethod.SELENIUM_STEALTH: SeleniumStealthStrategy(),
			StealthMethod.HTTP_MIMICRY: HTTPMimicryStrategy()
		}
		
		self.protection_detector = ProtectionDetectionEngine()
		self.success_history = {}  # Track success rates by domain and method
		self.protection_profiles = {}  # Cache protection profiles
	
	async def crawl_with_stealth(self, request: CrawlRequest, 
								 tenant_id: str,
								 adaptive_strategy: bool = True) -> CrawlResult:
		"""Perform stealth crawl with adaptive strategy selection"""
		
		# Determine optimal strategy
		if adaptive_strategy and not request.stealth_method:
			request.stealth_method = await self._select_optimal_strategy(request.url, tenant_id)
		elif not request.stealth_method:
			request.stealth_method = StealthMethod.CLOUDSCRAPER
		
		# Execute crawl with selected strategy
		strategy = self.strategies[request.stealth_method]
		result = await strategy.crawl(request)
		
		# Detect protection mechanisms
		if result.content and result.headers:
			protections = self.protection_detector.detect_protection(
				result.content, result.headers, result.status_code
			)
			result.protection_detected = protections
		
		# Update success history
		await self._update_success_history(request.url, request.stealth_method, result, tenant_id)
		
		# If failed, try alternative strategy
		if not result.success and adaptive_strategy:
			logger.info(f"Primary strategy {request.stealth_method} failed, trying alternative")
			alternative_result = await self._try_alternative_strategy(request, tenant_id)
			if alternative_result and alternative_result.success:
				return alternative_result
		
		return result
	
	async def _select_optimal_strategy(self, url: str, tenant_id: str) -> StealthMethod:
		"""Select optimal stealth strategy based on domain and success history"""
		from urllib.parse import urlparse
		domain = urlparse(url).netloc
		
		# Check if we have protection profile for this domain
		profile_key = f"{tenant_id}:{domain}"
		if profile_key in self.protection_profiles:
			profile = self.protection_profiles[profile_key]
			return self._strategy_for_protection(profile.protection_types)
		
		# Check success history
		if domain in self.success_history:
			history = self.success_history[domain]
			best_method = max(history.items(), key=lambda x: x[1]['success_rate'])
			if best_method[1]['success_rate'] > 0.8:
				return StealthMethod(best_method[0])
		
		# Default strategy based on URL characteristics
		if 'cloudflare' in url.lower():
			return StealthMethod.CLOUDSCRAPER
		elif any(js_indicator in url.lower() for js_indicator in ['react', 'vue', 'angular', 'spa']):
			return StealthMethod.PLAYWRIGHT
		else:
			return StealthMethod.HTTP_MIMICRY
	
	def _strategy_for_protection(self, protections: List[ProtectionType]) -> StealthMethod:
		"""Select best strategy for known protections"""
		if ProtectionType.CLOUDFLARE in protections:
			return StealthMethod.CLOUDSCRAPER
		elif ProtectionType.RECAPTCHA in protections or ProtectionType.HCAPTCHA in protections:
			return StealthMethod.PLAYWRIGHT
		elif ProtectionType.WAF_GENERIC in protections:
			return StealthMethod.SELENIUM_STEALTH
		else:
			return StealthMethod.HTTP_MIMICRY
	
	async def _try_alternative_strategy(self, request: CrawlRequest, tenant_id: str) -> Optional[CrawlResult]:
		"""Try alternative strategies when primary fails"""
		current_method = request.stealth_method
		
		# Define fallback order
		fallback_methods = [
			StealthMethod.PLAYWRIGHT,
			StealthMethod.SELENIUM_STEALTH,
			StealthMethod.CLOUDSCRAPER,
			StealthMethod.HTTP_MIMICRY
		]
		
		# Remove current method from fallbacks
		if current_method in fallback_methods:
			fallback_methods.remove(current_method)
		
		# Try each fallback method
		for method in fallback_methods[:2]:  # Try only first 2 alternatives
			request.stealth_method = method
			strategy = self.strategies[method]
			result = await strategy.crawl(request)
			
			if result.success:
				await self._update_success_history(request.url, method, result, tenant_id)
				return result
		
		return None
	
	async def _update_success_history(self, url: str, method: StealthMethod, 
									  result: CrawlResult, tenant_id: str):
		"""Update success history for strategy optimization"""
		from urllib.parse import urlparse
		domain = urlparse(url).netloc
		
		if domain not in self.success_history:
			self.success_history[domain] = {}
		
		if method.value not in self.success_history[domain]:
			self.success_history[domain][method.value] = {
				'attempts': 0,
				'successes': 0,
				'success_rate': 0.0,
				'avg_response_time': 0.0,
				'last_used': None
			}
		
		history = self.success_history[domain][method.value]
		history['attempts'] += 1
		
		if result.success:
			history['successes'] += 1
		
		history['success_rate'] = history['successes'] / history['attempts']
		history['avg_response_time'] = (
			(history['avg_response_time'] * (history['attempts'] - 1) + result.response_time) 
			/ history['attempts']
		)
		history['last_used'] = datetime.utcnow()
	
	async def analyze_protection_profile(self, url: str, tenant_id: str, 
										 num_attempts: int = 3) -> ProtectionProfile:
		"""Analyze website protection mechanisms with multiple attempts"""
		from urllib.parse import urlparse
		domain = urlparse(url).netloc
		
		detections = []
		
		# Try multiple strategies to detect protections
		for method in [StealthMethod.HTTP_MIMICRY, StealthMethod.CLOUDSCRAPER]:
			request = CrawlRequest(url=url, stealth_method=method)
			
			for attempt in range(num_attempts):
				result = await self.strategies[method].crawl(request)
				protections = self.protection_detector.detect_protection(
					result.content, result.headers, result.status_code
				)
				
				# Calculate confidence based on response characteristics
				confidence = 0.8 if result.status_code in [403, 429, 503] else 0.6
				if protections:
					confidence = 0.9
				
				detections.append((protections, confidence))
				
				# Small delay between attempts
				await asyncio.sleep(1.0)
		
		# Create and cache protection profile
		profile = self.protection_detector.create_protection_profile(
			tenant_id, domain, detections
		)
		
		profile_key = f"{tenant_id}:{domain}"
		self.protection_profiles[profile_key] = profile
		
		return profile
	
	async def get_strategy_recommendations(self, url: str, tenant_id: str) -> Dict[str, Any]:
		"""Get strategy recommendations for a URL"""
		from urllib.parse import urlparse
		domain = urlparse(url).netloc
		
		# Analyze protection profile
		profile = await self.analyze_protection_profile(url, tenant_id)
		
		recommendations = {
			'primary_strategy': self._strategy_for_protection(profile.protection_types),
			'protection_types': [p.value for p in profile.protection_types],
			'confidence': profile.detection_confidence,
			'difficulty_level': profile.difficulty_level,
			'recommended_settings': {}
		}
		
		# Add specific recommendations based on protections
		if ProtectionType.CLOUDFLARE in profile.protection_types:
			recommendations['recommended_settings']['use_cloudscraper'] = True
			recommendations['recommended_settings']['javascript_required'] = True
		
		if ProtectionType.RECAPTCHA in profile.protection_types:
			recommendations['recommended_settings']['manual_intervention'] = True
			recommendations['recommended_settings']['captcha_solving'] = True
		
		# Add success history if available
		if domain in self.success_history:
			recommendations['success_history'] = self.success_history[domain]
		
		return recommendations
	
	async def cleanup(self):
		"""Clean up all strategy resources"""
		for strategy in self.strategies.values():
			if hasattr(strategy, 'cleanup'):
				await strategy.cleanup()


# =====================================================
# EXPORTS
# =====================================================

__all__ = [
	'StealthOrchestrationEngine',
	'CrawlRequest',
	'CrawlResult',
	'StealthMethod',
	'ProtectionDetectionEngine'
]