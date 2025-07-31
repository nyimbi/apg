#!/usr/bin/env python3
"""
Enhanced Stealth Main Orchestrator
=================================

The main orchestrator class that combines all enhanced stealth techniques
including Cloudflare bypasses, advanced fingerprinting protection, session
management, proxy rotation, and intelligent timing.
"""

import asyncio
import logging
import random
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from urllib.parse import urlparse
import uuid

# Import components from the enhanced stealth system
try:
    from .enhanced_stealth_continuation import (
        EnhancedStealthHeaders,
        ResidentialProxyManager,
        SessionManager,
        CloudflareRateLimiter,
        HumanBehaviorSimulator,
        create_browser_config
    )
    from .crawl4ai_timeout_fix import TimeoutManager, patch_crawl4ai_timeouts
except ImportError:
    from enhanced_stealth_continuation import (
        EnhancedStealthHeaders,
        ResidentialProxyManager,
        SessionManager,
        CloudflareRateLimiter,
        HumanBehaviorSimulator,
        create_browser_config
    )
    from crawl4ai_timeout_fix import TimeoutManager, patch_crawl4ai_timeouts

# Apply timeout fix before any crawl4ai operations
patch_crawl4ai_timeouts()

logger = logging.getLogger(__name__)


class EnhancedStealthOrchestrator:
    """
    Main orchestrator for enhanced stealth techniques with comprehensive
    Cloudflare bypass capabilities and advanced anti-detection features.
    """

    def __init__(self, session_dir: str = "./browser_sessions"):
        # Initialize timeout management first
        self.timeout_manager = TimeoutManager()
        self.timeout_manager.apply_crawl4ai_fix()

        # Initialize all components with timeout-aware configuration
        self.browser_config = create_browser_config()
        self.header_manager = EnhancedStealthHeaders()
        self.proxy_manager = ResidentialProxyManager()
        self.session_manager = SessionManager(session_dir)
        self.rate_limiter = CloudflareRateLimiter()

        # Enhanced metrics tracking
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'blocked_requests': 0,
            'cloudflare_challenges': 0,
            'cloudflare_bypasses': 0,
            'stealth_activations': 0,
            'fingerprint_protections': 0,
            'session_rotations': 0,
            'tls_bypasses': 0,
            'proxy_rotations': 0,
            'challenge_solve_times': [],
            'domain_performance': {},
            'timeout_warnings': 0,
            'timeout_fixes_applied': 0
        }

        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.active_sessions = {}
        self.domain_configs = {}

        # Advanced state management
        self.last_request_times = {}
        self.domain_challenge_counts = {}
        self.successful_bypasses = {}

        # Store timeout configuration for stealth operations
        self.stealth_timeouts = self.timeout_manager.get_stealth_timeouts()

        logger.info(f"Enhanced stealth orchestrator initialized (session: {self.session_id[:8]})")
        logger.info(f"Timeout configuration: {self.stealth_timeouts}")

    async def configure_enhanced_stealth(self, url: str, force_cloudflare: bool = False) -> Dict[str, Any]:
        """
        Configure comprehensive stealth measures for the given URL with
        Cloudflare-specific enhancements.
        """
        try:
            domain = urlparse(url).netloc.lower()

            # Detect if this is likely a Cloudflare-protected site
            is_cloudflare = self._is_cloudflare_domain(domain) or force_cloudflare

            # Determine stealth level
            stealth_level = self._get_enhanced_stealth_level(domain, is_cloudflare)

            # Get or create session for this domain
            session_id = await self._get_or_create_session(domain)

            # Get appropriate proxy if needed
            proxy_config = None
            if stealth_level in ['maximum', 'cloudflare']:
                proxy_config = self.proxy_manager.get_proxy_for_domain(domain)
                if proxy_config:
                    self.metrics['proxy_rotations'] += 1

            # Generate enhanced headers
            headers = self.header_manager.get_enhanced_stealth_headers(url, is_cloudflare)

            # Generate JavaScript protection scripts
            js_scripts = self._generate_enhanced_js_protection(domain, is_cloudflare)

            # Generate TLS configuration
            tls_config = self._generate_tls_config(is_cloudflare)

            # Generate browser arguments for advanced stealth
            browser_args = self._generate_browser_arguments(is_cloudflare)

            # Create viewport configuration
            viewport_config = {
                'width': self.browser_config.width,
                'height': self.browser_config.height,
                'deviceScaleFactor': self.browser_config.pixel_ratio
            }

            # Create comprehensive stealth configuration
            config = {
                'url': url,
                'domain': domain,
                'session_id': session_id,
                'stealth_level': stealth_level,
                'is_cloudflare': is_cloudflare,
                'headers': headers,
                'js_scripts': js_scripts,
                'tls_config': tls_config,
                'browser_args': browser_args,
                'viewport_config': viewport_config,
                'proxy_config': proxy_config,
                'timing_config': self._get_timing_config(domain, is_cloudflare),
                'human_behavior': await HumanBehaviorSimulator.simulate_human_interaction(
                    self.browser_config.width,
                    self.browser_config.height
                ),
                'challenge_handling': {
                    'enabled': is_cloudflare,
                    'max_wait_time': 30,
                    'retry_attempts': 3
                }
            }

            # Update metrics
            self.metrics['stealth_activations'] += 1
            self.metrics['fingerprint_protections'] += len(js_scripts)

            # Store configuration for this domain
            self.domain_configs[domain] = config

            logger.info(f"Enhanced stealth configured for {domain} (level: {stealth_level}, CF: {is_cloudflare})")
            return config

        except Exception as e:
            logger.error(f"Failed to configure enhanced stealth for {url}: {e}")
            return {'error': str(e)}

    async def apply_intelligent_timing(self, url: str, attempt: int = 0, challenge_detected: bool = False) -> None:
        """Apply intelligent timing delays with Cloudflare awareness."""
        domain = urlparse(url).netloc.lower()

        # Check if this is a Cloudflare domain
        is_cloudflare = self._is_cloudflare_domain(domain)

        # Check for Ethiopian advanced sites
        is_ethiopian_advanced = 'addisstandard.com' in domain

        if is_ethiopian_advanced:
            # Ethiopian sites need longer delays due to sophisticated challenges
            base_delay = 25.0 if challenge_detected else 15.0
            jitter = random.uniform(3.0, 8.0)
            delay = min(base_delay + (attempt * 5.0) + jitter, 60.0)
            logger.info(f"Applying Ethiopian advanced delay of {delay:.2f}s for {domain}")
            await asyncio.sleep(delay)
        elif is_cloudflare or challenge_detected:
            await self.rate_limiter.cloudflare_delay(domain, attempt, challenge_detected)
        else:
            # Standard intelligent delay
            await self._apply_standard_delay(domain, attempt)

        self.metrics['stealth_activations'] += 1
        # Update last request time
        self.last_request_times[domain] = time.time()

    async def handle_cloudflare_challenge(self, url: str, response_content: str,
                                        response_headers: Dict[str, str]) -> Dict[str, Any]:
        """Handle Cloudflare challenge detection and resolution."""
        try:
            from .lightweight_stealth import CloudflareDetector
        except ImportError:
            from lightweight_stealth import CloudflareDetector

        domain = urlparse(url).netloc.lower()

        # Detect if this is a Cloudflare challenge
        is_challenge = CloudflareDetector.detect_cloudflare(response_content, response_headers)

        if is_challenge:
            self.metrics['cloudflare_challenges'] += 1
            self.rate_limiter.record_challenge(domain)

            # Update domain challenge count
            self.domain_challenge_counts[domain] = self.domain_challenge_counts.get(domain, 0) + 1

            # Get challenge wait time
            wait_time = CloudflareDetector.get_challenge_wait_time(response_content)

            logger.info(f"Cloudflare challenge detected for {domain}, waiting {wait_time}s")

            return {
                'is_challenge': True,
                'wait_time': wait_time,
                'challenge_type': 'javascript',
                'recommended_action': 'wait_and_retry',
                'retry_delay': wait_time + random.uniform(2, 5)
            }

        return {'is_challenge': False}

    def record_success(self, url: str, response_time: float = None) -> None:
        """Record successful request with enhanced metrics."""
        domain = urlparse(url).netloc.lower()

        # Update general metrics
        self.metrics['total_requests'] += 1
        self.metrics['successful_requests'] += 1

        # Update domain-specific performance
        if domain not in self.metrics['domain_performance']:
            self.metrics['domain_performance'][domain] = {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'avg_response_time': 0,
                'challenge_encounters': 0
            }

        domain_perf = self.metrics['domain_performance'][domain]
        domain_perf['requests'] += 1
        domain_perf['successes'] += 1

        if response_time:
            domain_perf['avg_response_time'] = (
                (domain_perf['avg_response_time'] * (domain_perf['requests'] - 1) + response_time) /
                domain_perf['requests']
            )

        # Update rate limiter
        self.rate_limiter.record_success(domain)

        # Update session manager
        if domain in self.active_sessions:
            self.session_manager.record_success(self.active_sessions[domain])

        # Track successful bypasses
        self.successful_bypasses[domain] = self.successful_bypasses.get(domain, 0) + 1

        logger.debug(f"Success recorded for {domain} (total: {domain_perf['successes']})")

    def record_failure(self, url: str, error_code: int = None, error_type: str = None) -> None:
        """Record failed request with detailed error tracking."""
        domain = urlparse(url).netloc.lower()

        # Update general metrics
        self.metrics['total_requests'] += 1
        self.metrics['blocked_requests'] += 1

        # Update domain-specific performance
        if domain not in self.metrics['domain_performance']:
            self.metrics['domain_performance'][domain] = {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'avg_response_time': 0,
                'challenge_encounters': 0,
                'error_types': {}
            }

        domain_perf = self.metrics['domain_performance'][domain]
        domain_perf['requests'] += 1
        domain_perf['failures'] += 1

        # Track error types
        if 'error_types' not in domain_perf:
            domain_perf['error_types'] = {}

        error_key = f"{error_code}_{error_type}" if error_type else str(error_code)
        domain_perf['error_types'][error_key] = domain_perf['error_types'].get(error_key, 0) + 1

        # Handle specific error codes
        if error_code == 403:
            # Increase stealth level for 403 errors
            self._escalate_stealth_level(domain)
        elif error_code == 503:
            # Might be rate limited
            self.rate_limiter.record_challenge(domain)

        # Update session manager
        if domain in self.active_sessions:
            self.session_manager.record_failure(self.active_sessions[domain])

        # Mark proxy as failed if using one
        if domain in self.domain_configs:
            proxy_config = self.domain_configs[domain].get('proxy_config')
            if proxy_config:
                self.proxy_manager.mark_proxy_failed(proxy_config)

        logger.warning(f"Failure recorded for {domain}: {error_code} ({error_type})")

    def record_cloudflare_bypass(self, url: str, solve_time: float = None) -> None:
        """Record successful Cloudflare challenge bypass."""
        domain = urlparse(url).netloc.lower()

        self.metrics['cloudflare_bypasses'] += 1

        if solve_time:
            self.metrics['challenge_solve_times'].append(solve_time)

        # Update domain performance
        if domain in self.metrics['domain_performance']:
            self.metrics['domain_performance'][domain]['challenge_encounters'] += 1

        logger.info(f"Cloudflare bypass successful for {domain} (solve time: {solve_time:.2f}s)")

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive stealth performance report."""
        total_requests = self.metrics['total_requests']

        # Calculate rates
        success_rate = (self.metrics['successful_requests'] / max(1, total_requests)) * 100.0

        cloudflare_bypass_rate = 0.0
        if self.metrics['cloudflare_challenges'] > 0:
            cloudflare_bypass_rate = (
                self.metrics['cloudflare_bypasses'] / self.metrics['cloudflare_challenges']
            ) * 100.0

        # Calculate stealth effectiveness
        stealth_effectiveness = self._calculate_stealth_effectiveness()

        # Average challenge solve time
        avg_solve_time = 0.0
        if self.metrics['challenge_solve_times']:
            avg_solve_time = sum(self.metrics['challenge_solve_times']) / len(self.metrics['challenge_solve_times'])

        return {
            'session_id': self.session_id,
            'total_requests': total_requests,
            'successful_requests': self.metrics['successful_requests'],
            'blocked_requests': self.metrics['blocked_requests'],
            'success_rate': success_rate,
            'cloudflare_challenges': self.metrics['cloudflare_challenges'],
            'cloudflare_bypasses': self.metrics['cloudflare_bypasses'],
            'cloudflare_bypass_rate': cloudflare_bypass_rate,
            'stealth_effectiveness': stealth_effectiveness,
            'avg_challenge_solve_time': avg_solve_time,
            'stealth_activations': self.metrics['stealth_activations'],
            'fingerprint_protections': self.metrics['fingerprint_protections'],
            'proxy_rotations': self.metrics['proxy_rotations'],
            'session_rotations': self.metrics['session_rotations'],
            'tls_bypasses': self.metrics['tls_bypasses'],
            'active_domains': len(self.domain_configs),
            'domain_performance': self.metrics['domain_performance'],
            'top_performing_domains': self._get_top_performing_domains(),
            'challenge_rates': self._get_domain_challenge_rates()
        }

    # Private helper methods

    def _is_cloudflare_domain(self, domain: str) -> bool:
        """Detect if domain is likely protected by Cloudflare."""
        # Known Cloudflare indicators for East African news sites
        cloudflare_domains = [
            'nation.co.ke',
            'standardmedia.co.ke',
            'monitor.co.ug',
            'theeastafrican.co.ke',
            'addisstandard.com'
        ]

        return any(cf_domain in domain for cf_domain in cloudflare_domains)

    def _get_enhanced_stealth_level(self, domain: str, is_cloudflare: bool) -> str:
        """Determine enhanced stealth level for domain."""
        # Ethiopian sites with advanced Cloudflare protection
        ethiopian_advanced_sites = ['addisstandard.com']

        if any(eth_site in domain for eth_site in ethiopian_advanced_sites):
            return "ethiopian_advanced"
        elif is_cloudflare:
            return "cloudflare"

        # High protection domains
        high_protection_domains = [
            'nation.co.ke', 'standardmedia.co.ke', 'monitor.co.ug',
            'theeastafrican.co.ke', 'newtimes.co.rw'
        ]

        if any(protected in domain for protected in high_protection_domains):
            return "maximum"
        elif any(tld in domain for tld in ['.co.ke', '.co.ug', '.et', '.so', '.rw']):
            return "high"
        else:
            return "standard"

    async def _get_or_create_session(self, domain: str) -> str:
        """Get existing session or create new one for domain."""
        # Try to get best existing session
        existing_session = self.session_manager.get_best_session(domain)

        if existing_session:
            self.active_sessions[domain] = existing_session
            return existing_session

        # Create new session
        new_session = self.session_manager.create_session(domain)
        self.active_sessions[domain] = new_session
        self.metrics['session_rotations'] += 1

        return new_session

    def _generate_enhanced_js_protection(self, domain: str, is_cloudflare: bool) -> Dict[str, str]:
        """Generate enhanced JavaScript protection scripts."""
        try:
            from .lightweight_stealth import EnhancedJavaScriptStealth
        except ImportError:
            from lightweight_stealth import EnhancedJavaScriptStealth

        js_stealth = EnhancedJavaScriptStealth(self.browser_config)

        scripts = {
            'canvas_protection': js_stealth._generate_enhanced_canvas_protection(),
            'webgl_protection': js_stealth._generate_enhanced_webgl_protection(),
            'timing_protection': js_stealth._generate_enhanced_timing_protection(),
            'browser_protection': js_stealth._generate_advanced_browser_protection(),
            'font_protection': js_stealth._generate_font_fingerprinting_protection(),
            'audio_protection': js_stealth._generate_audio_fingerprinting_protection()
        }

        # Add Cloudflare-specific protection
        if is_cloudflare:
            scripts['cloudflare_solver'] = js_stealth._generate_cloudflare_challenge_solver()

        # Add Ethiopian advanced site protection
        if 'addisstandard.com' in domain:
            scripts['ethiopian_advanced'] = js_stealth._generate_ethiopian_advanced_protection()
            scripts['managed_challenge'] = js_stealth._generate_managed_challenge_solver()
            scripts['token_handler'] = js_stealth._generate_token_handler_protection()

        return scripts

    def _generate_tls_config(self, is_cloudflare: bool) -> Dict[str, Any]:
        """Generate TLS configuration for fingerprinting bypass."""
        if is_cloudflare:
            return {
                "cipher_suite_blacklist": ["0x0035", "0x0084", "0x009c", "0x009d"],
                "ssl_version_min": "tls1.2",
                "tls13_variant": "draft28",
                "enable_tls_bypass": True
            }

        return {
            "ssl_version_min": "tls1.2",
            "enable_tls_bypass": False
        }

    def _generate_browser_arguments(self, is_cloudflare: bool) -> List[str]:
        """Generate browser arguments for enhanced stealth."""
        args = [
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows",
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--disable-ipc-flooding-protection",
            "--disable-features=VizDisplayCompositor",
            "--disable-extensions-except=",
            "--disable-extensions",
            f"--user-agent={random.choice(self.header_manager.user_agents)}"
        ]

        if is_cloudflare:
            # Additional arguments for Cloudflare bypass
            args.extend([
                "--cipher-suite-blacklist=0x0035,0x0084,0x009c,0x009d",
                "--ssl-version-min=tls1.2",
                "--tls13-variant=draft28",
                "--disable-features=VizDisplayCompositor,CalculateNativeWinOcclusion",
                "--enable-features=NetworkService,NetworkServiceLogging"
            ])

        return args

    def _get_timing_config(self, domain: str, is_cloudflare: bool) -> Dict[str, float]:
        """Get timing configuration for domain."""
        if is_cloudflare:
            return {
                'min_delay': 8.0,
                'max_delay': 20.0,
                'challenge_wait': 15.0,
                'retry_delay': 10.0
            }
        elif domain in ['nation.co.ke', 'standardmedia.co.ke', 'monitor.co.ug']:
            return {
                'min_delay': 4.0,
                'max_delay': 10.0,
                'challenge_wait': 5.0,
                'retry_delay': 6.0
            }
        else:
            return {
                'min_delay': 1.0,
                'max_delay': 4.0,
                'challenge_wait': 3.0,
                'retry_delay': 3.0
            }

    async def _apply_standard_delay(self, domain: str, attempt: int) -> None:
        """Apply standard intelligent delay."""
        timing_config = self._get_timing_config(domain, False)

        min_delay = timing_config['min_delay']
        max_delay = timing_config['max_delay']

        # Increase delay based on attempt
        multiplier = 1.2 ** min(attempt, 5)

        # Calculate delay with jitter
        delay = min_delay * multiplier + random.uniform(0, (max_delay - min_delay) * multiplier)
        delay = min(delay, max_delay * 2)  # Cap maximum delay

        logger.debug(f"Applying standard delay of {delay:.2f}s for {domain}")
        await asyncio.sleep(delay)

    def _escalate_stealth_level(self, domain: str) -> None:
        """Escalate stealth level for problematic domain."""
        if domain in self.domain_configs:
            current_level = self.domain_configs[domain]['stealth_level']

            if current_level == 'standard':
                self.domain_configs[domain]['stealth_level'] = 'high'
            elif current_level == 'high':
                self.domain_configs[domain]['stealth_level'] = 'maximum'
            elif current_level == 'maximum':
                self.domain_configs[domain]['stealth_level'] = 'cloudflare'

            logger.info(f"Escalated stealth level for {domain} to {self.domain_configs[domain]['stealth_level']}")

    def _calculate_stealth_effectiveness(self) -> float:
        """Calculate overall stealth effectiveness score."""
        if self.metrics['stealth_activations'] == 0:
            return 0.0

        # Multi-factor effectiveness calculation
        success_factor = (self.metrics['successful_requests'] / max(1, self.metrics['total_requests'])) * 0.4
        cf_factor = (self.metrics['cloudflare_bypasses'] / max(1, self.metrics['cloudflare_challenges'])) * 0.3
        protection_factor = (self.metrics['fingerprint_protections'] / max(1, self.metrics['stealth_activations'])) * 0.2
        stability_factor = (1.0 - (self.metrics['blocked_requests'] / max(1, self.metrics['total_requests']))) * 0.1

        effectiveness = success_factor + cf_factor + protection_factor + stability_factor
        return min(100.0, effectiveness * 100.0)

    def _get_top_performing_domains(self) -> List[Tuple[str, float]]:
        """Get top performing domains by success rate."""
        domain_rates = []

        for domain, perf in self.metrics['domain_performance'].items():
            if perf['requests'] > 0:
                success_rate = (perf['successes'] / perf['requests']) * 100.0
                domain_rates.append((domain, success_rate))

        return sorted(domain_rates, key=lambda x: x[1], reverse=True)[:5]

    def _get_domain_challenge_rates(self) -> Dict[str, float]:
        """Get challenge encounter rates by domain."""
        rates = {}

        for domain, perf in self.metrics['domain_performance'].items():
            if perf['requests'] > 0:
                challenge_rate = (perf.get('challenge_encounters', 0) / perf['requests']) * 100.0
                rates[domain] = challenge_rate

        return rates


# Factory function
def create_enhanced_stealth_orchestrator(session_dir: str = "./browser_sessions") -> EnhancedStealthOrchestrator:
    """Create an enhanced stealth orchestrator instance."""
    return EnhancedStealthOrchestrator(session_dir)


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Test the enhanced stealth orchestrator."""
        orchestrator = create_enhanced_stealth_orchestrator()

        # Test URLs
        test_urls = [
            "https://nation.co.ke/kenya/news",
            "https://www.standardmedia.co.ke/kenya/article",
            "https://www.monitor.co.ug/uganda/news",
            "https://www.theeastafrican.co.ke/tea/news"
        ]

        print("Enhanced Stealth Orchestrator Test")
        print("=" * 40)

        for url in test_urls:
            print(f"\nTesting: {url}")

            # Configure stealth
            config = await orchestrator.configure_enhanced_stealth(url)

            if 'error' not in config:
                print(f"✓ Stealth Level: {config['stealth_level']}")
                print(f"✓ Cloudflare Mode: {config['is_cloudflare']}")
                print(f"✓ Headers: {len(config['headers'])} configured")
                print(f"✓ JS Scripts: {len(config['js_scripts'])} loaded")
                print(f"✓ Proxy: {'Enabled' if config['proxy_config'] else 'Disabled'}")

                # Apply timing
                await orchestrator.apply_intelligent_timing(url)

                # Simulate success (for testing)
                orchestrator.record_success(url, 1.5)

            else:
                print(f"✗ Error: {config['error']}")

        # Get final report
        report = orchestrator.get_comprehensive_report()

        print(f"\nFinal Performance Report")
        print("=" * 40)
        print(f"Total Requests: {report['total_requests']}")
        print(f"Success Rate: {report['success_rate']:.1f}%")
        print(f"Cloudflare Bypass Rate: {report['cloudflare_bypass_rate']:.1f}%")
        print(f"Stealth Effectiveness: {report['stealth_effectiveness']:.1f}%")
        print(f"Active Domains: {report['active_domains']}")

        if report['top_performing_domains']:
            print(f"\nTop Performing Domains:")
            for domain, rate in report['top_performing_domains']:
                print(f"  {domain}: {rate:.1f}%")

    asyncio.run(main())
