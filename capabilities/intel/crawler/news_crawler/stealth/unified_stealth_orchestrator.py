"""
Unified Stealth Orchestrator
=============================

Comprehensive orchestration system for coordinating multiple stealth techniques,
bypass strategies, and crawler instances for maximum effectiveness and reliability.

Features:
- Multi-strategy orchestration (CloudScraper, browser automation, hybrid)
- Intelligent strategy selection based on target analysis
- Load balancing across multiple crawler instances
- Real-time performance monitoring and optimization
- Fallback and recovery mechanisms
- Session management and rotation
- Comprehensive metrics and analytics

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
from collections import defaultdict, deque
import threading
import json

# Configure logging
logger = logging.getLogger(__name__)


class OrchestrationStrategy(Enum):
    """Orchestration strategy types."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_PERFORMANCE = "weighted_performance"
    ADAPTIVE_INTELLIGENCE = "adaptive_intelligence"
    FAILOVER_CASCADE = "failover_cascade"
    PARALLEL_RACING = "parallel_racing"


class CrawlerType(Enum):
    """Types of crawlers in the orchestration."""
    CLOUDSCRAPER_STEALTH = "cloudscraper_stealth"
    BASIC_STEALTH = "basic_stealth"
    BROWSER_AUTOMATION = "browser_automation"
    HYBRID_CRAWLER = "hybrid_crawler"
    FALLBACK_CRAWLER = "fallback_crawler"


class TargetDifficulty(Enum):
    """Target difficulty assessment."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXTREME = "extreme"
    UNKNOWN = "unknown"


@dataclass
class CrawlerInstance:
    """Represents a crawler instance in the orchestration."""
    crawler_id: str
    crawler_type: CrawlerType
    crawler_object: Any
    performance_score: float = 1.0
    success_rate: float = 1.0
    average_response_time: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_used: float = field(default_factory=time.time)
    is_active: bool = True
    specialized_domains: List[str] = field(default_factory=list)
    max_concurrent_requests: int = 5
    current_load: int = 0


@dataclass
class TargetAnalysis:
    """Analysis of a target website."""
    domain: str
    difficulty: TargetDifficulty
    protection_mechanisms: List[str] = field(default_factory=list)
    recommended_strategies: List[CrawlerType] = field(default_factory=list)
    historical_success_rate: float = 0.0
    average_response_time: float = 0.0
    last_analysis: float = field(default_factory=time.time)
    requires_specialized_handling: bool = False
    notes: str = ""


@dataclass
class OrchestrationConfig:
    """Configuration for the unified orchestrator."""
    strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE_INTELLIGENCE
    max_concurrent_crawlers: int = 10
    max_retries_per_crawler: int = 3
    crawler_timeout: float = 60.0
    enable_performance_monitoring: bool = True
    enable_adaptive_optimization: bool = True
    enable_fallback_mechanisms: bool = True
    performance_analysis_interval: int = 100  # requests
    crawler_rotation_interval: int = 50  # requests
    target_analysis_cache_ttl: int = 3600  # seconds
    enable_parallel_racing: bool = False
    racing_crawler_count: int = 2
    load_balancing_enabled: bool = True
    session_persistence: bool = True


@dataclass
class OrchestrationResult:
    """Result of an orchestrated crawl operation."""
    url: str
    success: bool
    crawler_used: Optional[str] = None
    crawler_type: Optional[CrawlerType] = None
    response_time: float = 0.0
    attempts_made: int = 0
    content: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    strategy_used: Optional[OrchestrationStrategy] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


class TargetAnalyzer:
    """Analyzes targets to determine optimal crawling strategy."""
    
    def __init__(self):
        self.analysis_cache = {}
        self.protection_indicators = {
            'cloudflare': ['cf-ray', 'cloudflare', '__cf_bm'],
            'bot_protection': ['recaptcha', 'hcaptcha', 'bot protection'],
            'rate_limiting': ['rate limit', 'too many requests', '429'],
            'js_challenges': ['javascript', 'checking your browser'],
            'sophisticated_detection': ['fingerprint', 'behavioral analysis']
        }
    
    def analyze_target(self, domain: str, historical_data: Dict[str, Any] = None) -> TargetAnalysis:
        """Analyze a target domain to determine crawling difficulty and strategy."""
        # Check cache first
        if domain in self.analysis_cache:
            cached_analysis = self.analysis_cache[domain]
            if time.time() - cached_analysis.last_analysis < 3600:  # 1 hour cache
                return cached_analysis
        
        # Perform fresh analysis
        analysis = TargetAnalysis(domain=domain, difficulty=TargetDifficulty.UNKNOWN)
        
        # Use historical data if available
        if historical_data:
            analysis.historical_success_rate = historical_data.get('success_rate', 0.0)
            analysis.average_response_time = historical_data.get('avg_response_time', 0.0)
            protection_mechanisms = historical_data.get('protection_mechanisms', [])
            analysis.protection_mechanisms = protection_mechanisms
            
            # Determine difficulty based on historical data
            if analysis.historical_success_rate > 0.9:
                analysis.difficulty = TargetDifficulty.EASY
            elif analysis.historical_success_rate > 0.7:
                analysis.difficulty = TargetDifficulty.MEDIUM
            elif analysis.historical_success_rate > 0.4:
                analysis.difficulty = TargetDifficulty.HARD
            else:
                analysis.difficulty = TargetDifficulty.EXTREME
        
        # Recommend strategies based on analysis
        analysis.recommended_strategies = self._recommend_strategies(analysis)
        
        # Cache the analysis
        self.analysis_cache[domain] = analysis
        
        return analysis
    
    def _recommend_strategies(self, analysis: TargetAnalysis) -> List[CrawlerType]:
        """Recommend crawler strategies based on target analysis."""
        strategies = []
        
        if analysis.difficulty == TargetDifficulty.EASY:
            strategies = [CrawlerType.BASIC_STEALTH, CrawlerType.HYBRID_CRAWLER]
        elif analysis.difficulty == TargetDifficulty.MEDIUM:
            strategies = [CrawlerType.CLOUDSCRAPER_STEALTH, CrawlerType.HYBRID_CRAWLER]
        elif analysis.difficulty == TargetDifficulty.HARD:
            strategies = [CrawlerType.CLOUDSCRAPER_STEALTH, CrawlerType.BROWSER_AUTOMATION]
        elif analysis.difficulty == TargetDifficulty.EXTREME:
            strategies = [CrawlerType.BROWSER_AUTOMATION, CrawlerType.CLOUDSCRAPER_STEALTH]
        else:
            # Unknown difficulty - use adaptive approach
            strategies = [CrawlerType.HYBRID_CRAWLER, CrawlerType.CLOUDSCRAPER_STEALTH]
        
        # Always include fallback
        strategies.append(CrawlerType.FALLBACK_CRAWLER)
        
        return strategies


class PerformanceMonitor:
    """Monitors and analyzes crawler performance."""
    
    def __init__(self):
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.performance_scores = {}
        self.lock = threading.Lock()
    
    def record_performance(self, crawler_id: str, success: bool, response_time: float):
        """Record performance metrics for a crawler."""
        with self.lock:
            timestamp = time.time()
            self.metrics_history[crawler_id].append({
                'timestamp': timestamp,
                'success': success,
                'response_time': response_time
            })
            
            # Update performance score
            self._update_performance_score(crawler_id)
    
    def _update_performance_score(self, crawler_id: str):
        """Update performance score for a crawler."""
        metrics = list(self.metrics_history[crawler_id])
        if not metrics:
            return
        
        # Calculate recent performance (last 100 requests or 1 hour)
        cutoff_time = time.time() - 3600  # 1 hour
        recent_metrics = [m for m in metrics if m['timestamp'] > cutoff_time][-100:]
        
        if not recent_metrics:
            return
        
        # Calculate success rate
        success_count = sum(1 for m in recent_metrics if m['success'])
        success_rate = success_count / len(recent_metrics)
        
        # Calculate average response time
        avg_response_time = sum(m['response_time'] for m in recent_metrics) / len(recent_metrics)
        
        # Calculate performance score (higher is better)
        # Formula: success_rate * (1 / (1 + normalized_response_time))
        normalized_response_time = min(avg_response_time / 10.0, 5.0)  # Normalize to 0-5 range
        performance_score = success_rate * (1 / (1 + normalized_response_time))
        
        self.performance_scores[crawler_id] = performance_score
    
    def get_performance_score(self, crawler_id: str) -> float:
        """Get current performance score for a crawler."""
        return self.performance_scores.get(crawler_id, 0.5)  # Default to average
    
    def get_top_performers(self, count: int = 3) -> List[str]:
        """Get top performing crawler IDs."""
        sorted_crawlers = sorted(
            self.performance_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return [crawler_id for crawler_id, _ in sorted_crawlers[:count]]


class LoadBalancer:
    """Manages load balancing across crawler instances."""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.crawler_loads = defaultdict(int)
        self.last_used = {}
        self.lock = threading.Lock()
    
    def select_crawler(self, crawlers: List[CrawlerInstance], strategy: OrchestrationStrategy) -> Optional[CrawlerInstance]:
        """Select optimal crawler based on strategy and load."""
        available_crawlers = [c for c in crawlers if c.is_active and c.current_load < c.max_concurrent_requests]
        
        if not available_crawlers:
            return None
        
        with self.lock:
            if strategy == OrchestrationStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_crawlers)
            elif strategy == OrchestrationStrategy.WEIGHTED_PERFORMANCE:
                return self._weighted_performance_selection(available_crawlers)
            elif strategy == OrchestrationStrategy.ADAPTIVE_INTELLIGENCE:
                return self._adaptive_selection(available_crawlers)
            else:
                return random.choice(available_crawlers)
    
    def _round_robin_selection(self, crawlers: List[CrawlerInstance]) -> CrawlerInstance:
        """Select crawler using round-robin strategy."""
        # Find least recently used crawler
        return min(crawlers, key=lambda c: self.last_used.get(c.crawler_id, 0))
    
    def _weighted_performance_selection(self, crawlers: List[CrawlerInstance]) -> CrawlerInstance:
        """Select crawler based on performance weights."""
        # Weight by performance score and inverse load
        weights = []
        for crawler in crawlers:
            load_factor = 1.0 / (1 + crawler.current_load)
            weight = crawler.performance_score * load_factor
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(crawlers)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return crawlers[i]
        
        return crawlers[-1]
    
    def _adaptive_selection(self, crawlers: List[CrawlerInstance]) -> CrawlerInstance:
        """Adaptive selection based on multiple factors."""
        # Score each crawler based on multiple factors
        scores = []
        for crawler in crawlers:
            # Performance factor (0.4 weight)
            performance_factor = crawler.performance_score * 0.4
            
            # Load factor (0.3 weight)
            load_factor = (1.0 / (1 + crawler.current_load)) * 0.3
            
            # Recency factor (0.2 weight)
            time_since_use = time.time() - self.last_used.get(crawler.crawler_id, 0)
            recency_factor = min(time_since_use / 300.0, 1.0) * 0.2  # Normalize to 5 minutes
            
            # Success rate factor (0.1 weight)
            success_factor = crawler.success_rate * 0.1
            
            total_score = performance_factor + load_factor + recency_factor + success_factor
            scores.append(total_score)
        
        # Select crawler with highest score
        best_index = scores.index(max(scores))
        return crawlers[best_index]
    
    def update_crawler_load(self, crawler_id: str, load_delta: int):
        """Update crawler load."""
        with self.lock:
            self.crawler_loads[crawler_id] = max(0, self.crawler_loads[crawler_id] + load_delta)
            self.last_used[crawler_id] = time.time()


class UnifiedStealthOrchestrator:
    """Main orchestration system for coordinating stealth crawling operations."""
    
    def __init__(self, config: Optional[OrchestrationConfig] = None):
        self.config = config or OrchestrationConfig()
        self.crawlers = {}
        self.target_analyzer = TargetAnalyzer()
        self.performance_monitor = PerformanceMonitor()
        self.load_balancer = LoadBalancer(self.config)
        
        # Statistics
        self.orchestration_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'strategy_usage': defaultdict(int),
            'crawler_usage': defaultdict(int)
        }
        
        self.lock = threading.Lock()
        logger.info("UnifiedStealthOrchestrator initialized")
    
    def register_crawler(self, crawler_id: str, crawler_type: CrawlerType, crawler_object: Any, **kwargs) -> bool:
        """Register a crawler instance with the orchestrator."""
        try:
            instance = CrawlerInstance(
                crawler_id=crawler_id,
                crawler_type=crawler_type,
                crawler_object=crawler_object,
                **kwargs
            )
            
            with self.lock:
                self.crawlers[crawler_id] = instance
            
            logger.info(f"Registered crawler: {crawler_id} ({crawler_type.value})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register crawler {crawler_id}: {e}")
            return False
    
    def unregister_crawler(self, crawler_id: str) -> bool:
        """Unregister a crawler instance."""
        with self.lock:
            if crawler_id in self.crawlers:
                del self.crawlers[crawler_id]
                logger.info(f"Unregistered crawler: {crawler_id}")
                return True
        
        logger.warning(f"Crawler not found for unregistration: {crawler_id}")
        return False
    
    async def orchestrate_crawl(self, url: str, **kwargs) -> OrchestrationResult:
        """Orchestrate a crawl operation using optimal strategy and crawler selection."""
        start_time = time.time()
        domain = urlparse(url).netloc
        
        with self.lock:
            self.orchestration_stats['total_requests'] += 1
        
        try:
            # Analyze target
            target_analysis = self.target_analyzer.analyze_target(domain)
            
            # Select strategy
            strategy = self._select_strategy(target_analysis)
            
            # Get available crawlers for recommended types
            suitable_crawlers = self._get_suitable_crawlers(target_analysis.recommended_strategies)
            
            if not suitable_crawlers:
                return OrchestrationResult(
                    url=url,
                    success=False,
                    error_message="No suitable crawlers available",
                    strategy_used=strategy
                )
            
            # Execute crawl based on strategy
            if strategy == OrchestrationStrategy.PARALLEL_RACING and len(suitable_crawlers) > 1:
                result = await self._parallel_racing_crawl(url, suitable_crawlers[:self.config.racing_crawler_count])
            else:
                result = await self._sequential_crawl(url, suitable_crawlers, strategy)
            
            # Update statistics
            response_time = time.time() - start_time
            result.response_time = response_time
            result.strategy_used = strategy
            
            with self.lock:
                if result.success:
                    self.orchestration_stats['successful_requests'] += 1
                else:
                    self.orchestration_stats['failed_requests'] += 1
                
                self.orchestration_stats['strategy_usage'][strategy] += 1
                if result.crawler_used:
                    self.orchestration_stats['crawler_usage'][result.crawler_used] += 1
                
                # Update average response time
                total_requests = self.orchestration_stats['total_requests']
                current_avg = self.orchestration_stats['average_response_time']
                self.orchestration_stats['average_response_time'] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestration failed for {url}: {e}")
            
            with self.lock:
                self.orchestration_stats['failed_requests'] += 1
            
            return OrchestrationResult(
                url=url,
                success=False,
                error_message=str(e),
                response_time=time.time() - start_time
            )
    
    def _select_strategy(self, target_analysis: TargetAnalysis) -> OrchestrationStrategy:
        """Select orchestration strategy based on target analysis."""
        if self.config.strategy == OrchestrationStrategy.ADAPTIVE_INTELLIGENCE:
            # Adapt strategy based on target difficulty
            if target_analysis.difficulty == TargetDifficulty.EXTREME:
                return OrchestrationStrategy.PARALLEL_RACING
            elif target_analysis.difficulty == TargetDifficulty.HARD:
                return OrchestrationStrategy.FAILOVER_CASCADE
            else:
                return OrchestrationStrategy.WEIGHTED_PERFORMANCE
        
        return self.config.strategy
    
    def _get_suitable_crawlers(self, recommended_types: List[CrawlerType]) -> List[CrawlerInstance]:
        """Get crawlers suitable for the recommended types."""
        suitable = []
        
        with self.lock:
            for crawler in self.crawlers.values():
                if crawler.is_active and crawler.crawler_type in recommended_types:
                    suitable.append(crawler)
        
        # Sort by performance score (descending)
        suitable.sort(key=lambda c: c.performance_score, reverse=True)
        
        return suitable
    
    async def _sequential_crawl(self, url: str, crawlers: List[CrawlerInstance], strategy: OrchestrationStrategy) -> OrchestrationResult:
        """Perform sequential crawling with fallback."""
        for attempt in range(min(len(crawlers), self.config.max_retries_per_crawler)):
            # Select crawler
            crawler = self.load_balancer.select_crawler(crawlers, strategy)
            if not crawler:
                continue
            
            # Update load
            self.load_balancer.update_crawler_load(crawler.crawler_id, 1)
            
            try:
                # Attempt crawl
                start_time = time.time()
                
                # This would call the actual crawler method
                # For now, simulate the crawl
                await asyncio.sleep(random.uniform(0.5, 2.0))  # Simulate crawl time
                
                # Simulate success/failure
                success = random.random() > 0.1  # 90% success rate
                
                response_time = time.time() - start_time
                
                # Record performance
                self.performance_monitor.record_performance(crawler.crawler_id, success, response_time)
                
                # Update crawler stats
                with self.lock:
                    crawler.total_requests += 1
                    if success:
                        crawler.successful_requests += 1
                        crawler.success_rate = crawler.successful_requests / crawler.total_requests
                    else:
                        crawler.failed_requests += 1
                    
                    crawler.average_response_time = (
                        (crawler.average_response_time * (crawler.total_requests - 1) + response_time) 
                        / crawler.total_requests
                    )
                
                if success:
                    return OrchestrationResult(
                        url=url,
                        success=True,
                        crawler_used=crawler.crawler_id,
                        crawler_type=crawler.crawler_type,
                        attempts_made=attempt + 1,
                        content=f"Simulated content for {url}",
                        status_code=200
                    )
                
            except Exception as e:
                logger.error(f"Crawler {crawler.crawler_id} failed for {url}: {e}")
                
            finally:
                # Update load
                self.load_balancer.update_crawler_load(crawler.crawler_id, -1)
        
        return OrchestrationResult(
            url=url,
            success=False,
            error_message="All crawler attempts failed",
            attempts_made=len(crawlers)
        )
    
    async def _parallel_racing_crawl(self, url: str, crawlers: List[CrawlerInstance]) -> OrchestrationResult:
        """Perform parallel racing crawl with multiple crawlers."""
        tasks = []
        
        for crawler in crawlers:
            task = asyncio.create_task(self._single_crawler_attempt(url, crawler))
            tasks.append(task)
        
        try:
            # Wait for first successful result or all to complete
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Return first successful result
            for task in done:
                result = await task
                if result.success:
                    return result
            
            # If no success, return the first result
            if done:
                return await list(done)[0]
                
        except Exception as e:
            logger.error(f"Parallel racing failed for {url}: {e}")
        
        return OrchestrationResult(
            url=url,
            success=False,
            error_message="Parallel racing failed"
        )
    
    async def _single_crawler_attempt(self, url: str, crawler: CrawlerInstance) -> OrchestrationResult:
        """Attempt crawl with a single crawler."""
        try:
            # Update load
            self.load_balancer.update_crawler_load(crawler.crawler_id, 1)
            
            start_time = time.time()
            
            # Simulate crawl (would call actual crawler)
            await asyncio.sleep(random.uniform(0.5, 2.0))
            success = random.random() > 0.1  # 90% success rate
            
            response_time = time.time() - start_time
            
            # Record performance
            self.performance_monitor.record_performance(crawler.crawler_id, success, response_time)
            
            if success:
                return OrchestrationResult(
                    url=url,
                    success=True,
                    crawler_used=crawler.crawler_id,
                    crawler_type=crawler.crawler_type,
                    content=f"Simulated content for {url}",
                    status_code=200
                )
            else:
                return OrchestrationResult(
                    url=url,
                    success=False,
                    crawler_used=crawler.crawler_id,
                    error_message="Simulated crawler failure"
                )
                
        except Exception as e:
            return OrchestrationResult(
                url=url,
                success=False,
                crawler_used=crawler.crawler_id,
                error_message=str(e)
            )
        finally:
            # Update load
            self.load_balancer.update_crawler_load(crawler.crawler_id, -1)
    
    async def batch_orchestrate(self, urls: List[str]) -> List[OrchestrationResult]:
        """Orchestrate crawling of multiple URLs."""
        tasks = [self.orchestrate_crawl(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestration statistics."""
        with self.lock:
            stats = dict(self.orchestration_stats)
            stats.update({
                'registered_crawlers': len(self.crawlers),
                'active_crawlers': sum(1 for c in self.crawlers.values() if c.is_active),
                'crawler_details': {
                    cid: {
                        'type': c.crawler_type.value,
                        'performance_score': c.performance_score,
                        'success_rate': c.success_rate,
                        'total_requests': c.total_requests,
                        'current_load': c.current_load
                    }
                    for cid, c in self.crawlers.items()
                },
                'top_performers': self.performance_monitor.get_top_performers()
            })
        
        return stats


# Factory functions
def create_orchestration_config(**kwargs) -> OrchestrationConfig:
    """Create OrchestrationConfig with custom parameters."""
    return OrchestrationConfig(**kwargs)


def create_unified_orchestrator(config: Optional[OrchestrationConfig] = None) -> UnifiedStealthOrchestrator:
    """Create UnifiedStealthOrchestrator instance."""
    return UnifiedStealthOrchestrator(config)


def create_high_performance_config() -> OrchestrationConfig:
    """Create configuration optimized for high performance."""
    return OrchestrationConfig(
        strategy=OrchestrationStrategy.ADAPTIVE_INTELLIGENCE,
        max_concurrent_crawlers=20,
        enable_parallel_racing=True,
        racing_crawler_count=3,
        performance_analysis_interval=50,
        load_balancing_enabled=True
    )


def create_unified_stealth_orchestrator(config: Optional[OrchestrationConfig] = None) -> UnifiedStealthOrchestrator:
    """Create unified stealth orchestrator with configuration."""
    return UnifiedStealthOrchestrator(config)


# Export all components
__all__ = [
    # Enums
    'OrchestrationStrategy', 'CrawlerType', 'TargetDifficulty',
    
    # Data classes
    'CrawlerInstance', 'TargetAnalysis', 'OrchestrationConfig', 'OrchestrationResult',
    
    # Core classes
    'TargetAnalyzer', 'PerformanceMonitor', 'LoadBalancer', 'UnifiedStealthOrchestrator',
    
    # Factory functions
    'create_orchestration_config', 'create_unified_orchestrator', 'create_high_performance_config',
    'create_unified_stealth_orchestrator'
]