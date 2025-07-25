"""
Digital Twin Marketplace and Ecosystem Platform

This module provides a comprehensive marketplace ecosystem for digital twins,
enabling discovery, sharing, collaboration, and monetization of digital twin
models, algorithms, and services across organizations and communities.
"""

import asyncio
import json
import logging
import uuid
import random
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("digital_twin_marketplace")

class TwinCategory(str, Enum):
	"""Categories of digital twins in the marketplace"""
	MANUFACTURING = "manufacturing"
	HEALTHCARE = "healthcare"
	SMART_CITIES = "smart_cities"
	AEROSPACE = "aerospace"
	AUTOMOTIVE = "automotive"
	ENERGY = "energy"
	AGRICULTURE = "agriculture"
	LOGISTICS = "logistics"
	CONSTRUCTION = "construction"
	RETAIL = "retail"
	FINANCE = "finance"
	EDUCATION = "education"

class MarketplaceItemType(str, Enum):
	"""Types of items in the marketplace"""
	DIGITAL_TWIN = "digital_twin"
	ALGORITHM = "algorithm"
	DATA_MODEL = "data_model"
	VISUALIZATION = "visualization"
	SIMULATION = "simulation"
	API_SERVICE = "api_service"
	TEMPLATE = "template"
	PLUGIN = "plugin"

class LicenseType(str, Enum):
	"""License types for marketplace items"""
	OPEN_SOURCE = "open_source"
	COMMERCIAL = "commercial"
	FREEMIUM = "freemium"
	SUBSCRIPTION = "subscription"
	PAY_PER_USE = "pay_per_use"
	ENTERPRISE = "enterprise"

class QualityTier(str, Enum):
	"""Quality tiers for marketplace items"""
	COMMUNITY = "community"
	VERIFIED = "verified"
	CERTIFIED = "certified"
	ENTERPRISE = "enterprise"
	PREMIUM = "premium"

@dataclass
class MarketplaceUser:
	"""User in the digital twin marketplace"""
	user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	username: str = ""
	email: str = ""
	organization: str = ""
	user_type: str = "individual"  # individual, organization, enterprise
	reputation_score: float = 0.0
	contributions: int = 0
	downloads: int = 0
	created_at: datetime = field(default_factory=datetime.utcnow)
	verified: bool = False
	subscription_tier: str = "basic"
	api_key: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class MarketplaceItem:
	"""Item in the digital twin marketplace"""
	item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	name: str = ""
	description: str = ""
	category: TwinCategory = TwinCategory.MANUFACTURING
	item_type: MarketplaceItemType = MarketplaceItemType.DIGITAL_TWIN
	license_type: LicenseType = LicenseType.OPEN_SOURCE
	quality_tier: QualityTier = QualityTier.COMMUNITY
	
	# Metadata
	version: str = "1.0.0"
	author_id: str = ""
	organization: str = ""
	tags: List[str] = field(default_factory=list)
	keywords: List[str] = field(default_factory=list)
	
	# Content
	source_code_url: str = ""
	documentation_url: str = ""
	demo_url: str = ""
	api_endpoint: str = ""
	model_file_url: str = ""
	
	# Metrics
	downloads: int = 0
	rating: float = 0.0
	reviews_count: int = 0
	popularity_score: float = 0.0
	last_updated: datetime = field(default_factory=datetime.utcnow)
	created_at: datetime = field(default_factory=datetime.utcnow)
	
	# Pricing
	price: float = 0.0
	currency: str = "USD"
	pricing_model: str = "free"  # free, one_time, subscription, pay_per_use
	
	# Technical specifications
	requirements: Dict[str, Any] = field(default_factory=dict)
	compatibility: List[str] = field(default_factory=list)
	performance_metrics: Dict[str, float] = field(default_factory=dict)
	
	# Quality assurance
	tested: bool = False
	certified: bool = False
	security_scanned: bool = False
	performance_benchmarked: bool = False

@dataclass
class MarketplaceReview:
	"""Review for marketplace items"""
	review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	item_id: str = ""
	user_id: str = ""
	rating: int = 5  # 1-5 stars
	title: str = ""
	content: str = ""
	helpful_votes: int = 0
	verified_purchase: bool = False
	created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class MarketplaceTransaction:
	"""Transaction record in the marketplace"""
	transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
	item_id: str = ""
	buyer_id: str = ""
	seller_id: str = ""
	amount: float = 0.0
	currency: str = "USD"
	transaction_type: str = "purchase"  # purchase, subscription, rental
	status: str = "completed"  # pending, completed, failed, refunded
	created_at: datetime = field(default_factory=datetime.utcnow)
	metadata: Dict[str, Any] = field(default_factory=dict)

class DigitalTwinMarketplace:
	"""Main marketplace platform for digital twins"""
	
	def __init__(self):
		self.users: Dict[str, MarketplaceUser] = {}
		self.items: Dict[str, MarketplaceItem] = {}
		self.reviews: Dict[str, List[MarketplaceReview]] = {}
		self.transactions: List[MarketplaceTransaction] = []
		self.categories: Dict[TwinCategory, List[str]] = {}
		
		# Search and recommendation engine
		self.search_index: Dict[str, Set[str]] = {}
		self.recommendation_engine = RecommendationEngine()
		
		# Analytics and metrics
		self.marketplace_metrics = {
			"total_items": 0,
			"total_users": 0,
			"total_downloads": 0,
			"total_revenue": 0.0,
			"average_rating": 0.0,
			"active_contributors": 0,
			"daily_active_users": 0
		}
		
		# Quality assurance
		self.quality_checker = QualityAssuranceEngine()
		
		# Initialize with sample data
		asyncio.create_task(self._initialize_sample_data())
		
		logger.info("Digital Twin Marketplace initialized")
	
	async def _initialize_sample_data(self):
		"""Initialize marketplace with sample digital twins and users"""
		
		# Create sample users
		sample_users = [
			{
				"username": "factory_innovator",
				"email": "innovator@smartfactory.com",
				"organization": "Smart Factory Solutions",
				"user_type": "organization",
				"reputation_score": 4.8,
				"contributions": 12,
				"verified": True,
				"subscription_tier": "premium"
			},
			{
				"username": "healthcare_researcher",
				"email": "researcher@medtech.org",
				"organization": "MedTech Research Institute",
				"user_type": "organization",
				"reputation_score": 4.9,
				"contributions": 8,
				"verified": True,
				"subscription_tier": "enterprise"
			},
			{
				"username": "quantum_dev",
				"email": "dev@quantumtwins.ai",
				"organization": "Quantum Twins AI",
				"user_type": "enterprise",
				"reputation_score": 4.7,
				"contributions": 15,
				"verified": True,
				"subscription_tier": "enterprise"
			},
			{
				"username": "open_source_contributor",
				"email": "contributor@opensource.dev",
				"organization": "Open Source Digital Twins",
				"user_type": "individual",
				"reputation_score": 4.5,
				"contributions": 25,
				"verified": True,
				"subscription_tier": "basic"
			}
		]
		
		for user_data in sample_users:
			user = MarketplaceUser(**user_data)
			await self.register_user(user)
		
		# Create sample digital twin items
		sample_items = [
			{
				"name": "Smart Factory Production Line Twin",
				"description": "Complete digital twin for automotive production line with real-time monitoring, predictive maintenance, and optimization algorithms.",
				"category": TwinCategory.MANUFACTURING,
				"item_type": MarketplaceItemType.DIGITAL_TWIN,
				"license_type": LicenseType.COMMERCIAL,
				"quality_tier": QualityTier.CERTIFIED,
				"version": "2.1.0",
				"author_id": list(self.users.keys())[0],
				"tags": ["production", "automotive", "predictive-maintenance", "optimization"],
				"keywords": ["factory", "assembly", "efficiency", "quality"],
				"downloads": 1250,
				"rating": 4.8,
				"reviews_count": 47,
				"price": 2999.0,
				"pricing_model": "one_time",
				"requirements": {
					"cpu_cores": 8,
					"memory_gb": 16,
					"storage_gb": 100,
					"python_version": ">=3.8"
				},
				"compatibility": ["Python", "Docker", "Kubernetes"],
				"performance_metrics": {
					"accuracy": 0.96,
					"latency_ms": 15.2,
					"throughput_rps": 850
				},
				"tested": True,
				"certified": True,
				"security_scanned": True,
				"performance_benchmarked": True
			},
			{
				"name": "Hospital Patient Flow Optimizer",
				"description": "Digital twin for optimizing patient flow in hospitals, reducing wait times and improving resource utilization.",
				"category": TwinCategory.HEALTHCARE,
				"item_type": MarketplaceItemType.DIGITAL_TWIN,
				"license_type": LicenseType.SUBSCRIPTION,
				"quality_tier": QualityTier.ENTERPRISE,
				"version": "1.5.2",
				"author_id": list(self.users.keys())[1],
				"tags": ["healthcare", "patient-flow", "optimization", "resource-management"],
				"keywords": ["hospital", "queue", "efficiency", "capacity"],
				"downloads": 180,
				"rating": 4.9,
				"reviews_count": 23,
				"price": 499.0,
				"pricing_model": "subscription",
				"requirements": {
					"cpu_cores": 4,
					"memory_gb": 8,
					"storage_gb": 50,
					"hipaa_compliant": True
				},
				"compatibility": ["Python", "R", "AWS", "Azure"],
				"performance_metrics": {
					"wait_time_reduction": 0.35,
					"resource_utilization": 0.87,
					"patient_satisfaction": 0.92
				},
				"tested": True,
				"certified": True,
				"security_scanned": True,
				"performance_benchmarked": True
			},
			{
				"name": "Quantum Molecular Simulation Engine",
				"description": "Quantum-enhanced digital twin for molecular simulation and drug discovery applications.",
				"category": TwinCategory.HEALTHCARE,
				"item_type": MarketplaceItemType.ALGORITHM,
				"license_type": LicenseType.ENTERPRISE,
				"quality_tier": QualityTier.PREMIUM,
				"version": "3.0.0",
				"author_id": list(self.users.keys())[2],
				"tags": ["quantum", "molecular", "simulation", "drug-discovery"],
				"keywords": ["quantum", "chemistry", "pharmaceuticals", "modeling"],
				"downloads": 85,
				"rating": 4.7,
				"reviews_count": 12,
				"price": 15000.0,
				"pricing_model": "one_time",
				"requirements": {
					"quantum_backend": True,
					"cpu_cores": 16,
					"memory_gb": 64,
					"gpu_required": True
				},
				"compatibility": ["Qiskit", "Cirq", "Python", "CUDA"],
				"performance_metrics": {
					"quantum_advantage": 2.8,
					"simulation_accuracy": 0.98,
					"convergence_rate": 0.89
				},
				"tested": True,
				"certified": False,
				"security_scanned": True,
				"performance_benchmarked": True
			},
			{
				"name": "Smart City Traffic Management",
				"description": "Open-source digital twin for urban traffic optimization and smart city planning.",
				"category": TwinCategory.SMART_CITIES,
				"item_type": MarketplaceItemType.DIGITAL_TWIN,
				"license_type": LicenseType.OPEN_SOURCE,
				"quality_tier": QualityTier.VERIFIED,
				"version": "1.8.1",
				"author_id": list(self.users.keys())[3],
				"tags": ["traffic", "smart-city", "optimization", "open-source"],
				"keywords": ["urban", "transportation", "planning", "sustainability"],
				"downloads": 3200,
				"rating": 4.5,
				"reviews_count": 156,
				"price": 0.0,
				"pricing_model": "free",
				"requirements": {
					"cpu_cores": 4,
					"memory_gb": 8,
					"storage_gb": 20,
					"python_version": ">=3.7"
				},
				"compatibility": ["Python", "Docker", "PostgreSQL", "Redis"],
				"performance_metrics": {
					"traffic_reduction": 0.28,
					"fuel_savings": 0.22,
					"emissions_reduction": 0.31
				},
				"tested": True,
				"certified": False,
				"security_scanned": True,
				"performance_benchmarked": False
			},
			{
				"name": "Edge Computing Optimization Suite",
				"description": "Comprehensive suite for optimizing edge computing deployments in digital twin architectures.",
				"category": TwinCategory.MANUFACTURING,
				"item_type": MarketplaceItemType.PLUGIN,
				"license_type": LicenseType.FREEMIUM,
				"quality_tier": QualityTier.VERIFIED,
				"version": "2.3.0",
				"author_id": list(self.users.keys())[0],
				"tags": ["edge-computing", "optimization", "latency", "performance"],
				"keywords": ["edge", "real-time", "distributed", "iot"],
				"downloads": 890,
				"rating": 4.6,
				"reviews_count": 34,
				"price": 199.0,
				"pricing_model": "freemium",
				"requirements": {
					"edge_nodes": True,
					"kubernetes": True,
					"docker": True
				},
				"compatibility": ["Kubernetes", "Docker", "AWS IoT", "Azure IoT"],
				"performance_metrics": {
					"latency_reduction": 0.65,
					"bandwidth_savings": 0.42,
					"cost_reduction": 0.35
				},
				"tested": True,
				"certified": False,
				"security_scanned": True,
				"performance_benchmarked": True
			}
		]
		
		for item_data in sample_items:
			item = MarketplaceItem(**item_data)
			await self.publish_item(item)
		
		# Generate sample reviews
		await self._generate_sample_reviews()
		
		logger.info(f"Initialized marketplace with {len(self.users)} users and {len(self.items)} items")
	
	async def _generate_sample_reviews(self):
		"""Generate sample reviews for marketplace items"""
		
		review_templates = [
			{
				"rating": 5,
				"title": "Excellent digital twin solution!",
				"content": "This digital twin exceeded our expectations. Easy to integrate, well-documented, and provides significant value. Highly recommended for production environments."
			},
			{
				"rating": 4,
				"title": "Great performance, minor issues",
				"content": "Overall very satisfied with this solution. Performance is excellent and it integrates well with our existing systems. Some minor documentation gaps but support is responsive."
			},
			{
				"rating": 5,
				"title": "Game-changer for our operations",
				"content": "This digital twin completely transformed our operational efficiency. The predictive capabilities are outstanding and the ROI was achieved within 6 months."
			},
			{
				"rating": 4,
				"title": "Solid implementation",
				"content": "Well-architected solution with good performance characteristics. The visualization features are particularly impressive. Would benefit from more customization options."
			},
			{
				"rating": 5,
				"title": "Best-in-class quality",
				"content": "Professional-grade digital twin with enterprise features. Excellent security, scalability, and reliability. Worth every penny for serious applications."
			}
		]
		
		user_ids = list(self.users.keys())
		
		for item_id in self.items:
			num_reviews = random.randint(2, 5)
			
			for _ in range(num_reviews):
				template = random.choice(review_templates)
				review = MarketplaceReview(
					item_id=item_id,
					user_id=random.choice(user_ids),
					rating=template["rating"],
					title=template["title"],
					content=template["content"],
					helpful_votes=random.randint(0, 25),
					verified_purchase=random.choice([True, False])
				)
				
				await self.add_review(review)
	
	async def register_user(self, user: MarketplaceUser) -> bool:
		"""Register a new user in the marketplace"""
		
		if user.user_id in self.users:
			logger.warning(f"User {user.user_id} already exists")
			return False
		
		self.users[user.user_id] = user
		self.marketplace_metrics["total_users"] += 1
		
		logger.info(f"Registered user {user.username} ({user.organization})")
		return True
	
	async def publish_item(self, item: MarketplaceItem) -> bool:
		"""Publish an item to the marketplace"""
		
		# Quality check
		quality_result = await self.quality_checker.assess_item(item)
		if not quality_result["approved"]:
			logger.warning(f"Item {item.name} failed quality check: {quality_result['issues']}")
			return False
		
		# Update item with quality assessment
		item.tested = quality_result.get("tested", False)
		item.security_scanned = quality_result.get("security_scanned", False)
		item.performance_benchmarked = quality_result.get("performance_benchmarked", False)
		
		# Add to marketplace
		self.items[item.item_id] = item
		self.marketplace_metrics["total_items"] += 1
		
		# Update search index
		await self._update_search_index(item)
		
		# Update category index
		if item.category not in self.categories:
			self.categories[item.category] = []
		self.categories[item.category].append(item.item_id)
		
		logger.info(f"Published item {item.name} by {item.organization}")
		return True
	
	async def _update_search_index(self, item: MarketplaceItem):
		"""Update search index with item keywords"""
		
		searchable_terms = (
			[item.name.lower()] +
			item.tags +
			item.keywords +
			[item.category.value] +
			[item.item_type.value] +
			item.description.lower().split()[:10]  # First 10 words of description
		)
		
		for term in searchable_terms:
			if term not in self.search_index:
				self.search_index[term] = set()
			self.search_index[term].add(item.item_id)
	
	async def search_items(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[MarketplaceItem]:
		"""Search for items in the marketplace"""
		
		query_terms = query.lower().split()
		matching_items = set()
		
		# Find items matching query terms
		for term in query_terms:
			if term in self.search_index:
				matching_items.update(self.search_index[term])
		
		# If no direct matches, try partial matching
		if not matching_items:
			for indexed_term, item_ids in self.search_index.items():
				for query_term in query_terms:
					if query_term in indexed_term or indexed_term in query_term:
						matching_items.update(item_ids)
		
		# Apply filters
		if filters:
			filtered_items = set()
			for item_id in matching_items:
				item = self.items[item_id]
				
				# Category filter
				if "category" in filters and item.category.value != filters["category"]:
					continue
				
				# Item type filter
				if "item_type" in filters and item.item_type.value != filters["item_type"]:
					continue
				
				# Price range filter
				if "min_price" in filters and item.price < filters["min_price"]:
					continue
				if "max_price" in filters and item.price > filters["max_price"]:
					continue
				
				# Rating filter
				if "min_rating" in filters and item.rating < filters["min_rating"]:
					continue
				
				# License type filter
				if "license_type" in filters and item.license_type.value != filters["license_type"]:
					continue
				
				filtered_items.add(item_id)
			
			matching_items = filtered_items
		
		# Return sorted results
		results = [self.items[item_id] for item_id in matching_items]
		results.sort(key=lambda x: (x.popularity_score, x.rating, x.downloads), reverse=True)
		
		return results
	
	async def get_recommendations(self, user_id: str, num_recommendations: int = 5) -> List[MarketplaceItem]:
		"""Get personalized recommendations for a user"""
		
		if user_id not in self.users:
			# Return popular items for anonymous users
			return await self.get_popular_items(num_recommendations)
		
		return await self.recommendation_engine.get_recommendations(
			user_id, self.users, self.items, self.transactions, num_recommendations
		)
	
	async def get_popular_items(self, limit: int = 10) -> List[MarketplaceItem]:
		"""Get most popular items in the marketplace"""
		
		items = list(self.items.values())
		items.sort(key=lambda x: (x.popularity_score, x.downloads, x.rating), reverse=True)
		
		return items[:limit]
	
	async def get_trending_items(self, days: int = 7, limit: int = 10) -> List[MarketplaceItem]:
		"""Get trending items based on recent activity"""
		
		# In a real implementation, this would analyze recent downloads/purchases
		# For simulation, we'll return items with recent updates
		
		cutoff_date = datetime.utcnow() - timedelta(days=days)
		recent_items = [
			item for item in self.items.values()
			if item.last_updated >= cutoff_date
		]
		
		recent_items.sort(key=lambda x: (x.downloads / max(1, (datetime.utcnow() - x.created_at).days)), reverse=True)
		
		return recent_items[:limit]
	
	async def add_review(self, review: MarketplaceReview) -> bool:
		"""Add a review for an item"""
		
		if review.item_id not in self.items:
			logger.warning(f"Cannot review non-existent item {review.item_id}")
			return False
		
		if review.user_id not in self.users:
			logger.warning(f"Cannot add review from non-existent user {review.user_id}")
			return False
		
		# Add review
		if review.item_id not in self.reviews:
			self.reviews[review.item_id] = []
		
		self.reviews[review.item_id].append(review)
		
		# Update item rating
		item = self.items[review.item_id]
		item.reviews_count += 1
		
		# Recalculate average rating
		all_ratings = [r.rating for r in self.reviews[review.item_id]]
		item.rating = sum(all_ratings) / len(all_ratings)
		
		# Update popularity score
		item.popularity_score = self._calculate_popularity_score(item)
		
		logger.info(f"Added review for {item.name}: {review.rating}/5 stars")
		return True
	
	def _calculate_popularity_score(self, item: MarketplaceItem) -> float:
		"""Calculate popularity score for an item"""
		
		# Weighted formula considering downloads, rating, and recency
		download_score = min(item.downloads / 1000, 10)  # Cap at 10
		rating_score = item.rating * 2  # 0-10 scale
		recency_score = max(0, 10 - (datetime.utcnow() - item.last_updated).days / 30)  # Decay over months
		
		return (download_score * 0.4 + rating_score * 0.4 + recency_score * 0.2)
	
	async def purchase_item(self, buyer_id: str, item_id: str) -> Dict[str, Any]:
		"""Purchase an item from the marketplace"""
		
		if buyer_id not in self.users:
			return {"success": False, "error": "User not found"}
		
		if item_id not in self.items:
			return {"success": False, "error": "Item not found"}
		
		item = self.items[item_id]
		buyer = self.users[buyer_id]
		
		# Create transaction
		transaction = MarketplaceTransaction(
			item_id=item_id,
			buyer_id=buyer_id,
			seller_id=item.author_id,
			amount=item.price,
			currency=item.currency,
			transaction_type="purchase",
			status="completed"
		)
		
		self.transactions.append(transaction)
		
		# Update metrics
		item.downloads += 1
		self.marketplace_metrics["total_downloads"] += 1
		self.marketplace_metrics["total_revenue"] += item.price
		
		# Update popularity
		item.popularity_score = self._calculate_popularity_score(item)
		
		logger.info(f"User {buyer.username} purchased {item.name} for {item.price} {item.currency}")
		
		return {
			"success": True,
			"transaction_id": transaction.transaction_id,
			"item": {
				"name": item.name,
				"version": item.version,
				"download_url": f"https://marketplace.api/download/{item.item_id}/{transaction.transaction_id}",
				"api_key": buyer.api_key,
				"license_key": self._generate_license_key(transaction)
			}
		}
	
	def _generate_license_key(self, transaction: MarketplaceTransaction) -> str:
		"""Generate a license key for a transaction"""
		
		key_data = f"{transaction.transaction_id}:{transaction.item_id}:{transaction.buyer_id}:{datetime.utcnow().timestamp()}"
		key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:32]
		
		return f"DTMP-{key_hash[:8]}-{key_hash[8:16]}-{key_hash[16:24]}-{key_hash[24:32]}"
	
	async def get_marketplace_analytics(self) -> Dict[str, Any]:
		"""Get comprehensive marketplace analytics"""
		
		# Category distribution
		category_stats = {}
		for category, item_ids in self.categories.items():
			category_stats[category.value] = {
				"item_count": len(item_ids),
				"total_downloads": sum(self.items[item_id].downloads for item_id in item_ids),
				"avg_rating": sum(self.items[item_id].rating for item_id in item_ids) / len(item_ids) if item_ids else 0
			}
		
		# Revenue by item type
		revenue_by_type = {}
		for item in self.items.values():
			item_type = item.item_type.value
			if item_type not in revenue_by_type:
				revenue_by_type[item_type] = {"items": 0, "revenue": 0.0, "downloads": 0}
			
			revenue_by_type[item_type]["items"] += 1
			revenue_by_type[item_type]["revenue"] += item.price * item.downloads
			revenue_by_type[item_type]["downloads"] += item.downloads
		
		# Top contributors
		contributor_stats = {}
		for item in self.items.values():
			author_id = item.author_id
			if author_id in self.users:
				username = self.users[author_id].username
				if username not in contributor_stats:
					contributor_stats[username] = {"items": 0, "downloads": 0, "revenue": 0.0}
				
				contributor_stats[username]["items"] += 1
				contributor_stats[username]["downloads"] += item.downloads
				contributor_stats[username]["revenue"] += item.price * item.downloads
		
		# Quality metrics
		quality_stats = {
			"tested_items": sum(1 for item in self.items.values() if item.tested),
			"certified_items": sum(1 for item in self.items.values() if item.certified),
			"security_scanned": sum(1 for item in self.items.values() if item.security_scanned),
			"avg_rating": sum(item.rating for item in self.items.values()) / len(self.items) if self.items else 0
		}
		
		return {
			"overview": self.marketplace_metrics,
			"category_distribution": category_stats,
			"revenue_by_type": revenue_by_type,
			"top_contributors": dict(sorted(contributor_stats.items(), 
										   key=lambda x: x[1]["downloads"], reverse=True)[:10]),
			"quality_metrics": quality_stats,
			"growth_metrics": {
				"items_this_month": len([item for item in self.items.values() 
										if (datetime.utcnow() - item.created_at).days <= 30]),
				"new_users_this_month": len([user for user in self.users.values() 
											if (datetime.utcnow() - user.created_at).days <= 30]),
				"transactions_this_month": len([tx for tx in self.transactions 
											   if (datetime.utcnow() - tx.created_at).days <= 30])
			}
		}

class RecommendationEngine:
	"""AI-powered recommendation engine for marketplace items"""
	
	async def get_recommendations(self, user_id: str, users: Dict, items: Dict, 
								  transactions: List, num_recommendations: int = 5) -> List[MarketplaceItem]:
		"""Generate personalized recommendations using collaborative filtering"""
		
		user = users[user_id]
		
		# Get user's purchase history
		user_purchases = set(tx.item_id for tx in transactions if tx.buyer_id == user_id)
		
		# Find similar users based on organization and interests
		similar_users = []
		for other_user_id, other_user in users.items():
			if other_user_id == user_id:
				continue
			
			similarity_score = self._calculate_user_similarity(user, other_user)
			if similarity_score > 0.3:  # Threshold for similarity
				similar_users.append((other_user_id, similarity_score))
		
		similar_users.sort(key=lambda x: x[1], reverse=True)
		
		# Collect items purchased by similar users
		recommended_items = {}
		for similar_user_id, similarity in similar_users[:10]:  # Top 10 similar users
			similar_purchases = set(tx.item_id for tx in transactions if tx.buyer_id == similar_user_id)
			
			# Recommend items not already purchased by the user
			for item_id in similar_purchases - user_purchases:
				if item_id in items:
					if item_id not in recommended_items:
						recommended_items[item_id] = 0
					recommended_items[item_id] += similarity
		
		# Sort by recommendation score and return top items
		sorted_recommendations = sorted(recommended_items.items(), key=lambda x: x[1], reverse=True)
		
		result = []
		for item_id, score in sorted_recommendations[:num_recommendations]:
			result.append(items[item_id])
		
		# Fill remaining slots with popular items if needed
		if len(result) < num_recommendations:
			popular_items = sorted(items.values(), key=lambda x: x.popularity_score, reverse=True)
			for item in popular_items:
				if len(result) >= num_recommendations:
					break
				if item.item_id not in user_purchases and item not in result:
					result.append(item)
		
		return result[:num_recommendations]
	
	def _calculate_user_similarity(self, user1: MarketplaceUser, user2: MarketplaceUser) -> float:
		"""Calculate similarity between two users"""
		
		similarity = 0.0
		
		# Organization similarity
		if user1.organization == user2.organization:
			similarity += 0.5
		
		# User type similarity
		if user1.user_type == user2.user_type:
			similarity += 0.2
		
		# Subscription tier similarity
		if user1.subscription_tier == user2.subscription_tier:
			similarity += 0.1
		
		# Reputation similarity (closer scores = higher similarity)
		reputation_diff = abs(user1.reputation_score - user2.reputation_score)
		reputation_similarity = max(0, 1 - reputation_diff / 5)  # Max difference of 5
		similarity += reputation_similarity * 0.2
		
		return min(similarity, 1.0)

class QualityAssuranceEngine:
	"""Quality assurance engine for marketplace items"""
	
	async def assess_item(self, item: MarketplaceItem) -> Dict[str, Any]:
		"""Perform quality assessment on a marketplace item"""
		
		issues = []
		approved = True
		
		# Basic validation
		if len(item.name) < 5:
			issues.append("Item name too short")
			approved = False
		
		if len(item.description) < 50:
			issues.append("Description too brief")
			approved = False
		
		if not item.tags:
			issues.append("No tags provided")
		
		if not item.keywords:
			issues.append("No keywords provided")
		
		# Content validation
		if item.license_type == LicenseType.COMMERCIAL and item.price <= 0:
			issues.append("Commercial items must have a price > 0")
			approved = False
		
		if item.quality_tier in [QualityTier.CERTIFIED, QualityTier.ENTERPRISE] and not item.author_id:
			issues.append("High-tier items require verified authors")
			approved = False
		
		# Simulate additional checks
		await asyncio.sleep(0.1)  # Simulate processing time
		
		# Assign quality scores
		security_scanned = random.choice([True, True, False])  # 67% pass rate
		tested = random.choice([True, True, True, False])  # 75% pass rate
		performance_benchmarked = random.choice([True, False])  # 50% pass rate
		
		if not security_scanned:
			issues.append("Security scan failed")
		
		if not tested:
			issues.append("Automated tests failed")
		
		return {
			"approved": approved,
			"issues": issues,
			"security_scanned": security_scanned,
			"tested": tested,
			"performance_benchmarked": performance_benchmarked,
			"quality_score": max(0, 100 - len(issues) * 10)
		}

# Example usage and demonstration
async def demonstrate_digital_twin_marketplace():
	"""Demonstrate digital twin marketplace capabilities"""
	
	print("🏪 DIGITAL TWIN MARKETPLACE DEMONSTRATION")
	print("=" * 55)
	
	# Create marketplace
	marketplace = DigitalTwinMarketplace()
	
	# Wait for initialization
	await asyncio.sleep(0.5)
	
	print(f"\n📊 Marketplace Overview:")
	analytics = await marketplace.get_marketplace_analytics()
	print(f"   Total Items: {analytics['overview']['total_items']}")
	print(f"   Total Users: {analytics['overview']['total_users']}")
	print(f"   Total Downloads: {analytics['overview']['total_downloads']}")
	print(f"   Average Rating: {analytics['quality_metrics']['avg_rating']:.1f}/5.0")
	
	# Demonstrate search functionality
	print(f"\n🔍 Search Results for 'factory optimization':")
	search_results = await marketplace.search_items("factory optimization")
	for i, item in enumerate(search_results[:3], 1):
		print(f"   {i}. {item.name}")
		print(f"      Category: {item.category.value}")
		print(f"      Rating: {item.rating:.1f}/5 ({item.reviews_count} reviews)")
		print(f"      Price: ${item.price:.2f} ({item.pricing_model})")
		print(f"      Downloads: {item.downloads:,}")
	
	# Demonstrate filtering
	print(f"\n🔽 Filtered Search (Healthcare, Free items):")
	filtered_results = await marketplace.search_items("", {
		"category": "healthcare",
		"max_price": 0
	})
	for item in filtered_results:
		print(f"   • {item.name} - {item.license_type.value}")
	
	# Demonstrate recommendations
	print(f"\n🎯 Personalized Recommendations:")
	user_id = list(marketplace.users.keys())[0]
	username = marketplace.users[user_id].username
	recommendations = await marketplace.get_recommendations(user_id)
	print(f"   For user: {username}")
	for i, item in enumerate(recommendations[:3], 1):
		print(f"   {i}. {item.name}")
		print(f"      Reason: Popular in {item.category.value}")
		print(f"      Rating: {item.rating:.1f}/5")
	
	# Demonstrate popular and trending items
	print(f"\n📈 Popular Items:")
	popular = await marketplace.get_popular_items(3)
	for i, item in enumerate(popular, 1):
		print(f"   {i}. {item.name} ({item.downloads:,} downloads)")
	
	print(f"\n🔥 Trending Items (Last 7 days):")
	trending = await marketplace.get_trending_items(7, 3)
	for i, item in enumerate(trending, 1):
		print(f"   {i}. {item.name} (v{item.version})")
	
	# Demonstrate purchase flow
	print(f"\n💳 Purchase Simulation:")
	buyer_id = list(marketplace.users.keys())[1]
	item_id = list(marketplace.items.keys())[0]
	
	purchase_result = await marketplace.purchase_item(buyer_id, item_id)
	if purchase_result["success"]:
		print(f"   ✅ Purchase successful!")
		print(f"   Transaction ID: {purchase_result['transaction_id']}")
		print(f"   License Key: {purchase_result['item']['license_key']}")
		print(f"   Download URL: {purchase_result['item']['download_url']}")
	
	# Show comprehensive analytics
	print(f"\n📈 Marketplace Analytics:")
	analytics = await marketplace.get_marketplace_analytics()
	
	print(f"   Category Distribution:")
	for category, stats in list(analytics['category_distribution'].items())[:3]:
		print(f"     {category.title()}: {stats['item_count']} items, {stats['total_downloads']:,} downloads")
	
	print(f"   Top Contributors:")
	for contributor, stats in list(analytics['top_contributors'].items())[:3]:
		print(f"     {contributor}: {stats['items']} items, ${stats['revenue']:,.2f} revenue")
	
	print(f"   Quality Metrics:")
	quality = analytics['quality_metrics']
	print(f"     Tested Items: {quality['tested_items']}/{analytics['overview']['total_items']}")
	print(f"     Certified Items: {quality['certified_items']}/{analytics['overview']['total_items']}")
	print(f"     Security Scanned: {quality['security_scanned']}/{analytics['overview']['total_items']}")
	
	print(f"   Growth Metrics:")
	growth = analytics['growth_metrics']
	print(f"     New Items (30 days): {growth['items_this_month']}")
	print(f"     New Users (30 days): {growth['new_users_this_month']}")
	print(f"     Transactions (30 days): {growth['transactions_this_month']}")
	
	print(f"\n✅ Digital Twin Marketplace demonstration completed!")
	print("   Key Features Demonstrated:")
	print("   • Multi-category item marketplace with 5 sample digital twins")
	print("   • Advanced search and filtering capabilities")
	print("   • AI-powered personalized recommendations")
	print("   • Quality assurance and certification system")
	print("   • User reviews and rating system")
	print("   • Transaction processing and license management")
	print("   • Comprehensive analytics and insights")
	print("   • Support for multiple license types and pricing models")

if __name__ == "__main__":
	asyncio.run(demonstrate_digital_twin_marketplace())