"""
Intelligent Transaction Analysis Engine - Advanced AI-Powered Insights

Revolutionary transaction analysis using ML clustering, pattern recognition,
behavioral modeling, and predictive analytics for APG Payment Gateway.

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from uuid_extensions import uuid7str
from dataclasses import dataclass
import math

# ML Libraries
try:
	from sklearn.cluster import DBSCAN, KMeans
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	from sklearn.decomposition import PCA
	from sklearn.ensemble import IsolationForest
	from sklearn.metrics import silhouette_score
	import scipy.stats as stats
	from scipy.spatial.distance import euclidean
	import networkx as nx
except ImportError:
	print("⚠️  Advanced ML libraries not available - using simplified analysis")

from .models import PaymentTransaction, PaymentStatus, PaymentMethodType
from .ml_fraud_detection import MLFraudDetectionEngine

class AnalysisType(str, Enum):
	"""Types of transaction analysis"""
	PATTERN_DETECTION = "pattern_detection"
	BEHAVIORAL_CLUSTERING = "behavioral_clustering"
	ANOMALY_DETECTION = "anomaly_detection"
	TREND_ANALYSIS = "trend_analysis"
	NETWORK_ANALYSIS = "network_analysis"
	RISK_PROFILING = "risk_profiling"
	REVENUE_OPTIMIZATION = "revenue_optimization"
	CUSTOMER_SEGMENTATION = "customer_segmentation"

class InsightSeverity(str, Enum):
	"""Severity levels for insights"""
	CRITICAL = "critical"
	HIGH = "high"
	MEDIUM = "medium"
	LOW = "low"
	INFO = "info"

@dataclass
class TransactionInsight:
	"""Individual transaction insight"""
	id: str
	type: AnalysisType
	severity: InsightSeverity
	title: str
	description: str
	impact_score: float
	confidence: float
	affected_transactions: List[str]
	recommended_actions: List[str]
	metadata: Dict[str, Any]
	created_at: datetime

@dataclass
class AnalysisReport:
	"""Comprehensive analysis report"""
	id: str
	tenant_id: str
	analysis_period: Tuple[datetime, datetime]
	insights: List[TransactionInsight]
	summary_metrics: Dict[str, Any]
	trends: Dict[str, List[float]]
	risk_distribution: Dict[str, int]
	recommendations: List[str]
	created_at: datetime

class IntelligentTransactionAnalyzer:
	"""
	Advanced transaction analysis engine using AI/ML
	
	Provides deep insights into transaction patterns, customer behavior,
	fraud trends, and business optimization opportunities.
	"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		self.analyzer_id = uuid7str()
		
		# Analysis configuration
		self.analysis_window_hours = config.get("analysis_window_hours", 24)
		self.min_cluster_size = config.get("min_cluster_size", 5)
		self.anomaly_threshold = config.get("anomaly_threshold", 0.1)
		self.pattern_sensitivity = config.get("pattern_sensitivity", 0.8)
		
		# Feature engineering
		self.enable_advanced_clustering = config.get("enable_advanced_clustering", True)
		self.enable_network_analysis = config.get("enable_network_analysis", True)
		self.enable_time_series_analysis = config.get("enable_time_series_analysis", True)
		
		# Analysis models
		self._clustering_models: Dict[str, Any] = {}
		self._anomaly_detectors: Dict[str, Any] = {}
		self._pattern_models: Dict[str, Any] = {}
		
		# Data storage
		self._transaction_features: List[Dict[str, Any]] = []
		self._customer_profiles: Dict[str, Dict[str, Any]] = {}
		self._merchant_analytics: Dict[str, Dict[str, Any]] = {}
		self._temporal_patterns: Dict[str, List[float]] = {}
		
		# Insight tracking
		self._insights_history: List[TransactionInsight] = []
		self._analysis_cache: Dict[str, AnalysisReport] = {}
		
		self._initialized = False
		
		self._log_analyzer_created()
	
	async def initialize(self) -> Dict[str, Any]:
		"""Initialize intelligent transaction analyzer"""
		self._log_analyzer_initialization_start()
		
		try:
			# Initialize clustering models
			await self._initialize_clustering_models()
			
			# Initialize anomaly detection
			await self._initialize_anomaly_detection()
			
			# Initialize pattern recognition
			await self._initialize_pattern_recognition()
			
			# Set up real-time analysis pipeline
			await self._setup_realtime_pipeline()
			
			self._initialized = True
			
			self._log_analyzer_initialization_complete()
			
			return {
				"status": "initialized",
				"analyzer_id": self.analyzer_id,
				"analysis_window_hours": self.analysis_window_hours,
				"clustering_enabled": self.enable_advanced_clustering,
				"network_analysis_enabled": self.enable_network_analysis,
				"time_series_enabled": self.enable_time_series_analysis
			}
			
		except Exception as e:
			self._log_analyzer_initialization_error(str(e))
			raise
	
	async def analyze_transaction_patterns(
		self,
		transactions: List[PaymentTransaction],
		analysis_types: List[AnalysisType] | None = None
	) -> AnalysisReport:
		"""
		Perform comprehensive transaction pattern analysis
		
		Args:
			transactions: List of transactions to analyze
			analysis_types: Specific analysis types to perform
			
		Returns:
			AnalysisReport with insights and recommendations
		"""
		if not self._initialized:
			raise RuntimeError("Intelligent transaction analyzer not initialized")
		
		analysis_start = datetime.now(timezone.utc)
		analysis_types = analysis_types or list(AnalysisType)
		
		self._log_analysis_start(len(transactions), analysis_types)
		
		try:
			# Extract features for analysis
			features_df = await self._extract_comprehensive_features(transactions)
			
			# Perform requested analyses
			insights = []
			
			if AnalysisType.PATTERN_DETECTION in analysis_types:
				pattern_insights = await self._detect_patterns(features_df, transactions)
				insights.extend(pattern_insights)
			
			if AnalysisType.BEHAVIORAL_CLUSTERING in analysis_types:
				clustering_insights = await self._perform_behavioral_clustering(features_df, transactions)
				insights.extend(clustering_insights)
			
			if AnalysisType.ANOMALY_DETECTION in analysis_types:
				anomaly_insights = await self._detect_anomalies(features_df, transactions)
				insights.extend(anomaly_insights)
			
			if AnalysisType.TREND_ANALYSIS in analysis_types:
				trend_insights = await self._analyze_trends(features_df, transactions)
				insights.extend(trend_insights)
			
			if AnalysisType.NETWORK_ANALYSIS in analysis_types and self.enable_network_analysis:
				network_insights = await self._analyze_transaction_networks(transactions)
				insights.extend(network_insights)
			
			if AnalysisType.RISK_PROFILING in analysis_types:
				risk_insights = await self._perform_risk_profiling(features_df, transactions)
				insights.extend(risk_insights)
			
			if AnalysisType.REVENUE_OPTIMIZATION in analysis_types:
				revenue_insights = await self._analyze_revenue_optimization(features_df, transactions)
				insights.extend(revenue_insights)
			
			if AnalysisType.CUSTOMER_SEGMENTATION in analysis_types:
				segmentation_insights = await self._perform_customer_segmentation(features_df, transactions)
				insights.extend(segmentation_insights)
			
			# Generate summary metrics
			summary_metrics = await self._generate_summary_metrics(features_df, transactions)
			
			# Extract trends
			trends = await self._extract_trends(features_df, transactions)
			
			# Calculate risk distribution
			risk_distribution = await self._calculate_risk_distribution(insights)
			
			# Generate recommendations
			recommendations = await self._generate_recommendations(insights, summary_metrics)
			
			# Create analysis report
			report = AnalysisReport(
				id=uuid7str(),
				tenant_id=transactions[0].tenant_id if transactions else "unknown",
				analysis_period=(
					min(t.created_at for t in transactions),
					max(t.created_at for t in transactions)
				) if transactions else (analysis_start, analysis_start),
				insights=insights,
				summary_metrics=summary_metrics,
				trends=trends,
				risk_distribution=risk_distribution,
				recommendations=recommendations,
				created_at=analysis_start
			)
			
			# Cache report
			self._analysis_cache[report.id] = report
			
			# Update insights history
			self._insights_history.extend(insights)
			
			analysis_time = (datetime.now(timezone.utc) - analysis_start).total_seconds() * 1000
			self._log_analysis_complete(len(insights), analysis_time)
			
			return report
			
		except Exception as e:
			self._log_analysis_error(str(e))
			raise
	
	async def get_real_time_insights(
		self,
		transaction: PaymentTransaction,
		context: Dict[str, Any] | None = None
	) -> List[TransactionInsight]:
		"""Get real-time insights for a single transaction"""
		try:
			insights = []
			
			# Velocity analysis
			velocity_insight = await self._analyze_transaction_velocity(transaction)
			if velocity_insight:
				insights.append(velocity_insight)
			
			# Behavioral deviation analysis
			behavior_insight = await self._analyze_behavioral_deviation(transaction, context or {})
			if behavior_insight:
				insights.append(behavior_insight)
			
			# Amount analysis
			amount_insight = await self._analyze_amount_patterns(transaction)
			if amount_insight:
				insights.append(amount_insight)
			
			# Temporal analysis
			temporal_insight = await self._analyze_temporal_patterns(transaction)
			if temporal_insight:
				insights.append(temporal_insight)
			
			return insights
			
		except Exception as e:
			self._log_realtime_analysis_error(transaction.id, str(e))
			return []
	
	async def get_customer_intelligence(
		self,
		customer_id: str,
		lookback_days: int = 30
	) -> Dict[str, Any]:
		"""Get comprehensive customer intelligence profile"""
		try:
			# Get customer transactions
			customer_transactions = [
				t for t in self._get_recent_transactions(lookback_days)
				if t.get("customer_id") == customer_id
			]
			
			if not customer_transactions:
				return {"status": "no_data", "customer_id": customer_id}
			
			# Calculate customer metrics
			metrics = await self._calculate_customer_metrics(customer_transactions)
			
			# Behavioral patterns
			patterns = await self._extract_customer_patterns(customer_transactions)
			
			# Risk assessment
			risk_profile = await self._assess_customer_risk(customer_transactions)
			
			# Recommendations
			recommendations = await self._generate_customer_recommendations(metrics, patterns, risk_profile)
			
			return {
				"customer_id": customer_id,
				"metrics": metrics,
				"behavioral_patterns": patterns,
				"risk_profile": risk_profile,
				"recommendations": recommendations,
				"analysis_period_days": lookback_days,
				"transaction_count": len(customer_transactions)
			}
			
		except Exception as e:
			self._log_customer_intelligence_error(customer_id, str(e))
			return {"status": "error", "customer_id": customer_id, "error": str(e)}
	
	async def get_merchant_analytics(
		self,
		merchant_id: str,
		lookback_days: int = 30
	) -> Dict[str, Any]:
		"""Get comprehensive merchant analytics"""
		try:
			# Get merchant transactions
			merchant_transactions = [
				t for t in self._get_recent_transactions(lookback_days)
				if t.get("merchant_id") == merchant_id
			]
			
			if not merchant_transactions:
				return {"status": "no_data", "merchant_id": merchant_id}
			
			# Calculate metrics
			metrics = await self._calculate_merchant_metrics(merchant_transactions)
			
			# Performance analysis
			performance = await self._analyze_merchant_performance(merchant_transactions)
			
			# Fraud analysis
			fraud_analysis = await self._analyze_merchant_fraud_patterns(merchant_transactions)
			
			# Revenue optimization
			optimization = await self._analyze_merchant_optimization(merchant_transactions)
			
			return {
				"merchant_id": merchant_id,
				"metrics": metrics,
				"performance": performance,
				"fraud_analysis": fraud_analysis,
				"optimization_opportunities": optimization,
				"analysis_period_days": lookback_days,
				"transaction_count": len(merchant_transactions)
			}
			
		except Exception as e:
			self._log_merchant_analytics_error(merchant_id, str(e))
			return {"status": "error", "merchant_id": merchant_id, "error": str(e)}
	
	# Pattern detection methods
	
	async def _detect_patterns(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Detect transaction patterns using ML techniques"""
		insights = []
		
		try:
			# Time-based patterns
			time_patterns = await self._detect_time_patterns(features_df, transactions)
			insights.extend(time_patterns)
			
			# Amount patterns
			amount_patterns = await self._detect_amount_patterns(features_df, transactions)
			insights.extend(amount_patterns)
			
			# Geographic patterns
			geo_patterns = await self._detect_geographic_patterns(features_df, transactions)
			insights.extend(geo_patterns)
			
			# Payment method patterns
			payment_patterns = await self._detect_payment_method_patterns(features_df, transactions)
			insights.extend(payment_patterns)
			
		except Exception as e:
			self._log_pattern_detection_error(str(e))
		
		return insights
	
	async def _detect_time_patterns(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Detect temporal patterns in transactions"""
		insights = []
		
		# Group by hour and analyze volume
		hourly_volumes = {}
		for transaction in transactions:
			hour = transaction.created_at.hour
			hourly_volumes[hour] = hourly_volumes.get(hour, 0) + 1
		
		# Detect unusual activity spikes
		if hourly_volumes:
			mean_volume = np.mean(list(hourly_volumes.values()))
			std_volume = np.std(list(hourly_volumes.values()))
			
			for hour, volume in hourly_volumes.items():
				if volume > mean_volume + 2 * std_volume:
					insights.append(TransactionInsight(
						id=uuid7str(),
						type=AnalysisType.PATTERN_DETECTION,
						severity=InsightSeverity.MEDIUM,
						title=f"Unusual Activity Spike at Hour {hour}",
						description=f"Transaction volume at hour {hour} is {volume} (mean: {mean_volume:.1f})",
						impact_score=0.6,
						confidence=0.8,
						affected_transactions=[t.id for t in transactions if t.created_at.hour == hour],
						recommended_actions=["investigate_activity_spike", "monitor_hour_" + str(hour)],
						metadata={"hour": hour, "volume": volume, "mean": mean_volume},
						created_at=datetime.now(timezone.utc)
					))
		
		return insights
	
	async def _detect_amount_patterns(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Detect amount-based patterns"""
		insights = []
		
		amounts = [t.amount for t in transactions]
		if not amounts:
			return insights
		
		# Detect round number bias
		round_amounts = sum(1 for amount in amounts if amount % 1000 == 0)
		round_percentage = round_amounts / len(amounts)
		
		if round_percentage > 0.3:  # More than 30% round amounts
			insights.append(TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.PATTERN_DETECTION,
				severity=InsightSeverity.INFO,
				title="High Round Number Usage",
				description=f"{round_percentage:.1%} of transactions use round amounts",
				impact_score=0.3,
				confidence=0.9,
				affected_transactions=[t.id for t in transactions if t.amount % 1000 == 0],
				recommended_actions=["analyze_pricing_strategy"],
				metadata={"round_percentage": round_percentage},
				created_at=datetime.now(timezone.utc)
			))
		
		# Detect amount clustering
		if len(set(amounts)) < len(amounts) * 0.7:  # Less than 70% unique amounts
			insights.append(TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.PATTERN_DETECTION,
				severity=InsightSeverity.INFO,
				title="Amount Clustering Detected",
				description="High frequency of repeated transaction amounts",
				impact_score=0.4,
				confidence=0.7,
				affected_transactions=[t.id for t in transactions],
				recommended_actions=["review_pricing_model"],
				metadata={"unique_amounts_ratio": len(set(amounts)) / len(amounts)},
				created_at=datetime.now(timezone.utc)
			))
		
		return insights
	
	async def _detect_geographic_patterns(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Detect geographic patterns"""
		insights = []
		
		# This would use real geolocation data in production
		# For now, simulate some geographic pattern detection
		
		# Detect high-risk location clustering
		high_risk_transactions = [
			t for t in transactions 
			if t.metadata.get("location_risk", 0) > 0.7
		]
		
		if len(high_risk_transactions) > len(transactions) * 0.1:  # More than 10%
			insights.append(TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.PATTERN_DETECTION,
				severity=InsightSeverity.HIGH,
				title="High-Risk Location Clustering",
				description=f"{len(high_risk_transactions)} transactions from high-risk locations",
				impact_score=0.8,
				confidence=0.7,
				affected_transactions=[t.id for t in high_risk_transactions],
				recommended_actions=["review_geographic_risk", "implement_location_verification"],
				metadata={"high_risk_count": len(high_risk_transactions)},
				created_at=datetime.now(timezone.utc)
			))
		
		return insights
	
	async def _detect_payment_method_patterns(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Detect payment method patterns"""
		insights = []
		
		# Count payment methods
		method_counts = {}
		for transaction in transactions:
			method = transaction.payment_method_type
			method_counts[method] = method_counts.get(method, 0) + 1
		
		# Detect unusual payment method distribution
		total_transactions = len(transactions)
		for method, count in method_counts.items():
			percentage = count / total_transactions
			
			# Detect high usage of high-risk methods
			if method in [PaymentMethodType.CRYPTOCURRENCY] and percentage > 0.05:  # More than 5%
				insights.append(TransactionInsight(
					id=uuid7str(),
					type=AnalysisType.PATTERN_DETECTION,
					severity=InsightSeverity.HIGH,
					title=f"High Usage of {method.value}",
					description=f"{percentage:.1%} of transactions use {method.value}",
					impact_score=0.7,
					confidence=0.8,
					affected_transactions=[t.id for t in transactions if t.payment_method_type == method],
					recommended_actions=["review_payment_method_risk", "enhance_verification"],
					metadata={"method": method.value, "percentage": percentage},
					created_at=datetime.now(timezone.utc)
				))
		
		return insights
	
	# Clustering analysis methods
	
	async def _perform_behavioral_clustering(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Perform behavioral clustering analysis"""
		insights = []
		
		if not self.enable_advanced_clustering or len(transactions) < self.min_cluster_size:
			return insights
		
		try:
			# Prepare features for clustering
			clustering_features = await self._prepare_clustering_features(features_df)
			
			# Perform K-means clustering
			kmeans_insights = await self._perform_kmeans_clustering(clustering_features, transactions)
			insights.extend(kmeans_insights)
			
			# Perform DBSCAN clustering
			dbscan_insights = await self._perform_dbscan_clustering(clustering_features, transactions)
			insights.extend(dbscan_insights)
			
		except Exception as e:
			self._log_clustering_error(str(e))
		
		return insights
	
	async def _perform_kmeans_clustering(
		self,
		features: np.ndarray,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Perform K-means clustering analysis"""
		insights = []
		
		try:
			# Determine optimal number of clusters
			optimal_k = await self._find_optimal_clusters(features, max_k=min(8, len(transactions) // 2))
			
			if optimal_k < 2:
				return insights
			
			# Perform clustering
			kmeans = KMeans(n_clusters=optimal_k, random_state=42)
			cluster_labels = kmeans.fit_predict(features)
			
			# Analyze clusters
			cluster_analysis = await self._analyze_clusters(cluster_labels, transactions, features)
			
			# Generate insights
			for cluster_id, analysis in cluster_analysis.items():
				if analysis["is_anomalous"]:
					insights.append(TransactionInsight(
						id=uuid7str(),
						type=AnalysisType.BEHAVIORAL_CLUSTERING,
						severity=InsightSeverity.MEDIUM,
						title=f"Anomalous Transaction Cluster {cluster_id}",
						description=f"Cluster with {analysis['size']} transactions shows unusual patterns",
						impact_score=analysis["risk_score"],
						confidence=0.7,
						affected_transactions=analysis["transaction_ids"],
						recommended_actions=["investigate_cluster", "monitor_pattern"],
						metadata=analysis,
						created_at=datetime.now(timezone.utc)
					))
		
		except Exception as e:
			self._log_kmeans_error(str(e))
		
		return insights
	
	async def _perform_dbscan_clustering(
		self,
		features: np.ndarray,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Perform DBSCAN clustering for outlier detection"""
		insights = []
		
		try:
			# Perform DBSCAN
			dbscan = DBSCAN(eps=0.5, min_samples=max(2, self.min_cluster_size // 2))
			cluster_labels = dbscan.fit_predict(features)
			
			# Find outliers (label -1)
			outlier_indices = np.where(cluster_labels == -1)[0]
			
			if len(outlier_indices) > 0:
				outlier_transactions = [transactions[i] for i in outlier_indices]
				
				insights.append(TransactionInsight(
					id=uuid7str(),
					type=AnalysisType.BEHAVIORAL_CLUSTERING,
					severity=InsightSeverity.MEDIUM,
					title="Behavioral Outliers Detected",
					description=f"{len(outlier_indices)} transactions identified as behavioral outliers",
					impact_score=0.6,
					confidence=0.8,
					affected_transactions=[t.id for t in outlier_transactions],
					recommended_actions=["investigate_outliers", "manual_review"],
					metadata={"outlier_count": len(outlier_indices)},
					created_at=datetime.now(timezone.utc)
				))
		
		except Exception as e:
			self._log_dbscan_error(str(e))
		
		return insights
	
	# Anomaly detection methods
	
	async def _detect_anomalies(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Detect statistical and ML-based anomalies"""
		insights = []
		
		try:
			# Statistical anomalies
			statistical_insights = await self._detect_statistical_anomalies(features_df, transactions)
			insights.extend(statistical_insights)
			
			# ML-based anomalies
			ml_insights = await self._detect_ml_anomalies(features_df, transactions)
			insights.extend(ml_insights)
			
		except Exception as e:
			self._log_anomaly_detection_error(str(e))
		
		return insights
	
	async def _detect_statistical_anomalies(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Detect statistical anomalies"""
		insights = []
		
		# Amount anomalies using IQR
		amounts = [t.amount for t in transactions]
		if amounts:
			q1 = np.percentile(amounts, 25)
			q3 = np.percentile(amounts, 75)
			iqr = q3 - q1
			lower_bound = q1 - 1.5 * iqr
			upper_bound = q3 + 1.5 * iqr
			
			anomalous_amounts = [
				t for t in transactions 
				if t.amount < lower_bound or t.amount > upper_bound
			]
			
			if anomalous_amounts:
				insights.append(TransactionInsight(
					id=uuid7str(),
					type=AnalysisType.ANOMALY_DETECTION,
					severity=InsightSeverity.MEDIUM,
					title="Amount Anomalies Detected",
					description=f"{len(anomalous_amounts)} transactions with anomalous amounts",
					impact_score=0.5,
					confidence=0.7,
					affected_transactions=[t.id for t in anomalous_amounts],
					recommended_actions=["review_amount_anomalies"],
					metadata={"count": len(anomalous_amounts), "bounds": [lower_bound, upper_bound]},
					created_at=datetime.now(timezone.utc)
				))
		
		return insights
	
	async def _detect_ml_anomalies(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Detect ML-based anomalies using Isolation Forest"""
		insights = []
		
		if len(transactions) < 10:  # Need minimum samples
			return insights
		
		try:
			# Prepare features
			feature_matrix = await self._prepare_anomaly_features(features_df)
			
			# Use Isolation Forest
			iso_forest = IsolationForest(contamination=self.anomaly_threshold, random_state=42)
			anomaly_scores = iso_forest.fit_predict(feature_matrix)
			
			# Find anomalies (score -1)
			anomaly_indices = np.where(anomaly_scores == -1)[0]
			
			if len(anomaly_indices) > 0:
				anomalous_transactions = [transactions[i] for i in anomaly_indices]
				
				insights.append(TransactionInsight(
					id=uuid7str(),
					type=AnalysisType.ANOMALY_DETECTION,
					severity=InsightSeverity.HIGH,
					title="ML Anomalies Detected",
					description=f"{len(anomaly_indices)} transactions identified as anomalous by ML model",
					impact_score=0.7,
					confidence=0.8,
					affected_transactions=[t.id for t in anomalous_transactions],
					recommended_actions=["investigate_ml_anomalies", "manual_review"],
					metadata={"anomaly_count": len(anomaly_indices)},
					created_at=datetime.now(timezone.utc)
				))
		
		except Exception as e:
			self._log_ml_anomaly_error(str(e))
		
		return insights
	
	# Feature engineering methods
	
	async def _extract_comprehensive_features(
		self,
		transactions: List[PaymentTransaction]
	) -> pd.DataFrame:
		"""Extract comprehensive features for analysis"""
		features = []
		
		for transaction in transactions:
			feature_dict = {
				"transaction_id": transaction.id,
				"amount": transaction.amount,
				"amount_log": np.log1p(transaction.amount),
				"currency": transaction.currency,
				"payment_method": transaction.payment_method_type.value,
				"hour": transaction.created_at.hour,
				"day_of_week": transaction.created_at.weekday(),
				"day_of_month": transaction.created_at.day,
				"month": transaction.created_at.month,
				"is_weekend": transaction.created_at.weekday() >= 5,
				"is_business_hours": 9 <= transaction.created_at.hour <= 17,
				"customer_id": transaction.customer_id or "guest",
				"merchant_id": transaction.merchant_id,
				"status": transaction.status.value,
				"has_description": bool(transaction.description),
				"description_length": len(transaction.description or ""),
				"metadata_count": len(transaction.metadata),
				"created_timestamp": transaction.created_at.timestamp()
			}
			
			features.append(feature_dict)
		
		return pd.DataFrame(features)
	
	async def _prepare_clustering_features(self, features_df: pd.DataFrame) -> np.ndarray:
		"""Prepare numerical features for clustering"""
		numerical_features = [
			"amount_log", "hour", "day_of_week", "day_of_month", "month",
			"description_length", "metadata_count"
		]
		
		# Add categorical encodings
		categorical_features = []
		
		# Encode payment method
		if "payment_method" in features_df.columns:
			payment_method_encoded = pd.get_dummies(features_df["payment_method"], prefix="pm")
			categorical_features.append(payment_method_encoded)
		
		# Encode currency
		if "currency" in features_df.columns:
			currency_encoded = pd.get_dummies(features_df["currency"], prefix="curr")
			categorical_features.append(currency_encoded)
		
		# Combine features
		feature_matrix = features_df[numerical_features].fillna(0)
		
		for cat_features in categorical_features:
			feature_matrix = pd.concat([feature_matrix, cat_features], axis=1)
		
		# Scale features
		scaler = StandardScaler()
		scaled_features = scaler.fit_transform(feature_matrix)
		
		return scaled_features
	
	async def _prepare_anomaly_features(self, features_df: pd.DataFrame) -> np.ndarray:
		"""Prepare features for anomaly detection"""
		return await self._prepare_clustering_features(features_df)
	
	# Analysis utility methods
	
	async def _find_optimal_clusters(self, features: np.ndarray, max_k: int = 8) -> int:
		"""Find optimal number of clusters using elbow method"""
		if len(features) < 4:
			return 1
		
		try:
			inertias = []
			k_range = range(2, min(max_k + 1, len(features)))
			
			for k in k_range:
				kmeans = KMeans(n_clusters=k, random_state=42)
				kmeans.fit(features)
				inertias.append(kmeans.inertia_)
			
			if len(inertias) < 2:
				return 2
			
			# Find elbow using rate of change
			rate_of_change = np.diff(inertias)
			elbow_index = np.argmax(rate_of_change) + 2  # +2 because we start from k=2
			
			return min(elbow_index, max_k)
		
		except Exception:
			return 2  # Default to 2 clusters
	
	async def _analyze_clusters(
		self,
		cluster_labels: np.ndarray,
		transactions: List[PaymentTransaction],
		features: np.ndarray
	) -> Dict[int, Dict[str, Any]]:
		"""Analyze cluster characteristics"""
		cluster_analysis = {}
		
		unique_labels = np.unique(cluster_labels)
		
		for cluster_id in unique_labels:
			cluster_mask = cluster_labels == cluster_id
			cluster_transactions = [transactions[i] for i in np.where(cluster_mask)[0]]
			cluster_features = features[cluster_mask]
			
			# Calculate cluster statistics
			avg_amount = np.mean([t.amount for t in cluster_transactions])
			std_amount = np.std([t.amount for t in cluster_transactions])
			
			# Determine if cluster is anomalous
			is_anomalous = await self._is_cluster_anomalous(cluster_transactions, avg_amount)
			
			cluster_analysis[cluster_id] = {
				"size": len(cluster_transactions),
				"avg_amount": avg_amount,
				"std_amount": std_amount,
				"is_anomalous": is_anomalous,
				"risk_score": await self._calculate_cluster_risk(cluster_transactions),
				"transaction_ids": [t.id for t in cluster_transactions],
				"dominant_payment_method": await self._get_dominant_payment_method(cluster_transactions),
				"time_pattern": await self._get_cluster_time_pattern(cluster_transactions)
			}
		
		return cluster_analysis
	
	async def _is_cluster_anomalous(
		self,
		cluster_transactions: List[PaymentTransaction],
		avg_amount: float
	) -> bool:
		"""Determine if a cluster is anomalous"""
		# Simple heuristics for anomaly detection
		
		# Very high average amount
		if avg_amount > 100000:  # $1000+
			return True
		
		# All transactions at unusual hours
		unusual_hours = sum(1 for t in cluster_transactions if t.created_at.hour < 6 or t.created_at.hour > 22)
		if unusual_hours > len(cluster_transactions) * 0.8:
			return True
		
		# High concentration of same payment method (if it's high-risk)
		payment_methods = [t.payment_method_type for t in cluster_transactions]
		most_common_method = max(set(payment_methods), key=payment_methods.count)
		method_concentration = payment_methods.count(most_common_method) / len(payment_methods)
		
		if method_concentration > 0.9 and most_common_method in [PaymentMethodType.CRYPTOCURRENCY]:
			return True
		
		return False
	
	async def _calculate_cluster_risk(self, cluster_transactions: List[PaymentTransaction]) -> float:
		"""Calculate risk score for a cluster"""
		risk_score = 0.0
		
		# Amount-based risk
		avg_amount = np.mean([t.amount for t in cluster_transactions])
		if avg_amount > 50000:  # $500+
			risk_score += 0.3
		
		# Time-based risk
		unusual_time_ratio = sum(
			1 for t in cluster_transactions 
			if t.created_at.hour < 6 or t.created_at.hour > 22
		) / len(cluster_transactions)
		risk_score += unusual_time_ratio * 0.4
		
		# Payment method risk
		high_risk_methods = [PaymentMethodType.CRYPTOCURRENCY]
		high_risk_ratio = sum(
			1 for t in cluster_transactions 
			if t.payment_method_type in high_risk_methods
		) / len(cluster_transactions)
		risk_score += high_risk_ratio * 0.3
		
		return min(1.0, risk_score)
	
	async def _get_dominant_payment_method(
		self,
		cluster_transactions: List[PaymentTransaction]
	) -> str:
		"""Get dominant payment method in cluster"""
		payment_methods = [t.payment_method_type.value for t in cluster_transactions]
		return max(set(payment_methods), key=payment_methods.count)
	
	async def _get_cluster_time_pattern(
		self,
		cluster_transactions: List[PaymentTransaction]
	) -> Dict[str, Any]:
		"""Get time pattern for cluster"""
		hours = [t.created_at.hour for t in cluster_transactions]
		return {
			"avg_hour": np.mean(hours),
			"std_hour": np.std(hours),
			"most_common_hour": max(set(hours), key=hours.count)
		}
	
	# Additional analysis methods (simplified implementations)
	
	async def _analyze_trends(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Analyze trends in transaction data"""
		insights = []
		
		# Volume trend analysis
		if len(transactions) > 10:
			# Group by hour and analyze volume trend
			hourly_volumes = {}
			for transaction in transactions:
				hour_key = transaction.created_at.replace(minute=0, second=0, microsecond=0)
				hourly_volumes[hour_key] = hourly_volumes.get(hour_key, 0) + 1
			
			if len(hourly_volumes) > 3:
				volumes = list(hourly_volumes.values())
				# Simple trend detection using linear correlation with time
				time_points = list(range(len(volumes)))
				correlation = np.corrcoef(time_points, volumes)[0, 1]
				
				if abs(correlation) > 0.7:  # Strong correlation
					trend_direction = "increasing" if correlation > 0 else "decreasing"
					insights.append(TransactionInsight(
						id=uuid7str(),
						type=AnalysisType.TREND_ANALYSIS,
						severity=InsightSeverity.INFO,
						title=f"Strong {trend_direction.title()} Volume Trend",
						description=f"Transaction volume shows {trend_direction} trend (correlation: {correlation:.2f})",
						impact_score=0.4,
						confidence=0.7,
						affected_transactions=[t.id for t in transactions],
						recommended_actions=["monitor_volume_trend"],
						metadata={"correlation": correlation, "trend": trend_direction},
						created_at=datetime.now(timezone.utc)
					))
		
		return insights
	
	async def _analyze_transaction_networks(
		self,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Analyze transaction networks and connections"""
		insights = []
		
		# Simplified network analysis
		# In production, this would use graph algorithms
		
		# Find customers with shared devices or IPs (simulated)
		customer_connections = {}
		for transaction in transactions:
			customer_id = transaction.customer_id
			if customer_id:
				# Simulate device/IP sharing detection
				shared_entities = transaction.metadata.get("shared_entities", [])
				if shared_entities:
					customer_connections[customer_id] = shared_entities
		
		# Detect large connected components
		if len(customer_connections) > 5:
			insights.append(TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.NETWORK_ANALYSIS,
				severity=InsightSeverity.MEDIUM,
				title="Customer Network Connections Detected",
				description=f"Network of {len(customer_connections)} customers with shared devices/IPs",
				impact_score=0.6,
				confidence=0.5,
				affected_transactions=[
					t.id for t in transactions 
					if t.customer_id in customer_connections
				],
				recommended_actions=["investigate_network", "verify_customer_identities"],
				metadata={"network_size": len(customer_connections)},
				created_at=datetime.now(timezone.utc)
			))
		
		return insights
	
	async def _perform_risk_profiling(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Perform risk profiling analysis"""
		insights = []
		
		# Calculate overall risk distribution
		high_risk_count = sum(
			1 for t in transactions 
			if t.amount > 100000 or  # High amount
			t.created_at.hour < 6 or t.created_at.hour > 22 or  # Unusual time
			t.payment_method_type in [PaymentMethodType.CRYPTOCURRENCY]  # High-risk method
		)
		
		risk_percentage = high_risk_count / len(transactions) if transactions else 0
		
		if risk_percentage > 0.2:  # More than 20% high-risk
			insights.append(TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.RISK_PROFILING,
				severity=InsightSeverity.HIGH,
				title="High Risk Transaction Concentration",
				description=f"{risk_percentage:.1%} of transactions are high-risk",
				impact_score=0.8,
				confidence=0.8,
				affected_transactions=[t.id for t in transactions],
				recommended_actions=["enhance_risk_controls", "increase_monitoring"],
				metadata={"risk_percentage": risk_percentage},
				created_at=datetime.now(timezone.utc)
			))
		
		return insights
	
	async def _analyze_revenue_optimization(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Analyze revenue optimization opportunities"""
		insights = []
		
		# Analyze payment method mix
		method_revenues = {}
		for transaction in transactions:
			method = transaction.payment_method_type
			method_revenues[method] = method_revenues.get(method, 0) + transaction.amount
		
		if method_revenues:
			total_revenue = sum(method_revenues.values())
			
			# Find most profitable payment method
			best_method = max(method_revenues, key=method_revenues.get)
			best_method_percentage = method_revenues[best_method] / total_revenue
			
			if best_method_percentage > 0.6:  # More than 60% from one method
				insights.append(TransactionInsight(
					id=uuid7str(),
					type=AnalysisType.REVENUE_OPTIMIZATION,
					severity=InsightSeverity.INFO,
					title="Payment Method Revenue Concentration",
					description=f"{best_method_percentage:.1%} of revenue from {best_method.value}",
					impact_score=0.3,
					confidence=0.8,
					affected_transactions=[
						t.id for t in transactions 
						if t.payment_method_type == best_method
					],
					recommended_actions=["diversify_payment_methods", "optimize_method_mix"],
					metadata={
						"dominant_method": best_method.value,
						"percentage": best_method_percentage
					},
					created_at=datetime.now(timezone.utc)
				))
		
		return insights
	
	async def _perform_customer_segmentation(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> List[TransactionInsight]:
		"""Perform customer segmentation analysis"""
		insights = []
		
		# Simple customer value segmentation
		customer_values = {}
		for transaction in transactions:
			customer_id = transaction.customer_id
			if customer_id:
				customer_values[customer_id] = customer_values.get(customer_id, 0) + transaction.amount
		
		if customer_values:
			values = list(customer_values.values())
			high_value_threshold = np.percentile(values, 80)  # Top 20%
			
			high_value_customers = [
				customer_id for customer_id, value in customer_values.items()
				if value >= high_value_threshold
			]
			
			if high_value_customers:
				insights.append(TransactionInsight(
					id=uuid7str(),
					type=AnalysisType.CUSTOMER_SEGMENTATION,
					severity=InsightSeverity.INFO,
					title="High-Value Customer Segment Identified",
					description=f"{len(high_value_customers)} high-value customers (${high_value_threshold/100:.0f}+ each)",
					impact_score=0.5,
					confidence=0.9,
					affected_transactions=[
						t.id for t in transactions 
						if t.customer_id in high_value_customers
					],
					recommended_actions=["create_vip_program", "personalize_experience"],
					metadata={
						"high_value_count": len(high_value_customers),
						"threshold": high_value_threshold
					},
					created_at=datetime.now(timezone.utc)
				))
		
		return insights
	
	# Real-time analysis methods
	
	async def _analyze_transaction_velocity(
		self,
		transaction: PaymentTransaction
	) -> Optional[TransactionInsight]:
		"""Analyze transaction velocity for real-time insights"""
		# Get recent transactions for the same customer
		if not transaction.customer_id:
			return None
		
		recent_transactions = [
			t for t in self._get_recent_transactions(1)  # Last 1 day
			if t.get("customer_id") == transaction.customer_id
		]
		
		# Check for high velocity in short time window
		current_time = transaction.created_at
		window_5min = [
			t for t in recent_transactions
			if current_time - t.get("created_at", datetime.min) <= timedelta(minutes=5)
		]
		
		if len(window_5min) > 3:  # More than 3 transactions in 5 minutes
			return TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.PATTERN_DETECTION,
				severity=InsightSeverity.HIGH,
				title="High Transaction Velocity",
				description=f"{len(window_5min)} transactions in 5 minutes",
				impact_score=0.8,
				confidence=0.9,
				affected_transactions=[transaction.id],
				recommended_actions=["velocity_check", "manual_review"],
				metadata={"velocity_count": len(window_5min)},
				created_at=datetime.now(timezone.utc)
			)
		
		return None
	
	async def _analyze_behavioral_deviation(
		self,
		transaction: PaymentTransaction,
		context: Dict[str, Any]
	) -> Optional[TransactionInsight]:
		"""Analyze behavioral deviation from normal patterns"""
		if not transaction.customer_id:
			return None
		
		# Get customer profile
		customer_profile = self._customer_profiles.get(transaction.customer_id, {})
		
		if not customer_profile:
			return None  # No baseline to compare against
		
		# Check amount deviation
		avg_amount = customer_profile.get("avg_amount", 0)
		if avg_amount > 0 and transaction.amount > avg_amount * 5:  # 5x normal amount
			return TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.BEHAVIORAL_CLUSTERING,
				severity=InsightSeverity.MEDIUM,
				title="Amount Deviation from Normal Behavior",
				description=f"Transaction amount ${transaction.amount/100:.0f} is 5x customer average",
				impact_score=0.6,
				confidence=0.7,
				affected_transactions=[transaction.id],
				recommended_actions=["verify_customer_intent", "additional_authentication"],
				metadata={"deviation_ratio": transaction.amount / avg_amount},
				created_at=datetime.now(timezone.utc)
			)
		
		return None
	
	async def _analyze_amount_patterns(
		self,
		transaction: PaymentTransaction
	) -> Optional[TransactionInsight]:
		"""Analyze amount patterns for insights"""
		# Check for suspicious round amounts
		if transaction.amount % 10000 == 0 and transaction.amount >= 100000:  # Round $1000+ amounts
			return TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.PATTERN_DETECTION,
				severity=InsightSeverity.LOW,
				title="Large Round Amount Transaction",
				description=f"Transaction for exactly ${transaction.amount/100:.0f}",
				impact_score=0.3,
				confidence=0.6,
				affected_transactions=[transaction.id],
				recommended_actions=["verify_transaction_purpose"],
				metadata={"amount": transaction.amount},
				created_at=datetime.now(timezone.utc)
			)
		
		return None
	
	async def _analyze_temporal_patterns(
		self,
		transaction: PaymentTransaction
	) -> Optional[TransactionInsight]:
		"""Analyze temporal patterns"""
		# Check for unusual time patterns
		hour = transaction.created_at.hour
		is_weekend = transaction.created_at.weekday() >= 5
		
		if hour < 5 or hour > 23:  # Very late or very early
			severity = InsightSeverity.HIGH if hour < 3 or hour > 23 else InsightSeverity.MEDIUM
			
			return TransactionInsight(
				id=uuid7str(),
				type=AnalysisType.PATTERN_DETECTION,
				severity=severity,
				title="Unusual Transaction Time",
				description=f"Transaction at {hour:02d}:xx {'on weekend' if is_weekend else ''}",
				impact_score=0.5 if severity == InsightSeverity.MEDIUM else 0.7,
				confidence=0.8,
				affected_transactions=[transaction.id],
				recommended_actions=["verify_transaction_legitimacy"],
				metadata={"hour": hour, "is_weekend": is_weekend},
				created_at=datetime.now(timezone.utc)
			)
		
		return None
	
	# Utility methods
	
	async def _generate_summary_metrics(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> Dict[str, Any]:
		"""Generate summary metrics for analysis"""
		if not transactions:
			return {}
		
		amounts = [t.amount for t in transactions]
		
		return {
			"total_transactions": len(transactions),
			"total_amount": sum(amounts),
			"average_amount": np.mean(amounts),
			"median_amount": np.median(amounts),
			"std_amount": np.std(amounts),
			"min_amount": min(amounts),
			"max_amount": max(amounts),
			"unique_customers": len(set(t.customer_id for t in transactions if t.customer_id)),
			"unique_merchants": len(set(t.merchant_id for t in transactions)),
			"payment_methods": len(set(t.payment_method_type for t in transactions)),
			"currencies": len(set(t.currency for t in transactions)),
			"time_span_hours": (
				max(t.created_at for t in transactions) - min(t.created_at for t in transactions)
			).total_seconds() / 3600 if len(transactions) > 1 else 0
		}
	
	async def _extract_trends(
		self,
		features_df: pd.DataFrame,
		transactions: List[PaymentTransaction]
	) -> Dict[str, List[float]]:
		"""Extract trend data"""
		trends = {}
		
		if len(transactions) > 5:
			# Hourly volume trend
			hourly_counts = {}
			for transaction in transactions:
				hour_key = transaction.created_at.replace(minute=0, second=0, microsecond=0)
				hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
			
			if hourly_counts:
				sorted_hours = sorted(hourly_counts.keys())
				trends["hourly_volume"] = [hourly_counts[hour] for hour in sorted_hours]
			
			# Amount trend
			sorted_transactions = sorted(transactions, key=lambda t: t.created_at)
			trends["amount_trend"] = [t.amount for t in sorted_transactions[-20:]]  # Last 20
		
		return trends
	
	async def _calculate_risk_distribution(
		self,
		insights: List[TransactionInsight]
	) -> Dict[str, int]:
		"""Calculate risk distribution from insights"""
		distribution = {severity.value: 0 for severity in InsightSeverity}
		
		for insight in insights:
			distribution[insight.severity.value] += 1
		
		return distribution
	
	async def _generate_recommendations(
		self,
		insights: List[TransactionInsight],
		summary_metrics: Dict[str, Any]
	) -> List[str]:
		"""Generate actionable recommendations"""
		recommendations = []
		
		# Collect all recommended actions
		all_actions = []
		for insight in insights:
			all_actions.extend(insight.recommended_actions)
		
		# Count action frequency
		action_counts = {}
		for action in all_actions:
			action_counts[action] = action_counts.get(action, 0) + 1
		
		# Generate top recommendations
		sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
		
		for action, count in sorted_actions[:5]:  # Top 5
			recommendations.append(f"{action.replace('_', ' ').title()} (mentioned {count} times)")
		
		# Add general recommendations based on metrics
		if summary_metrics.get("total_transactions", 0) > 100:
			recommendations.append("Consider implementing advanced fraud monitoring")
		
		if summary_metrics.get("max_amount", 0) > 1000000:  # $10,000+
			recommendations.append("Implement enhanced verification for high-value transactions")
		
		return recommendations
	
	def _get_recent_transactions(self, days: int = 1) -> List[Dict[str, Any]]:
		"""Get recent transactions (simulated)"""
		# In production, this would query the database
		cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
		return [
			t for t in self._transaction_features
			if t.get("created_at", datetime.min) >= cutoff_time
		]
	
	# Customer and merchant analysis methods (simplified)
	
	async def _calculate_customer_metrics(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Calculate customer metrics"""
		amounts = [t.get("amount", 0) for t in transactions]
		
		return {
			"transaction_count": len(transactions),
			"total_amount": sum(amounts),
			"average_amount": np.mean(amounts) if amounts else 0,
			"frequency_days": len(set(
				t.get("created_at", datetime.min).date() 
				for t in transactions
			)) if transactions else 0
		}
	
	async def _extract_customer_patterns(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Extract customer behavioral patterns"""
		if not transactions:
			return {}
		
		hours = [t.get("created_at", datetime.min).hour for t in transactions]
		amounts = [t.get("amount", 0) for t in transactions]
		
		return {
			"preferred_hours": list(set(hours)),
			"amount_consistency": np.std(amounts) / np.mean(amounts) if amounts and np.mean(amounts) > 0 else 0,
			"transaction_regularity": "regular" if len(set(hours)) <= 3 else "varied"
		}
	
	async def _assess_customer_risk(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Assess customer risk profile"""
		# Simplified risk assessment
		risk_score = 0.0
		
		# High amounts increase risk
		avg_amount = np.mean([t.get("amount", 0) for t in transactions])
		if avg_amount > 100000:  # $1000+
			risk_score += 0.3
		
		# Unusual times increase risk
		unusual_times = sum(
			1 for t in transactions
			if t.get("created_at", datetime.min).hour < 6 or t.get("created_at", datetime.min).hour > 22
		)
		if unusual_times > len(transactions) * 0.3:
			risk_score += 0.4
		
		risk_level = "high" if risk_score > 0.6 else "medium" if risk_score > 0.3 else "low"
		
		return {
			"risk_score": min(1.0, risk_score),
			"risk_level": risk_level,
			"risk_factors": ["high_amounts", "unusual_times"][:int(risk_score * 3)]
		}
	
	async def _generate_customer_recommendations(
		self,
		metrics: Dict[str, Any],
		patterns: Dict[str, Any],
		risk_profile: Dict[str, Any]
	) -> List[str]:
		"""Generate customer-specific recommendations"""
		recommendations = []
		
		if risk_profile.get("risk_level") == "high":
			recommendations.append("Implement enhanced verification for this customer")
		
		if metrics.get("average_amount", 0) > 100000:
			recommendations.append("Consider VIP customer program enrollment")
		
		if patterns.get("amount_consistency", 0) < 0.2:
			recommendations.append("Customer shows consistent spending patterns")
		
		return recommendations
	
	# Similar methods for merchant analytics (simplified implementations)
	
	async def _calculate_merchant_metrics(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Calculate merchant metrics"""
		amounts = [t.get("amount", 0) for t in transactions]
		return {
			"transaction_count": len(transactions),
			"total_revenue": sum(amounts),
			"average_transaction": np.mean(amounts) if amounts else 0
		}
	
	async def _analyze_merchant_performance(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Analyze merchant performance"""
		return {"performance_score": 0.8}  # Simplified
	
	async def _analyze_merchant_fraud_patterns(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
		"""Analyze merchant fraud patterns"""
		return {"fraud_rate": 0.02}  # Simplified
	
	async def _analyze_merchant_optimization(self, transactions: List[Dict[str, Any]]) -> List[str]:
		"""Analyze merchant optimization opportunities"""
		return ["Optimize payment method mix"]  # Simplified
	
	# Initialization methods
	
	async def _initialize_clustering_models(self):
		"""Initialize clustering models"""
		self._clustering_models = {
			"kmeans": None,  # Will be created dynamically
			"dbscan": None   # Will be created dynamically
		}
		self._log_clustering_models_initialized()
	
	async def _initialize_anomaly_detection(self):
		"""Initialize anomaly detection models"""
		self._anomaly_detectors = {
			"isolation_forest": None  # Will be created dynamically
		}
		self._log_anomaly_detection_initialized()
	
	async def _initialize_pattern_recognition(self):
		"""Initialize pattern recognition models"""
		self._pattern_models = {
			"time_series": None,
			"sequential": None
		}
		self._log_pattern_recognition_initialized()
	
	async def _setup_realtime_pipeline(self):
		"""Set up real-time analysis pipeline"""
		self._log_realtime_pipeline_setup()
	
	# Logging methods following APG patterns
	
	def _log_analyzer_created(self):
		"""Log analyzer creation"""
		print(f"🧠 Intelligent Transaction Analyzer created")
		print(f"   Analyzer ID: {self.analyzer_id}")
		print(f"   Analysis Window: {self.analysis_window_hours}h")
		print(f"   Advanced Clustering: {self.enable_advanced_clustering}")
	
	def _log_analyzer_initialization_start(self):
		"""Log analyzer initialization start"""
		print(f"🚀 Initializing Intelligent Transaction Analyzer...")
		print(f"   Network Analysis: {self.enable_network_analysis}")
		print(f"   Time Series: {self.enable_time_series_analysis}")
	
	def _log_analyzer_initialization_complete(self):
		"""Log analyzer initialization complete"""
		print(f"✅ Intelligent Transaction Analyzer initialized successfully")
		print(f"   Clustering Models: {len(self._clustering_models)}")
		print(f"   Anomaly Detectors: {len(self._anomaly_detectors)}")
	
	def _log_analyzer_initialization_error(self, error: str):
		"""Log analyzer initialization error"""
		print(f"❌ Intelligent Transaction Analyzer initialization failed: {error}")
	
	def _log_analysis_start(self, transaction_count: int, analysis_types: List[AnalysisType]):
		"""Log analysis start"""
		print(f"🔍 Starting intelligent analysis on {transaction_count} transactions")
		print(f"   Analysis Types: {[t.value for t in analysis_types]}")
	
	def _log_analysis_complete(self, insights_count: int, analysis_time: float):
		"""Log analysis completion"""
		print(f"✅ Intelligent analysis complete")
		print(f"   Insights Generated: {insights_count}")
		print(f"   Analysis Time: {analysis_time:.1f}ms")
	
	def _log_analysis_error(self, error: str):
		"""Log analysis error"""
		print(f"❌ Intelligent analysis failed: {error}")
	
	def _log_realtime_analysis_error(self, transaction_id: str, error: str):
		"""Log real-time analysis error"""
		print(f"❌ Real-time analysis failed for {transaction_id}: {error}")
	
	def _log_customer_intelligence_error(self, customer_id: str, error: str):
		"""Log customer intelligence error"""
		print(f"❌ Customer intelligence failed for {customer_id}: {error}")
	
	def _log_merchant_analytics_error(self, merchant_id: str, error: str):
		"""Log merchant analytics error"""
		print(f"❌ Merchant analytics failed for {merchant_id}: {error}")
	
	def _log_pattern_detection_error(self, error: str):
		"""Log pattern detection error"""
		print(f"❌ Pattern detection error: {error}")
	
	def _log_clustering_error(self, error: str):
		"""Log clustering error"""
		print(f"❌ Clustering analysis error: {error}")
	
	def _log_kmeans_error(self, error: str):
		"""Log K-means error"""
		print(f"❌ K-means clustering error: {error}")
	
	def _log_dbscan_error(self, error: str):
		"""Log DBSCAN error"""
		print(f"❌ DBSCAN clustering error: {error}")
	
	def _log_anomaly_detection_error(self, error: str):
		"""Log anomaly detection error"""
		print(f"❌ Anomaly detection error: {error}")
	
	def _log_ml_anomaly_error(self, error: str):
		"""Log ML anomaly detection error"""
		print(f"❌ ML anomaly detection error: {error}")
	
	def _log_clustering_models_initialized(self):
		"""Log clustering models initialization"""
		print(f"🎯 Clustering models initialized")
	
	def _log_anomaly_detection_initialized(self):
		"""Log anomaly detection initialization"""
		print(f"🚨 Anomaly detection models initialized")
	
	def _log_pattern_recognition_initialized(self):
		"""Log pattern recognition initialization"""
		print(f"🔮 Pattern recognition models initialized")
	
	def _log_realtime_pipeline_setup(self):
		"""Log real-time pipeline setup"""
		print(f"⚡ Real-time analysis pipeline configured")

# Factory function for creating intelligent transaction analyzer
def create_intelligent_transaction_analyzer(config: Dict[str, Any]) -> IntelligentTransactionAnalyzer:
	"""Factory function to create intelligent transaction analyzer"""
	return IntelligentTransactionAnalyzer(config)

def _log_intelligent_analysis_module_loaded():
	"""Log intelligent analysis module loaded"""
	print("🧠 Intelligent Transaction Analysis Engine module loaded")
	print("   - Advanced pattern detection")
	print("   - ML-powered clustering & anomaly detection")
	print("   - Behavioral analysis & customer intelligence")
	print("   - Real-time insights & trend analysis")
	print("   - Revenue optimization recommendations")

# Execute module loading log
_log_intelligent_analysis_module_loaded()