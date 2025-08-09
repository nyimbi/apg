"""
APG Billing ML Churn Prediction

Machine learning-based churn prediction system using real ML models
with feature engineering, model training, and real-time prediction capabilities.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

from .models import BLCustomer, BLSubscription, BLPayment, BLUsage, BLInvoice, SubscriptionStatus, PaymentStatus


class ChurnPredictionEngine:
	"""ML-based churn prediction engine with real model training and inference"""
	
	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.ChurnPredictionEngine")
		
		# ML models
		self.models = {
			'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
			'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
			'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
		}
		self.best_model = None
		self.scaler = StandardScaler()
		self.label_encoders = {}
		self.feature_columns = []
		
		# Model performance tracking
		self.model_metrics = {}
		self.training_history = []
		
		# Feature importance
		self.feature_importance = {}
		
		# Prediction thresholds
		self.churn_threshold = 0.7  # High risk threshold
		self.warning_threshold = 0.4  # Warning threshold
		
		# Model versioning
		self.model_version = "1.0.0"
		self.last_training_date = None
		
		# AI orchestration integration
		self._ai_orchestration_available = False
		asyncio.create_task(self._initialize_ai_integration())
	
	async def _initialize_ai_integration(self) -> None:
		"""Initialize AI orchestration for advanced ML features"""
		try:
			from capabilities.common.ai_orchestration import get_orchestration_service
			self.ai_orchestration = get_orchestration_service()
			self._ai_orchestration_available = True
			self.logger.info("✅ AI orchestration integration initialized for churn prediction")
		except ImportError:
			self.logger.warning("⚠️  AI orchestration not available for advanced ML features")
	
	def extract_features(self, customers: List[BLCustomer], subscriptions: List[BLSubscription],
						payments: List[BLPayment], usage_records: List[BLUsage], 
						invoices: List[BLInvoice]) -> pd.DataFrame:
		"""Extract features for churn prediction"""
		try:
			features_list = []
			
			for customer in customers:
				# Get customer's subscriptions
				customer_subscriptions = [s for s in subscriptions if s.customer_id == customer.id]
				customer_payments = [p for p in payments if p.customer_id == customer.id]
				customer_usage = [u for u in usage_records if u.customer_id == customer.id]
				customer_invoices = [i for i in invoices if i.customer_id == customer.id]
				
				if not customer_subscriptions:
					continue
				
				# Extract features for each subscription
				for subscription in customer_subscriptions:
					features = self._extract_subscription_features(
						customer, subscription, customer_payments, customer_usage, customer_invoices
					)
					if features:
						features_list.append(features)
			
			if not features_list:
				return pd.DataFrame()
			
			df = pd.DataFrame(features_list)
			self.feature_columns = df.columns.tolist()
			
			# Remove target column if present
			if 'churned' in df.columns:
				self.feature_columns.remove('churned')
			
			return df
		
		except Exception as e:
			self.logger.error(f"Feature extraction failed: {e}")
			raise
	
	def _extract_subscription_features(self, customer: BLCustomer, subscription: BLSubscription,
									  payments: List[BLPayment], usage_records: List[BLUsage],
									  invoices: List[BLInvoice]) -> Optional[Dict[str, Any]]:
		"""Extract features for a single subscription"""
		try:
			now = datetime.utcnow()
			subscription_age_days = (now - subscription.created_at).days
			
			# Basic subscription features
			features = {
				'customer_id': customer.id,
				'subscription_id': subscription.id,
				'subscription_age_days': subscription_age_days,
				'billing_period': subscription.billing_period.value if subscription.billing_period else 'monthly',
				'plan_type': getattr(subscription, 'plan_type', 'standard'),
				'mrr': float(getattr(subscription, 'mrr', 0)),
				'contract_term_months': subscription.contract_term_months or 0,
				'auto_renewal': subscription.auto_renewal,
				'trial_used': subscription.trial_start is not None,
				'trial_converted': subscription.trial_start is not None and subscription.status == SubscriptionStatus.ACTIVE
			}
			
			# Customer features
			features.update({
				'customer_tier': getattr(customer, 'tier', 'standard'),
				'company_size': getattr(customer, 'company_size', 'unknown'),
				'industry': getattr(customer, 'industry', 'unknown'),
				'signup_channel': getattr(customer, 'signup_channel', 'unknown')
			})
			
			# Payment behavior features
			subscription_payments = [p for p in payments if p.subscription_id == subscription.id]
			features.update(self._extract_payment_features(subscription_payments))
			
			# Usage behavior features
			subscription_usage = [u for u in usage_records if u.subscription_id == subscription.id]
			features.update(self._extract_usage_features(subscription_usage))
			
			# Invoice features
			subscription_invoices = [i for i in invoices if i.subscription_id == subscription.id]
			features.update(self._extract_invoice_features(subscription_invoices))
			
			# Target variable (for training)
			features['churned'] = subscription.status in [SubscriptionStatus.CANCELLED, SubscriptionStatus.EXPIRED]
			
			return features
		
		except Exception as e:
			self.logger.error(f"Subscription feature extraction failed: {e}")
			return None
	
	def _extract_payment_features(self, payments: List[BLPayment]) -> Dict[str, Any]:
		"""Extract payment-related features"""
		if not payments:
			return {
				'total_payments': 0,
				'failed_payments': 0,
				'payment_failure_rate': 0.0,
				'avg_payment_amount': 0.0,
				'days_since_last_payment': 999,
				'payment_method_changes': 0
			}
		
		# Sort payments by date
		payments.sort(key=lambda p: p.created_at)
		
		failed_payments = [p for p in payments if p.status == PaymentStatus.FAILED]
		successful_payments = [p for p in payments if p.status == PaymentStatus.SUCCEEDED]
		
		# Payment failure rate
		failure_rate = len(failed_payments) / len(payments) if payments else 0
		
		# Average payment amount
		avg_amount = float(sum(p.amount for p in successful_payments) / len(successful_payments)) if successful_payments else 0
		
		# Days since last payment
		last_payment_date = max(p.created_at for p in payments)
		days_since_last = (datetime.utcnow() - last_payment_date).days
		
		# Payment method changes (simplified)
		unique_methods = len(set(p.payment_method for p in payments if p.payment_method))
		
		return {
			'total_payments': len(payments),
			'failed_payments': len(failed_payments),
			'payment_failure_rate': failure_rate,
			'avg_payment_amount': avg_amount,
			'days_since_last_payment': days_since_last,
			'payment_method_changes': max(0, unique_methods - 1)
		}
	
	def _extract_usage_features(self, usage_records: List[BLUsage]) -> Dict[str, Any]:
		"""Extract usage-related features"""
		if not usage_records:
			return {
				'total_usage_events': 0,
				'usage_trend': 0.0,
				'days_since_last_usage': 999,
				'unique_metrics_used': 0,
				'avg_usage_per_day': 0.0
			}
		
		# Sort usage by timestamp
		usage_records.sort(key=lambda u: u.timestamp)
		
		# Usage trend (comparing first half to second half)
		mid_point = len(usage_records) // 2
		if mid_point > 0:
			early_usage = sum(u.quantity for u in usage_records[:mid_point])
			late_usage = sum(u.quantity for u in usage_records[mid_point:])
			usage_trend = (late_usage - early_usage) / max(early_usage, 1)
		else:
			usage_trend = 0.0
		
		# Days since last usage
		last_usage_date = max(u.timestamp for u in usage_records)
		days_since_last = (datetime.utcnow() - last_usage_date).days
		
		# Unique metrics used
		unique_metrics = len(set(u.metric_name for u in usage_records))
		
		# Average usage per day
		if usage_records:
			first_usage = min(u.timestamp for u in usage_records)
			last_usage = max(u.timestamp for u in usage_records)
			usage_days = max(1, (last_usage - first_usage).days)
			avg_usage_per_day = float(sum(u.quantity for u in usage_records)) / usage_days
		else:
			avg_usage_per_day = 0.0
		
		return {
			'total_usage_events': len(usage_records),
			'usage_trend': usage_trend,
			'days_since_last_usage': days_since_last,
			'unique_metrics_used': unique_metrics,
			'avg_usage_per_day': avg_usage_per_day
		}
	
	def _extract_invoice_features(self, invoices: List[BLInvoice]) -> Dict[str, Any]:
		"""Extract invoice-related features"""
		if not invoices:
			return {
				'total_invoices': 0,
				'overdue_invoices': 0,
				'avg_time_to_pay': 0.0,
				'total_amount_billed': 0.0
			}
		
		overdue_invoices = [i for i in invoices if i.due_date < datetime.utcnow() and i.amount_due > 0]
		paid_invoices = [i for i in invoices if i.paid_at is not None]
		
		# Average time to pay
		if paid_invoices:
			pay_times = [(i.paid_at - i.invoice_date).days for i in paid_invoices if i.paid_at]
			avg_time_to_pay = sum(pay_times) / len(pay_times) if pay_times else 0
		else:
			avg_time_to_pay = 0
		
		# Total amount billed
		total_billed = float(sum(i.total for i in invoices))
		
		return {
			'total_invoices': len(invoices),
			'overdue_invoices': len(overdue_invoices),
			'avg_time_to_pay': avg_time_to_pay,
			'total_amount_billed': total_billed
		}
	
	def preprocess_features(self, df: pd.DataFrame, fit_encoders: bool = False) -> pd.DataFrame:
		"""Preprocess features for ML training/prediction"""
		try:
			df_processed = df.copy()
			
			# Handle categorical variables
			categorical_columns = ['billing_period', 'plan_type', 'customer_tier', 'company_size', 'industry', 'signup_channel']
			
			for col in categorical_columns:
				if col in df_processed.columns:
					if fit_encoders:
						if col not in self.label_encoders:
							self.label_encoders[col] = LabelEncoder()
						df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
					else:
						if col in self.label_encoders:
							# Handle unseen categories
							unique_values = set(df_processed[col].astype(str))
							known_values = set(self.label_encoders[col].classes_)
							unknown_values = unique_values - known_values
							
							if unknown_values:
								# Map unknown values to the most common class
								most_common = self.label_encoders[col].classes_[0]
								df_processed[col] = df_processed[col].astype(str).replace(list(unknown_values), most_common)
							
							df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
			
			# Handle boolean columns
			boolean_columns = ['auto_renewal', 'trial_used', 'trial_converted']
			for col in boolean_columns:
				if col in df_processed.columns:
					df_processed[col] = df_processed[col].astype(int)
			
			# Fill missing values
			numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
			df_processed[numeric_columns] = df_processed[numeric_columns].fillna(0)
			
			return df_processed
		
		except Exception as e:
			self.logger.error(f"Feature preprocessing failed: {e}")
			raise
	
	async def train_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
		"""Train churn prediction models"""
		try:
			if training_data.empty or 'churned' not in training_data.columns:
				raise ValueError("Training data must contain 'churned' target column")
			
			# Preprocess features
			processed_data = self.preprocess_features(training_data, fit_encoders=True)
			
			# Separate features and target
			feature_cols = [col for col in processed_data.columns if col not in ['customer_id', 'subscription_id', 'churned']]
			X = processed_data[feature_cols]
			y = processed_data['churned']
			
			# Split data
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
			
			# Scale features
			X_train_scaled = self.scaler.fit_transform(X_train)
			X_test_scaled = self.scaler.transform(X_test)
			
			# Train models
			model_results = {}
			
			for model_name, model in self.models.items():
				self.logger.info(f"Training {model_name} model...")
				
				# Train model
				model.fit(X_train_scaled, y_train)
				
				# Make predictions
				y_pred = model.predict(X_test_scaled)
				y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
				
				# Calculate metrics
				metrics = {
					'accuracy': accuracy_score(y_test, y_pred),
					'precision': precision_score(y_test, y_pred, zero_division=0),
					'recall': recall_score(y_test, y_pred, zero_division=0),
					'f1_score': f1_score(y_test, y_pred, zero_division=0),
					'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
				}
				
				# Cross-validation
				cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
				metrics['cv_auc_mean'] = cv_scores.mean()
				metrics['cv_auc_std'] = cv_scores.std()
				
				model_results[model_name] = metrics
				self.model_metrics[model_name] = metrics
				
				# Feature importance
				if hasattr(model, 'feature_importances_'):
					importance = dict(zip(feature_cols, model.feature_importances_))
					self.feature_importance[model_name] = importance
			
			# Select best model based on ROC AUC
			best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['roc_auc'])
			self.best_model = self.models[best_model_name]
			
			# Save training history
			training_record = {
				'timestamp': datetime.utcnow().isoformat(),
				'model_version': self.model_version,
				'training_samples': len(training_data),
				'churn_rate': float(y.mean()),
				'best_model': best_model_name,
				'model_metrics': model_results
			}
			self.training_history.append(training_record)
			self.last_training_date = datetime.utcnow()
			
			self.logger.info(f"Model training completed. Best model: {best_model_name}")
			
			return {
				'training_completed': True,
				'best_model': best_model_name,
				'model_metrics': model_results,
				'feature_importance': self.feature_importance.get(best_model_name, {}),
				'training_samples': len(training_data),
				'churn_rate': float(y.mean())
			}
		
		except Exception as e:
			self.logger.error(f"Model training failed: {e}")
			raise
	
	async def predict_churn_probability(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Predict churn probability for a customer/subscription"""
		try:
			if self.best_model is None:
				return {
					'error': 'No trained model available',
					'churn_probability': 0.5,
					'risk_level': 'unknown'
				}
			
			# Convert to DataFrame
			df = pd.DataFrame([customer_data])
			
			# Preprocess
			processed_df = self.preprocess_features(df, fit_encoders=False)
			
			# Select features
			feature_cols = [col for col in processed_df.columns if col in self.feature_columns]
			X = processed_df[feature_cols]
			
			# Ensure all required features are present
			missing_features = set(self.feature_columns) - set(feature_cols)
			if missing_features:
				for feature in missing_features:
					X[feature] = 0
			
			# Reorder columns to match training
			X = X[self.feature_columns]
			
			# Scale features
			X_scaled = self.scaler.transform(X)
			
			# Predict
			churn_probability = self.best_model.predict_proba(X_scaled)[0, 1]
			
			# Determine risk level
			if churn_probability >= self.churn_threshold:
				risk_level = 'high'
			elif churn_probability >= self.warning_threshold:
				risk_level = 'medium'
			else:
				risk_level = 'low'
			
			# Generate recommendations
			recommendations = self._generate_retention_recommendations(churn_probability, customer_data)
			
			return {
				'churn_probability': float(churn_probability),
				'risk_level': risk_level,
				'recommendations': recommendations,
				'model_version': self.model_version,
				'prediction_timestamp': datetime.utcnow().isoformat()
			}
		
		except Exception as e:
			self.logger.error(f"Churn prediction failed: {e}")
			return {
				'error': str(e),
				'churn_probability': 0.5,
				'risk_level': 'unknown'
			}
	
	def _generate_retention_recommendations(self, churn_probability: float, customer_data: Dict[str, Any]) -> List[str]:
		"""Generate retention recommendations based on churn risk"""
		recommendations = []
		
		if churn_probability >= 0.8:
			recommendations.extend([
				"Immediate intervention required - contact customer within 24 hours",
				"Offer personalized discount or contract renegotiation",
				"Schedule executive-level check-in call"
			])
		elif churn_probability >= 0.6:
			recommendations.extend([
				"Schedule customer success review meeting",
				"Provide additional training or onboarding resources",
				"Offer premium support or dedicated account manager"
			])
		elif churn_probability >= 0.4:
			recommendations.extend([
				"Monitor usage patterns more closely",
				"Send targeted educational content",
				"Invite to user community or events"
			])
		
		# Feature-specific recommendations
		if customer_data.get('payment_failure_rate', 0) > 0.2:
			recommendations.append("Address payment method issues - contact billing team")
		
		if customer_data.get('usage_trend', 0) < -0.3:
			recommendations.append("Usage declining - provide feature adoption assistance")
		
		if customer_data.get('days_since_last_usage', 0) > 30:
			recommendations.append("Re-engagement campaign needed - customer inactive")
		
		return recommendations[:5]  # Return top 5 recommendations
	
	async def batch_predict_churn(self, customers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
		"""Predict churn for multiple customers"""
		predictions = []
		
		for customer_data in customers_data:
			try:
				prediction = await self.predict_churn_probability(customer_data)
				prediction['customer_id'] = customer_data.get('customer_id')
				prediction['subscription_id'] = customer_data.get('subscription_id')
				predictions.append(prediction)
			except Exception as e:
				self.logger.error(f"Batch prediction failed for customer {customer_data.get('customer_id')}: {e}")
				predictions.append({
					'customer_id': customer_data.get('customer_id'),
					'subscription_id': customer_data.get('subscription_id'),
					'error': str(e),
					'churn_probability': 0.5,
					'risk_level': 'unknown'
				})
		
		return predictions
	
	def save_model(self, filepath: str) -> bool:
		"""Save trained model to disk"""
		try:
			model_data = {
				'best_model': self.best_model,
				'scaler': self.scaler,
				'label_encoders': self.label_encoders,
				'feature_columns': self.feature_columns,
				'model_metrics': self.model_metrics,
				'feature_importance': self.feature_importance,
				'model_version': self.model_version,
				'last_training_date': self.last_training_date,
				'churn_threshold': self.churn_threshold,
				'warning_threshold': self.warning_threshold
			}
			
			joblib.dump(model_data, filepath)
			self.logger.info(f"Model saved to {filepath}")
			return True
		
		except Exception as e:
			self.logger.error(f"Model save failed: {e}")
			return False
	
	def load_model(self, filepath: str) -> bool:
		"""Load trained model from disk"""
		try:
			model_data = joblib.load(filepath)
			
			self.best_model = model_data['best_model']
			self.scaler = model_data['scaler']
			self.label_encoders = model_data['label_encoders']
			self.feature_columns = model_data['feature_columns']
			self.model_metrics = model_data['model_metrics']
			self.feature_importance = model_data['feature_importance']
			self.model_version = model_data['model_version']
			self.last_training_date = model_data['last_training_date']
			self.churn_threshold = model_data.get('churn_threshold', 0.7)
			self.warning_threshold = model_data.get('warning_threshold', 0.4)
			
			self.logger.info(f"Model loaded from {filepath}")
			return True
		
		except Exception as e:
			self.logger.error(f"Model load failed: {e}")
			return False
	
	async def get_model_performance_report(self) -> Dict[str, Any]:
		"""Generate model performance report"""
		try:
			if not self.model_metrics:
				return {'error': 'No trained models available'}
			
			# Find best performing model
			best_model_name = max(self.model_metrics.keys(), key=lambda k: self.model_metrics[k]['roc_auc'])
			best_metrics = self.model_metrics[best_model_name]
			
			return {
				'model_version': self.model_version,
				'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
				'best_model': best_model_name,
				'performance_metrics': best_metrics,
				'feature_importance': self.feature_importance.get(best_model_name, {}),
				'model_comparison': self.model_metrics,
				'training_history_count': len(self.training_history),
				'thresholds': {
					'churn_threshold': self.churn_threshold,
					'warning_threshold': self.warning_threshold
				}
			}
		
		except Exception as e:
			self.logger.error(f"Performance report generation failed: {e}")
			return {'error': str(e)}


# Global churn prediction engine
_churn_engine_instance: Optional[ChurnPredictionEngine] = None

def get_churn_prediction_engine() -> ChurnPredictionEngine:
	"""Get global churn prediction engine instance"""
	global _churn_engine_instance
	if _churn_engine_instance is None:
		_churn_engine_instance = ChurnPredictionEngine()
	return _churn_engine_instance


__all__ = [
	'ChurnPredictionEngine',
	'get_churn_prediction_engine'
]