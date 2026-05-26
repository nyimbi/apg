"""
Stripe Reporting and Analytics Service - APG Payment Gateway

Complete reporting and analytics implementation for Stripe integration:
- Payment analytics and transaction reporting
- Revenue analytics and financial metrics
- Customer analytics and behavior insights
- Subscription analytics and churn analysis
- Fraud detection and risk analytics
- Performance monitoring and health metrics
- Custom report generation and data exports

© 2025 Datacraft. All rights reserved.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import csv
import io
import zipfile
from xml.sax.saxutils import escape as xml_escape

import stripe
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class ReportPeriod(str, Enum):
    """Report time period options"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Report output format options"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"


class MetricType(str, Enum):
    """Types of metrics to track"""
    REVENUE = "revenue"
    VOLUME = "volume"
    COUNT = "count"
    PERCENTAGE = "percentage"
    AVERAGE = "average"
    MEDIAN = "median"
    GROWTH_RATE = "growth_rate"


@dataclass
class ReportFilter:
    """Report filtering options"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    currency: Optional[str] = None
    customer_ids: Optional[List[str]] = None
    product_ids: Optional[List[str]] = None
    payment_method_types: Optional[List[str]] = None
    status: Optional[List[str]] = None
    metadata: Optional[Dict[str, str]] = None
    min_amount: Optional[int] = None  # in cents
    max_amount: Optional[int] = None  # in cents


@dataclass
class MetricValue:
    """Metric value with metadata"""
    value: Union[int, float, Decimal]
    type: MetricType
    period: ReportPeriod
    timestamp: datetime
    currency: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class PaymentAnalytics:
    """Payment analytics data"""
    total_revenue: Decimal
    total_transactions: int
    successful_transactions: int
    failed_transactions: int
    average_transaction_value: Decimal
    median_transaction_value: Decimal
    success_rate: float
    failure_rate: float
    revenue_by_currency: Dict[str, Decimal]
    transactions_by_payment_method: Dict[str, int]
    revenue_by_payment_method: Dict[str, Decimal]
    top_customers: List[Dict[str, Any]]
    period_over_period_growth: float
    chargeback_rate: float
    refund_rate: float


@dataclass
class CustomerAnalytics:
    """Customer analytics data"""
    total_customers: int
    new_customers: int
    active_customers: int
    customers_with_failed_payments: int
    customer_acquisition_cost: Decimal
    customer_lifetime_value: Decimal
    average_revenue_per_customer: Decimal
    customer_retention_rate: float
    customer_churn_rate: float
    top_customers_by_revenue: List[Dict[str, Any]]
    customer_segmentation: Dict[str, int]
    payment_method_adoption: Dict[str, float]


@dataclass
class SubscriptionAnalytics:
    """Subscription analytics data"""
    total_subscriptions: int
    active_subscriptions: int
    canceled_subscriptions: int
    paused_subscriptions: int
    past_due_subscriptions: int
    trial_subscriptions: int
    monthly_recurring_revenue: Decimal
    annual_recurring_revenue: Decimal
    average_revenue_per_user: Decimal
    churn_rate: float
    growth_rate: float
    lifetime_value: Decimal
    trial_conversion_rate: float
    subscription_upgrades: int
    subscription_downgrades: int
    revenue_by_plan: Dict[str, Decimal]


@dataclass
class FraudAnalytics:
    """Fraud and risk analytics"""
    total_disputed_transactions: int
    total_dispute_amount: Decimal
    dispute_rate: float
    chargeback_rate: float
    fraud_detection_accuracy: float
    blocked_transactions: int
    blocked_amount: Decimal
    false_positive_rate: float
    high_risk_transactions: int
    radar_risk_score_distribution: Dict[str, int]
    top_fraud_indicators: List[Dict[str, Any]]


class StripeReportingService:
    """Complete Stripe reporting and analytics service"""
    
    def __init__(
        self,
        stripe_client: stripe.StripeClient,
        config: Dict[str, Any]
    ):
        self.stripe = stripe_client
        self.config = config
        self._cache = {}
        self._cache_ttl = config.get("cache_ttl", 300)  # 5 minutes
        
        logger.info("Stripe reporting service initialized")
    
    async def generate_payment_analytics(
        self,
        period: ReportPeriod = ReportPeriod.MONTH,
        filters: Optional[ReportFilter] = None
    ) -> PaymentAnalytics:
        """
        Generate comprehensive payment analytics
        
        Args:
            period: Report period
            filters: Optional filters to apply
            
        Returns:
            PaymentAnalytics object with all metrics
        """
        try:
            logger.info(f"Generating payment analytics for period: {period.value}")
            
            # Set default date range if not provided
            if not filters:
                filters = ReportFilter()
            
            if not filters.start_date or not filters.end_date:
                filters.start_date, filters.end_date = self._get_period_dates(period)
            
            # Get payment intents data
            payment_intents_data = await self._get_payment_intents_data(filters)
            
            # Get charges data for more detailed analysis
            charges_data = await self._get_charges_data(filters)
            
            # Calculate metrics
            total_revenue = self._calculate_total_revenue(payment_intents_data)
            total_transactions = len(payment_intents_data)
            successful_transactions = len([p for p in payment_intents_data if p.status == 'succeeded'])
            failed_transactions = total_transactions - successful_transactions
            
            success_rate = successful_transactions / total_transactions if total_transactions > 0 else 0.0
            failure_rate = 1.0 - success_rate
            
            # Calculate transaction values
            successful_amounts = [p.amount for p in payment_intents_data if p.status == 'succeeded']
            average_transaction_value = Decimal(sum(successful_amounts) / len(successful_amounts)) / 100 if successful_amounts else Decimal(0)
            median_transaction_value = Decimal(sorted(successful_amounts)[len(successful_amounts) // 2]) / 100 if successful_amounts else Decimal(0)
            
            # Revenue by currency
            revenue_by_currency = self._calculate_revenue_by_currency(payment_intents_data)
            
            # Transactions by payment method
            transactions_by_payment_method = self._calculate_transactions_by_payment_method(charges_data)
            revenue_by_payment_method = self._calculate_revenue_by_payment_method(charges_data)
            
            # Top customers
            top_customers = await self._get_top_customers(payment_intents_data)
            
            # Period over period growth
            previous_period_filters = self._get_previous_period_filters(filters, period)
            previous_revenue = await self._get_previous_period_revenue(previous_period_filters)
            period_over_period_growth = self._calculate_growth_rate(total_revenue, previous_revenue)
            
            # Chargeback and refund rates
            chargeback_rate = await self._calculate_chargeback_rate(filters)
            refund_rate = await self._calculate_refund_rate(filters)
            
            return PaymentAnalytics(
                total_revenue=total_revenue,
                total_transactions=total_transactions,
                successful_transactions=successful_transactions,
                failed_transactions=failed_transactions,
                average_transaction_value=average_transaction_value,
                median_transaction_value=median_transaction_value,
                success_rate=success_rate,
                failure_rate=failure_rate,
                revenue_by_currency=revenue_by_currency,
                transactions_by_payment_method=transactions_by_payment_method,
                revenue_by_payment_method=revenue_by_payment_method,
                top_customers=top_customers,
                period_over_period_growth=period_over_period_growth,
                chargeback_rate=chargeback_rate,
                refund_rate=refund_rate
            )
            
        except Exception as e:
            logger.error(f"Error generating payment analytics: {str(e)}")
            raise
    
    async def generate_customer_analytics(
        self,
        period: ReportPeriod = ReportPeriod.MONTH,
        filters: Optional[ReportFilter] = None
    ) -> CustomerAnalytics:
        """
        Generate comprehensive customer analytics
        
        Args:
            period: Report period
            filters: Optional filters to apply
            
        Returns:
            CustomerAnalytics object with all metrics
        """
        try:
            logger.info(f"Generating customer analytics for period: {period.value}")
            
            if not filters:
                filters = ReportFilter()
            
            if not filters.start_date or not filters.end_date:
                filters.start_date, filters.end_date = self._get_period_dates(period)
            
            # Get customers data
            customers_data = await self._get_customers_data(filters)
            
            # Get payment data for customer analysis
            payment_intents_data = await self._get_payment_intents_data(filters)
            
            # Calculate basic metrics
            total_customers = len(customers_data)
            new_customers = len([c for c in customers_data if c.created >= int(filters.start_date.timestamp())])
            
            # Active customers (customers with successful payments)
            customer_payment_map = self._map_customers_to_payments(customers_data, payment_intents_data)
            active_customers = len([c for c, payments in customer_payment_map.items() 
                                 if any(p.status == 'succeeded' for p in payments)])
            
            # Customers with failed payments
            customers_with_failed_payments = len([c for c, payments in customer_payment_map.items()
                                                if any(p.status == 'requires_payment_method' for p in payments)])
            
            # Revenue metrics
            total_revenue = self._calculate_total_revenue(payment_intents_data)
            customer_acquisition_cost = await self._calculate_customer_acquisition_cost(filters)
            customer_lifetime_value = await self._calculate_customer_lifetime_value(filters)
            average_revenue_per_customer = total_revenue / total_customers if total_customers > 0 else Decimal(0)
            
            # Retention and churn
            customer_retention_rate = await self._calculate_retention_rate(filters, period)
            customer_churn_rate = 1.0 - customer_retention_rate
            
            # Top customers by revenue
            top_customers_by_revenue = await self._get_top_customers_by_revenue(customer_payment_map)
            
            # Customer segmentation
            customer_segmentation = self._segment_customers(customer_payment_map)
            
            # Payment method adoption
            payment_method_adoption = await self._calculate_payment_method_adoption(customers_data)
            
            return CustomerAnalytics(
                total_customers=total_customers,
                new_customers=new_customers,
                active_customers=active_customers,
                customers_with_failed_payments=customers_with_failed_payments,
                customer_acquisition_cost=customer_acquisition_cost,
                customer_lifetime_value=customer_lifetime_value,
                average_revenue_per_customer=average_revenue_per_customer,
                customer_retention_rate=customer_retention_rate,
                customer_churn_rate=customer_churn_rate,
                top_customers_by_revenue=top_customers_by_revenue,
                customer_segmentation=customer_segmentation,
                payment_method_adoption=payment_method_adoption
            )
            
        except Exception as e:
            logger.error(f"Error generating customer analytics: {str(e)}")
            raise
    
    async def generate_subscription_analytics(
        self,
        period: ReportPeriod = ReportPeriod.MONTH,
        filters: Optional[ReportFilter] = None
    ) -> SubscriptionAnalytics:
        """
        Generate comprehensive subscription analytics
        
        Args:
            period: Report period
            filters: Optional filters to apply
            
        Returns:
            SubscriptionAnalytics object with all metrics
        """
        try:
            logger.info(f"Generating subscription analytics for period: {period.value}")
            
            if not filters:
                filters = ReportFilter()
            
            if not filters.start_date or not filters.end_date:
                filters.start_date, filters.end_date = self._get_period_dates(period)
            
            # Get subscriptions data
            subscriptions_data = await self._get_subscriptions_data(filters)
            
            # Basic counts
            total_subscriptions = len(subscriptions_data)
            active_subscriptions = len([s for s in subscriptions_data if s.status == 'active'])
            canceled_subscriptions = len([s for s in subscriptions_data if s.status == 'canceled'])
            paused_subscriptions = len([s for s in subscriptions_data if s.status == 'paused'])
            past_due_subscriptions = len([s for s in subscriptions_data if s.status == 'past_due'])
            trial_subscriptions = len([s for s in subscriptions_data if s.status == 'trialing'])
            
            # Revenue metrics
            monthly_recurring_revenue = await self._calculate_mrr(subscriptions_data)
            annual_recurring_revenue = monthly_recurring_revenue * 12
            average_revenue_per_user = monthly_recurring_revenue / active_subscriptions if active_subscriptions > 0 else Decimal(0)
            
            # Churn and growth
            churn_rate = await self._calculate_subscription_churn_rate(filters, period)
            growth_rate = await self._calculate_subscription_growth_rate(filters, period)
            
            # Lifetime value
            lifetime_value = await self._calculate_subscription_ltv(subscriptions_data)
            
            # Trial conversion
            trial_conversion_rate = await self._calculate_trial_conversion_rate(filters)
            
            # Upgrades/downgrades
            subscription_changes = await self._get_subscription_changes(filters)
            subscription_upgrades = subscription_changes.get('upgrades', 0)
            subscription_downgrades = subscription_changes.get('downgrades', 0)
            
            # Revenue by plan
            revenue_by_plan = await self._calculate_revenue_by_plan(subscriptions_data)
            
            return SubscriptionAnalytics(
                total_subscriptions=total_subscriptions,
                active_subscriptions=active_subscriptions,
                canceled_subscriptions=canceled_subscriptions,
                paused_subscriptions=paused_subscriptions,
                past_due_subscriptions=past_due_subscriptions,
                trial_subscriptions=trial_subscriptions,
                monthly_recurring_revenue=monthly_recurring_revenue,
                annual_recurring_revenue=annual_recurring_revenue,
                average_revenue_per_user=average_revenue_per_user,
                churn_rate=churn_rate,
                growth_rate=growth_rate,
                lifetime_value=lifetime_value,
                trial_conversion_rate=trial_conversion_rate,
                subscription_upgrades=subscription_upgrades,
                subscription_downgrades=subscription_downgrades,
                revenue_by_plan=revenue_by_plan
            )
            
        except Exception as e:
            logger.error(f"Error generating subscription analytics: {str(e)}")
            raise
    
    async def generate_fraud_analytics(
        self,
        period: ReportPeriod = ReportPeriod.MONTH,
        filters: Optional[ReportFilter] = None
    ) -> FraudAnalytics:
        """
        Generate comprehensive fraud and risk analytics
        
        Args:
            period: Report period
            filters: Optional filters to apply
            
        Returns:
            FraudAnalytics object with all metrics
        """
        try:
            logger.info(f"Generating fraud analytics for period: {period.value}")
            
            if not filters:
                filters = ReportFilter()
            
            if not filters.start_date or not filters.end_date:
                filters.start_date, filters.end_date = self._get_period_dates(period)
            
            # Get dispute data
            disputes_data = await self._get_disputes_data(filters)
            
            # Get charges for dispute rate calculation
            charges_data = await self._get_charges_data(filters)
            
            # Basic dispute metrics
            total_disputed_transactions = len(disputes_data)
            total_dispute_amount = Decimal(sum(d.amount for d in disputes_data)) / 100
            
            total_charge_amount = Decimal(sum(c.amount for c in charges_data if c.status == 'succeeded')) / 100
            dispute_rate = total_dispute_amount / total_charge_amount if total_charge_amount > 0 else 0.0
            chargeback_rate = total_disputed_transactions / len(charges_data) if len(charges_data) > 0 else 0.0
            
            # Radar analytics (if enabled)
            radar_analytics = await self._get_radar_analytics(filters)
            fraud_detection_accuracy = radar_analytics.get('accuracy', 0.0)
            blocked_transactions = radar_analytics.get('blocked_transactions', 0)
            blocked_amount = Decimal(radar_analytics.get('blocked_amount', 0)) / 100
            false_positive_rate = radar_analytics.get('false_positive_rate', 0.0)
            high_risk_transactions = radar_analytics.get('high_risk_transactions', 0)
            
            # Risk score distribution
            radar_risk_score_distribution = radar_analytics.get('risk_score_distribution', {})
            
            # Top fraud indicators
            top_fraud_indicators = await self._get_top_fraud_indicators(disputes_data)
            
            return FraudAnalytics(
                total_disputed_transactions=total_disputed_transactions,
                total_dispute_amount=total_dispute_amount,
                dispute_rate=dispute_rate,
                chargeback_rate=chargeback_rate,
                fraud_detection_accuracy=fraud_detection_accuracy,
                blocked_transactions=blocked_transactions,
                blocked_amount=blocked_amount,
                false_positive_rate=false_positive_rate,
                high_risk_transactions=high_risk_transactions,
                radar_risk_score_distribution=radar_risk_score_distribution,
                top_fraud_indicators=top_fraud_indicators
            )
            
        except Exception as e:
            logger.error(f"Error generating fraud analytics: {str(e)}")
            raise
    
    async def generate_custom_report(
        self,
        metrics: List[str],
        period: ReportPeriod = ReportPeriod.MONTH,
        filters: Optional[ReportFilter] = None,
        format: ReportFormat = ReportFormat.JSON
    ) -> Dict[str, Any]:
        """
        Generate custom report with specified metrics
        
        Args:
            metrics: List of metric names to include
            period: Report period
            filters: Optional filters to apply
            format: Output format
            
        Returns:
            Report data in specified format
        """
        try:
            logger.info(f"Generating custom report with metrics: {metrics}")
            
            report_data = {
                "report_id": f"custom_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.utcnow().isoformat(),
                "period": period.value,
                "filters": self._serialize_filters(filters) if filters else None,
                "metrics": {}
            }
            
            # Generate requested metrics
            for metric in metrics:
                try:
                    metric_value = await self._calculate_custom_metric(metric, period, filters)
                    report_data["metrics"][metric] = metric_value
                except Exception as e:
                    logger.warning(f"Failed to calculate metric {metric}: {str(e)}")
                    report_data["metrics"][metric] = {
                        "error": str(e),
                        "value": None
                    }
            
            # Format output
            if format == ReportFormat.JSON:
                return report_data
            elif format == ReportFormat.CSV:
                return self._format_as_csv(report_data)
            elif format == ReportFormat.EXCEL:
                return self._format_as_excel(report_data)
            else:
                return report_data
            
        except Exception as e:
            logger.error(f"Error generating custom report: {str(e)}")
            raise
    
    async def export_transaction_data(
        self,
        filters: Optional[ReportFilter] = None,
        format: ReportFormat = ReportFormat.CSV,
        limit: int = 10000
    ) -> Union[str, bytes]:
        """
        Export raw transaction data
        
        Args:
            filters: Optional filters to apply
            format: Output format
            limit: Maximum number of records
            
        Returns:
            Exported data in specified format
        """
        try:
            logger.info(f"Exporting transaction data in {format.value} format")
            
            # Get payment intents data
            payment_intents = await self._get_payment_intents_data(filters, limit=limit)
            
            # Prepare data for export
            export_data = []
            for payment_intent in payment_intents:
                export_data.append({
                    "id": payment_intent.id,
                    "amount": payment_intent.amount / 100,  # Convert from cents
                    "currency": payment_intent.currency,
                    "status": payment_intent.status,
                    "created": datetime.fromtimestamp(payment_intent.created).isoformat(),
                    "customer_id": payment_intent.customer,
                    "payment_method": payment_intent.payment_method_types[0] if payment_intent.payment_method_types else None,
                    "description": payment_intent.description,
                    "metadata": json.dumps(payment_intent.metadata) if payment_intent.metadata else None
                })
            
            # Format output
            if format == ReportFormat.CSV:
                return self._export_as_csv(export_data)
            elif format == ReportFormat.JSON:
                return json.dumps(export_data, indent=2)
            else:
                return json.dumps(export_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error exporting transaction data: {str(e)}")
            raise
    
    # Private helper methods
    
    def _get_period_dates(self, period: ReportPeriod) -> Tuple[datetime, datetime]:
        """Get start and end dates for period"""
        now = datetime.now(timezone.utc)
        
        if period == ReportPeriod.HOUR:
            start = now.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif period == ReportPeriod.DAY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == ReportPeriod.WEEK:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=now.weekday())
            end = start + timedelta(weeks=1)
        elif period == ReportPeriod.MONTH:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        elif period == ReportPeriod.QUARTER:
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            start = now.replace(month=quarter_start_month, day=1, hour=0, minute=0, second=0, microsecond=0)
            if quarter_start_month == 10:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=quarter_start_month + 3)
        elif period == ReportPeriod.YEAR:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(year=start.year + 1)
        else:
            # Default to current month
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        
        return start, end
    
    async def _get_payment_intents_data(
        self,
        filters: Optional[ReportFilter] = None,
        limit: int = 100
    ) -> List[stripe.PaymentIntent]:
        """Get payment intents data with filters"""
        try:
            params = {"limit": limit}
            
            if filters:
                if filters.start_date:
                    params["created"] = {"gte": int(filters.start_date.timestamp())}
                if filters.end_date:
                    if "created" not in params:
                        params["created"] = {}
                    params["created"]["lte"] = int(filters.end_date.timestamp())
                
                if filters.customer_ids:
                    # Note: Stripe API doesn't support filtering by multiple customers directly
                    # This would need to be done client-side or via multiple API calls
                    pass
            
            payment_intents = []
            async for payment_intent in self.stripe.PaymentIntent.list(**params).auto_paging_iter():
                payment_intents.append(payment_intent)
                if len(payment_intents) >= limit:
                    break
            
            return payment_intents
            
        except Exception as e:
            logger.error(f"Error getting payment intents data: {str(e)}")
            return []
    
    async def _get_charges_data(
        self,
        filters: Optional[ReportFilter] = None,
        limit: int = 100
    ) -> List[stripe.Charge]:
        """Get charges data with filters"""
        try:
            params = {"limit": limit}
            
            if filters:
                if filters.start_date:
                    params["created"] = {"gte": int(filters.start_date.timestamp())}
                if filters.end_date:
                    if "created" not in params:
                        params["created"] = {}
                    params["created"]["lte"] = int(filters.end_date.timestamp())
            
            charges = []
            async for charge in self.stripe.Charge.list(**params).auto_paging_iter():
                charges.append(charge)
                if len(charges) >= limit:
                    break
            
            return charges
            
        except Exception as e:
            logger.error(f"Error getting charges data: {str(e)}")
            return []
    
    async def _get_customers_data(
        self,
        filters: Optional[ReportFilter] = None,
        limit: int = 100
    ) -> List[stripe.Customer]:
        """Get customers data with filters"""
        try:
            params = {"limit": limit}
            
            if filters:
                if filters.start_date:
                    params["created"] = {"gte": int(filters.start_date.timestamp())}
                if filters.end_date:
                    if "created" not in params:
                        params["created"] = {}
                    params["created"]["lte"] = int(filters.end_date.timestamp())
            
            customers = []
            async for customer in self.stripe.Customer.list(**params).auto_paging_iter():
                customers.append(customer)
                if len(customers) >= limit:
                    break
            
            return customers
            
        except Exception as e:
            logger.error(f"Error getting customers data: {str(e)}")
            return []
    
    async def _get_subscriptions_data(
        self,
        filters: Optional[ReportFilter] = None,
        limit: int = 100
    ) -> List[stripe.Subscription]:
        """Get subscriptions data with filters"""
        try:
            params = {"limit": limit, "status": "all"}
            
            if filters:
                if filters.start_date:
                    params["created"] = {"gte": int(filters.start_date.timestamp())}
                if filters.end_date:
                    if "created" not in params:
                        params["created"] = {}
                    params["created"]["lte"] = int(filters.end_date.timestamp())
            
            subscriptions = []
            async for subscription in self.stripe.Subscription.list(**params).auto_paging_iter():
                subscriptions.append(subscription)
                if len(subscriptions) >= limit:
                    break
            
            return subscriptions
            
        except Exception as e:
            logger.error(f"Error getting subscriptions data: {str(e)}")
            return []
    
    async def _get_disputes_data(
        self,
        filters: Optional[ReportFilter] = None,
        limit: int = 100
    ) -> List[stripe.Dispute]:
        """Get disputes data with filters"""
        try:
            params = {"limit": limit}
            
            if filters:
                if filters.start_date:
                    params["created"] = {"gte": int(filters.start_date.timestamp())}
                if filters.end_date:
                    if "created" not in params:
                        params["created"] = {}
                    params["created"]["lte"] = int(filters.end_date.timestamp())
            
            disputes = []
            async for dispute in self.stripe.Dispute.list(**params).auto_paging_iter():
                disputes.append(dispute)
                if len(disputes) >= limit:
                    break
            
            return disputes
            
        except Exception as e:
            logger.error(f"Error getting disputes data: {str(e)}")
            return []
    
    def _calculate_total_revenue(self, payment_intents: List[stripe.PaymentIntent]) -> Decimal:
        """Calculate total revenue from payment intents"""
        successful_payments = [p for p in payment_intents if p.status == 'succeeded']
        total_cents = sum(p.amount for p in successful_payments)
        return Decimal(total_cents) / 100
    
    def _calculate_revenue_by_currency(
        self, 
        payment_intents: List[stripe.PaymentIntent]
    ) -> Dict[str, Decimal]:
        """Calculate revenue by currency"""
        revenue_by_currency = {}
        
        for payment_intent in payment_intents:
            if payment_intent.status == 'succeeded':
                currency = payment_intent.currency.upper()
                amount = Decimal(payment_intent.amount) / 100
                
                if currency not in revenue_by_currency:
                    revenue_by_currency[currency] = Decimal(0)
                
                revenue_by_currency[currency] += amount
        
        return revenue_by_currency
    
    def _calculate_transactions_by_payment_method(
        self,
        charges: List[stripe.Charge]
    ) -> Dict[str, int]:
        """Calculate transaction count by payment method"""
        transactions_by_method = {}
        
        for charge in charges:
            if hasattr(charge, 'payment_method_details') and charge.payment_method_details:
                method_type = charge.payment_method_details.type
                transactions_by_method[method_type] = transactions_by_method.get(method_type, 0) + 1
        
        return transactions_by_method
    
    def _calculate_revenue_by_payment_method(
        self,
        charges: List[stripe.Charge]
    ) -> Dict[str, Decimal]:
        """Calculate revenue by payment method"""
        revenue_by_method = {}
        
        for charge in charges:
            if charge.status == 'succeeded' and hasattr(charge, 'payment_method_details') and charge.payment_method_details:
                method_type = charge.payment_method_details.type
                amount = Decimal(charge.amount) / 100
                
                if method_type not in revenue_by_method:
                    revenue_by_method[method_type] = Decimal(0)
                
                revenue_by_method[method_type] += amount
        
        return revenue_by_method
    
    def _export_as_csv(self, data: List[Dict[str, Any]]) -> str:
        """Export data as CSV string"""
        if not data:
            return ""
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    # Derived calculations that combine Stripe resource snapshots
    
    async def _get_top_customers(self, payment_intents: List[stripe.PaymentIntent]) -> List[Dict[str, Any]]:
        """Get top customers by revenue"""
        customer_revenue = {}
        
        for payment_intent in payment_intents:
            if payment_intent.status == 'succeeded' and payment_intent.customer:
                customer_id = payment_intent.customer
                amount = Decimal(payment_intent.amount) / 100
                
                if customer_id not in customer_revenue:
                    customer_revenue[customer_id] = Decimal(0)
                
                customer_revenue[customer_id] += amount
        
        # Sort by revenue and take top 10
        top_customers = sorted(customer_revenue.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return [{"customer_id": customer_id, "revenue": float(revenue)} 
                for customer_id, revenue in top_customers]
    
    def _get_previous_period_filters(
        self, 
        current_filters: ReportFilter, 
        period: ReportPeriod
    ) -> ReportFilter:
        """Get filters for previous period comparison"""
        if not current_filters.start_date or not current_filters.end_date:
            return current_filters
        
        period_length = current_filters.end_date - current_filters.start_date
        
        previous_filters = ReportFilter(
            start_date=current_filters.start_date - period_length,
            end_date=current_filters.start_date,
            currency=current_filters.currency,
            customer_ids=current_filters.customer_ids,
            product_ids=current_filters.product_ids,
            payment_method_types=current_filters.payment_method_types,
            status=current_filters.status,
            metadata=current_filters.metadata,
            min_amount=current_filters.min_amount,
            max_amount=current_filters.max_amount
        )
        
        return previous_filters
    
    async def _get_previous_period_revenue(self, filters: ReportFilter) -> Decimal:
        """Get revenue for previous period"""
        payment_intents = await self._get_payment_intents_data(filters)
        return self._calculate_total_revenue(payment_intents)
    
    def _calculate_growth_rate(self, current: Decimal, previous: Decimal) -> float:
        """Calculate period-over-period growth rate"""
        if previous == 0:
            return 1.0 if current > 0 else 0.0
        
        return float((current - previous) / previous)
    
    # Additional metric helpers

    def _attr(self, value: Any, name: str, default: Any = None) -> Any:
        """Read an attribute from either a Stripe object or dict."""
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _nested_attr(self, value: Any, path: str, default: Any = None) -> Any:
        """Read a dotted attribute path from mixed Stripe objects and dicts."""
        current = value
        for part in path.split("."):
            current = self._attr(current, part, None)
            if current is None:
                return default
        return current

    def _amount_to_decimal(self, amount: Any) -> Decimal:
        """Convert Stripe minor-unit amounts to major-unit Decimal values."""
        if amount is None:
            return Decimal("0")
        return Decimal(str(amount)) / Decimal("100")

    def _payment_revenue(self, payment_intent: stripe.PaymentIntent) -> Decimal:
        if self._attr(payment_intent, "status") != "succeeded":
            return Decimal("0")
        return self._amount_to_decimal(self._attr(payment_intent, "amount", 0))

    def _charge_revenue(self, charge: stripe.Charge) -> Decimal:
        if self._attr(charge, "status") != "succeeded":
            return Decimal("0")
        return self._amount_to_decimal(self._attr(charge, "amount", 0))

    def _subscription_monthly_amount(self, subscription: stripe.Subscription) -> Decimal:
        """Calculate normalized monthly recurring revenue for a subscription."""
        items = self._nested_attr(subscription, "items.data", []) or []
        total = Decimal("0")
        for item in items:
            price = self._attr(item, "price", {})
            recurring = self._attr(price, "recurring", {}) or {}
            interval = self._attr(recurring, "interval", "month")
            interval_count = int(self._attr(recurring, "interval_count", 1) or 1)
            unit_amount = self._attr(price, "unit_amount", None)
            if unit_amount is None:
                unit_amount = self._attr(price, "unit_amount_decimal", 0)
            quantity = int(self._attr(item, "quantity", 1) or 1)
            amount = self._amount_to_decimal(unit_amount) * quantity
            if interval == "year":
                amount = amount / Decimal(max(interval_count * 12, 1))
            elif interval == "week":
                amount = amount * Decimal("52") / Decimal(max(interval_count * 12, 1))
            elif interval == "day":
                amount = amount * Decimal("365") / Decimal(max(interval_count * 12, 1))
            elif interval == "month" and interval_count > 1:
                amount = amount / Decimal(interval_count)
            total += amount
        return total
    
    async def _calculate_chargeback_rate(self, filters: ReportFilter) -> float:
        """Calculate chargeback rate as disputes per successful charge."""
        charges, disputes = await asyncio.gather(
            self._get_charges_data(filters),
            self._get_disputes_data(filters),
        )
        successful_charges = [charge for charge in charges if self._attr(charge, "status") == "succeeded"]
        return len(disputes) / len(successful_charges) if successful_charges else 0.0
    
    async def _calculate_refund_rate(self, filters: ReportFilter) -> float:
        """Calculate refund rate as refunded amount divided by successful charge amount."""
        charges = await self._get_charges_data(filters)
        successful_charges = [charge for charge in charges if self._attr(charge, "status") == "succeeded"]
        charged_amount = sum((self._attr(charge, "amount", 0) or 0) for charge in successful_charges)
        refunded_amount = sum((self._attr(charge, "amount_refunded", 0) or 0) for charge in successful_charges)
        return refunded_amount / charged_amount if charged_amount else 0.0
    
    async def _calculate_customer_acquisition_cost(self, filters: ReportFilter) -> Decimal:
        """Calculate customer acquisition cost from configured acquisition spend."""
        customers = await self._get_customers_data(filters)
        if not customers:
            return Decimal("0")
        metadata = filters.metadata if filters and filters.metadata else {}
        spend_value = (
            metadata.get("acquisition_spend")
            or metadata.get("marketing_spend")
            or self.config.get("acquisition_spend")
            or self.config.get("marketing_spend")
            or 0
        )
        acquisition_spend = Decimal(str(spend_value))
        return acquisition_spend / Decimal(len(customers)) if acquisition_spend else Decimal("0")
    
    async def _calculate_customer_lifetime_value(self, filters: ReportFilter) -> Decimal:
        """Calculate observed customer lifetime value from successful payments."""
        payment_intents = await self._get_payment_intents_data(filters)
        customer_revenue: Dict[str, Decimal] = {}
        for payment_intent in payment_intents:
            customer_id = self._attr(payment_intent, "customer")
            revenue = self._payment_revenue(payment_intent)
            if customer_id and revenue:
                customer_revenue[customer_id] = customer_revenue.get(customer_id, Decimal("0")) + revenue
        if not customer_revenue:
            return Decimal("0")
        return sum(customer_revenue.values(), Decimal("0")) / Decimal(len(customer_revenue))
    
    async def _calculate_retention_rate(self, filters: ReportFilter, period: ReportPeriod) -> float:
        """Calculate retained paying customers across current and previous periods."""
        if not filters.start_date or not filters.end_date:
            filters.start_date, filters.end_date = self._get_period_dates(period)
        previous_filters = self._get_previous_period_filters(filters, period)
        current_payments, previous_payments = await asyncio.gather(
            self._get_payment_intents_data(filters),
            self._get_payment_intents_data(previous_filters),
        )
        current_customers = {
            self._attr(payment, "customer")
            for payment in current_payments
            if self._payment_revenue(payment) > 0 and self._attr(payment, "customer")
        }
        previous_customers = {
            self._attr(payment, "customer")
            for payment in previous_payments
            if self._payment_revenue(payment) > 0 and self._attr(payment, "customer")
        }
        if not previous_customers:
            return 1.0 if current_customers else 0.0
        return len(current_customers & previous_customers) / len(previous_customers)
    
    def _map_customers_to_payments(
        self,
        customers: List[stripe.Customer],
        payment_intents: List[stripe.PaymentIntent]
    ) -> Dict[str, List[stripe.PaymentIntent]]:
        """Map customers to their payments"""
        customer_payments = {}
        
        for customer in customers:
            customer_payments[customer.id] = []
        
        for payment_intent in payment_intents:
            if payment_intent.customer and payment_intent.customer in customer_payments:
                customer_payments[payment_intent.customer].append(payment_intent)
        
        return customer_payments
    
    async def _get_top_customers_by_revenue(self, customer_payment_map) -> List[Dict[str, Any]]:
        """Return customers ranked by successful payment revenue."""
        ranked = []
        for customer_id, payments in customer_payment_map.items():
            revenue = sum((self._payment_revenue(payment) for payment in payments), Decimal("0"))
            transaction_count = len([payment for payment in payments if self._payment_revenue(payment) > 0])
            if revenue:
                ranked.append({
                    "customer_id": customer_id,
                    "revenue": float(revenue),
                    "transaction_count": transaction_count,
                })
        return sorted(ranked, key=lambda item: item["revenue"], reverse=True)[:10]
    
    def _segment_customers(self, customer_payment_map) -> Dict[str, int]:
        """Segment customers by observed successful payment revenue."""
        segments = {"high_value": 0, "medium_value": 0, "low_value": 0, "inactive": 0}
        high_threshold = Decimal(str(self.config.get("high_value_customer_threshold", "1000")))
        medium_threshold = Decimal(str(self.config.get("medium_value_customer_threshold", "100")))
        for payments in customer_payment_map.values():
            revenue = sum((self._payment_revenue(payment) for payment in payments), Decimal("0"))
            if revenue >= high_threshold:
                segments["high_value"] += 1
            elif revenue >= medium_threshold:
                segments["medium_value"] += 1
            elif revenue > 0:
                segments["low_value"] += 1
            else:
                segments["inactive"] += 1
        return segments
    
    async def _calculate_payment_method_adoption(self, customers) -> Dict[str, float]:
        """Calculate customer default payment method adoption ratios."""
        method_counts: Dict[str, int] = {}
        for customer in customers:
            method_type = self._nested_attr(customer, "invoice_settings.default_payment_method.type")
            if not method_type:
                method_type = self._attr(self._attr(customer, "metadata", {}) or {}, "default_payment_method_type")
            if not method_type:
                source = self._attr(customer, "default_source")
                method_type = self._attr(source, "object", None) if source else "unknown"
            method_counts[method_type or "unknown"] = method_counts.get(method_type or "unknown", 0) + 1
        total = sum(method_counts.values())
        return {method: count / total for method, count in method_counts.items()} if total else {}
    
    async def _calculate_mrr(self, subscriptions) -> Decimal:
        """Calculate monthly recurring revenue from active subscriptions."""
        return sum(
            (
                self._subscription_monthly_amount(subscription)
                for subscription in subscriptions
                if self._attr(subscription, "status") in {"active", "trialing", "past_due"}
            ),
            Decimal("0"),
        )
    
    async def _calculate_subscription_churn_rate(self, filters, period) -> float:
        """Calculate current-period subscription churn rate."""
        subscriptions = await self._get_subscriptions_data(filters)
        if not subscriptions:
            return 0.0
        canceled = [subscription for subscription in subscriptions if self._attr(subscription, "status") == "canceled"]
        return len(canceled) / len(subscriptions)
    
    async def _calculate_subscription_growth_rate(self, filters, period) -> float:
        """Calculate active subscription growth against the previous period."""
        if filters and (not filters.start_date or not filters.end_date):
            filters.start_date, filters.end_date = self._get_period_dates(period)
        previous_filters = self._get_previous_period_filters(filters, period) if filters else None
        current_subscriptions, previous_subscriptions = await asyncio.gather(
            self._get_subscriptions_data(filters),
            self._get_subscriptions_data(previous_filters),
        )
        current_active = len([s for s in current_subscriptions if self._attr(s, "status") == "active"])
        previous_active = len([s for s in previous_subscriptions if self._attr(s, "status") == "active"])
        if previous_active == 0:
            return 1.0 if current_active > 0 else 0.0
        return (current_active - previous_active) / previous_active
    
    async def _calculate_subscription_ltv(self, subscriptions) -> Decimal:
        """Calculate observed average subscription lifetime value."""
        if not subscriptions:
            return Decimal("0")
        lifetime_values = []
        now_ts = int(datetime.now(timezone.utc).timestamp())
        for subscription in subscriptions:
            created = int(self._attr(subscription, "created", now_ts) or now_ts)
            ended = int(self._attr(subscription, "ended_at", 0) or now_ts)
            active_months = max(Decimal(ended - created) / Decimal(30 * 24 * 60 * 60), Decimal("1"))
            lifetime_values.append(self._subscription_monthly_amount(subscription) * active_months)
        return sum(lifetime_values, Decimal("0")) / Decimal(len(lifetime_values))
    
    async def _calculate_trial_conversion_rate(self, filters) -> float:
        """Calculate the ratio of trial subscriptions that became active."""
        subscriptions = await self._get_subscriptions_data(filters)
        trialed = [
            subscription for subscription in subscriptions
            if self._attr(subscription, "trial_start") or self._attr(subscription, "trial_end")
        ]
        if not trialed:
            return 0.0
        converted = [subscription for subscription in trialed if self._attr(subscription, "status") == "active"]
        return len(converted) / len(trialed)
    
    async def _get_subscription_changes(self, filters) -> Dict[str, int]:
        """Infer subscription upgrades and downgrades from metadata when available."""
        subscriptions = await self._get_subscriptions_data(filters)
        changes = {"upgrades": 0, "downgrades": 0}
        for subscription in subscriptions:
            metadata = self._attr(subscription, "metadata", {}) or {}
            change_type = self._attr(metadata, "change_type")
            if change_type in changes:
                changes[change_type] += 1
        return changes
    
    async def _calculate_revenue_by_plan(self, subscriptions) -> Dict[str, Decimal]:
        """Calculate MRR grouped by Stripe price/product plan."""
        revenue_by_plan: Dict[str, Decimal] = {}
        for subscription in subscriptions:
            if self._attr(subscription, "status") not in {"active", "trialing", "past_due"}:
                continue
            for item in self._nested_attr(subscription, "items.data", []) or []:
                price = self._attr(item, "price", {})
                plan = (
                    self._attr(price, "nickname")
                    or self._attr(price, "lookup_key")
                    or self._attr(price, "id")
                    or "unknown"
                )
                item_subscription = type("SubscriptionShim", (), {"items": {"data": [item]}})()
                revenue_by_plan[plan] = revenue_by_plan.get(plan, Decimal("0")) + self._subscription_monthly_amount(item_subscription)
        return revenue_by_plan
    
    async def _get_radar_analytics(self, filters) -> Dict[str, Any]:
        """Calculate Radar-style risk analytics from charge outcomes."""
        charges = await self._get_charges_data(filters)
        distribution = {"low": 0, "medium": 0, "high": 0}
        blocked_transactions = 0
        blocked_amount = 0
        high_risk_transactions = 0
        reviewed_transactions = 0
        false_positives = 0
        for charge in charges:
            outcome = self._attr(charge, "outcome", {}) or {}
            risk_score = int(self._attr(outcome, "risk_score", 0) or 0)
            seller_message = (self._attr(outcome, "seller_message", "") or "").lower()
            if risk_score >= 75:
                distribution["high"] += 1
                high_risk_transactions += 1
            elif risk_score >= 40:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
            if self._attr(charge, "status") == "failed" or "blocked" in seller_message:
                blocked_transactions += 1
                blocked_amount += self._attr(charge, "amount", 0) or 0
            if self._attr(outcome, "type"):
                reviewed_transactions += 1
                if risk_score >= 75 and self._attr(charge, "status") == "succeeded":
                    false_positives += 1
        total = len(charges)
        return {
            "accuracy": 1.0 - (false_positives / reviewed_transactions) if reviewed_transactions else 0.0,
            "blocked_transactions": blocked_transactions,
            "blocked_amount": blocked_amount,
            "false_positive_rate": false_positives / reviewed_transactions if reviewed_transactions else 0.0,
            "high_risk_transactions": high_risk_transactions,
            "risk_score_distribution": distribution,
            "reviewed_transactions": reviewed_transactions,
            "coverage": reviewed_transactions / total if total else 0.0,
        }
    
    async def _get_top_fraud_indicators(self, disputes) -> List[Dict[str, Any]]:
        """Return dispute reasons ranked as fraud indicators."""
        counts: Dict[str, int] = {}
        for dispute in disputes:
            reason = self._attr(dispute, "reason", None) or self._attr(dispute, "status", None) or "unknown"
            counts[reason] = counts.get(reason, 0) + 1
        return [
            {"indicator": indicator, "count": count}
            for indicator, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:10]
        ]
    
    async def _calculate_custom_metric(self, metric, period, filters) -> Dict[str, Any]:
        """Calculate supported custom metrics from live report data."""
        metric_name = str(metric).lower()
        payment_intents = await self._get_payment_intents_data(filters)
        successful_payments = [payment for payment in payment_intents if self._payment_revenue(payment) > 0]
        total_revenue = sum((self._payment_revenue(payment) for payment in successful_payments), Decimal("0"))
        value: Union[int, float, Decimal]
        metric_type = MetricType.COUNT
        if metric_name in {"revenue", "total_revenue"}:
            value = total_revenue
            metric_type = MetricType.REVENUE
        elif metric_name in {"transactions", "transaction_count", "count"}:
            value = len(payment_intents)
            metric_type = MetricType.COUNT
        elif metric_name in {"successful_transactions", "successful_count"}:
            value = len(successful_payments)
            metric_type = MetricType.COUNT
        elif metric_name in {"average_transaction_value", "atv"}:
            value = total_revenue / Decimal(len(successful_payments)) if successful_payments else Decimal("0")
            metric_type = MetricType.AVERAGE
        elif metric_name == "chargeback_rate":
            value = await self._calculate_chargeback_rate(filters)
            metric_type = MetricType.PERCENTAGE
        elif metric_name == "refund_rate":
            value = await self._calculate_refund_rate(filters)
            metric_type = MetricType.PERCENTAGE
        else:
            raise ValueError(f"Unsupported custom metric: {metric}")
        return {
            "value": float(value) if isinstance(value, Decimal) else value,
            "type": metric_type.value,
            "period": period.value,
            "calculated_at": datetime.utcnow().isoformat()
        }
    
    def _serialize_filters(self, filters: ReportFilter) -> Dict[str, Any]:
        """Serialize filters for JSON output"""
        return {
            "start_date": filters.start_date.isoformat() if filters.start_date else None,
            "end_date": filters.end_date.isoformat() if filters.end_date else None,
            "currency": filters.currency,
            "customer_ids": filters.customer_ids,
            "product_ids": filters.product_ids,
            "payment_method_types": filters.payment_method_types,
            "status": filters.status,
            "metadata": filters.metadata,
            "min_amount": filters.min_amount,
            "max_amount": filters.max_amount
        }
    
    def _format_as_csv(self, report_data: Dict[str, Any]) -> str:
        """Format report data as CSV"""
        # Simplified CSV formatting
        return "metric,value\n" + "\n".join([f"{k},{v}" for k, v in report_data["metrics"].items()])
    
    def _format_as_excel(self, report_data: Dict[str, Any]) -> bytes:
        """Format report data as a minimal XLSX workbook."""
        rows = [["metric", "value", "type", "period"]]
        for metric_name, metric_value in report_data.get("metrics", {}).items():
            if isinstance(metric_value, dict):
                rows.append([
                    metric_name,
                    metric_value.get("value"),
                    metric_value.get("type"),
                    metric_value.get("period", report_data.get("period")),
                ])
            else:
                rows.append([metric_name, metric_value, None, report_data.get("period")])

        def cell_ref(row_index: int, column_index: int) -> str:
            return f"{chr(ord('A') + column_index)}{row_index}"

        sheet_rows = []
        for row_index, row in enumerate(rows, start=1):
            cells = []
            for column_index, value in enumerate(row):
                text = "" if value is None else str(value)
                cells.append(
                    f'<c r="{cell_ref(row_index, column_index)}" t="inlineStr">'
                    f'<is><t>{xml_escape(text)}</t></is></c>'
                )
            sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

        worksheet = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetData>{"".join(sheet_rows)}</sheetData>'
            '</worksheet>'
        )
        workbook = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Report" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>'
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '</Types>'
        )
        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        )
        workbook_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '</Relationships>'
        )
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as workbook_zip:
            workbook_zip.writestr("[Content_Types].xml", content_types)
            workbook_zip.writestr("_rels/.rels", root_rels)
            workbook_zip.writestr("xl/workbook.xml", workbook)
            workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
            workbook_zip.writestr("xl/worksheets/sheet1.xml", worksheet)
        return output.getvalue()


async def create_stripe_reporting_service(
    stripe_client: stripe.StripeClient,
    config: Optional[Dict[str, Any]] = None
) -> StripeReportingService:
    """
    Factory function to create Stripe reporting service
    
    Args:
        stripe_client: Initialized Stripe client
        config: Optional configuration dictionary
        
    Returns:
        Configured StripeReportingService instance
    """
    if not config:
        config = {}
    
    service = StripeReportingService(stripe_client, config)
    
    logger.info("Stripe reporting service created successfully")
    return service
