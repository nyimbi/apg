"""APG Cash Management - Advanced Risk Analytics Engine

World-class risk analytics with sophisticated risk measurement, stress testing,
scenario analysis, and regulatory compliance monitoring capabilities.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# Advanced risk analytics
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, t, chi2, jarque_bera, kstest
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.stats.diagnostic import het_breuschpagan
from arch import arch_model
from arch.univariate import ConstantMean, GARCH, Normal

# Monte Carlo and simulation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance, LedoitWolf

# Network analysis for contagion risk
import networkx as nx
from collections import defaultdict

# Advanced statistics
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import BetaUnivariate, GammaUnivariate
import pymc3 as pm

from .cache import CashCacheManager
from .events import CashEventManager

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)

def _log_risk_metric(metric_name: str, value: float, threshold: float) -> str:
	"""Log risk metrics with APG formatting"""
	status = "BREACH" if value > threshold else "NORMAL"
	return f"RISK_METRIC | metric={metric_name} | value={value:.6f} | threshold={threshold:.6f} | status={status}"

def _log_stress_test(scenario: str, loss_amount: float, severity: str) -> str:
	"""Log stress test results"""
	return f"STRESS_TEST | scenario={scenario} | loss=${loss_amount:,.2f} | severity={severity}"

# ============================================================================
# Risk Analytics Enums and Data Classes
# ============================================================================

class RiskMetric(str, Enum):
	"""Types of risk metrics"""
	VALUE_AT_RISK = "value_at_risk"
	CONDITIONAL_VAR = "conditional_var"
	EXPECTED_SHORTFALL = "expected_shortfall"
	MAXIMUM_DRAWDOWN = "maximum_drawdown"
	SHARPE_RATIO = "sharpe_ratio"
	SORTINO_RATIO = "sortino_ratio"
	CALMAR_RATIO = "calmar_ratio"
	VOLATILITY = "volatility"
	SKEWNESS = "skewness"
	KURTOSIS = "kurtosis"
	BETA = "beta"
	TRACKING_ERROR = "tracking_error"
	INFORMATION_RATIO = "information_ratio"

class RiskModel(str, Enum):
	"""Risk modeling approaches"""
	PARAMETRIC = "parametric"
	HISTORICAL_SIMULATION = "historical_simulation"
	MONTE_CARLO = "monte_carlo"
	EXTREME_VALUE_THEORY = "extreme_value_theory"
	COPULA_BASED = "copula_based"
	GARCH = "garch"
	BAYESIAN = "bayesian"

class StressTestType(str, Enum):
	"""Types of stress tests"""
	HISTORICAL_SCENARIO = "historical_scenario"
	HYPOTHETICAL_SCENARIO = "hypothetical_scenario"
	SENSITIVITY_ANALYSIS = "sensitivity_analysis"
	REVERSE_STRESS_TEST = "reverse_stress_test"
	EXTREME_MOVEMENT = "extreme_movement"
	LIQUIDITY_STRESS = "liquidity_stress"
	CREDIT_STRESS = "credit_stress"
	OPERATIONAL_STRESS = "operational_stress"

class RegulatoryFramework(str, Enum):
	"""Regulatory compliance frameworks"""
	BASEL_III = "basel_iii"
	CCAR = "ccar"
	DFAST = "dfast"
	ICAAP = "icaap"
	ORSA = "orsa"
	SOLVENCY_II = "solvency_ii"
	IFRS_9 = "ifrs_9"
	CECL = "cecl"

@dataclass
class RiskMeasurement:
	"""Risk measurement result"""
	metric: RiskMetric
	value: float
	confidence_level: float
	time_horizon: int  # days
	calculation_method: RiskModel
	timestamp: datetime
	breakdown: Dict[str, float]
	warnings: List[str]

@dataclass
class StressTestResult:
	"""Stress test scenario result"""
	scenario_name: str
	test_type: StressTestType
	loss_amount: float
	loss_percentage: float
	confidence_level: float
	affected_accounts: List[str]
	recovery_time_days: int
	mitigation_actions: List[str]
	regulatory_impact: Dict[str, float]

@dataclass
class RiskAlert:
	"""Risk monitoring alert"""
	alert_id: str
	risk_type: str
	severity: str  # LOW, MEDIUM, HIGH, CRITICAL
	threshold_breached: float
	current_value: float
	affected_entities: List[str]
	recommendation: str
	regulatory_implication: str
	timestamp: datetime

@dataclass
class LiquidityRisk:
	"""Liquidity risk assessment"""
	total_liquid_assets: float
	liquidity_coverage_ratio: float
	net_stable_funding_ratio: float
	cash_flow_gap_7d: float
	cash_flow_gap_30d: float
	funding_concentration: Dict[str, float]
	liquidity_buffers: Dict[str, float]
	stress_test_survival_days: int

# ============================================================================
# Value at Risk (VaR) Models
# ============================================================================

class VaRCalculator:
	"""Advanced Value at Risk calculation engine"""
	
	def __init__(self):
		self.models = {}
		self.calibration_data = {}
		
	async def calculate_parametric_var(
		self, 
		returns: np.ndarray, 
		confidence_level: float = 0.95,
		holding_period: int = 1
	) -> Tuple[float, Dict[str, Any]]:
		"""Calculate VaR using parametric approach"""
		
		# Calculate mean and standard deviation
		mean_return = np.mean(returns)
		std_return = np.std(returns, ddof=1)
		
		# Normal distribution assumption
		z_score = norm.ppf(confidence_level)
		
		# Scale for holding period
		scaled_mean = mean_return * holding_period
		scaled_std = std_return * np.sqrt(holding_period)
		
		# VaR calculation
		var = -(scaled_mean - z_score * scaled_std)
		
		# Additional statistics
		details = {
			'mean_return': mean_return,
			'volatility': std_return,
			'z_score': z_score,
			'scaled_mean': scaled_mean,
			'scaled_std': scaled_std,
			'distribution': 'normal',
			'normality_test': self._test_normality(returns)
		}
		
		return var, details
	
	async def calculate_historical_var(
		self, 
		returns: np.ndarray, 
		confidence_level: float = 0.95,
		holding_period: int = 1
	) -> Tuple[float, Dict[str, Any]]:
		"""Calculate VaR using historical simulation"""
		
		# Scale returns for holding period
		if holding_period > 1:
			scaled_returns = self._scale_returns_for_holding_period(returns, holding_period)
		else:
			scaled_returns = returns
		
		# Sort returns
		sorted_returns = np.sort(scaled_returns)
		
		# Calculate percentile
		alpha = 1 - confidence_level
		index = int(alpha * len(sorted_returns))
		
		# VaR is the negative of the percentile
		var = -sorted_returns[index]
		
		# Additional statistics
		details = {
			'data_points': len(returns),
			'percentile_index': index,
			'actual_percentile': sorted_returns[index],
			'min_return': np.min(returns),
			'max_return': np.max(returns),
			'holding_period': holding_period
		}
		
		return var, details
	
	async def calculate_monte_carlo_var(
		self, 
		returns: np.ndarray, 
		confidence_level: float = 0.95,
		holding_period: int = 1,
		num_simulations: int = 10000
	) -> Tuple[float, Dict[str, Any]]:
		"""Calculate VaR using Monte Carlo simulation"""
		
		# Fit distribution to returns
		mu = np.mean(returns)
		sigma = np.std(returns, ddof=1)
		
		# Generate random scenarios
		np.random.seed(42)  # For reproducibility
		simulated_returns = np.random.normal(mu, sigma, num_simulations)
		
		# Scale for holding period
		if holding_period > 1:
			simulated_returns = simulated_returns * np.sqrt(holding_period)
		
		# Calculate VaR
		alpha = 1 - confidence_level
		var = -np.percentile(simulated_returns, alpha * 100)
		
		# Additional statistics
		details = {
			'num_simulations': num_simulations,
			'fitted_mean': mu,
			'fitted_std': sigma,
			'simulated_mean': np.mean(simulated_returns),
			'simulated_std': np.std(simulated_returns),
			'holding_period': holding_period
		}
		
		return var, details
	
	async def calculate_garch_var(
		self, 
		returns: np.ndarray, 
		confidence_level: float = 0.95,
		holding_period: int = 1
	) -> Tuple[float, Dict[str, Any]]:
		"""Calculate VaR using GARCH model for volatility clustering"""
		
		# Fit GARCH(1,1) model
		garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='Normal')
		garch_fitted = garch_model.fit(disp='off')
		
		# Get conditional volatility forecast
		forecast = garch_fitted.forecast(horizon=holding_period)
		conditional_vol = np.sqrt(forecast.variance.iloc[-1, :].sum()) / 100
		
		# Calculate VaR
		z_score = norm.ppf(confidence_level)
		var = z_score * conditional_vol
		
		# Additional statistics
		details = {
			'garch_params': garch_fitted.params.to_dict(),
			'conditional_volatility': conditional_vol,
			'log_likelihood': garch_fitted.loglikelihood,
			'aic': garch_fitted.aic,
			'bic': garch_fitted.bic,
			'holding_period': holding_period
		}
		
		return var, details
	
	async def calculate_extreme_value_var(
		self, 
		returns: np.ndarray, 
		confidence_level: float = 0.95,
		threshold_percentile: float = 0.95
	) -> Tuple[float, Dict[str, Any]]:
		"""Calculate VaR using Extreme Value Theory (EVT)"""
		
		# Define threshold for extreme values
		threshold = np.percentile(np.abs(returns), threshold_percentile * 100)
		
		# Extract exceedances (losses beyond threshold)
		losses = -returns[returns < -threshold]
		exceedances = losses - threshold
		
		if len(exceedances) < 10:
			# Fall back to parametric method if insufficient extreme values
			return await self.calculate_parametric_var(returns, confidence_level)
		
		# Fit Generalized Pareto Distribution to exceedances
		from scipy.stats import genpareto
		shape, loc, scale = genpareto.fit(exceedances, floc=0)
		
		# Calculate VaR using EVT
		n = len(returns)
		n_exceedances = len(exceedances)
		
		# Probability of exceedance
		prob_exceed = n_exceedances / n
		
		# VaR calculation
		alpha = 1 - confidence_level
		if alpha < prob_exceed:
			# Use EVT formula
			var = threshold + (scale / shape) * (((n * alpha) / n_exceedances) ** (-shape) - 1)
		else:
			# Use empirical quantile
			var = -np.percentile(returns, alpha * 100)
		
		details = {
			'threshold': threshold,
			'num_exceedances': n_exceedances,
			'exceedance_rate': prob_exceed,
			'gpd_shape': shape,
			'gpd_scale': scale,
			'threshold_percentile': threshold_percentile
		}
		
		return var, details
	
	def _test_normality(self, returns: np.ndarray) -> Dict[str, Any]:
		"""Test for normality of returns distribution"""
		
		# Jarque-Bera test
		jb_stat, jb_pvalue = jarque_bera(returns)
		
		# Kolmogorov-Smirnov test
		ks_stat, ks_pvalue = kstest(returns, 'norm', args=(np.mean(returns), np.std(returns)))
		
		# Shapiro-Wilk test (for smaller samples)
		if len(returns) <= 5000:
			from scipy.stats import shapiro
			sw_stat, sw_pvalue = shapiro(returns)
		else:
			sw_stat, sw_pvalue = None, None
		
		return {
			'jarque_bera_stat': jb_stat,
			'jarque_bera_pvalue': jb_pvalue,
			'ks_stat': ks_stat,
			'ks_pvalue': ks_pvalue,
			'shapiro_wilk_stat': sw_stat,
			'shapiro_wilk_pvalue': sw_pvalue,
			'is_normal_jb': jb_pvalue > 0.05,
			'is_normal_ks': ks_pvalue > 0.05,
			'is_normal_sw': sw_pvalue > 0.05 if sw_pvalue else None
		}
	
	def _scale_returns_for_holding_period(self, returns: np.ndarray, holding_period: int) -> np.ndarray:
		"""Scale returns for multi-day holding period"""
		
		if holding_period == 1:
			return returns
		
		# Overlapping approach for historical simulation
		scaled_returns = []
		for i in range(len(returns) - holding_period + 1):
			period_return = np.prod(1 + returns[i:i + holding_period]) - 1
			scaled_returns.append(period_return)
		
		return np.array(scaled_returns)

# ============================================================================
# Expected Shortfall (Conditional VaR)
# ============================================================================

class ExpectedShortfallCalculator:
	"""Expected Shortfall (ES) / Conditional Value at Risk calculator"""
	
	async def calculate_parametric_es(
		self, 
		returns: np.ndarray, 
		confidence_level: float = 0.95,
		holding_period: int = 1
	) -> Tuple[float, Dict[str, Any]]:
		"""Calculate Expected Shortfall using parametric approach"""
		
		mean_return = np.mean(returns)
		std_return = np.std(returns, ddof=1)
		
		# Scale for holding period
		scaled_mean = mean_return * holding_period
		scaled_std = std_return * np.sqrt(holding_period)
		
		# Calculate ES for normal distribution
		alpha = 1 - confidence_level
		z_alpha = norm.ppf(alpha)
		
		# ES formula for normal distribution
		es = -(scaled_mean - scaled_std * norm.pdf(z_alpha) / alpha)
		
		details = {
			'mean_return': mean_return,
			'volatility': std_return,
			'z_alpha': z_alpha,
			'scaled_mean': scaled_mean,
			'scaled_std': scaled_std,
			'confidence_level': confidence_level
		}
		
		return es, details
	
	async def calculate_historical_es(
		self, 
		returns: np.ndarray, 
		confidence_level: float = 0.95,
		holding_period: int = 1
	) -> Tuple[float, Dict[str, Any]]:
		"""Calculate Expected Shortfall using historical simulation"""
		
		# Scale returns for holding period
		if holding_period > 1:
			scaled_returns = self._scale_returns_for_holding_period(returns, holding_period)
		else:
			scaled_returns = returns
		
		# Sort returns
		sorted_returns = np.sort(scaled_returns)
		
		# Find tail losses
		alpha = 1 - confidence_level
		cutoff_index = int(alpha * len(sorted_returns))
		
		# Expected Shortfall is the mean of tail losses
		tail_losses = sorted_returns[:cutoff_index]
		if len(tail_losses) > 0:
			es = -np.mean(tail_losses)
		else:
			es = -sorted_returns[0]  # Worst case
		
		details = {
			'tail_observations': len(tail_losses),
			'worst_loss': -sorted_returns[0] if len(sorted_returns) > 0 else 0,
			'cutoff_index': cutoff_index,
			'confidence_level': confidence_level
		}
		
		return es, details
	
	def _scale_returns_for_holding_period(self, returns: np.ndarray, holding_period: int) -> np.ndarray:
		"""Scale returns for multi-day holding period"""
		
		if holding_period == 1:
			return returns
		
		scaled_returns = []
		for i in range(len(returns) - holding_period + 1):
			period_return = np.prod(1 + returns[i:i + holding_period]) - 1
			scaled_returns.append(period_return)
		
		return np.array(scaled_returns)

# ============================================================================
# Stress Testing Engine
# ============================================================================

class StressTestingEngine:
	"""Advanced stress testing and scenario analysis"""
	
	def __init__(self, cache_manager: CashCacheManager):
		self.cache = cache_manager
		self.historical_scenarios = {}
		self.stress_models = {}
		
	async def run_historical_stress_test(
		self,
		portfolio_data: Dict[str, Any],
		scenario_name: str,
		scenario_shocks: Dict[str, float]
	) -> StressTestResult:
		"""Run stress test based on historical scenario"""
		
		logger.info(f"Running historical stress test: {scenario_name}")
		
		# Apply shocks to portfolio
		stressed_values = {}
		total_loss = 0.0
		affected_accounts = []
		
		for account_id, current_value in portfolio_data.items():
			if isinstance(current_value, dict):
				value = float(current_value.get('balance', 0))
				account_type = current_value.get('type', 'cash')
			else:
				value = float(current_value)
				account_type = 'cash'
			
			# Apply appropriate shock based on account type
			shock = self._get_shock_for_account_type(account_type, scenario_shocks)
			
			# Calculate stressed value
			stressed_value = value * (1 + shock)
			loss = value - stressed_value
			
			if loss > 0:
				total_loss += loss
				affected_accounts.append(account_id)
			
			stressed_values[account_id] = {
				'original_value': value,
				'stressed_value': stressed_value,
				'loss': loss,
				'shock_applied': shock
			}
		
		# Calculate percentage loss
		total_portfolio_value = sum(
			float(v.get('balance', 0)) if isinstance(v, dict) else float(v) 
			for v in portfolio_data.values()
		)
		
		loss_percentage = (total_loss / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0
		
		# Estimate recovery time
		recovery_time = self._estimate_recovery_time(scenario_name, loss_percentage)
		
		# Generate mitigation actions
		mitigation_actions = self._generate_mitigation_actions(scenario_name, loss_percentage)
		
		# Calculate regulatory impact
		regulatory_impact = await self._calculate_regulatory_impact(loss_percentage, total_loss)
		
		return StressTestResult(
			scenario_name=scenario_name,
			test_type=StressTestType.HISTORICAL_SCENARIO,
			loss_amount=total_loss,
			loss_percentage=loss_percentage,
			confidence_level=0.99,  # Historical scenarios typically high confidence
			affected_accounts=affected_accounts,
			recovery_time_days=recovery_time,
			mitigation_actions=mitigation_actions,
			regulatory_impact=regulatory_impact
		)
	
	async def run_monte_carlo_stress_test(
		self,
		portfolio_data: Dict[str, Any],
		num_simulations: int = 10000,
		confidence_levels: List[float] = [0.95, 0.99, 0.999]
	) -> Dict[float, StressTestResult]:
		"""Run Monte Carlo stress testing"""
		
		logger.info(f"Running Monte Carlo stress test with {num_simulations} simulations")
		
		# Set up simulation parameters
		np.random.seed(42)
		
		# Generate correlated shocks for different asset classes
		correlation_matrix = self._get_asset_correlation_matrix()
		
		results = {}
		
		for confidence_level in confidence_levels:
			# Run simulations
			simulation_losses = []
			
			for _ in range(num_simulations):
				# Generate correlated random shocks
				shocks = self._generate_correlated_shocks(correlation_matrix)
				
				# Calculate portfolio loss for this simulation
				total_loss = 0.0
				affected_accounts = []
				
				for account_id, current_value in portfolio_data.items():
					if isinstance(current_value, dict):
						value = float(current_value.get('balance', 0))
						account_type = current_value.get('type', 'cash')
					else:
						value = float(current_value)
						account_type = 'cash'
					
					# Apply shock
					shock = shocks.get(account_type, 0.0)
					stressed_value = value * (1 + shock)
					loss = max(0, value - stressed_value)
					
					if loss > 0:
						total_loss += loss
						if account_id not in affected_accounts:
							affected_accounts.append(account_id)
				
				simulation_losses.append(total_loss)
			
			# Calculate VaR at confidence level
			alpha = 1 - confidence_level
			var_loss = np.percentile(simulation_losses, confidence_level * 100)
			
			# Calculate total portfolio value
			total_portfolio_value = sum(
				float(v.get('balance', 0)) if isinstance(v, dict) else float(v) 
				for v in portfolio_data.values()
			)
			
			loss_percentage = (var_loss / total_portfolio_value * 100) if total_portfolio_value > 0 else 0.0
			
			# Estimate recovery time and mitigation actions
			recovery_time = self._estimate_recovery_time("monte_carlo", loss_percentage)
			mitigation_actions = self._generate_mitigation_actions("monte_carlo", loss_percentage)
			regulatory_impact = await self._calculate_regulatory_impact(loss_percentage, var_loss)
			
			results[confidence_level] = StressTestResult(
				scenario_name=f"Monte Carlo ({confidence_level:.1%} confidence)",
				test_type=StressTestType.HYPOTHETICAL_SCENARIO,
				loss_amount=var_loss,
				loss_percentage=loss_percentage,
				confidence_level=confidence_level,
				affected_accounts=list(set(affected_accounts)),
				recovery_time_days=recovery_time,
				mitigation_actions=mitigation_actions,
				regulatory_impact=regulatory_impact
			)
		
		return results
	
	async def run_liquidity_stress_test(
		self,
		portfolio_data: Dict[str, Any],
		scenarios: List[str] = ["mild", "moderate", "severe", "extreme"]
	) -> Dict[str, StressTestResult]:
		"""Run liquidity stress testing"""
		
		results = {}
		
		# Define liquidity stress scenarios
		liquidity_scenarios = {
			"mild": {"funding_gap": 0.05, "market_liquidity": 0.9, "duration": 7},
			"moderate": {"funding_gap": 0.15, "market_liquidity": 0.7, "duration": 14},
			"severe": {"funding_gap": 0.30, "market_liquidity": 0.5, "duration": 30},
			"extreme": {"funding_gap": 0.50, "market_liquidity": 0.3, "duration": 60}
		}
		
		for scenario_name in scenarios:
			scenario_params = liquidity_scenarios[scenario_name]
			
			# Calculate liquidity impact
			total_liquid_assets = 0.0
			available_liquidity = 0.0
			funding_gap = 0.0
			
			for account_id, current_value in portfolio_data.items():
				if isinstance(current_value, dict):
					value = float(current_value.get('balance', 0))
					account_type = current_value.get('type', 'cash')
					liquidity_score = current_value.get('liquidity_score', 1.0)
				else:
					value = float(current_value)
					account_type = 'cash'
					liquidity_score = 1.0 if account_type == 'cash' else 0.8
				
				total_liquid_assets += value
				
				# Apply liquidity stress
				stressed_liquidity = liquidity_score * scenario_params["market_liquidity"]
				available_liquidity += value * stressed_liquidity
			
			# Calculate funding gap
			funding_gap = total_liquid_assets * scenario_params["funding_gap"]
			liquidity_shortfall = max(0, funding_gap - available_liquidity)
			
			loss_percentage = (liquidity_shortfall / total_liquid_assets * 100) if total_liquid_assets > 0 else 0.0
			
			# Generate stress test result
			recovery_time = scenario_params["duration"]
			mitigation_actions = [
				"Activate emergency credit facilities",
				"Liquidate high-quality liquid assets",
				"Reduce discretionary outflows",
				"Negotiate payment deferrals"
			]
			
			regulatory_impact = await self._calculate_regulatory_impact(loss_percentage, liquidity_shortfall)
			
			results[scenario_name] = StressTestResult(
				scenario_name=f"Liquidity Stress - {scenario_name.title()}",
				test_type=StressTestType.LIQUIDITY_STRESS,
				loss_amount=liquidity_shortfall,
				loss_percentage=loss_percentage,
				confidence_level=0.95,
				affected_accounts=list(portfolio_data.keys()),
				recovery_time_days=recovery_time,
				mitigation_actions=mitigation_actions,
				regulatory_impact=regulatory_impact
			)
		
		return results
	
	def _get_shock_for_account_type(self, account_type: str, scenario_shocks: Dict[str, float]) -> float:
		"""Get appropriate shock value for account type"""
		
		shock_mapping = {
			'cash': scenario_shocks.get('cash', 0.0),
			'checking': scenario_shocks.get('cash', 0.0),
			'savings': scenario_shocks.get('cash', 0.0),
			'money_market': scenario_shocks.get('short_term_rates', -0.02),
			'investment': scenario_shocks.get('equity', -0.20),
			'bond': scenario_shocks.get('bonds', -0.10),
			'cd': scenario_shocks.get('short_term_rates', -0.01)
		}
		
		return shock_mapping.get(account_type, scenario_shocks.get('general', -0.05))
	
	def _estimate_recovery_time(self, scenario_name: str, loss_percentage: float) -> int:
		"""Estimate recovery time based on scenario and loss severity"""
		
		base_recovery_times = {
			"2008_financial_crisis": 720,  # 2 years
			"covid_pandemic": 180,         # 6 months
			"dot_com_crash": 540,          # 1.5 years
			"monte_carlo": 90,             # 3 months
			"liquidity_stress": 30         # 1 month
		}
		
		base_time = base_recovery_times.get(scenario_name, 90)
		
		# Adjust based on loss severity
		if loss_percentage > 50:
			multiplier = 2.0
		elif loss_percentage > 30:
			multiplier = 1.5
		elif loss_percentage > 10:
			multiplier = 1.2
		else:
			multiplier = 1.0
		
		return int(base_time * multiplier)
	
	def _generate_mitigation_actions(self, scenario_name: str, loss_percentage: float) -> List[str]:
		"""Generate mitigation actions based on scenario and severity"""
		
		base_actions = [
			"Monitor market conditions closely",
			"Review and update risk limits",
			"Enhance liquidity management",
			"Diversify funding sources"
		]
		
		if loss_percentage > 10:
			base_actions.extend([
				"Implement capital conservation measures",
				"Reduce discretionary spending",
				"Activate contingency funding plans"
			])
		
		if loss_percentage > 30:
			base_actions.extend([
				"Consider asset disposals",
				"Negotiate with creditors",
				"Seek additional capital",
				"Implement crisis management protocols"
			])
		
		scenario_specific = {
			"2008_financial_crisis": [
				"Enhance counterparty monitoring",
				"Reduce interbank exposures",
				"Strengthen capital buffers"
			],
			"liquidity_stress": [
				"Activate emergency credit lines",
				"Postpone non-essential payments",
				"Optimize cash flow timing"
			]
		}
		
		if scenario_name in scenario_specific:
			base_actions.extend(scenario_specific[scenario_name])
		
		return base_actions
	
	async def _calculate_regulatory_impact(self, loss_percentage: float, loss_amount: float) -> Dict[str, float]:
		"""Calculate impact on regulatory ratios and requirements"""
		
		regulatory_impact = {}
		
		# Capital adequacy impact (simplified)
		if loss_percentage > 0:
			# Tier 1 capital ratio impact
			regulatory_impact['tier1_capital_impact'] = -loss_percentage * 0.1
			
			# Leverage ratio impact
			regulatory_impact['leverage_ratio_impact'] = -loss_percentage * 0.05
			
			# Liquidity coverage ratio impact
			regulatory_impact['lcr_impact'] = -loss_percentage * 0.2
		
		# Regulatory reporting requirements
		if loss_percentage > 5:
			regulatory_impact['requires_supervisory_notification'] = 1.0
		
		if loss_percentage > 15:
			regulatory_impact['requires_capital_plan'] = 1.0
		
		if loss_percentage > 30:
			regulatory_impact['requires_recovery_plan'] = 1.0
		
		return regulatory_impact
	
	def _get_asset_correlation_matrix(self) -> np.ndarray:
		"""Get correlation matrix for different asset classes"""
		
		# Simplified correlation matrix
		# In practice, this would be estimated from historical data
		asset_classes = ['cash', 'short_term_rates', 'equity', 'bonds']
		
		correlation_matrix = np.array([
			[1.00, 0.10, -0.05, 0.05],  # cash
			[0.10, 1.00, -0.20, 0.60],  # short_term_rates
			[-0.05, -0.20, 1.00, -0.30], # equity
			[0.05, 0.60, -0.30, 1.00]   # bonds
		])
		
		return correlation_matrix
	
	def _generate_correlated_shocks(self, correlation_matrix: np.ndarray) -> Dict[str, float]:
		"""Generate correlated random shocks"""
		
		# Generate independent normal random variables
		independent_shocks = np.random.normal(0, 1, correlation_matrix.shape[0])
		
		# Apply correlation using Cholesky decomposition
		chol_matrix = np.linalg.cholesky(correlation_matrix)
		correlated_shocks = chol_matrix @ independent_shocks
		
		# Scale shocks and map to asset classes
		asset_classes = ['cash', 'short_term_rates', 'equity', 'bonds']
		shock_scales = [0.001, 0.02, 0.15, 0.08]  # Typical daily volatilities
		
		shocks = {}
		for i, asset_class in enumerate(asset_classes):
			shocks[asset_class] = correlated_shocks[i] * shock_scales[i]
		
		return shocks

# ============================================================================
# Liquidity Risk Analytics
# ============================================================================

class LiquidityRiskAnalyzer:
	"""Advanced liquidity risk measurement and monitoring"""
	
	def __init__(self):
		self.liquidity_metrics = {}
		
	async def calculate_liquidity_coverage_ratio(
		self,
		high_quality_liquid_assets: float,
		net_cash_outflows_30d: float
	) -> float:
		"""Calculate Liquidity Coverage Ratio (LCR)"""
		
		if net_cash_outflows_30d <= 0:
			return float('inf')  # No outflows
		
		lcr = high_quality_liquid_assets / net_cash_outflows_30d
		return lcr
	
	async def calculate_net_stable_funding_ratio(
		self,
		available_stable_funding: float,
		required_stable_funding: float
	) -> float:
		"""Calculate Net Stable Funding Ratio (NSFR)"""
		
		if required_stable_funding <= 0:
			return float('inf')
		
		nsfr = available_stable_funding / required_stable_funding
		return nsfr
	
	async def analyze_cash_flow_gaps(
		self,
		cash_flows: pd.DataFrame,
		time_buckets: List[int] = [1, 7, 30, 90, 180, 365]
	) -> Dict[int, float]:
		"""Analyze cash flow gaps across time buckets"""
		
		gaps = {}
		
		for bucket in time_buckets:
			# Filter cash flows for time bucket
			bucket_flows = cash_flows[cash_flows['days_to_maturity'] <= bucket]
			
			# Calculate inflows and outflows
			inflows = bucket_flows[bucket_flows['amount'] > 0]['amount'].sum()
			outflows = abs(bucket_flows[bucket_flows['amount'] < 0]['amount'].sum())
			
			# Calculate gap
			gap = inflows - outflows
			gaps[bucket] = gap
		
		return gaps
	
	async def calculate_funding_concentration(
		self,
		funding_sources: Dict[str, float]
	) -> Dict[str, float]:
		"""Calculate funding concentration metrics"""
		
		total_funding = sum(funding_sources.values())
		
		if total_funding <= 0:
			return {}
		
		# Calculate concentration percentages
		concentrations = {
			source: amount / total_funding 
			for source, amount in funding_sources.items()
		}
		
		# Calculate Herfindahl-Hirschman Index
		hhi = sum(conc**2 for conc in concentrations.values())
		
		# Find largest concentrations
		sorted_concentrations = sorted(concentrations.items(), key=lambda x: x[1], reverse=True)
		
		metrics = {
			'hhi': hhi,
			'largest_source': sorted_concentrations[0][1] if sorted_concentrations else 0.0,
			'top_3_concentration': sum(conc for _, conc in sorted_concentrations[:3]),
			'effective_number_sources': 1.0 / hhi if hhi > 0 else 0.0
		}
		
		return metrics
	
	async def assess_liquidity_risk(
		self,
		portfolio_data: Dict[str, Any],
		stress_scenarios: List[str] = ["normal", "stressed", "severely_stressed"]
	) -> LiquidityRisk:
		"""Comprehensive liquidity risk assessment"""
		
		# Calculate liquid assets
		total_liquid_assets = 0.0
		for account_id, account_data in portfolio_data.items():
			if isinstance(account_data, dict):
				value = float(account_data.get('balance', 0))
				liquidity_score = account_data.get('liquidity_score', 1.0)
			else:
				value = float(account_data)
				liquidity_score = 0.9  # Default
			
			total_liquid_assets += value * liquidity_score
		
		# Calculate funding concentration
		funding_sources = {
			'deposits': total_liquid_assets * 0.7,
			'wholesale': total_liquid_assets * 0.2,
			'other': total_liquid_assets * 0.1
		}
		
		funding_concentration = await self.calculate_funding_concentration(funding_sources)
		
		# Simulate cash flow gaps
		sample_cash_flows = self._generate_sample_cash_flows(portfolio_data)
		cash_flow_gaps = await self.analyze_cash_flow_gaps(sample_cash_flows)
		
		# Calculate regulatory ratios
		net_outflows_30d = abs(cash_flow_gaps.get(30, 0))
		lcr = await self.calculate_liquidity_coverage_ratio(total_liquid_assets, net_outflows_30d)
		
		# Estimate NSFR (simplified)
		available_stable_funding = total_liquid_assets * 0.8
		required_stable_funding = total_liquid_assets * 0.7
		nsfr = await self.calculate_net_stable_funding_ratio(available_stable_funding, required_stable_funding)
		
		# Calculate liquidity buffers
		liquidity_buffers = {
			'primary': total_liquid_assets * 0.3,
			'secondary': total_liquid_assets * 0.2,
			'tertiary': total_liquid_assets * 0.1
		}
		
		# Stress test survival days
		daily_outflow_rate = abs(cash_flow_gaps.get(1, 0))
		if daily_outflow_rate > 0:
			survival_days = int(total_liquid_assets / daily_outflow_rate)
		else:
			survival_days = 365  # More than a year
		
		return LiquidityRisk(
			total_liquid_assets=total_liquid_assets,
			liquidity_coverage_ratio=lcr,
			net_stable_funding_ratio=nsfr,
			cash_flow_gap_7d=cash_flow_gaps.get(7, 0),
			cash_flow_gap_30d=cash_flow_gaps.get(30, 0),
			funding_concentration=funding_concentration,
			liquidity_buffers=liquidity_buffers,
			stress_test_survival_days=survival_days
		)
	
	def _generate_sample_cash_flows(self, portfolio_data: Dict[str, Any]) -> pd.DataFrame:
		"""Generate sample cash flows for analysis"""
		
		# This would typically use real cash flow projections
		# For now, generate representative sample data
		
		cash_flows = []
		base_date = datetime.now()
		
		for days in range(1, 366):  # One year of daily cash flows
			# Generate random inflows and outflows
			inflow = np.random.exponential(5000) if np.random.random() > 0.7 else 0
			outflow = -np.random.exponential(3000) if np.random.random() > 0.6 else 0
			
			if inflow > 0:
				cash_flows.append({
					'date': base_date + timedelta(days=days),
					'days_to_maturity': days,
					'amount': inflow,
					'type': 'inflow'
				})
			
			if outflow < 0:
				cash_flows.append({
					'date': base_date + timedelta(days=days),
					'days_to_maturity': days,
					'amount': outflow,
					'type': 'outflow'
				})
		
		return pd.DataFrame(cash_flows)

# ============================================================================
# Advanced Risk Analytics Engine
# ============================================================================

class AdvancedRiskAnalyticsEngine:
	"""Comprehensive risk analytics and monitoring system"""
	
	def __init__(
		self, 
		tenant_id: str, 
		cache_manager: CashCacheManager, 
		event_manager: CashEventManager
	):
		self.tenant_id = tenant_id
		self.cache = cache_manager
		self.events = event_manager
		
		# Initialize component analyzers
		self.var_calculator = VaRCalculator()
		self.es_calculator = ExpectedShortfallCalculator()
		self.stress_testing_engine = StressTestingEngine(cache_manager)
		self.liquidity_analyzer = LiquidityRiskAnalyzer()
		
		# Risk monitoring
		self.risk_alerts: List[RiskAlert] = []
		self.risk_thresholds = self._initialize_risk_thresholds()
		
	def _initialize_risk_thresholds(self) -> Dict[str, float]:
		"""Initialize default risk thresholds"""
		
		return {
			'var_95': 0.05,          # 5% VaR threshold
			'var_99': 0.10,          # 10% VaR threshold
			'expected_shortfall': 0.15, # 15% ES threshold
			'max_drawdown': 0.20,    # 20% max drawdown
			'liquidity_ratio': 1.0,  # 100% liquidity coverage
			'concentration_limit': 0.25, # 25% single exposure limit
			'volatility': 0.30       # 30% annualized volatility
		}
	
	async def calculate_comprehensive_risk_metrics(
		self,
		portfolio_data: Dict[str, Any],
		returns_data: np.ndarray,
		confidence_levels: List[float] = [0.95, 0.99],
		holding_periods: List[int] = [1, 10]
	) -> Dict[str, Any]:
		"""Calculate comprehensive risk metrics"""
		
		logger.info(f"Calculating comprehensive risk metrics for tenant {self.tenant_id}")
		
		risk_metrics = {}
		
		# Value at Risk calculations
		var_results = {}
		for confidence_level in confidence_levels:
			for holding_period in holding_periods:
				key = f"var_{int(confidence_level*100)}_{holding_period}d"
				
				# Parametric VaR
				param_var, param_details = await self.var_calculator.calculate_parametric_var(
					returns_data, confidence_level, holding_period
				)
				
				# Historical VaR
				hist_var, hist_details = await self.var_calculator.calculate_historical_var(
					returns_data, confidence_level, holding_period
				)
				
				# Monte Carlo VaR
				mc_var, mc_details = await self.var_calculator.calculate_monte_carlo_var(
					returns_data, confidence_level, holding_period
				)
				
				var_results[key] = {
					'parametric': {'value': param_var, 'details': param_details},
					'historical': {'value': hist_var, 'details': hist_details},
					'monte_carlo': {'value': mc_var, 'details': mc_details}
				}
		
		risk_metrics['value_at_risk'] = var_results
		
		# Expected Shortfall calculations
		es_results = {}
		for confidence_level in confidence_levels:
			for holding_period in holding_periods:
				key = f"es_{int(confidence_level*100)}_{holding_period}d"
				
				param_es, param_es_details = await self.es_calculator.calculate_parametric_es(
					returns_data, confidence_level, holding_period
				)
				
				hist_es, hist_es_details = await self.es_calculator.calculate_historical_es(
					returns_data, confidence_level, holding_period
				)
				
				es_results[key] = {
					'parametric': {'value': param_es, 'details': param_es_details},
					'historical': {'value': hist_es, 'details': hist_es_details}
				}
		
		risk_metrics['expected_shortfall'] = es_results
		
		# Additional risk metrics
		risk_metrics['descriptive_statistics'] = await self._calculate_descriptive_statistics(returns_data)
		risk_metrics['performance_ratios'] = await self._calculate_performance_ratios(returns_data)
		risk_metrics['tail_risk_metrics'] = await self._calculate_tail_risk_metrics(returns_data)
		
		# Liquidity risk assessment
		liquidity_risk = await self.liquidity_analyzer.assess_liquidity_risk(portfolio_data)
		risk_metrics['liquidity_risk'] = {
			'total_liquid_assets': liquidity_risk.total_liquid_assets,
			'liquidity_coverage_ratio': liquidity_risk.liquidity_coverage_ratio,
			'net_stable_funding_ratio': liquidity_risk.net_stable_funding_ratio,
			'cash_flow_gap_7d': liquidity_risk.cash_flow_gap_7d,
			'cash_flow_gap_30d': liquidity_risk.cash_flow_gap_30d,
			'funding_concentration': liquidity_risk.funding_concentration,
			'stress_test_survival_days': liquidity_risk.stress_test_survival_days
		}
		
		# Check risk thresholds and generate alerts
		await self._monitor_risk_thresholds(risk_metrics)
		
		# Cache results
		await self._cache_risk_metrics(risk_metrics)
		
		return risk_metrics
	
	async def run_comprehensive_stress_tests(
		self,
		portfolio_data: Dict[str, Any]
	) -> Dict[str, Any]:
		"""Run comprehensive stress testing suite"""
		
		logger.info(f"Running comprehensive stress tests for tenant {self.tenant_id}")
		
		stress_test_results = {}
		
		# Historical scenario stress tests
		historical_scenarios = {
			"2008_financial_crisis": {
				"equity": -0.45, "bonds": -0.15, "short_term_rates": -0.03, "cash": 0.0
			},
			"covid_pandemic": {
				"equity": -0.35, "bonds": 0.05, "short_term_rates": -0.015, "cash": 0.0
			},
			"dot_com_crash": {
				"equity": -0.50, "bonds": 0.10, "short_term_rates": -0.02, "cash": 0.0
			}
		}
		
		historical_results = {}
		for scenario_name, shocks in historical_scenarios.items():
			result = await self.stress_testing_engine.run_historical_stress_test(
				portfolio_data, scenario_name, shocks
			)
			historical_results[scenario_name] = result
		
		stress_test_results['historical_scenarios'] = historical_results
		
		# Monte Carlo stress tests
		mc_results = await self.stress_testing_engine.run_monte_carlo_stress_test(portfolio_data)
		stress_test_results['monte_carlo'] = mc_results
		
		# Liquidity stress tests
		liquidity_results = await self.stress_testing_engine.run_liquidity_stress_test(portfolio_data)
		stress_test_results['liquidity_stress'] = liquidity_results
		
		# Evaluate overall stress test results
		stress_summary = await self._summarize_stress_test_results(stress_test_results)
		stress_test_results['summary'] = stress_summary
		
		# Cache results
		await self._cache_stress_test_results(stress_test_results)
		
		return stress_test_results
	
	async def _calculate_descriptive_statistics(self, returns: np.ndarray) -> Dict[str, float]:
		"""Calculate descriptive statistics for returns"""
		
		return {
			'mean': np.mean(returns),
			'median': np.median(returns),
			'std': np.std(returns, ddof=1),
			'variance': np.var(returns, ddof=1),
			'skewness': stats.skew(returns),
			'kurtosis': stats.kurtosis(returns),
			'min': np.min(returns),
			'max': np.max(returns),
			'range': np.max(returns) - np.min(returns),
			'percentile_1': np.percentile(returns, 1),
			'percentile_5': np.percentile(returns, 5),
			'percentile_95': np.percentile(returns, 95),
			'percentile_99': np.percentile(returns, 99)
		}
	
	async def _calculate_performance_ratios(self, returns: np.ndarray) -> Dict[str, float]:
		"""Calculate performance and risk-adjusted ratios"""
		
		mean_return = np.mean(returns)
		std_return = np.std(returns, ddof=1)
		
		# Annualize metrics (assuming daily returns)
		annual_return = mean_return * 252
		annual_volatility = std_return * np.sqrt(252)
		
		# Sharpe ratio (assuming risk-free rate of 2%)
		risk_free_rate = 0.02
		sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
		
		# Sortino ratio (downside deviation)
		negative_returns = returns[returns < 0]
		downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(252) if len(negative_returns) > 0 else 0
		sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
		
		# Maximum drawdown
		cumulative_returns = np.cumprod(1 + returns)
		peak = np.maximum.accumulate(cumulative_returns)
		drawdown = (cumulative_returns - peak) / peak
		max_drawdown = np.min(drawdown)
		
		# Calmar ratio
		calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
		
		return {
			'annual_return': annual_return,
			'annual_volatility': annual_volatility,
			'sharpe_ratio': sharpe_ratio,
			'sortino_ratio': sortino_ratio,
			'calmar_ratio': calmar_ratio,
			'max_drawdown': max_drawdown,
			'downside_deviation': downside_deviation
		}
	
	async def _calculate_tail_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
		"""Calculate tail risk metrics"""
		
		# Tail ratio (95th percentile / 5th percentile)
		p95 = np.percentile(returns, 95)
		p5 = np.percentile(returns, 5)
		tail_ratio = p95 / abs(p5) if p5 != 0 else 0
		
		# Gain-to-pain ratio
		positive_returns = returns[returns > 0]
		negative_returns = returns[returns < 0]
		
		gain_sum = np.sum(positive_returns) if len(positive_returns) > 0 else 0
		pain_sum = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 1e-8
		gain_to_pain = gain_sum / pain_sum
		
		# Omega ratio (probability weighted ratio of gains vs losses)
		threshold = 0.0  # Use zero as threshold
		gains = returns[returns > threshold]
		losses = returns[returns <= threshold]
		
		omega_numerator = np.sum(gains - threshold) if len(gains) > 0 else 0
		omega_denominator = abs(np.sum(losses - threshold)) if len(losses) > 0 else 1e-8
		omega_ratio = omega_numerator / omega_denominator
		
		return {
			'tail_ratio': tail_ratio,
			'gain_to_pain_ratio': gain_to_pain,
			'omega_ratio': omega_ratio,
			'positive_periods': len(positive_returns) / len(returns),
			'negative_periods': len(negative_returns) / len(returns),
			'average_gain': np.mean(positive_returns) if len(positive_returns) > 0 else 0,
			'average_loss': np.mean(negative_returns) if len(negative_returns) > 0 else 0
		}
	
	async def _monitor_risk_thresholds(self, risk_metrics: Dict[str, Any]) -> None:
		"""Monitor risk thresholds and generate alerts"""
		
		current_alerts = []
		
		# Check VaR thresholds
		var_data = risk_metrics.get('value_at_risk', {})
		for var_key, var_results in var_data.items():
			historical_var = var_results.get('historical', {}).get('value', 0)
			
			if '95' in var_key and historical_var > self.risk_thresholds['var_95']:
				alert = RiskAlert(
					alert_id=f"var_95_breach_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
					risk_type="market_risk",
					severity="HIGH",
					threshold_breached=self.risk_thresholds['var_95'],
					current_value=historical_var,
					affected_entities=[self.tenant_id],
					recommendation="Reduce portfolio risk exposure and review hedging strategies",
					regulatory_implication="May require additional capital buffer",
					timestamp=datetime.now()
				)
				current_alerts.append(alert)
		
		# Check liquidity thresholds
		liquidity_data = risk_metrics.get('liquidity_risk', {})
		lcr = liquidity_data.get('liquidity_coverage_ratio', float('inf'))
		
		if lcr < self.risk_thresholds['liquidity_ratio']:
			alert = RiskAlert(
				alert_id=f"lcr_breach_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
				risk_type="liquidity_risk",
				severity="CRITICAL",
				threshold_breached=self.risk_thresholds['liquidity_ratio'],
				current_value=lcr,
				affected_entities=[self.tenant_id],
				recommendation="Increase liquid asset holdings and diversify funding sources",
				regulatory_implication="Regulatory reporting required; potential supervisory action",
				timestamp=datetime.now()
			)
			current_alerts.append(alert)
		
		# Store alerts
		self.risk_alerts.extend(current_alerts)
		
		# Emit alert events
		for alert in current_alerts:
			await self.events.emit_risk_alert(self.tenant_id, alert)
			logger.warning(f"Risk alert generated: {alert.alert_id} - {alert.risk_type}")
	
	async def _summarize_stress_test_results(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
		"""Summarize stress test results"""
		
		summary = {
			'worst_case_loss': 0.0,
			'worst_case_scenario': '',
			'average_loss': 0.0,
			'scenarios_tested': 0,
			'scenarios_passed': 0,
			'regulatory_breaches': 0
		}
		
		all_losses = []
		
		# Collect losses from all scenarios
		for category, results in stress_results.items():
			if category == 'summary':
				continue
				
			if isinstance(results, dict):
				for scenario_name, result in results.items():
					if hasattr(result, 'loss_amount'):
						loss = result.loss_amount
						all_losses.append(loss)
						
						if loss > summary['worst_case_loss']:
							summary['worst_case_loss'] = loss
							summary['worst_case_scenario'] = scenario_name
						
						# Check if scenario "passed" (loss < 15% threshold)
						if result.loss_percentage < 15.0:
							summary['scenarios_passed'] += 1
						
						# Check regulatory breaches
						if result.regulatory_impact:
							summary['regulatory_breaches'] += len(result.regulatory_impact)
		
		summary['scenarios_tested'] = len(all_losses)
		summary['average_loss'] = np.mean(all_losses) if all_losses else 0.0
		
		return summary
	
	async def _cache_risk_metrics(self, risk_metrics: Dict[str, Any]) -> None:
		"""Cache risk metrics results"""
		
		cache_key = f"risk_metrics:{self.tenant_id}:{datetime.now().strftime('%Y%m%d')}"
		
		cache_data = {
			'risk_metrics': risk_metrics,
			'calculation_timestamp': datetime.now().isoformat(),
			'tenant_id': self.tenant_id
		}
		
		await self.cache.set(cache_key, cache_data, ttl=86400)  # 24 hours
		logger.info(f"Cached risk metrics for tenant {self.tenant_id}")
	
	async def _cache_stress_test_results(self, stress_results: Dict[str, Any]) -> None:
		"""Cache stress test results"""
		
		cache_key = f"stress_tests:{self.tenant_id}:{datetime.now().strftime('%Y%m%d')}"
		
		# Convert StressTestResult objects to dictionaries for caching
		serializable_results = {}
		
		for category, results in stress_results.items():
			if isinstance(results, dict):
				serializable_results[category] = {}
				for scenario_name, result in results.items():
					if hasattr(result, '__dict__'):
						serializable_results[category][scenario_name] = result.__dict__
					else:
						serializable_results[category][scenario_name] = result
			else:
				serializable_results[category] = results
		
		cache_data = {
			'stress_test_results': serializable_results,
			'calculation_timestamp': datetime.now().isoformat(),
			'tenant_id': self.tenant_id
		}
		
		await self.cache.set(cache_key, cache_data, ttl=86400)  # 24 hours
		logger.info(f"Cached stress test results for tenant {self.tenant_id}")
	
	async def get_risk_dashboard_data(self) -> Dict[str, Any]:
		"""Get comprehensive risk dashboard data"""
		
		# Retrieve cached data
		risk_metrics_key = f"risk_metrics:{self.tenant_id}:{datetime.now().strftime('%Y%m%d')}"
		stress_tests_key = f"stress_tests:{self.tenant_id}:{datetime.now().strftime('%Y%m%d')}"
		
		risk_metrics = await self.cache.get(risk_metrics_key)
		stress_tests = await self.cache.get(stress_tests_key)
		
		# Get recent alerts
		recent_alerts = [
			alert for alert in self.risk_alerts 
			if (datetime.now() - alert.timestamp).days <= 7
		]
		
		dashboard_data = {
			'risk_metrics': risk_metrics.get('risk_metrics', {}) if risk_metrics else {},
			'stress_tests': stress_tests.get('stress_test_results', {}) if stress_tests else {},
			'recent_alerts': [alert.__dict__ for alert in recent_alerts],
			'risk_summary': {
				'total_alerts': len(recent_alerts),
				'critical_alerts': len([a for a in recent_alerts if a.severity == 'CRITICAL']),
				'last_calculation': risk_metrics.get('calculation_timestamp') if risk_metrics else None,
				'risk_score': self._calculate_overall_risk_score(risk_metrics, recent_alerts)
			}
		}
		
		return dashboard_data
	
	def _calculate_overall_risk_score(
		self, 
		risk_metrics: Optional[Dict[str, Any]], 
		alerts: List[RiskAlert]
	) -> float:
		"""Calculate overall risk score (0-100, higher is riskier)"""
		
		base_score = 20.0  # Base risk score
		
		# Adjust for alerts
		alert_penalty = len(alerts) * 5.0
		critical_penalty = len([a for a in alerts if a.severity == 'CRITICAL']) * 15.0
		
		# Adjust for risk metrics if available
		metrics_penalty = 0.0
		if risk_metrics:
			var_data = risk_metrics.get('risk_metrics', {}).get('value_at_risk', {})
			if var_data:
				# Use 1-day 95% VaR as proxy
				var_95_1d = var_data.get('var_95_1d', {})
				if var_95_1d:
					hist_var = var_95_1d.get('historical', {}).get('value', 0)
					if hist_var > 0.05:  # 5% threshold
						metrics_penalty = (hist_var - 0.05) * 200  # Scale penalty
		
		total_score = min(100.0, base_score + alert_penalty + critical_penalty + metrics_penalty)
		return total_score

# ============================================================================
# Export
# ============================================================================

__all__ = [
	'AdvancedRiskAnalyticsEngine',
	'VaRCalculator',
	'ExpectedShortfallCalculator',
	'StressTestingEngine',
	'LiquidityRiskAnalyzer',
	'RiskMetric',
	'RiskModel',
	'StressTestType',
	'RegulatoryFramework',
	'RiskMeasurement',
	'StressTestResult',
	'RiskAlert',
	'LiquidityRisk'
]