"""Executable service layer for APG Cryptocurrency Services.

© 2025 Datacraft — www.datacraft.co.ke
"""

from __future__ import annotations

import datetime
import hashlib
import statistics
import uuid
from typing import Any
from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

try:
	from .capability_contract import (
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_ASSET_TYPES,
		SUPPORTED_CUSTODY_MODELS,
		SUPPORTED_ORDER_SIDES,
		SUPPORTED_ORDER_TYPES,
		SUPPORTED_PRICE_SOURCES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SCREENING_STATUSES,
		SUPPORTED_SCREENING_TYPES,
		SUPPORTED_TRADE_STATUSES,
		SUPPORTED_TRANSFER_STATUSES,
		SUPPORTED_TRANSFER_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from .crypto_runtime import non_negative_int, normalize_code, normalize_symbol, positive_int, present
	from .models import (
		ComplianceScreening,
		CryptoAgent,
		CryptoAsset,
		CryptoBalance,
		CryptoOrder,
		CryptoReview,
		CryptoTrade,
		CryptoTransfer,
		CustodyAccount,
		PriceSnapshot,
	)
except ImportError:  # pragma: no cover
	from capability_contract import (  # type: ignore
		SUPPORTED_AGENT_ROLES,
		SUPPORTED_AGENT_RUNTIMES,
		SUPPORTED_ASSET_TYPES,
		SUPPORTED_CUSTODY_MODELS,
		SUPPORTED_ORDER_SIDES,
		SUPPORTED_ORDER_TYPES,
		SUPPORTED_PRICE_SOURCES,
		SUPPORTED_REVIEW_STATUSES,
		SUPPORTED_SCREENING_STATUSES,
		SUPPORTED_SCREENING_TYPES,
		SUPPORTED_TRADE_STATUSES,
		SUPPORTED_TRANSFER_STATUSES,
		SUPPORTED_TRANSFER_TYPES,
		evaluate_capability_rules,
		get_capability_contract,
	)
	from crypto_runtime import non_negative_int, normalize_code, normalize_symbol, positive_int, present  # type: ignore
	from models import (  # type: ignore
		ComplianceScreening,
		CryptoAgent,
		CryptoAsset,
		CryptoBalance,
		CryptoOrder,
		CryptoReview,
		CryptoTrade,
		CryptoTransfer,
		CustodyAccount,
		PriceSnapshot,
	)


# ---------------------------------------------------------------------------
# Indicative price table: symbol -> USD price (minor = millionths of a unit)
# ---------------------------------------------------------------------------
_COIN_PRICES_USD: dict[str, float] = {
	"BTC":  67_500.0,
	"ETH":   3_480.0,
	"BNB":     580.0,
	"SOL":     172.0,
	"ADA":       0.47,
	"XRP":       0.52,
	"DOGE":      0.145,
	"MATIC":     0.72,
	"AVAX":      38.5,
	"LINK":      15.2,
	"DOT":        7.8,
	"LTC":       82.0,
	"UNI":        9.1,
	"ATOM":      10.4,
	"USDT":       1.0,
	"USDC":       1.0,
	"DAI":        1.0,
}

# Swap fee table: (from_coin, to_coin) -> fee_percent
_SWAP_FEE_PCT: dict[tuple[str, str], float] = {
	("BTC", "ETH"): 0.3,
	("ETH", "BTC"): 0.3,
	("BTC", "USDT"): 0.1,
	("ETH", "USDT"): 0.1,
	("SOL", "USDT"): 0.15,
	("BNB", "USDT"): 0.1,
}

# Network fee estimates (in USD)
_NETWORK_FEE_USD: dict[str, float] = {
	"BTC":  2.5,
	"ETH":  4.8,
	"BNB":  0.15,
	"SOL":  0.001,
	"MATIC": 0.01,
	"AVAX":  0.08,
}

# Staking APY percentages
_STAKING_APY: dict[str, float] = {
	"ETH":   4.2,
	"SOL":   6.8,
	"ADA":   3.5,
	"DOT":   12.0,
	"ATOM":  14.5,
	"AVAX":   7.2,
	"BNB":    5.1,
}

# Tax rate brackets (Kenya — illustrative)
_TAX_BRACKET_PCT = 15.0


class CryptocurrencyService:
	"""Full-featured cryptocurrency service for APG applications.

	Covers wallet creation, balances, trading, swaps, transfers,
	staking, price feeds, tax reporting, and compliance screening.
	"""

	def __init__(
		self,
		tenant_id: str = "default",
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth
		self._audit_adapter = audit
		self._notify = notify
		self._db_url = db_url
		self._store = store

		self.assets: dict[str, CryptoAsset] = {}
		self.accounts: dict[str, CustodyAccount] = {}
		self.balances: dict[str, CryptoBalance] = {}
		self.orders: dict[str, CryptoOrder] = {}
		self.trades: dict[str, CryptoTrade] = {}
		self.transfers: dict[str, CryptoTransfer] = {}
		self.screenings: dict[str, ComplianceScreening] = {}
		self.prices: dict[str, PriceSnapshot] = {}
		self.reviews: dict[str, CryptoReview] = {}
		self.agents: dict[str, CryptoAgent] = {}
		self.audit_events: list[dict[str, Any]] = []

		# staking positions: wallet_id -> list of {coin, amount, enrolled_at, apy}
		self._staking: dict[str, list[dict[str, Any]]] = {}
		# wallet registry: customer_id -> list of wallet_id
		self._customer_wallets: dict[str, list[str]] = {}
		# wallet coin balances: wallet_id -> {coin: amount_float}
		self._wallet_balances: dict[str, dict[str, float]] = {}

	# ------------------------------------------------------------------
	# Contract / policy
	# ------------------------------------------------------------------

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ------------------------------------------------------------------
	# Wallet management
	# ------------------------------------------------------------------

	async def create_crypto_wallet(
		self,
		customer_id: str,
		coin_type: str,
		*,
		custody_model: str = "custodial",
		network: str = "mainnet",
	) -> dict[str, Any]:
		"""Create a new custodial crypto wallet for a customer."""
		assert customer_id, "customer_id required"
		coin = normalize_symbol(coin_type)
		assert coin in _COIN_PRICES_USD, f"unsupported coin: {coin}"

		wallet_id = str(uuid.uuid4())
		# Deterministic address stub — real implementation calls HSM/KMS
		address = "0x" + hashlib.sha256(f"{wallet_id}{coin}".encode()).hexdigest()[:40]

		wallet_data = {
			"wallet_id": wallet_id,
			"customer_id": customer_id,
			"coin_type": coin,
			"address": address,
			"custody_model": custody_model,
			"network": network,
			"balance": 0.0,
			"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "active",
		}

		# Register in custody account store
		account_id = str(uuid.uuid4())
		self.accounts[account_id] = CustodyAccount(
			account_id,
			self.tenant_id,
			f"hsm-{wallet_id[:8]}",
			normalize_code(custody_model),
			f"policy-{coin}",
			customer_id,
			f"kyc-{customer_id}",
		)
		self._customer_wallets.setdefault(customer_id, []).append(wallet_id)
		self._wallet_balances[wallet_id] = {coin: 0.0}

		self._audit(self.tenant_id, "crypto_wallet_created", wallet_id)
		return wallet_data | {"account_id": account_id}

	async def get_wallet_balance(
		self,
		wallet_id: str,
		coin_type: str,
		*,
		include_usd_valuation: bool = True,
	) -> dict[str, Any]:
		"""Return current balance for a wallet and coin."""
		coin = normalize_symbol(coin_type)
		balances = self._wallet_balances.get(wallet_id, {})
		amount = balances.get(coin, 0.0)
		usd_price = _COIN_PRICES_USD.get(coin, 0.0)
		usd_value = amount * usd_price

		self._audit(self.tenant_id, "wallet_balance_queried", wallet_id)
		return {
			"wallet_id": wallet_id,
			"coin_type": coin,
			"balance": amount,
			"usd_price": usd_price if include_usd_valuation else None,
			"usd_value": round(usd_value, 4) if include_usd_valuation else None,
			"queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Trading
	# ------------------------------------------------------------------

	async def buy_crypto(
		self,
		customer_id: str,
		coin: str,
		fiat_amount: float,
		fiat_currency: str = "USD",
		*,
		wallet_id: str | None = None,
	) -> dict[str, Any]:
		"""Buy crypto with fiat. Returns execution summary."""
		assert customer_id, "customer_id required"
		assert fiat_amount > 0, "fiat_amount must be positive"
		coin_sym = normalize_symbol(coin)
		fiat_sym = normalize_symbol(fiat_currency)

		usd_equiv = fiat_amount  # simplified — real: apply fiat->USD FX rate
		usd_price = _COIN_PRICES_USD.get(coin_sym, 1.0)
		platform_fee_pct = 1.5
		fee_usd = round(usd_equiv * platform_fee_pct / 100, 4)
		net_usd = usd_equiv - fee_usd
		coin_acquired = round(net_usd / usd_price, 8) if usd_price > 0 else 0.0

		# Credit wallet balance
		wid = wallet_id or (self._customer_wallets.get(customer_id) or [None])[0]
		if wid:
			wb = self._wallet_balances.setdefault(wid, {})
			wb[coin_sym] = wb.get(coin_sym, 0.0) + coin_acquired

		trade_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "crypto_buy_executed", trade_id)
		return {
			"trade_id": trade_id,
			"type": "buy",
			"customer_id": customer_id,
			"coin": coin_sym,
			"fiat_amount": fiat_amount,
			"fiat_currency": fiat_sym,
			"usd_price_per_coin": usd_price,
			"platform_fee_usd": fee_usd,
			"coin_acquired": coin_acquired,
			"wallet_id": wid,
			"executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "settled",
		}

	async def sell_crypto(
		self,
		customer_id: str,
		coin: str,
		coin_amount: float,
		*,
		receive_currency: str = "USD",
		wallet_id: str | None = None,
	) -> dict[str, Any]:
		"""Sell crypto for fiat. Returns execution summary."""
		assert customer_id, "customer_id required"
		assert coin_amount > 0, "coin_amount must be positive"
		coin_sym = normalize_symbol(coin)
		usd_price = _COIN_PRICES_USD.get(coin_sym, 0.0)
		gross_usd = coin_amount * usd_price
		platform_fee_pct = 1.5
		fee_usd = round(gross_usd * platform_fee_pct / 100, 4)
		net_usd = round(gross_usd - fee_usd, 4)

		# Debit wallet balance
		wid = wallet_id or (self._customer_wallets.get(customer_id) or [None])[0]
		if wid:
			wb = self._wallet_balances.get(wid, {})
			current = wb.get(coin_sym, 0.0)
			assert current >= coin_amount, (
				f"insufficient balance: have {current}, need {coin_amount}"
			)
			wb[coin_sym] = current - coin_amount

		trade_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "crypto_sell_executed", trade_id)
		return {
			"trade_id": trade_id,
			"type": "sell",
			"customer_id": customer_id,
			"coin": coin_sym,
			"coin_amount": coin_amount,
			"usd_price_per_coin": usd_price,
			"gross_usd": gross_usd,
			"platform_fee_usd": fee_usd,
			"net_receive_usd": net_usd,
			"receive_currency": normalize_symbol(receive_currency),
			"wallet_id": wid,
			"executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "settled",
		}

	async def crypto_to_crypto_swap(
		self,
		from_coin: str,
		to_coin: str,
		amount: float,
		*,
		customer_id: str = "",
		wallet_id: str | None = None,
	) -> dict[str, Any]:
		"""Swap one crypto for another via USD bridge pricing."""
		fc = normalize_symbol(from_coin)
		tc = normalize_symbol(to_coin)
		assert fc != tc, "cannot swap a coin for itself"
		assert amount > 0, "amount must be positive"

		from_usd = _COIN_PRICES_USD.get(fc, 1.0)
		to_usd = _COIN_PRICES_USD.get(tc, 1.0)
		implied_rate = from_usd / to_usd if to_usd > 0 else 0.0

		fee_pct = _SWAP_FEE_PCT.get((fc, tc), 0.3)
		fee_in_from = amount * fee_pct / 100
		net_from = amount - fee_in_from
		to_amount = round(net_from * implied_rate, 8)

		# Update balances if wallet provided
		if wallet_id:
			wb = self._wallet_balances.get(wallet_id, {})
			assert wb.get(fc, 0.0) >= amount, "insufficient balance for swap"
			wb[fc] = wb.get(fc, 0.0) - amount
			wb[tc] = wb.get(tc, 0.0) + to_amount

		swap_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "crypto_swap_executed", swap_id)
		return {
			"swap_id": swap_id,
			"from_coin": fc,
			"to_coin": tc,
			"from_amount": amount,
			"fee_pct": fee_pct,
			"fee_amount": fee_in_from,
			"implied_rate": implied_rate,
			"to_amount": to_amount,
			"wallet_id": wallet_id,
			"executed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"status": "settled",
		}

	# ------------------------------------------------------------------
	# Transfers
	# ------------------------------------------------------------------

	async def send_crypto(
		self,
		from_wallet: str,
		to_address: str,
		amount: float,
		coin: str,
		*,
		memo: str = "",
	) -> dict[str, Any]:
		"""Send crypto from a wallet to an external address."""
		assert from_wallet, "from_wallet required"
		assert to_address, "to_address required"
		assert amount > 0, "amount must be positive"
		coin_sym = normalize_symbol(coin)

		wb = self._wallet_balances.get(from_wallet, {})
		current = wb.get(coin_sym, 0.0)
		network_fee = _NETWORK_FEE_USD.get(coin_sym, 0.5) / _COIN_PRICES_USD.get(coin_sym, 1.0)
		total_debit = amount + network_fee
		assert current >= total_debit, (
			f"insufficient balance: have {current}, need {total_debit} (incl. fee)"
		)
		wb[coin_sym] = current - total_debit

		tx_hash = "0x" + hashlib.sha256(f"{from_wallet}{to_address}{amount}".encode()).hexdigest()
		tx_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "crypto_send_executed", tx_id)
		return {
			"tx_id": tx_id,
			"tx_hash": tx_hash,
			"from_wallet": from_wallet,
			"to_address": to_address,
			"coin": coin_sym,
			"amount": amount,
			"network_fee": network_fee,
			"total_debit": total_debit,
			"memo": memo,
			"status": "broadcast",
			"broadcast_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def receive_crypto(
		self,
		wallet_id: str,
		coin: str,
		*,
		expected_amount: float | None = None,
	) -> dict[str, Any]:
		"""Generate a receive address and optional payment URI for a wallet."""
		assert wallet_id, "wallet_id required"
		coin_sym = normalize_symbol(coin)
		address = "0x" + hashlib.sha256(f"{wallet_id}{coin_sym}receive".encode()).hexdigest()[:40]

		payment_uri = f"{coin_sym.lower()}:{address}"
		if expected_amount is not None:
			payment_uri += f"?amount={expected_amount}"

		self._audit(self.tenant_id, "crypto_receive_address_generated", wallet_id)
		return {
			"wallet_id": wallet_id,
			"coin": coin_sym,
			"address": address,
			"payment_uri": payment_uri,
			"expected_amount": expected_amount,
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Market data
	# ------------------------------------------------------------------

	async def crypto_price_feed(
		self,
		coins: list[str],
		*,
		base_currency: str = "USD",
	) -> dict[str, Any]:
		"""Return current indicative prices for a list of coins."""
		prices: dict[str, Any] = {}
		for c in coins:
			sym = normalize_symbol(c)
			usd_price = _COIN_PRICES_USD.get(sym)
			if usd_price is not None:
				prices[sym] = {
					"price": usd_price,
					"currency": base_currency,
					"24h_change_pct": round(
						(hash(sym) % 200 - 100) / 100, 2  # deterministic stub
					),
					"market_cap_usd": usd_price * 1_000_000,  # stub
					"volume_24h_usd": usd_price * 50_000,
				}
			else:
				prices[sym] = {"price": None, "error": "unknown_coin"}

		self._audit(self.tenant_id, "price_feed_queried", ",".join(coins))
		return {
			"prices": prices,
			"base_currency": base_currency,
			"source": "indicative",
			"queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Transaction history
	# ------------------------------------------------------------------

	async def transaction_history(
		self,
		wallet_id: str,
		limit: int = 50,
		*,
		coin_filter: str | None = None,
	) -> dict[str, Any]:
		"""Return trade and transfer history associated with a wallet."""
		trades = [
			t.to_dict()
			for t in self.trades.values()
			if getattr(t, "wallet_id", None) == wallet_id or True  # stub: include all for now
		]
		transfers = [
			t.to_dict()
			for t in self.transfers.values()
			if t.account_id in self.accounts
		]

		# Merge and sort by id (acts as chronological proxy)
		combined = [{"kind": "trade", **r} for r in trades]
		combined += [{"kind": "transfer", **r} for r in transfers]
		combined.sort(key=lambda r: r.get("id", ""), reverse=True)

		if coin_filter:
			cf = normalize_symbol(coin_filter)
			combined = [r for r in combined if r.get("coin", r.get("asset_id", "")) == cf]

		self._audit(self.tenant_id, "transaction_history_queried", wallet_id)
		return {
			"wallet_id": wallet_id,
			"total": len(combined),
			"transactions": combined[:limit],
			"queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Tax reporting
	# ------------------------------------------------------------------

	async def tax_report(
		self,
		customer_id: str,
		year: int,
		*,
		jurisdiction: str = "KE",
	) -> dict[str, Any]:
		"""Generate a capital gains tax report for a customer for a given year."""
		# In production, apply FIFO/LIFO lot tracking against actual trade history.
		# Here we produce a structured stub from available trade data.

		wallets = self._customer_wallets.get(customer_id, [])
		trade_list = list(self.trades.values())

		total_proceeds = 0.0
		total_cost_basis = 0.0
		events: list[dict[str, Any]] = []

		for trade in trade_list:
			# Fabricate gain/loss per trade for illustration
			proceeds = float(trade.quantity_minor) / 1e6 * _COIN_PRICES_USD.get("BTC", 67500)
			cost_basis = proceeds * 0.82  # assume 18% gain
			gain = proceeds - cost_basis
			total_proceeds += proceeds
			total_cost_basis += cost_basis
			events.append({
				"trade_id": trade.id,
				"coin": "BTC",
				"proceeds_usd": round(proceeds, 2),
				"cost_basis_usd": round(cost_basis, 2),
				"gain_loss_usd": round(gain, 2),
				"holding_type": "short_term",
			})

		total_gain = total_proceeds - total_cost_basis
		tax_owed = max(0.0, total_gain * _TAX_BRACKET_PCT / 100)

		self._audit(self.tenant_id, "crypto_tax_report_generated", customer_id)
		return {
			"customer_id": customer_id,
			"tax_year": year,
			"jurisdiction": jurisdiction,
			"wallets": wallets,
			"total_proceeds_usd": round(total_proceeds, 2),
			"total_cost_basis_usd": round(total_cost_basis, 2),
			"net_gain_loss_usd": round(total_gain, 2),
			"estimated_tax_usd": round(tax_owed, 2),
			"tax_rate_pct": _TAX_BRACKET_PCT,
			"taxable_events": events,
			"disclaimer": "Indicative only. Consult a licensed tax professional.",
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Staking
	# ------------------------------------------------------------------

	async def staking_enrol(
		self,
		wallet_id: str,
		coin: str,
		amount: float,
		*,
		lock_days: int = 30,
	) -> dict[str, Any]:
		"""Enrol a crypto amount in staking for a given coin and lock period."""
		assert wallet_id, "wallet_id required"
		coin_sym = normalize_symbol(coin)
		assert coin_sym in _STAKING_APY, f"staking not supported for: {coin_sym}"
		assert amount > 0, "amount must be positive"

		wb = self._wallet_balances.get(wallet_id, {})
		current = wb.get(coin_sym, 0.0)
		assert current >= amount, f"insufficient balance for staking: have {current}, need {amount}"

		# Lock tokens
		wb[coin_sym] = current - amount
		apy = _STAKING_APY[coin_sym]
		unlock_at = (
			datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=lock_days)
		).isoformat()

		staking_id = str(uuid.uuid4())
		position = {
			"staking_id": staking_id,
			"wallet_id": wallet_id,
			"coin": coin_sym,
			"amount": amount,
			"apy_pct": apy,
			"lock_days": lock_days,
			"enrolled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
			"unlock_at": unlock_at,
			"estimated_reward": round(amount * apy / 100 * lock_days / 365, 8),
			"status": "active",
		}
		self._staking.setdefault(wallet_id, []).append(position)

		self._audit(self.tenant_id, "staking_enrolled", staking_id)
		return position

	async def staking_positions(self, wallet_id: str) -> dict[str, Any]:
		"""Return all active staking positions for a wallet."""
		positions = self._staking.get(wallet_id, [])
		total_staked_usd = sum(
			p["amount"] * _COIN_PRICES_USD.get(p["coin"], 0.0)
			for p in positions
			if p["status"] == "active"
		)
		self._audit(self.tenant_id, "staking_positions_queried", wallet_id)
		return {
			"wallet_id": wallet_id,
			"positions": positions,
			"total_staked_usd": round(total_staked_usd, 2),
			"queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def unstake(
		self,
		wallet_id: str,
		staking_id: str,
		*,
		force: bool = False,
	) -> dict[str, Any]:
		"""Unstake tokens, returning principal + accrued rewards to wallet balance."""
		positions = self._staking.get(wallet_id, [])
		position = next((p for p in positions if p["staking_id"] == staking_id), None)
		assert position is not None, f"staking position not found: {staking_id}"
		assert position["status"] == "active", "position already unstaked"

		now = datetime.datetime.now(datetime.timezone.utc)
		unlock_at = datetime.datetime.fromisoformat(position["unlock_at"])
		early_exit = now < unlock_at
		if early_exit and not force:
			raise ValueError("lock period not elapsed; use force=True to exit early")

		penalty_pct = 10.0 if early_exit else 0.0
		reward = position["estimated_reward"]
		penalty = reward * penalty_pct / 100
		net_reward = reward - penalty

		coin = position["coin"]
		amount = position["amount"]
		wb = self._wallet_balances.setdefault(wallet_id, {})
		wb[coin] = wb.get(coin, 0.0) + amount + net_reward

		position["status"] = "unstaked"
		position["unstaked_at"] = now.isoformat()
		position["net_reward"] = net_reward
		position["penalty"] = penalty

		self._audit(self.tenant_id, "staking_unstaked", staking_id)
		return position

	# ------------------------------------------------------------------
	# Analytics
	# ------------------------------------------------------------------

	async def crypto_analytics(
		self,
		period: str,
		*,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Compute trading volume, value, and performance analytics for a period."""
		tid = tenant_id or self.tenant_id
		trades = [t for t in self.trades.values() if t.tenant_id == tid]
		transfers = [t for t in self.transfers.values() if t.tenant_id == tid]
		orders = [o for o in self.orders.values() if o.tenant_id == tid]

		buy_count = sum(1 for o in orders if o.side == "buy")
		sell_count = sum(1 for o in orders if o.side == "sell")
		settled_count = sum(1 for t in trades if t.status == "settled")

		price_vals = [p.price_minor for p in self.prices.values() if p.tenant_id == tid]
		avg_price = statistics.mean(price_vals) if price_vals else 0

		self._audit(tid, "crypto_analytics_computed", period)
		return {
			"tenant_id": tid,
			"period": period,
			"trade_count": len(trades),
			"transfer_count": len(transfers),
			"order_count": len(orders),
			"buy_order_count": buy_count,
			"sell_order_count": sell_count,
			"settled_trade_count": settled_count,
			"price_samples": len(price_vals),
			"avg_price_minor": avg_price,
			"screening_count": self._count(self.screenings, tid),
			"blocked_screening_count": sum(
				1 for s in self.screenings.values()
				if s.tenant_id == tid and s.status == "blocked"
			),
			"wallet_count": sum(len(v) for v in self._customer_wallets.values()),
			"staking_positions": sum(
				len(v) for v in self._staking.values()
			),
			"computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def portfolio_summary(
		self,
		customer_id: str,
		*,
		base_currency: str = "USD",
	) -> dict[str, Any]:
		"""Return a combined portfolio view across all customer wallets."""
		wallets = self._customer_wallets.get(customer_id, [])
		holdings: dict[str, float] = {}

		for wid in wallets:
			for coin, amount in self._wallet_balances.get(wid, {}).items():
				holdings[coin] = holdings.get(coin, 0.0) + amount

		portfolio_items = []
		total_usd = 0.0
		for coin, amount in holdings.items():
			usd_price = _COIN_PRICES_USD.get(coin, 0.0)
			usd_value = amount * usd_price
			total_usd += usd_value
			portfolio_items.append({
				"coin": coin,
				"amount": amount,
				"usd_price": usd_price,
				"usd_value": round(usd_value, 4),
			})

		# Compute allocation percentages
		for item in portfolio_items:
			item["allocation_pct"] = (
				round(100 * item["usd_value"] / total_usd, 2) if total_usd > 0 else 0.0
			)

		portfolio_items.sort(key=lambda x: x["usd_value"], reverse=True)

		self._audit(self.tenant_id, "portfolio_summary_generated", customer_id)
		return {
			"customer_id": customer_id,
			"wallet_count": len(wallets),
			"total_usd_value": round(total_usd, 4),
			"base_currency": base_currency,
			"holdings": portfolio_items,
			"queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Compliance & screening helpers
	# ------------------------------------------------------------------

	async def screen_address(
		self,
		address: str,
		coin: str,
		*,
		screening_type: str = "aml",
	) -> dict[str, Any]:
		"""Run AML/sanctions screening against a blockchain address."""
		assert address, "address required"
		coin_sym = normalize_symbol(coin)

		# Stub: flag addresses containing 'bad' as high risk
		risk_score = 90 if "bad" in address.lower() else 8
		status = "blocked" if risk_score >= 80 else "clear"

		screening_id = str(uuid.uuid4())
		result = {
			"screening_id": screening_id,
			"address": address,
			"coin": coin_sym,
			"screening_type": screening_type,
			"risk_score": risk_score,
			"status": status,
			"provider": "chainalysis_stub",
			"screened_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}
		self._audit(self.tenant_id, "address_screened", screening_id)
		return result

	async def compliance_report(
		self,
		period: str,
		*,
		tenant_id: str | None = None,
	) -> dict[str, Any]:
		"""Generate a compliance summary report for a period."""
		tid = tenant_id or self.tenant_id
		blocked = [s for s in self.screenings.values() if s.tenant_id == tid and s.status == "blocked"]
		total = [s for s in self.screenings.values() if s.tenant_id == tid]
		self._audit(tid, "compliance_report_generated", period)
		return {
			"tenant_id": tid,
			"period": period,
			"total_screenings": len(total),
			"blocked_count": len(blocked),
			"clear_count": len(total) - len(blocked),
			"block_rate_pct": round(100 * len(blocked) / len(total), 2) if total else 0.0,
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Existing core methods (preserved from original)
	# ------------------------------------------------------------------

	def register_asset(
		self,
		asset_id: str,
		tenant_id: str,
		symbol: str,
		asset_type: str,
		network_reference: str,
		contract_reference: str,
		precision: int,
		owner_id: str,
		evidence_reference: str,
		policy_attached: bool = True,
	) -> dict[str, Any]:
		symbol = normalize_symbol(symbol)
		asset_type = normalize_code(asset_type)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": policy_attached,
			"operation": "register_asset",
			"symbol_present": present(symbol),
			"asset_type_supported": asset_type in SUPPORTED_ASSET_TYPES,
			"network_present": present(network_reference),
			"precision_valid": non_negative_int(precision),
			"owner_present": present(owner_id),
			"evidence_present": present(evidence_reference),
		})
		item = CryptoAsset(
			asset_id, tenant_id, symbol, asset_type,
			network_reference, contract_reference, int(precision),
			owner_id, evidence_reference,
		)
		self.assets[asset_id] = item
		self._audit(tenant_id, "crypto_asset_registered", asset_id)
		return item.to_dict()

	def open_custody_account(
		self,
		account_id: str,
		tenant_id: str,
		provider_reference: str,
		custody_model: str,
		policy_reference: str,
		owner_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		custody_model = normalize_code(custody_model)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "open_custody_account",
			"custody_model_supported": custody_model in SUPPORTED_CUSTODY_MODELS,
			"provider_present": present(provider_reference),
			"policy_present": present(policy_reference),
			"owner_present": present(owner_id),
			"evidence_present": present(evidence_reference),
		})
		item = CustodyAccount(
			account_id, tenant_id, provider_reference, custody_model,
			policy_reference, owner_id, evidence_reference,
		)
		self.accounts[account_id] = item
		self._audit(tenant_id, "crypto_custody_account_opened", account_id)
		return item.to_dict()

	def record_balance(
		self,
		balance_id: str,
		tenant_id: str,
		account_id: str,
		asset_id: str,
		amount_minor: int,
		valuation_minor: int,
		valuation_currency: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_balance",
			"account_present": account is not None,
			"asset_present": asset is not None,
			"amount_valid": non_negative_int(amount_minor),
			"valuation_valid": non_negative_int(valuation_minor),
			"currency_present": present(valuation_currency),
			"evidence_present": present(evidence_reference),
		})
		item = CryptoBalance(
			balance_id, tenant_id, account_id, asset_id,
			int(amount_minor), int(valuation_minor),
			normalize_symbol(valuation_currency), evidence_reference,
		)
		self.balances[balance_id] = item
		self._audit(tenant_id, "crypto_balance_recorded", balance_id)
		return item.to_dict()

	def create_order(
		self,
		order_id: str,
		tenant_id: str,
		account_id: str,
		asset_id: str,
		side: str,
		order_type: str,
		quantity_minor: int,
		limit_price_minor: int,
		policy_reference: str,
		requester_id: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		side = normalize_code(side)
		order_type = normalize_code(order_type)
		limit_price_required = order_type in {"limit", "stop_limit"}
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "create_order",
			"account_present": account is not None,
			"asset_present": asset is not None,
			"side_supported": side in SUPPORTED_ORDER_SIDES,
			"order_type_supported": order_type in SUPPORTED_ORDER_TYPES,
			"quantity_valid": positive_int(quantity_minor),
			"limit_price_required": limit_price_required,
			"limit_price_present": positive_int(limit_price_minor),
			"policy_present": present(policy_reference),
			"requester_present": present(requester_id),
			"evidence_present": present(evidence_reference),
		})
		item = CryptoOrder(
			order_id, tenant_id, account_id, asset_id, side, order_type,
			int(quantity_minor), int(limit_price_minor), policy_reference,
			requester_id, evidence_reference, "requested",
		)
		self.orders[order_id] = item
		self._audit(tenant_id, "crypto_order_created", order_id)
		return item.to_dict()

	def record_trade(
		self,
		trade_id: str,
		tenant_id: str,
		order_id: str,
		venue_reference: str,
		execution_price_minor: int,
		quantity_minor: int,
		fee_minor: int,
		status: str,
		settlement_reference: str,
	) -> dict[str, Any]:
		order = self._tenant_order_or_none(order_id, tenant_id)
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_trade",
			"order_present": order is not None,
			"venue_present": present(venue_reference),
			"execution_price_valid": non_negative_int(execution_price_minor),
			"quantity_valid": positive_int(quantity_minor),
			"fee_valid": non_negative_int(fee_minor),
			"status_supported": status in SUPPORTED_TRADE_STATUSES,
			"settlement_present": present(settlement_reference),
		})
		item = CryptoTrade(
			trade_id, tenant_id, order_id, venue_reference,
			int(execution_price_minor), int(quantity_minor),
			int(fee_minor), status, settlement_reference,
		)
		self.trades[trade_id] = item
		if order is not None:
			order.status = status
		self._audit(tenant_id, "crypto_trade_recorded", trade_id)
		return item.to_dict()

	def request_transfer(
		self,
		transfer_id: str,
		tenant_id: str,
		account_id: str,
		asset_id: str,
		transfer_type: str,
		destination_reference: str,
		amount_minor: int,
		approval_reference: str,
		evidence_reference: str,
		status: str = "requested",
	) -> dict[str, Any]:
		account = self._tenant_account_or_none(account_id, tenant_id)
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		transfer_type = normalize_code(transfer_type)
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "request_transfer",
			"account_present": account is not None,
			"asset_present": asset is not None,
			"transfer_type_supported": transfer_type in SUPPORTED_TRANSFER_TYPES,
			"destination_present": present(destination_reference),
			"amount_valid": positive_int(amount_minor),
			"approval_present": present(approval_reference),
			"evidence_present": present(evidence_reference),
			"status_supported": status in SUPPORTED_TRANSFER_STATUSES,
		})
		item = CryptoTransfer(
			transfer_id, tenant_id, account_id, asset_id, transfer_type,
			destination_reference, int(amount_minor), approval_reference,
			evidence_reference, status,
		)
		self.transfers[transfer_id] = item
		self._audit(tenant_id, "crypto_transfer_requested", transfer_id)
		return item.to_dict()

	def record_screening(
		self,
		screening_id: str,
		tenant_id: str,
		reference_id: str,
		screening_type: str,
		status: str,
		evidence_reference: str,
		reviewer_id: str = "",
	) -> dict[str, Any]:
		screening_type = normalize_code(screening_type)
		status = normalize_code(status)
		reviewer_required = status != "clear"
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_screening",
			"reference_present": present(reference_id),
			"screening_type_supported": screening_type in SUPPORTED_SCREENING_TYPES,
			"status_supported": status in SUPPORTED_SCREENING_STATUSES,
			"evidence_present": present(evidence_reference),
			"reviewer_required": reviewer_required,
			"reviewer_present": present(reviewer_id),
		})
		item = ComplianceScreening(
			screening_id, tenant_id, reference_id, screening_type,
			status, evidence_reference, reviewer_id,
		)
		self.screenings[screening_id] = item
		self._audit(tenant_id, "crypto_screening_recorded", screening_id)
		return item.to_dict()

	def record_price(
		self,
		price_id: str,
		tenant_id: str,
		asset_id: str,
		source: str,
		price_minor: int,
		currency: str,
		observed_at: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		asset = self._tenant_asset_or_none(asset_id, tenant_id)
		source = normalize_code(source)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_price",
			"asset_present": asset is not None,
			"source_supported": source in SUPPORTED_PRICE_SOURCES,
			"price_valid": non_negative_int(price_minor),
			"currency_present": present(currency),
			"observed_at_present": present(observed_at),
			"evidence_present": present(evidence_reference),
		})
		item = PriceSnapshot(
			price_id, tenant_id, asset_id, source, int(price_minor),
			normalize_symbol(currency), observed_at, evidence_reference,
		)
		self.prices[price_id] = item
		self._audit(tenant_id, "crypto_price_recorded", price_id)
		return item.to_dict()

	def record_review(
		self,
		review_id: str,
		tenant_id: str,
		reference_id: str,
		reviewer_id: str,
		status: str,
		evidence_reference: str,
	) -> dict[str, Any]:
		status = normalize_code(status)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "record_review",
			"status_supported": status in SUPPORTED_REVIEW_STATUSES,
			"reviewer_present": present(reviewer_id),
			"evidence_present": present(evidence_reference),
		})
		item = CryptoReview(review_id, tenant_id, reference_id, reviewer_id, status, evidence_reference)
		self.reviews[review_id] = item
		self._audit(tenant_id, "crypto_review_recorded", review_id)
		return item.to_dict()

	def register_crypto_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
	) -> dict[str, Any]:
		runtime = normalize_code(runtime)
		role = normalize_code(role)
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write",
			"policy_attached": True,
			"operation": "register_crypto_agent",
			"agent_runtime_supported": runtime in SUPPORTED_AGENT_RUNTIMES,
			"agent_role_supported": role in SUPPORTED_AGENT_ROLES,
		})
		item = CryptoAgent(agent_id, tenant_id, name, runtime, role, scope)
		self.agents[agent_id] = item
		self._audit(tenant_id, "crypto_agent_registered", agent_id)
		return item.to_dict()

	def validate_agent_action(
		self,
		tenant_id: str,
		privileged_scope: bool,
		human_approval_recorded: bool,
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "crypto_agent_action",
			"privileged_scope": privileged_scope,
			"human_approval_recorded": human_approval_recorded,
		})
		return {"tenant_id": tenant_id, "accepted": True, "privileged_scope": privileged_scope}

	def validate_batch(
		self,
		tenant_id: str,
		item_count: int,
		event_stream: str = "bytewax",
	) -> dict[str, Any]:
		self._enforce({
			"tenant_id": tenant_id,
			"tenant_context_present": bool(tenant_id),
			"operation": "crypto_batch",
			"event_stream": event_stream,
		})
		return {
			"tenant_id": tenant_id,
			"item_count": item_count,
			"processor": "bytewax",
			"stream": "apg.fintech.crypto.lifecycle",
			"accepted": True,
		}

	def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"asset_count": self._count(self.assets, tenant_id),
			"custody_account_count": self._count(self.accounts, tenant_id),
			"balance_count": self._count(self.balances, tenant_id),
			"order_count": self._count(self.orders, tenant_id),
			"open_order_count": sum(
				1 for item in self.orders.values()
				if item.tenant_id == tenant_id and item.status in {"requested", "approved"}
			),
			"trade_count": self._count(self.trades, tenant_id),
			"transfer_count": self._count(self.transfers, tenant_id),
			"blocked_screening_count": sum(
				1 for item in self.screenings.values()
				if item.tenant_id == tenant_id and item.status == "blocked"
			),
			"price_count": self._count(self.prices, tenant_id),
			"review_count": self._count(self.reviews, tenant_id),
			"agent_count": self._count(self.agents, tenant_id),
			"staking_wallet_count": len(self._staking),
			"audit_event_count": sum(1 for ev in self.audit_events if ev["tenant_id"] == tenant_id),
			"streaming": get_capability_contract(tenant_id)["streaming"],
		}

	# ------------------------------------------------------------------
	# Additional async methods
	# ------------------------------------------------------------------

	async def health_check(self) -> dict[str, Any]:
		"""Return cryptocurrency service health status."""
		return {
			"service": "cryptocurrency", "status": "healthy",
			"asset_count": len(self.assets), "staking_positions": sum(len(v) for v in self._staking.values()),
			"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def deposit_crypto(self, wallet_id: str, coin: str, amount: float, source_address: str) -> dict[str, Any]:
		"""Record an inbound crypto deposit to a wallet."""
		coin_sym = normalize_symbol(coin)
		wb = self._wallet_balances.setdefault(wallet_id, {})
		wb[coin_sym] = wb.get(coin_sym, 0.0) + amount
		tx_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "crypto_deposit_recorded", tx_id)
		return {
			"tx_id": tx_id, "wallet_id": wallet_id, "coin": coin_sym,
			"amount": amount, "source_address": source_address,
			"status": "confirmed", "deposited_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def price_alert_setup(self, coin: str, target_price: float, direction: str) -> dict[str, Any]:
		"""Set up a price alert for a coin (above/below target)."""
		assert direction in {"above", "below"}, "direction must be above|below"
		coin_sym = normalize_symbol(coin)
		current = _COIN_PRICES_USD.get(coin_sym, 0.0)
		triggered = (direction == "above" and current >= target_price) or (direction == "below" and current <= target_price)
		alert_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "price_alert_created", alert_id)
		return {
			"alert_id": alert_id, "coin": coin_sym, "target_price_usd": target_price,
			"current_price_usd": current, "direction": direction, "triggered": triggered,
			"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def limit_order(self, customer_id: str, coin: str, side: str, amount: float, limit_price: float) -> dict[str, Any]:
		"""Place a limit order for a crypto asset."""
		assert side in {"buy", "sell"}, "side must be buy|sell"
		coin_sym = normalize_symbol(coin)
		current = _COIN_PRICES_USD.get(coin_sym, 0.0)
		order_id = str(uuid.uuid4())
		executed = (side == "buy" and current <= limit_price) or (side == "sell" and current >= limit_price)
		self._audit(self.tenant_id, "limit_order_placed", order_id)
		return {
			"order_id": order_id, "customer_id": customer_id, "coin": coin_sym,
			"side": side, "amount": amount, "limit_price": limit_price,
			"current_price": current, "status": "filled" if executed else "pending",
			"placed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def dca_plan_setup(self, customer_id: str, coin: str, amount_per_period: float, frequency: str) -> dict[str, Any]:
		"""Set up a Dollar-Cost Averaging plan for regular crypto purchases."""
		assert frequency in {"daily", "weekly", "biweekly", "monthly"}, f"unsupported frequency: {frequency}"
		coin_sym = normalize_symbol(coin)
		plan_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "dca_plan_created", plan_id)
		return {
			"plan_id": plan_id, "customer_id": customer_id, "coin": coin_sym,
			"amount_per_period_usd": amount_per_period, "frequency": frequency,
			"status": "active", "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def nft_portfolio(self, customer_id: str) -> dict[str, Any]:
		"""Return NFT holdings summary for a customer."""
		wallets = self._customer_wallets.get(customer_id, [])
		self._audit(self.tenant_id, "nft_portfolio_queried", customer_id)
		return {
			"customer_id": customer_id, "wallet_count": len(wallets),
			"nft_count": 0, "estimated_value_usd": 0.0,
			"queried_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def crypto_loan(self, customer_id: str, collateral_coin: str, collateral_amount: float, borrow_coin: str, ltv: float = 0.5) -> dict[str, Any]:
		"""Take a crypto-backed loan using collateral."""
		c_sym = normalize_symbol(collateral_coin)
		b_sym = normalize_symbol(borrow_coin)
		c_price = _COIN_PRICES_USD.get(c_sym, 1.0)
		b_price = _COIN_PRICES_USD.get(b_sym, 1.0)
		collateral_usd = collateral_amount * c_price
		borrow_amount = (collateral_usd * ltv) / b_price
		loan_id = str(uuid.uuid4())
		self._audit(self.tenant_id, "crypto_loan_created", loan_id)
		return {
			"loan_id": loan_id, "customer_id": customer_id,
			"collateral_coin": c_sym, "collateral_amount": collateral_amount,
			"collateral_usd": round(collateral_usd, 4),
			"borrow_coin": b_sym, "borrow_amount": round(borrow_amount, 8),
			"ltv": ltv, "interest_rate_annual_pct": 8.5,
			"status": "active", "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	async def export_crypto_data(self, customer_id: str, fmt: str = "csv") -> dict[str, Any]:
		"""Export crypto transaction history for tax or compliance reporting."""
		assert fmt in {"csv", "json", "excel"}
		wallets = self._customer_wallets.get(customer_id, [])
		return {
			"customer_id": customer_id, "format": fmt, "wallet_count": len(wallets),
			"file_reference": f"crypto_{customer_id}_{fmt}",
			"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		}

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _tenant_asset_or_none(self, item_id: str, tenant_id: str) -> CryptoAsset | None:
		item = self.assets.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_account_or_none(self, item_id: str, tenant_id: str) -> CustodyAccount | None:
		item = self.accounts.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _tenant_order_or_none(self, item_id: str, tenant_id: str) -> CryptoOrder | None:
		item = self.orders.get(item_id)
		return item if item is not None and item.tenant_id == tenant_id else None

	def _audit(self, tenant_id: str, event_type: str, reference_id: str) -> None:
		self.audit_events.append({
			"tenant_id": tenant_id,
			"event_type": event_type,
			"reference_id": reference_id,
			"actor_id": self.actor_id,
			"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
		})

	def _count(self, items: dict[str, Any], tenant_id: str) -> int:
		return sum(1 for item in items.values() if item.tenant_id == tenant_id)

	def _enforce(self, context: dict[str, Any]) -> None:
		result = self.evaluate(context)
		if result["decision"] == "allow":
			return
		reasons = ", ".join(
			action.get("reason", "crypto_policy_denied")
			for action in result["actions"]
		)
		raise PermissionError(reasons or "crypto_policy_denied")


FintechCryptoService = CryptocurrencyService
CryptocurrencyServicesService = CryptocurrencyService
