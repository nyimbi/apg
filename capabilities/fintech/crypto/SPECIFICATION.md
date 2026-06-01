# Cryptocurrency Services Specification

## Purpose

Cryptocurrency Services gives APG applications a first-class digital asset
operating surface. It makes assets, custody, balances, orders, trades,
transfers, screening, prices, reviews, and AI agents composable while keeping
live exchange, custody, market-data, signing, and chain-provider integrations
behind adapters.

## Functional Scope

- Register crypto assets with symbol, supported asset type, blockchain network
  reference, optional token contract, precision, owner, and evidence.
- Open custody accounts with provider reference, supported custody model,
  policy reference, owner, and evidence.
- Record balances with account, asset, amount, valuation, currency, and
  evidence.
- Create orders with account, asset, side, order type, quantity, limit price
  when required, policy, requester, and evidence.
- Record trades with order, venue, execution price, quantity, fee, status, and
  settlement reference.
- Request transfers with account, asset, type, destination, amount, approval,
  evidence, and status.
- Record compliance screening with reference, screening type, status, evidence,
  and reviewer when the result is not clear.
- Record price snapshots with asset, source, price, currency, observation time,
  and evidence.
- Record reviews for crypto artifacts.
- Register provider-neutral AI agents with supported runtimes and roles.
- Publish UI routes, theme metadata, and Bytewax lifecycle metadata.

## Guardrails

- Every write requires tenant context and policy evidence.
- Assets require symbol, supported type, network reference, non-negative
  precision, owner, and evidence.
- Custody accounts require supported custody model, provider, policy, owner,
  and evidence.
- Balances require existing account, existing asset, non-negative amount,
  non-negative valuation, currency, and evidence.
- Orders require existing account, existing asset, supported side, supported
  type, positive quantity, limit price for limit orders, policy, requester, and
  evidence.
- Trades require existing order, venue, non-negative execution price, positive
  quantity, non-negative fee, supported status, and settlement reference.
- Transfers require existing account, existing asset, supported type,
  destination, positive amount, approval, evidence, and supported status.
- Screening requires reference, supported type, supported status, evidence, and
  reviewer for non-clear results.
- Prices require existing asset, supported source, non-negative price,
  currency, observed timestamp, and evidence.
- Reviews require supported status, reviewer, and evidence.
- Batch lifecycle events require Bytewax routing.
- Privileged AI-agent actions require human approval.

## Non-Goals

- No live exchange connectivity, custody-provider API, order routing,
  transaction signing, private-key custody, chain RPC access, market-data feed,
  or durable worker topology is embedded in this package.
- External systems remain behind APG adapter contracts.
