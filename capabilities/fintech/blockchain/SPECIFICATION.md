# Blockchain Services Specification

## Purpose

Blockchain Services gives APG applications a first-class chain operations
surface. It makes blockchain networks, custody references, smart contracts,
transactions, evidence anchors, oracle feeds, node health, reviews, and AI
agents composable without embedding vendor-specific chain clients or private-key
material in generated applications.

## Functional Scope

- Register networks by chain family, environment, chain ID, endpoint reference,
  owner, and evidence.
- Register wallets with network, wallet reference, custody model, key policy,
  owner, and evidence.
- Record smart contract deployments with network, contract type, artifact,
  owner, approval, and evidence.
- Record chain transactions with hash, type, asset reference, amount, signer,
  evidence, settlement status, and optional high-value approval.
- Anchor APG evidence by payload hash, reference ID, timestamp, and evidence.
- Register oracle feeds by type, source, owner, and evidence.
- Record node health with endpoint reference, status, block height, and
  evidence.
- Record reviews for blockchain artifacts.
- Register provider-neutral AI agents with supported runtimes and roles.
- Publish UI routes, theme metadata, and Bytewax lifecycle metadata.

## Guardrails

- Every write requires tenant context and policy evidence.
- Networks require supported type, supported environment, chain ID, RPC
  reference, owner, and evidence.
- Wallets require existing network, wallet reference, supported custody model,
  key policy, owner, and evidence.
- Smart contracts require existing network, supported contract type, artifact,
  owner, approval, and evidence.
- Transactions require existing network, hash, supported type, asset reference,
  non-negative amount, signer, evidence, supported settlement status, and
  approval when high value.
- Evidence anchors require existing network, payload hash, reference ID,
  timestamp, and evidence.
- Oracle feeds require existing network, supported feed type, source, owner, and
  evidence.
- Node health requires existing network, endpoint, supported status,
  non-negative block height, and evidence.
- Reviews require supported status, reviewer, and evidence.
- Batch lifecycle events require Bytewax routing.
- Privileged AI-agent actions require human approval.

## Non-Goals

- No live chain RPC calls, transaction signing, custody-provider integration,
  chain indexer integration, bridge operation, oracle vendor connection, or
  durable worker topology is embedded in this package.
- External systems remain behind APG adapter contracts.
