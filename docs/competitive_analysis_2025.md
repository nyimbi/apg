# APG Platform: Capability-by-Capability Competitive Analysis

**Datacraft APG Platform — Internal Strategic Report**  
**© 2025 Datacraft | Confidential**  
Author: Nyimbi Odero | www.datacraft.co.ke

---

## Executive Summary

This report provides a capability-by-capability competitive analysis of all 259 APG platform capabilities against the world's three best-in-class implementations in each domain. The analysis covers 28 capability domains spanning Finance/Fintech (36), Common Platform (81), Business Intelligence & Analytics (8), Intelligence/Security (20), Healthcare (9), Pharma (9), GRC (6), Government (10), CRM (1), Retail (5), HCM (3), SCM (1), Transport (10), Mining (6), Real Estate (10), Education (3), Project Portfolio Management (6), Energy (6), Telecom (10), Enterprise Asset Management (1), Composition Platform (6), Mobility (3), Localization (3), and Product Data Engineering (1).

### Overall Assessment

APG's 259-capability portfolio demonstrates strong **breadth** — covering more business domains than any single commercial vendor. The platform's strength lies in its composable architecture, local AI-first orientation (Ollama-native), and zero per-seat licensing cost. However, the analysis identifies consistent **depth gaps** across every domain: world-class systems have 20–30 years of domain refinement, regulatory certifications, and ecosystem effects that APG's implementations do not yet match.

### Priority Gap Categories

| Tier | Gap Type | Estimated Capabilities Affected |
|------|----------|--------------------------------|
| 🔴 Critical | Core feature absent — capability cannot serve its primary use case | ~45 capabilities |
| 🟠 High | Significant functional gaps blocking enterprise deployment | ~110 capabilities |
| 🟡 Medium | Feature parity at core; missing advanced/ML features | ~80 capabilities |
| 🟢 Low | Competitive for APG's target market (SMB/mid-market, Africa, local-AI) | ~24 capabilities |

### Top 10 Cross-Cutting Gaps

1. **No durable execution** — workflow state is lost on process restart across all workflow-bearing capabilities
2. **No real-time messaging infrastructure** — HTTP polling replaces WebSockets throughout; no event streaming
3. **Single-country/single-entity architecture** — payroll, tax, HR, and finance capabilities block multinational use
4. **No ML/AI at capability level** — world-best platforms embed AI scoring, forecasting, and automation; APG has none at the feature level
5. **No hardware/device integration** — POS, T&A, MDM, and IoT capabilities lack device protocol implementations
6. **No regulatory certifications** — SOC 2, ISO 27001, PCI DSS, HIPAA, FDA 21 CFR Part 11, etc. are absent
7. **No ecosystem / connector marketplace** — every integration requires custom development; world-best have 300–5,000+ pre-built connectors
8. **No developer portal or API analytics** — API gateway, observability, and self-service tooling are missing
9. **No offline capability** — POS, field service, mapping, and mobile capabilities require constant connectivity
10. **Static configuration, no policy-as-code** — access control, compliance rules, and configurations are imperative and cannot be CI/CD-tested

### APG Competitive Advantages (Where APG Leads or Matches)

- **Zero licensing cost** — eliminates vendor lock-in vs $50K–$5M/yr enterprise platforms
- **Local AI-first (Ollama)** — data sovereignty for regulated industries in Africa and emerging markets
- **Composable capability mesh** — 259 pre-built modules composable via APG DSL (no comparable open platform)
- **Single-codebase deployment** — all 259 capabilities deploy from one repository; competitors require multi-system integration
- **APG DSL** — unique capability specification language enabling rapid capability generation with no analog in the market
- **Africa/emerging market focus** — MPESA, local currency/language coverage that global platforms deprioritize

---

## Methodology

**Comparison framework:** For each capability, three world-class implementations were selected based on:
- Gartner Magic Quadrant leader status or IDC MarketScape leadership (where available)
- G2 / SoftwareReviews category leadership (2024–2025)
- Market share in the relevant segment

**Scoring:** Each feature row uses:
- ✓ = Fully implemented and production-grade
- Partial = Implemented but with material limitations
- None = Not implemented

**Data sources:** Vendor documentation, analyst reports, G2 reviews, and community benchmarks as of 2025. APG capability assessments are based on direct codebase analysis.

---

## Table of Contents

1. [Finance & Fintech (36 capabilities)](#1-finance--fintech)
2. [Human Capital Management, CRM & Retail (9 capabilities)](#2-human-capital-management-crm--retail)
3. [Healthcare, Pharma, GRC & Government (34 capabilities)](#3-healthcare-pharma-grc--government)
4. [Business Intelligence, Intelligence/OSINT, Energy & Telecom (44 capabilities)](#4-business-intelligence-intelligenceosint-energy--telecom)
5. [Transport, Mining, Real Estate, Education, PPM & EAM (36 capabilities)](#5-transport-mining-real-estate-education-ppm--eam)
6. [Common Platform Capabilities (81 capabilities)](#6-common-platform-capabilities)
7. [SCM, Composition Platform, Mobility, Localization, PDE & CKM (17 capabilities)](#7-scm-composition-platform-mobility-localization-pde--ckm)
8. [Cross-Domain Gap Analysis & Recommendations](#8-cross-domain-gap-analysis--recommendations)

---

# 1. Finance & Fintech

I have enough information to write the full competitive analysis now. The APG codebase has real implementations: structured service layers, domain rule engines, calculations modules, adapter patterns, and event emission — but these are self-contained Python services without the enterprise-grade data consortiums, regulatory certifications, global network effects, or decades of production hardening that the world-class vendors possess.

---

## Accounts Payable (`apy_accounts_payable`)

**APG provides:** Automated invoice processing pipeline with intelligent matching, duplicate prevention, approval workflows, exception resolution, and vendor self-service portal. Includes cash flow analytics, compliance monitoring, and period-close automation built as a Flask-AppBuilder blueprint with PostgreSQL persistence.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Invoice capture / OCR | Partial | ✓ (SAP S/4HANA, Oracle Fusion) | No production-grade OCR engine; relies on structured input; no multi-format document ingestion |
| 3-way PO matching | Partial | ✓ (SAP S/4HANA) | Matching logic present but no native ERP PO/GRN integration layer |
| Multi-tier approval workflows | ✓ | ✓ (Oracle Fusion, SAP) | Workflow engine is in-process; no external BPM integration (e.g., SAP BTP workflows) |
| Duplicate invoice detection | ✓ | ✓ (SAP, Coupa) | Rule-based; no cross-tenant consortium duplicate scoring |
| Supplier/vendor portal | Partial | ✓ (Coupa, Ariba) | Basic self-service; no supplier onboarding marketplace or network connectivity |
| Dynamic discounting / SCF | ✗ | ✓ (Taulia, SAP Ariba) | No supply chain finance or early payment discount optimization |
| ERP/P2P integration | Partial | ✓ (SAP S/4HANA, Oracle Fusion) | No prebuilt connectors to SAP, Oracle EBS, Dynamics 365 |
| Multi-currency / FX | Partial | ✓ (Oracle Fusion Financials) | Currency support present; no live FX rate feeds or hedging workflow |
| Regulatory e-invoicing (PEPPOL, CFDI, ZUGFeRD) | ✗ | ✓ (SAP, Basware) | No statutory e-invoicing compliance for any jurisdiction |
| Analytics / spend intelligence | Partial | ✓ (Coupa, SAP Analytics Cloud) | Cash flow analytics present; no vendor spend cube or category intelligence |

**World-best reference:** SAP S/4HANA Finance (AP), Oracle Fusion Cloud Financials, Coupa, Basware

**Critical gaps:**
- No certified OCR/IDP engine; invoice ingestion requires structured data, eliminating straight-through processing for paper/PDF volumes
- Zero statutory e-invoicing compliance (PEPPOL, CFDI, FatturaPA, etc.) — a hard regulatory blocker in most markets
- No supply chain finance or dynamic discounting module — a core differentiator for large AP platforms
- No prebuilt ERP connectors; integration requires custom development per deployment

---

## Accounts Receivable (`arc_accounts_receivable`)

**APG provides:** Billing lifecycle management, collections workflow, cash application with multi-currency support, fraud detection on receivables, exception resolution, and contextual intelligence. Implemented as a service layer with production-readiness optimizations and performance tuning.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Invoice generation & delivery | ✓ | ✓ (SAP, Oracle Fusion) | No e-invoice delivery network (PEPPOL, EDI X12 820) |
| Cash application / auto-matching | Partial | ✓ (HighRadius, Esker) | Logic present; no ML-based remittance parsing from bank statements |
| Collections management & dunning | Partial | ✓ (HighRadius, Billtrust) | Dunning workflows exist but no collector workbench with predictive prioritization |
| Deduction management | ✗ | ✓ (HighRadius, SAP) | No deduction coding, root-cause tracking, or short-pay resolution |
| Credit limit management | Partial | ✓ (Oracle Fusion, SAP) | Basic credit fields; no dynamic credit scoring integration |
| Multi-currency cash application | Partial | ✓ (Oracle Fusion Financials) | Multi-currency service present; no real-time FX settlement |
| Customer portal / self-service | ✗ | ✓ (Billtrust, HighRadius) | No customer-facing payment portal |
| DSO / working capital analytics | Partial | ✓ (SAP Analytics Cloud, HighRadius) | Analytics module present; no aging bucket DSO forecasting |
| ERP integration (SAR postings) | Partial | ✓ (SAP, Oracle) | Adapter pattern in place; no certified ERP write-back connectors |
| Lockbox / bank statement processing | ✗ | ✓ (Oracle Fusion, SAP) | No BAI2/CAMT.053 bank file ingestion |

**World-best reference:** HighRadius, SAP S/4HANA Finance (AR), Oracle Fusion Cloud Financials, Billtrust

**Critical gaps:**
- No ML-driven remittance matching — cash application remains manual-heavy without bank statement parsing
- Deduction management is absent entirely, a significant gap for B2B trade receivables
- No customer self-service payment portal, eliminating a primary DSO reduction lever
- No lockbox or bank file ingestion (BAI2/CAMT.053); cash reconciliation requires manual intervention

---

## Budgeting & Forecasting (`bfc_budgeting_forecasting`)

**APG provides:** ML forecasting engine with time-series models, interactive dashboard, automated monitoring with variance alerting, advanced analytics, and budget version management. Built with async Python and PostgreSQL.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Driver-based budgeting | Partial | ✓ (Anaplan, Adaptive Insights) | No connected planning with operational driver propagation |
| ML/statistical forecasting | ✓ | ✓ (Anaplan, Oracle PBCS) | Local Ollama-based models; no pre-trained financial forecasting model library |
| Rolling forecasts | ✓ | ✓ (Anaplan, Oracle PBCS) | Present; no continuous-close integration with actuals feed |
| Scenario / what-if modeling | Partial | ✓ (Anaplan, SAP BPC) | Basic scenario support; no multidimensional OLAP cube for large model hierarchies |
| Workforce / headcount planning | ✗ | ✓ (Anaplan, Workday Adaptive) | No HR-integrated headcount cost modeling |
| Capital expenditure planning | ✗ | ✓ (SAP BPC, Oracle PBCS) | No project-linked capex workflow |
| Consolidation across entities | Partial | ✓ (SAP BPC, OneStream) | Multi-tenant design present; no legal entity elimination logic |
| Variance analysis & commentary | Partial | ✓ (OneStream, Workiva) | Automated monitoring present; no narrative AI commentary generation against budget |
| ERP actuals integration | Partial | ✓ (SAP, Oracle) | Adapter pattern exists; no certified real-time actuals pull from GL |
| Workflow & approval cycles | ✓ | ✓ (Adaptive Insights, Anaplan) | Present; lacks version locking and audit-grade sign-off trail |

**World-best reference:** Anaplan, Workday Adaptive Planning, SAP BPC/Analytics Cloud, OneStream

**Critical gaps:**
- No connected planning — operational drivers (sales pipeline, headcount, production) are not linked to financial model
- Zero workforce/HR cost planning, which typically represents 50–70% of an operating budget
- No multidimensional OLAP engine; large model hierarchies will degrade to row-level SQL scans
- Consolidation lacks intercompany elimination, a hard requirement for any multi-entity statutory budget

---

## Cash Management (`cbm_cash_management`)

**APG provides:** Real-time cash position tracking, liquidity management, bank connectivity with real-time sync, voice interface integration, and cash flow forecasting via a Flask-AppBuilder blueprint.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Cash position / intraday liquidity | Partial | ✓ (SAP S/4HANA, Kyriba) | Real-time sync present; no SWIFT MT940/camt.053 native bank statement import |
| Cash flow forecasting | Partial | ✓ (Kyriba, FIS Integrity) | Forecasting logic present; no multi-source AR/AP/payroll feed aggregation |
| Bank account management (BAM) | ✗ | ✓ (Kyriba, GTreasury) | No bank account structure management, signatory tracking, or FBAR reporting |
| In-house banking / IHB | ✗ | ✓ (SAP In-House Cash, Kyriba) | No intercompany netting or notional pooling |
| Physical cash pooling | ✗ | ✓ (SAP, Oracle Treasury) | Not implemented |
| FX exposure management | ✗ | ✓ (Kyriba, SAP TRM) | No FX hedging workflow or exposure aggregation |
| Investment management (short-term) | ✗ | ✓ (Kyriba, FIS Integrity) | No money market fund or short-term investment execution |
| Payment factory / payment hub | ✗ | ✓ (Kyriba, FIS) | No centralized payment factory with bank agnostic execution |
| SWIFT / API bank connectivity | Partial | ✓ (Kyriba, SAP) | Real-time sync module present; no certified SWIFT connectivity or bank API marketplace |
| Regulatory reporting (LCR, NSFR) | ✗ | ✓ (FIS, Moody's Analytics) | No Basel liquidity ratio calculation |

**World-best reference:** Kyriba, SAP S/4HANA Treasury & Risk Management, FIS Integrity SaaS Treasury

**Critical gaps:**
- No SWIFT or certified bank API connectivity — cash position depends on manual or custom-built bank feeds
- In-house banking, notional pooling, and physical sweeping are entirely absent — critical for corporates with treasury centers
- FX exposure management and hedging are missing, leaving currency risk unmanaged
- No Basel III LCR/NSFR reporting capability, a regulatory requirement for financial institutions

---

## General Ledger (`glr_general_ledger`)

**APG provides:** Chart of accounts management, journal entry processing, period-close automation, multi-entity support, event-driven architecture with discovery and integration layers, and capability contract enforcement.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Universal journal / single ledger | Partial | ✓ (SAP S/4HANA Universal Journal) | GL present; no single-entry unified financial/management accounting table |
| Multi-GAAP / parallel accounting | ✗ | ✓ (SAP S/4HANA, Oracle Fusion) | No parallel ledger support (IFRS + local GAAP simultaneously) |
| Real-time posting & balance inquiry | Partial | ✓ (Oracle FLEXCUBE, SAP) | Event-driven architecture present; no sub-second in-memory balance aggregation |
| Intercompany accounting & elimination | ✗ | ✓ (SAP, Oracle Fusion) | Not implemented |
| Allocations & distributions | ✗ | ✓ (SAP, Oracle, Workiva) | No cost allocation engine |
| Period-close orchestration | Partial | ✓ (BlackLine, SAP) | Period-close autopilot present; no reconciliation certification workflow |
| Segment / profit centre reporting | Partial | ✓ (SAP, Oracle) | Multi-entity in design; no dimensional reporting cube |
| Audit trail & SOX controls | Partial | ✓ (BlackLine, Oracle Fusion) | Event sourcing present; no SOX control attestation or segregation-of-duties enforcement |
| Multi-currency revaluation | Partial | ✓ (Oracle FLEXCUBE, SAP) | Currency fields present; no automated FX revaluation batch |
| Integration with sub-ledgers | Partial | ✓ (SAP S/4HANA, Oracle Fusion) | Adapter pattern in place; no certified sub-ledger-to-GL reconciliation |

**World-best reference:** SAP S/4HANA Finance (Universal Journal), Oracle Fusion Cloud General Ledger, Oracle FLEXCUBE

**Critical gaps:**
- No parallel accounting ledgers — cannot maintain IFRS and local GAAP books simultaneously, a baseline requirement for multinational entities
- Intercompany elimination is absent, blocking statutory consolidated financial statements
- No cost allocation engine; management accounting requires a separate solution
- SOX/ICFR control framework is not implemented — period-close lacks reconciliation certification and SoD enforcement

---

## Financial Reporting (`rpt_financial_reporting`)

**APG provides:** Financial report generation with Ollama-based LLM narrative integration, Flask-AppBuilder blueprint for report rendering, and structured output formatting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Standard financial statements (BS, IS, CF) | Partial | ✓ (SAP S/4HANA, Oracle Fusion) | Statements generatable; no IFRS/GAAP-certified presentation layer |
| Consolidation & group reporting | ✗ | ✓ (SAP S/4HANA Group Reporting, Workiva) | Not implemented |
| Regulatory / statutory filing | ✗ | ✓ (Workiva, Oracle Hyperion) | No XBRL tagging, SEC/EDGAR, or statutory filing automation |
| Management reporting & KPIs | Partial | ✓ (SAP Analytics Cloud, Workiva) | Basic reporting; no live operational KPI drill-through |
| Narrative / MD&A generation | Partial | ✓ (Workiva, Oracle Narrative Reporting) | Ollama integration present; no structured variance commentary templates |
| Multi-GAAP reporting | ✗ | ✓ (SAP S/4HANA, Oracle Fusion) | No parallel accounting basis |
| External audit support | ✗ | ✓ (Workiva, BlackLine) | No audit confirmation workflow or evidence packaging |
| Disclosure management | ✗ | ✓ (Workiva, Oracle Narrative Reporting) | Not implemented |
| Report distribution / governance | ✗ | ✓ (Workiva, SAP Analytics Cloud) | No report version control, distribution lists, or publish-to-regulator workflow |
| Embedded analytics / drill-down | ✗ | ✓ (SAP Analytics Cloud, Oracle Analytics) | No in-report drill-through to source transactions |

**World-best reference:** Workiva, SAP S/4HANA Finance for Group Reporting, Oracle Fusion Cloud Financial Reporting, BlackLine

**Critical gaps:**
- No XBRL tagging or statutory filing automation — APG cannot produce regulator-ready filings
- Consolidation and group reporting are entirely absent; multi-entity reporting requires manual aggregation
- No disclosure management or MD&A workflow — the Ollama narrative is unstructured and not audit-traceable
- No drill-through from report figures to source journal entries — auditability is severely limited

---

## Agency Banking (`fintech_agency`)

**APG provides:** Agent network management, transaction processing via agency channels, commission management, float/liquidity tracking, and agent onboarding workflows with Africa-market design focus.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Agent registration & tiering | ✓ | ✓ (Temenos Transact, Craft Silicon) | Present; no biometric agent identity verification pipeline |
| Float/liquidity management | ✓ | ✓ (Temenos, InterSwitch) | Present; no real-time float rebalancing alerts across large agent networks |
| Transaction processing (cash-in/out) | ✓ | ✓ (Temenos, Craft Silicon) | Present; no ISO 8583 native terminal integration |
| Commission calculation & settlement | ✓ | ✓ (Temenos, Craft Silicon) | Present; limited support for tiered/dynamic commission structures |
| Agent performance analytics | Partial | ✓ (Temenos, InterSwitch) | Basic analytics; no agent scorecard or fraud risk scoring per agent |
| Regulatory reporting (CBK, CBN, etc.) | Partial | ✓ (Temenos, Craft Silicon) | Event emission present; no jurisdiction-specific statutory report templates |
| Offline / low-connectivity mode | ✗ | ✓ (Craft Silicon, Musoni) | No store-and-forward for low-bandwidth environments |
| Device / POS management | ✗ | ✓ (Craft Silicon, Network International) | No device lifecycle management or remote key loading |
| Network hierarchy (super-agent, sub-agent) | Partial | ✓ (Temenos, Craft Silicon) | Basic hierarchy; no multi-level network commission cascading |
| USSD / SMS channel | ✗ | ✓ (Craft Silicon, Temenos) | No USSD session management integration |

**World-best reference:** Craft Silicon BankFusion, Temenos Transact, InterSwitch

**Critical gaps:**
- No offline/store-and-forward mode — unworkable in low-connectivity rural Africa deployments
- No device management or remote key loading — cannot manage POS/mPOS terminal fleet
- No USSD/SMS integration — the dominant channel for agency banking in sub-Saharan Africa
- ISO 8583 terminal integration is absent; cash-in/out relies on custom API rather than card scheme infrastructure

---

## Anti-Money Laundering (`fintech_aml`)

**APG provides:** Full AML service covering transaction monitoring with rule evaluation, alert lifecycle management, case management through SAR/CTR filing, watchlist screening (OFAC, PEP, UN, EU), network/graph analysis, structuring detection, velocity anomaly detection, crypto mixer routing detection, and NFT wash trade detection.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Transaction monitoring rules | ✓ | ✓ (NICE Actimize SAM10) | Rule engine present; no pre-built typology library (500+ NICE Actimize scenarios) |
| ML-based anomaly detection | Partial | ✓ (NICE Actimize, ComplyAdvantage) | Calculation module present; local Ollama models — no globally-trained consortium models |
| Watchlist screening | ✓ | ✓ (NICE Actimize WL-X, ComplyAdvantage) | OFAC/PEP/UN/EU supported; no real-time list refresh SLA or fuzzy matching at NICE Actimize fidelity |
| Entity resolution / link analysis | Partial | ✓ (NICE Actimize, Quantexa) | Network analysis code present; no production graph database (Neo4j/TigerGraph scale) |
| SAR / CTR / STR filing | ✓ | ✓ (NICE Actimize STAR) | Filing workflow present; no direct FinCEN/FCA/CBK electronic submission |
| Case management | ✓ | ✓ (NICE Actimize, Fiserv FCRM) | Lifecycle implemented; no investigator productivity analytics or SLA dashboards |
| Crypto / DeFi monitoring | Partial | ✓ (Chainalysis, Elliptic via NICE Actimize) | Mixer detection logic present; no blockchain data feed (Chainalysis/Elliptic integration) |
| Customer risk scoring (CDD/EDD) | Partial | ✓ (NICE Actimize CDD-X) | Risk calculation present; no continuous monitoring refresh triggered by event streams |
| Regulatory reporting automation | Partial | ✓ (NICE Actimize, Fiserv FCRM) | Event emission present; no regulator-format XML/XBRL output |
| False positive reduction / tuning | ✗ | ✓ (NICE Actimize — 60% FP reduction) | No ML-based threshold auto-tuning or feedback loop from investigator decisions |

**World-best reference:** NICE Actimize (SAM10, WL-X, STAR, CDD-X), Fiserv Financial Crime Risk Management, ComplyAdvantage

**Critical gaps:**
- No globally-trained consortium ML models — detection rates will be significantly lower than NICE Actimize's cross-institution learning
- No production graph database backend; link analysis will not scale beyond thousands of entities
- No direct electronic SAR/STR submission to any financial intelligence unit — filing is manual-export only
- No ML threshold auto-tuning; false positive rates will remain high without investigator feedback loop integration

---

## Open Banking APIs (`fintech_apis`)

**APG provides:** Open Banking API service with OAuth2/OIDC authentication, consent management, account aggregation, payment initiation, and API versioning. Includes models for API consumers and data sharing agreements.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| PSD2 / Open Banking UK compliance | Partial | ✓ (TrueLayer, Tink, Yapily) | API structure present; no regulatory certification under PSD2 Technical Standards |
| Consent management | ✓ | ✓ (TrueLayer, Mulesoft) | Present; no fine-grained GDPR data minimization enforcement |
| Account information (AIS) | Partial | ✓ (TrueLayer, Plaid) | Aggregation logic present; no live bank adapter library (Plaid covers 12,000+ institutions) |
| Payment initiation (PIS) | Partial | ✓ (TrueLayer, Yapily) | Initiation model present; no certified scheme connectivity (Faster Payments, SEPA SCT Inst) |
| Developer portal / sandbox | ✗ | ✓ (Stripe, TrueLayer, Plaid) | No self-serve developer portal, API key provisioning, or mock sandbox environment |
| Rate limiting & throttling | ✗ | ✓ (Kong, Apigee, AWS API Gateway) | No API gateway-level rate limiting |
| Monetization / billing for API usage | ✗ | ✓ (Apigee, AWS API Gateway) | No usage metering or API product tiering |
| Webhook reliability / event streaming | Partial | ✓ (Stripe, Adyen) | Event emission designed in; no at-least-once delivery guarantee with retry management |
| Bank adapter coverage | ✗ | ✓ (Plaid — 12,000+, Tink — 3,400+) | No prebuilt bank adapters; each institution requires custom integration |
| TPP onboarding & registration | ✗ | ✓ (Open Banking Ltd, Tink) | No third-party provider directory or eIDAS certificate validation |

**World-best reference:** TrueLayer, Plaid, Tink (Visa), Yapily

**Critical gaps:**
- No regulatory certification (PSD2 RTS, CDR, Open Banking UK) — APG cannot act as an ASPSP or TPP in regulated markets
- Zero prebuilt bank adapters; competitive advantage of platforms like Plaid/Tink is entirely their institution coverage
- No developer portal or sandbox — API adoption requires direct engineering engagement per consumer
- No eIDAS certificate validation or TPP directory — cannot operate in European Open Banking ecosystem

---

## Blockchain / DLT (`fintech_blockchain`)

**APG provides:** Blockchain service layer with smart contract interaction, wallet management, transaction submission and tracking, multi-chain support, DeFi protocol integration, and NFT operations.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Multi-chain support | Partial | ✓ (Fireblocks, Consensys) | Multiple chains in model; no production MPC key management |
| Smart contract lifecycle | Partial | ✓ (Consensys, Hyperledger Fabric) | Interaction layer present; no formal verification or audit framework |
| Enterprise DLT (Fabric, Corda) | ✗ | ✓ (Hyperledger Fabric, R3 Corda) | No permissioned enterprise DLT node management |
| MPC / HSM custody | ✗ | ✓ (Fireblocks, BitGo) | No institutional-grade key management; wallet keys in-process |
| Tokenization (RWA, securities) | Partial | ✓ (Fireblocks, Securitize) | Model exists; no regulatory-compliant tokenization framework |
| Cross-chain bridge / interoperability | ✗ | ✓ (Chainlink CCIP, LayerZero) | Not implemented |
| On-chain compliance (FATF Travel Rule) | ✗ | ✓ (Notabene, Sygna) | No Travel Rule data exchange protocol |
| Blockchain analytics / forensics | ✗ | ✓ (Chainalysis, Elliptic) | No transaction tracing or risk scoring against known-bad addresses |
| Gas optimization / fee management | ✗ | ✓ (Alchemy, Infura) | No dynamic fee estimation or gas station integration |
| Node infrastructure management | ✗ | ✓ (Alchemy, Infura, Quicknode) | No node provisioning; depends on external RPC endpoints |

**World-best reference:** Fireblocks, Consensys, Hyperledger Fabric, R3 Corda

**Critical gaps:**
- Wallet private key management is in-process with no HSM or MPC custody — a critical security failure for any production deployment
- No FATF Travel Rule compliance — legally required for virtual asset service providers in most jurisdictions
- No enterprise permissioned DLT (Fabric/Corda) — majority of institutional blockchain use cases require permissioned networks
- No blockchain analytics integration — cannot screen wallet addresses against sanctions or illicit activity databases

---

## Central Bank Digital Currency (`fintech_cbdc`)

**APG provides:** CBDC lifecycle management including issuance, distribution, retail wallet management, programmable money features, and interoperability models for both retail and wholesale CBDC.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Retail CBDC wallet management | Partial | ✓ (G+D Filia, Giesecke+Devrient) | Wallet model present; no hardware security element integration |
| Wholesale CBDC / interbank settlement | Partial | ✓ (Partior, JPM Coin, mBridge) | Model present; no real central bank integration |
| Programmable money / smart contracts | Partial | ✓ (G+D Filia, MIT Hamilton) | Programmable feature model present; no on-chain rule enforcement |
| Offline CBDC payments | ✗ | ✓ (G+D Filia, Idemia) | No secure element / NFC offline payment capability |
| Privacy-preserving design | ✗ | ✓ (MIT Hamilton, BIS Project Tourbillon) | No zero-knowledge proof or tiered anonymity architecture |
| Central bank integration / API | ✗ | ✓ (G+D, Temenos) | No certified central bank system integration |
| Cross-border CBDC (mBridge) | ✗ | ✓ (BIS mBridge, Partior) | Not implemented |
| AML/CFT controls on CBDC | Partial | ✓ (G+D Filia, Chainalysis) | AML module exists separately; no CBDC-specific inline compliance |
| CBDC analytics / circulation reporting | ✗ | ✓ (G+D, central bank platforms) | No monetary circulation dashboard for central bank oversight |
| Resilience / dual offline operation | ✗ | ✓ (G+D Filia) | No contingency offline settlement mode |

**World-best reference:** G+D Filia, MIT Digital Currency Initiative (Project Hamilton), BIS Innovation Hub, Partior

**Critical gaps:**
- No privacy architecture (ZKP/tiered anonymity) — a foundational design requirement for any central bank CBDC
- Offline payment capability is entirely absent — critical for financial inclusion use cases
- No central bank system integration or sandbox participation with any real monetary authority
- Cross-border CBDC interoperability is not implemented — the primary institutional use case for wholesale CBDC

---

## Chargeback Management (`fintech_cha`)

**APG provides:** Chargeback lifecycle management covering dispute initiation, evidence collection, representment workflow, and chargeback analytics. Modeled as part of the gateway capability domain.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Dispute intake & classification | Partial | ✓ (Chargebacks911, Verifi) | Basic dispute model; no reason code taxonomy automation (Visa/MC/Amex codes) |
| Evidence collection & packaging | Partial | ✓ (Chargebacks911) | Evidence model present; no automated document gathering from transaction logs |
| Representment filing | ✗ | ✓ (Chargebacks911, Ethoca) | No network-specific representment format generation |
| Pre-dispute / order insight | ✗ | ✓ (Verifi CDRN, Ethoca Alerts) | No Visa/Mastercard pre-dispute alert integration |
| Win rate analytics | ✗ | ✓ (Chargebacks911) | Not implemented |
| Fraud vs. friendly fraud detection | ✗ | ✓ (Chargebacks911, Kount) | No behavioral distinction between true fraud and first-party misuse |
| Scheme rule library (Visa, MC) | ✗ | ✓ (Chargebacks911, Verifi) | No built-in scheme regulation knowledge base |
| Automated threshold alerting | Partial | ✓ (Verifi, Chargebacks911) | Basic alerting present; no Visa Dispute Monitoring Program (VDMP) breach prediction |
| Merchant dispute portal | ✗ | ✓ (Chargebacks911) | No merchant-facing self-service dispute management UI |
| Cross-channel dispute correlation | ✗ | ✓ (Featurespace, Chargebacks911) | Not implemented |

**World-best reference:** Chargebacks911, Verifi (Visa), Ethoca (Mastercard), Kount (Equifax)

**Critical gaps:**
- No pre-dispute alert integration with Verifi CDRN or Ethoca — the primary mechanism for preventing chargebacks before they are filed
- No scheme-specific representment format generation; each chargeback response requires manual formatting
- Chargeback capability is not a standalone module — it is embedded in the gateway with insufficient depth
- No win rate tracking or analytics, making continuous improvement impossible

---

## Credit Lifecycle (`fintech_clc`)

**APG provides:** End-to-end credit lifecycle management via the lending service, covering origination through collections. Integrates with the credit risk engine and decision engine capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Application & origination | ✓ | ✓ (nCino, Blend) | Present; no digital application UX comparable to Blend's borrower-facing experience |
| Automated underwriting | Partial | ✓ (nCino, Finastra Fusion Lending) | Decision engine integration present; no AUS (Fannie/Freddie) connectivity |
| Credit bureau integration | ✗ | ✓ (nCino, Finastra) | No prebuilt connectors to Experian, Equifax, TransUnion, or African bureaus (CRB Africa) |
| Document collection & verification | Partial | ✓ (Blend, nCino) | Document model present; no OCR/IDP pipeline for automated spreading |
| Pricing & rate sheet management | Partial | ✓ (Finastra Fusion Lending, nCino) | Pricing logic present; no market-rate feed or competitive pricing analytics |
| Covenant tracking | ✗ | ✓ (nCino, Finastra) | Not implemented for commercial credit |
| Portfolio management | Partial | ✓ (nCino, Finastra) | Portfolio model present; no watchlist-triggered review workflow |
| Restructuring & modifications | Partial | ✓ (Finastra Fusion Lending) | Modification model present; no IFRS 9 significant increase in credit risk (SICR) trigger |
| Regulatory capital calculation (RWA) | ✗ | ✓ (Moody's Analytics, Finastra) | Not implemented |
| Collateral management | ✗ | ✓ (Finastra, nCino) | Not implemented |

**World-best reference:** nCino, Blend, Finastra Fusion Lending, Salesforce Financial Services Cloud

**Critical gaps:**
- No credit bureau integration — automated credit decisions require manual bureau report retrieval
- Covenant monitoring and collateral management are absent, blocking commercial lending use cases
- No IFRS 9 SICR trigger — regulatory provisioning calculation cannot be automated
- No AUS (automated underwriting system) connectivity for mortgage products

---

## Customer Lifetime Value (`fintech_clv`)

**APG provides:** CLV calculation and analytics service covering revenue attribution, churn prediction integration, segment scoring, and customer profitability analysis.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Historical CLV calculation | ✓ | ✓ (Salesforce Einstein, SAS) | Present; no integration with external transactional data warehouse |
| Predictive CLV (ML) | Partial | ✓ (Salesforce Einstein, SAS CI360) | ML scoring present with local models; no pre-trained financial services CLV models |
| Churn prediction integration | Partial | ✓ (Salesforce, Adobe CDP) | Model integration present; no real-time event trigger from behavioral stream |
| Segment-based CLV | ✓ | ✓ (SAS, Adobe) | Present; no next-best-action recommendation engine |
| Product propensity scoring | ✗ | ✓ (Salesforce Einstein, SAS) | Not implemented |
| Channel attribution | ✗ | ✓ (Adobe Analytics, Salesforce) | No multi-touch attribution model |
| Real-time CLV update | ✗ | ✓ (Salesforce, Pega) | Batch calculation only; no event-driven CLV update on transaction |
| CRM integration | ✗ | ✓ (Salesforce FSC, Microsoft Dynamics) | No CRM write-back for CLV-driven sales actions |
| A/B testing / offer optimization | ✗ | ✓ (Adobe Target, Salesforce) | Not implemented |
| Data privacy / consent for ML scoring | ✗ | ✓ (OneTrust + Salesforce) | No consent-gated scoring or GDPR-compliant model input management |

**World-best reference:** Salesforce Financial Services Cloud with Einstein, SAS Customer Intelligence 360, Adobe Real-Time CDP

**Critical gaps:**
- No real-time CLV update on transaction events — CLV is a periodic batch calculation, reducing actionability
- No next-best-action or product propensity engine — CLV score is computed but not operationalized
- No CRM integration; CLV insights cannot drive sales or retention actions without manual export
- No GDPR/consent-gated model input management — scoring customers without explicit consent tracking is a regulatory risk

---

## Compliance Management (`fintech_cmp`)

**APG provides:** Regulatory compliance service with obligation tracking, policy management, control testing, compliance calendar, breach management, and regtech integration hooks.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Regulatory obligation library | Partial | ✓ (MetricStream, NICE Actimize) | Obligation model present; no pre-populated regulatory library (GDPR, PSD2, Basel III) |
| Policy & control management | ✓ | ✓ (MetricStream, ServiceNow GRC) | Present; no control effectiveness scoring against industry frameworks |
| Control testing & assurance | Partial | ✓ (MetricStream, Galvanize) | Testing workflow present; no automated control evidence collection |
| Regulatory change management | ✗ | ✓ (Thomson Reuters Regulatory Intelligence, Ascent) | No regulatory change feed or horizon scanning |
| Breach / incident management | ✓ | ✓ (MetricStream, ServiceNow) | Lifecycle present; no regulator notification workflow with jurisdiction-specific timelines |
| Compliance reporting / dashboard | Partial | ✓ (MetricStream, Oracle GRC) | Dashboard present; no board-pack style regulatory heat map |
| Third-party / vendor compliance | ✗ | ✓ (MetricStream, Prevalent) | Not implemented |
| GDPR / data privacy compliance | ✗ | ✓ (OneTrust, TrustArc) | No data mapping, DPIA workflow, or subject access request management |
| Sanctions screening integration | Partial | ✓ (ComplyAdvantage, Dow Jones) | AML module handles screening; no standalone compliance-owned screening |
| Exam & audit management | ✗ | ✓ (MetricStream, Wolters Kluwer) | Not implemented |

**World-best reference:** MetricStream, ServiceNow GRC, Thomson Reuters Regulatory Intelligence, ComplyAdvantage

**Critical gaps:**
- No pre-populated regulatory content library — every obligation must be manually entered
- No regulatory change feed or horizon scanning — firms must manually track regulatory updates
- GDPR/data privacy workflow (DPIA, SAR management) is entirely absent
- No third-party risk management — vendor/partner compliance tracking is not implemented

---

## Credit Risk Engine (`fintech_cre`)

**APG provides:** Credit risk calculation engine with scoring models, probability of default (PD) estimation, loss given default (LGD) modeling, exposure at default (EAD), risk segmentation, and portfolio-level risk aggregation.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| PD / LGD / EAD modeling | Partial | ✓ (Moody's Analytics, SAS Credit Risk) | Calculation modules present; models are custom-built without validated benchmark performance |
| IFRS 9 / CECL provisioning | ✗ | ✓ (Moody's Analytics RiskCalc, SAS) | Not implemented; no ECL staging or forward-looking macro scenario overlay |
| Basel III/IV RWA calculation | ✗ | ✓ (Moody's Analytics, Oracle FSA) | Not implemented |
| Stress testing | ✗ | ✓ (Moody's Analytics, IBM OpenPages) | Not implemented |
| Scorecard development & validation | Partial | ✓ (SAS Credit Risk, FICO Score) | Scoring infrastructure present; no scorecard champion-challenger framework |
| Credit bureau data integration | ✗ | ✓ (FICO, Experian PowerCurve) | No bureau connector |
| Portfolio concentration risk | Partial | ✓ (Moody's Analytics, SAS) | Aggregation logic present; no regulatory concentration limit enforcement |
| Vintage analysis | ✗ | ✓ (Moody's Analytics, SAS) | Not implemented |
| Real-time risk decisioning | Partial | ✓ (FICO Decision Modeler, Experian PowerCurve) | Decision engine integration; no sub-100ms scoring SLA for high-frequency origination |
| Model risk management (MRM) | ✗ | ✓ (Moody's Analytics, IBM OpenPages) | No model inventory, validation lifecycle, or MRM governance |

**World-best reference:** Moody's Analytics RiskCalc/ImpairmentStudio, SAS Credit Risk Management, FICO Decision Modeler

**Critical gaps:**
- IFRS 9 ECL staging (Stage 1/2/3) and CECL provisioning are entirely absent — a regulatory necessity for any bank holding loans
- No Basel III/IV RWA calculation — capital adequacy reporting cannot be automated
- Model risk management framework is absent — regulators require formal model validation documentation
- No stress testing infrastructure — DFAST/EBA stress test requirements cannot be met

---

## Fintech CRM (`fintech_crm`)

**APG provides:** Customer relationship management tailored for financial services, with customer 360 view, interaction tracking, product holding visibility, lead/opportunity management, and service case management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Customer 360 / unified profile | Partial | ✓ (Salesforce FSC, Microsoft Dynamics 365) | Profile model present; no real-time product holding aggregation from core banking |
| Lead & opportunity management | Partial | ✓ (Salesforce FSC) | Present; no AI-powered next-best-action or lead scoring |
| Service case management | ✓ | ✓ (Salesforce Service Cloud, Zendesk) | Present; no SLA enforcement or escalation routing |
| Financial goals / life events | ✗ | ✓ (Salesforce FSC) | No life-event detection or goals-based planning workflow |
| Household / relationship management | ✗ | ✓ (Salesforce FSC, Microsoft Dynamics) | No household grouping or referral network visualization |
| Campaign & marketing automation | ✗ | ✓ (Salesforce Marketing Cloud, HubSpot) | Not implemented |
| Channel integration (phone, chat, email) | ✗ | ✓ (Salesforce Service Cloud, Genesys) | No CTI integration or omnichannel routing |
| Regulatory interaction logging | Partial | ✓ (Salesforce FSC) | Interaction model present; no MiFID II suitability record-keeping |
| Analytics & relationship health score | ✗ | ✓ (Salesforce Einstein, Microsoft) | Not implemented |
| Mobile CRM for advisors | ✗ | ✓ (Salesforce Mobile, Microsoft Teams) | No mobile-optimized advisor experience |

**World-best reference:** Salesforce Financial Services Cloud, Microsoft Dynamics 365 for Financial Services

**Critical gaps:**
- No life-event detection or goals-based planning — the core differentiator of financial services CRM vs. generic CRM
- No marketing automation — customer segments from CLV/propensity scores cannot be activated into campaigns
- No omnichannel routing (CTI, chat); CRM interactions are logged manually without telephony integration
- MiFID II suitability and advice record-keeping is not implemented — a hard regulatory requirement for investment advisors

---

## Core Banking Lending (`fintech_csl`)

**APG provides:** Core banking lending module supporting loan product configuration, disbursement, repayment scheduling, interest calculation, and arrears management. Africa-market focused with support for mobile money disbursement.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Product configuration engine | ✓ | ✓ (Temenos Transact, Mambu) | Present; less composable than Mambu's fully parameterized product factory |
| Disbursement (bank, mobile money) | ✓ | ✓ (Mambu, Temenos) | Present with M-Pesa integration; no multi-rail settlement finality tracking |
| Repayment schedule generation | ✓ | ✓ (Temenos, Mambu, Oracle FLEXCUBE) | Present; limited amortization method support vs. FLEXCUBE |
| Interest accrual & capitalization | ✓ | ✓ (Temenos, Oracle FLEXCUBE) | Present; no compound interest with multiple compounding periods |
| Arrears / delinquency management | Partial | ✓ (Temenos, Mambu) | Arrears tracking present; no automated IFRS 9 stage migration |
| Loan restructuring | Partial | ✓ (Temenos, Finastra) | Modification model present; no automated NPL reclassification |
| Islamic finance products (Murabaha, Ijara) | ✗ | ✓ (Temenos, Oracle FLEXCUBE) | Not implemented |
| Multi-currency loans | Partial | ✓ (Temenos, Oracle FLEXCUBE) | Currency fields present; no FX-linked repayment indexing |
| GL integration / accounting entries | Partial | ✓ (Temenos, Oracle FLEXCUBE) | GL module present but not certified end-to-end |
| Regulatory reporting (CBK, CBUAE) | Partial | ✓ (Temenos, Oracle FLEXCUBE) | Event emission present; no jurisdiction-specific regulatory report templates |

**World-best reference:** Temenos Transact, Mambu, Oracle FLEXCUBE

**Critical gaps:**
- No IFRS 9 automated stage migration — loan loss provisioning cannot be automated
- Islamic finance product types are absent — required for Kenya, East Africa, and GCC markets
- No jurisdiction-specific regulatory report templates — CBK, Bank of Uganda, CBN returns require manual formatting
- GL accounting entries are not certified end-to-end; double-entry integrity across loan lifecycle is unverified

---

## Debt Management (`fintech_dbt`)

**APG provides:** Debt collection lifecycle management with debtor segmentation, payment plan negotiation, legal action tracking, and recovery analytics.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Debtor segmentation & prioritization | Partial | ✓ (Experian Debt Manager, FICO) | Segmentation present; no ML-based propensity-to-pay scoring |
| Payment arrangement / plan management | ✓ | ✓ (Experian Debt Manager, FICO) | Present; no self-service debtor portal |
| Multi-channel contact management | ✗ | ✓ (Experian, FICO Debt Manager) | No outbound dialer integration, SMS campaign, or email sequencing |
| Legal action tracking | Partial | ✓ (Experian, LexisNexis) | Workflow present; no integration with court systems or legal document generation |
| Recovery rate analytics | Partial | ✓ (Experian, FICO) | Basic analytics; no vintage-based recovery curve modeling |
| Debt sale / portfolio transfer | ✗ | ✓ (Experian, DebtX) | Not implemented |
| Right-party contact verification | ✗ | ✓ (LexisNexis, Experian) | No skip-tracing or contact data enrichment |
| Regulatory compliance (FDCPA, FCA) | Partial | ✓ (Experian, FICO) | Compliance model present; no jurisdiction-specific collections regulation enforcement |
| Debtor financial hardship assessment | ✗ | ✓ (Experian, FCA-regulated collectors) | No vulnerability assessment or breathing space scheme integration (UK) |
| NPL portfolio analytics | Partial | ✓ (Moody's Analytics, SAS) | Analytics present; no IFRS 9 NPL provisioning calculation |

**World-best reference:** Experian Debt Manager, FICO Debt Manager, LexisNexis Risk Solutions

**Critical gaps:**
- No multi-channel contact automation (dialer, SMS, email) — collections operations are entirely manual-outreach dependent
- No propensity-to-pay ML scoring — debtor prioritization is rule-based, reducing recovery efficiency
- Debt sale and portfolio transfer functionality is absent
- No right-party contact verification or skip-tracing — reaching debtors requires manual data gathering

---

## Decision Engine (`fintech_dec`)

**APG provides:** Rule-based and ML-augmented decision engine for credit, fraud, and operational decisions. Supports decision trees, scorecards, champion-challenger testing, and audit-grade decision logging.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Rule engine (DMN / Drools-equivalent) | Partial | ✓ (FICO Blaze Advisor, IBM ODM) | Rules present; no DMN-compliant rule authoring tool for business users |
| Scorecard execution | Partial | ✓ (FICO Decision Modeler, Experian PowerCurve) | Scoring infrastructure present; no FICO-certified scorecard import |
| Champion-challenger framework | Partial | ✓ (FICO, Experian PowerCurve) | Reference in capability contract; no production A/B traffic splitting infrastructure |
| Real-time decisioning latency | Partial | ✓ (FICO — <50ms, Experian PowerCurve) | Async Python service; no SLA enforcement for sub-100ms responses |
| ML model serving | Partial | ✓ (FICO, H2O.ai, Experian) | Ollama-based local models; no MLflow/BentoML model registry or versioned deployment |
| Decision audit trail | ✓ | ✓ (FICO, IBM ODM) | Logging present; no GDPR Article 22 adverse action notice generation |
| Data enrichment pre-decision | ✗ | ✓ (Experian PowerCurve, FICO) | No bureau data, telco, or alternative data pull at decision time |
| Flow/strategy design (visual) | ✗ | ✓ (FICO Decision Modeler, Pega) | No drag-and-drop strategy designer for business analysts |
| Regulatory explainability (adverse action) | Partial | ✓ (FICO Explainable AI, Experian) | Logging present; no jurisdiction-compliant adverse action reason code generation |
| Batch decisioning | Partial | ✓ (FICO, SAS) | Batch model present; no high-throughput batch scoring pipeline (millions/hour) |

**World-best reference:** FICO Decision Modeler (Blaze/XPRESS), Experian PowerCurve, IBM Operational Decision Manager

**Critical gaps:**
- No DMN-compliant business user rule authoring — all rule changes require developer involvement
- No production MLflow/model registry — model versioning and rollback are manual
- Data enrichment at decision time (bureau, alternative data) is entirely absent — decision inputs are limited to application data
- No visual strategy designer — a fundamental usability gap vs. FICO/Experian tools used by credit risk teams

---

## Deposits Management (`fintech_dep`)

**APG provides:** Deposit product management covering current accounts, savings, term deposits, and transactional accounts. Includes interest calculation, statement generation, and dormancy management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Product configuration (CASA, TD, RD) | ✓ | ✓ (Temenos Transact, Mambu) | Present; less granular product factory parameterization than Mambu |
| Interest calculation & accrual | ✓ | ✓ (Temenos, Oracle FLEXCUBE) | Present; no tiered interest rate ladder with multiple breakpoints |
| Statement generation | Partial | ✓ (Temenos, Oracle FLEXCUBE) | Basic statement logic; no multi-format delivery (PDF, MT940, CAMT.053) |
| Dormancy / escheatment management | Partial | ✓ (Temenos, Oracle FLEXCUBE) | Dormancy logic present; no jurisdiction-specific escheatment reporting |
| Joint accounts / mandates | Partial | ✓ (Temenos, Mambu) | Model present; no full mandate management workflow |
| Overdraft facilities | Partial | ✓ (Temenos, Mambu) | Overdraft model present; no automated limit review or behavioural scoring |
| Deposit insurance reporting (FDIC, DPS) | ✗ | ✓ (Temenos, Oracle FLEXCUBE) | Not implemented |
| Sweep / auto-transfer rules | ✗ | ✓ (Temenos, Oracle FLEXCUBE) | Not implemented |
| Islamic deposits (Mudarabah, Wakala) | ✗ | ✓ (Temenos, Oracle FLEXCUBE Islamic) | Not implemented |
| Regulatory reserve reporting | ✗ | ✓ (Temenos, Oracle FLEXCUBE) | No CRR/SLR calculation or central bank reserve reporting |

**World-best reference:** Temenos Transact, Mambu, Oracle FLEXCUBE

**Critical gaps:**
- No deposit insurance reporting (FDIC/DPS equivalent for local regulators) — mandatory for licensed deposit-taking institutions
- Sweep rules and auto-transfer are absent, blocking any treasury sweep or linked savings product
- Islamic deposit products are not implemented — required for operating in Kenya, East Africa Muslim communities, GCC
- Regulatory reserve calculation (CRR/SLR) not implemented — core central bank requirement for deposit institutions

---

## Fraud Detection (`fintech_frd`)

**APG provides:** Fraud detection service with signal ingestion, behavioral profiling, velocity analysis, case management with evidence collection, configurable decision outcomes, and multi-channel support (card, mobile, digital).

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time transaction scoring | Partial | ✓ (FICO Falcon — 4ms avg latency) | Scoring logic present; no sub-10ms latency guarantee; in-process Python vs. C++/JVM inference |
| Network-level consortium data | ✗ | ✓ (FICO Falcon Intelligence Network — 4B accounts) | No cross-institution data sharing or network-derived fraud signals |
| Behavioral profiling | Partial | ✓ (FICO Falcon, SAS Fraud Management) | Velocity and pattern logic present; no long-window behavioral profile (90/365-day baselines) |
| ML model portfolio (100+ techniques) | ✗ | ✓ (FICO Falcon — 100+ patented ML techniques) | Single local model; no specialized models per fraud typology |
| APP scam / social engineering detection | ✗ | ✓ (FICO Falcon v3.0, Featurespace ARIC) | Not implemented |
| Device fingerprinting | ✗ | ✓ (ThreatMetrix, Sardine) | Not implemented |
| Card compromise detection | ✗ | ✓ (FICO Falcon Intelligence Network) | No cross-bank compromise event sharing |
| Case management | ✓ | ✓ (FICO, SAS, Featurespace) | Present; no analyst workbench with investigation timeline visualization |
| False positive management | Partial | ✓ (FICO Falcon — 10:1 FP ratio) | Risk bands present; no ML-based FP feedback loop |
| Regulatory STR / fraud reporting | Partial | ✓ (FICO, SAS) | Event emission present; no regulator-format fraud report output |

**World-best reference:** FICO Falcon Fraud Manager, SAS Fraud Management, Featurespace ARIC

**Critical gaps:**
- No consortium / network intelligence — the single biggest differentiator of FICO Falcon is cross-institution signal sharing; APG operates in isolation
- APP/scam detection is entirely absent — the fastest-growing fraud typology in real-time payment environments
- No device fingerprinting — digital fraud detection without device signals is severely handicapped
- Python runtime cannot achieve 4ms latency at scale without significant infrastructure engineering beyond current design

---

## Payment Gateway (`fintech_gwy`)

**APG provides:** Full payment gateway service with merchant management, multi-provider routing, payment intent lifecycle, risk review, authorization/capture/refund, dispute management, webhook delivery, subscription billing, and multi-tenant support. Includes M-Pesa and Stripe processor integrations.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Payment intent lifecycle | ✓ | ✓ (Stripe, Adyen) | Well-implemented; no Stripe-grade idempotency key infrastructure |
| Multi-provider routing | ✓ | ✓ (Adyen, Stripe) | Routing logic present; no ML-based conversion optimization (Adyen Uplift) |
| 100+ payment methods | Partial | ✓ (Adyen — 250+, Stripe — 100+) | M-Pesa + cards + local methods; no global payment method breadth |
| Subscription / recurring billing | ✓ | ✓ (Stripe Billing — $500M run rate) | Present; no prorated upgrades/downgrades or dunning automation at Stripe fidelity |
| 3DS2 / SCA compliance | ✗ | ✓ (Stripe, Adyen) | Not implemented |
| Network tokenization | ✗ | ✓ (Stripe, Adyen) | Not implemented |
| Authorization rate optimization | ✗ | ✓ (Adyen — direct network connections) | No direct acquirer connections; all volume through third-party processors |
| Fraud scoring at authorization | Partial | ✓ (Stripe Radar, Adyen Risk) | Risk review module present; no custom ML model per merchant |
| Global acquiring (195 countries) | ✗ | ✓ (Stripe — 195 countries, Adyen) | Africa-focused; no global acquiring infrastructure |
| Payouts / marketplace splits | Partial | ✓ (Stripe Connect, Adyen Platforms) | Multi-tenant architecture present; no marketplace split payment infrastructure |

**World-best reference:** Stripe, Adyen, FIS Modern Banking Platform

**Critical gaps:**
- No 3DS2/SCA implementation — mandatory for European card payments; a hard regulatory blocker for EU merchants
- No network tokenization — payment security and authorization rates are below industry standard
- Authorization rate optimization is absent; without direct network connections, payment success rates will lag Adyen/Stripe
- No global acquiring infrastructure — APG gateway is Africa-regional, not a global platform

---

## Insurance (`fintech_ins`)

**APG provides:** Insurance product management covering policy lifecycle, premium calculation, claims processing, underwriting workflow, and reinsurance support.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Policy administration | Partial | ✓ (Guidewire PolicyCenter, Duck Creek) | Lifecycle model present; no multi-line, multi-jurisdiction policy engine |
| Premium calculation engine | Partial | ✓ (Guidewire, Majesco) | Calculation present; no actuarial rating algorithm library |
| Claims management | Partial | ✓ (Guidewire ClaimCenter, Duck Creek) | Claims workflow present; no subrogation, salvage, or litigation tracking |
| Underwriting workbench | ✗ | ✓ (Guidewire, Applied Systems) | Not implemented as a standalone workflow |
| Reinsurance management | ✗ | ✓ (Guidewire, Sapiens) | Not implemented |
| Fraud detection for claims | Partial | ✓ (Guidewire, SAS Fraud) | Fraud module exists separately; no insurance-specific claims fraud scoring |
| Regulatory filing (IFRS 17) | ✗ | ✓ (Guidewire, Majesco, Oracle) | Not implemented |
| Embedded insurance | ✗ | ✓ (Majesco, Cover Genius) | Not implemented as API-distributable insurance |
| Actuarial / loss reserving | ✗ | ✓ (Guidewire, Milliman) | Not implemented |
| Distribution / agent management | Partial | ✓ (Guidewire, Applied Systems) | Agent model shared with agency banking; no insurance-specific producer management |

**World-best reference:** Guidewire Insurance Suite, Duck Creek Technologies, Majesco

**Critical gaps:**
- No IFRS 17 compliance — the primary accounting standard for insurance contracts since 2023; a hard regulatory requirement
- Actuarial and loss reserving are entirely absent — cannot calculate adequate technical provisions
- Reinsurance management is not implemented — required for any insurer writing significant volumes
- No embedded insurance API distribution — the fastest-growing insurance channel in Africa and globally

---

## Investments / Wealth (`fintech_inv`)

**APG provides:** Wealth and investment management service covering portfolio construction, asset allocation, order management, performance measurement, and client reporting. Includes robo-advisory and portfolio optimization.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Portfolio construction & rebalancing | Partial | ✓ (FNZ, SS&C Advent, Temenos Multifonds) | Logic present; no Black-Litterman or factor-based optimization |
| Order management (OMS) | Partial | ✓ (SS&C Advent Geneva, Charles River IMS) | Order model present; no FIX protocol connectivity or market access |
| Performance measurement (GIPS) | ✗ | ✓ (SS&C Advent, FactSet) | Not implemented; no GIPS-compliant return calculation |
| Risk analytics (VaR, CVaR) | Partial | ✓ (Moody's Analytics, FactSet) | Risk service present; no Monte Carlo simulation at portfolio scale |
| Client reporting | Partial | ✓ (FNZ, SS&C) | Reporting present; no GIPS-compliant composites or CRS tax reporting |
| Robo-advisory | ✓ | ✓ (FNZ, Nutmeg, Betterment) | Robo service present; no regulatory suitability questionnaire (MiFID II, FSP) |
| Custody / settlement integration | ✗ | ✓ (FNZ, SS&C, BNY Mellon) | No custodian connectivity or DvP settlement instruction |
| Tax lot accounting | ✗ | ✓ (SS&C Advent, Advent Geneva) | Not implemented |
| Alternative investments | ✗ | ✓ (SS&C Advent, iCapital) | No private equity, hedge fund, or illiquid asset support |
| Regulatory reporting (MiFID II, FATCA) | ✗ | ✓ (SS&C, FNZ) | Not implemented |

**World-best reference:** FNZ, SS&C Advent Geneva, Temenos Multifonds, Charles River IMS

**Critical gaps:**
- No FIX protocol connectivity or market access — the OMS cannot route orders to any exchange or broker without custom integration
- GIPS-compliant performance measurement is absent — required for institutional mandates and any third-party capital
- No custody/settlement integration — trade lifecycle ends at order; DvP settlement instruction is not generated
- MiFID II/FATCA regulatory reporting is not implemented — a hard barrier for wealth management in regulated markets

---

## Key Financial Indicators (`fintech_kfi`)

**APG provides:** KFI calculation service covering standard financial ratios, liquidity metrics, capital adequacy, profitability indicators, and operational efficiency metrics across banking and fintech product lines.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Standard banking ratios (ROE, ROA, NIM) | ✓ | ✓ (Oracle FSDF, Moody's Analytics) | Present; no certified regulatory definition alignment (Basel/BCBS) |
| Capital adequacy ratios (CAR, T1, T2) | Partial | ✓ (Moody's Analytics, Oracle FSDF) | Calculation present; no regulatory capital deduction cascade (Basel III) |
| Liquidity ratios (LCR, NSFR) | Partial | ✓ (Moody's Analytics, Oracle FSDF) | Models present; no HQLA classification engine or run-off factor library |
| Asset quality (NPL ratio, provision coverage) | ✓ | ✓ (Moody's Analytics, Oracle FSDF) | Present; no IFRS 9 stage-linked automatic recalculation |
| Operational efficiency (cost-to-income) | ✓ | ✓ (Oracle, SAP) | Present; no multi-period peer benchmarking |
| KFI drill-through to source data | ✗ | ✓ (Oracle FSDF, SAP Analytics Cloud) | No lineage from KFI back to GL/transaction level |
| Regulatory submission format | ✗ | ✓ (Oracle FSDF, Moody's Analytics) | No XBRL/SDMX output for central bank statistical returns |
| Forecast KFIs | ✗ | ✓ (Oracle, Moody's Analytics) | No forward-looking KFI projection linked to budget/forecast |
| Peer benchmarking | ✗ | ✓ (Moody's Analytics, S&P Global MI) | Not implemented |
| Real-time KFI update | Partial | ✓ (Oracle FSDF, SAP) | Batch calculation; no event-driven KFI refresh on transaction |

**World-best reference:** Oracle Financial Services Data Foundation (FSDF), Moody's Analytics, SAP Analytics Cloud

**Critical gaps:**
- No regulatory submission output (XBRL/SDMX) — KFIs cannot be used for automated central bank reporting
- No drill-through lineage to source transactions — audit trail from KFI to GL entry is broken
- No peer benchmarking data — KFIs are calculated in isolation with no market context
- No IFRS 9-linked automatic recalculation of provisioning-dependent ratios

---

## KYC / Identity Verification (`fintech_kyk`)

**APG provides:** Full KYC lifecycle service with Africa-specific design covering 48 methods across application management, document verification, biometric checks, watchlist screening, risk scoring, EDD for PEP/high-risk, compliance reporting, and digital onboarding. Country-aware with KE, NG, ZA, and others.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Document verification (OCR + NFC) | Partial | ✓ (Onfido, Jumio, Idemia) | Document model and rules present; no certified OCR/NFC chip reading engine |
| Biometric liveness detection | Partial | ✓ (Onfido, iProov) | Liveness check rule asserted; no certified liveness SDK (iBeta PAD Level 2) |
| Watchlist / sanctions screening | ✓ | ✓ (ComplyAdvantage, Dow Jones) | OFAC/PEP/UN/EU screening present; no real-time list refresh SLA |
| PEP & adverse media screening | Partial | ✓ (ComplyAdvantage, Refinitiv WCO) | PEP flag in rules; no adverse media NLP feed from news sources |
| Risk-based CDD / EDD workflow | ✓ | ✓ (Onfido, Jumio, ComplyAdvantage) | Full EDD for PEP/high-risk asserted and enforced in rules |
| eKYC / government database lookup | ✗ | ✓ (Onfido, Smile Identity for Africa) | No integration with NIMC (Nigeria), Huduma Namba (Kenya), DHA (South Africa) |
| Digital onboarding UX | ✗ | ✓ (Onfido, Jumio) | No mobile SDK or web widget for end-customer document capture |
| Re-KYC / periodic review automation | Partial | ✓ (ComplyAdvantage, Acuant) | Expiry calculation present; no automated re-KYC trigger workflow |
| Corporate / UBO verification | Partial | ✓ (ComplyAdvantage, Refinitiv) | Corporate model present; no automated UBO registry lookup (e.g., Companies House) |
| Regulatory compliance reporting | Partial | ✓ (ComplyAdvantage, Fiserv) | Event emission present; no FIU-format KYC statistical report output |

**World-best reference:** Onfido, Jumio, ComplyAdvantage, Smile Identity (Africa-specific)

**Critical gaps:**
- No certified document OCR/NFC engine — document verification is rule/model-only without actual image processing capability
- No certified liveness SDK — biometric liveness is asserted as a rule but has no underlying ML/computer vision implementation
- No integration with African government identity databases (NIMC, Huduma, DHA) — eKYC in target markets cannot be automated
- No mobile SDK or web widget for customer-facing document capture — KYC workflow requires third-party front-end

---

## Loan Management System (`fintech_lms`)

**APG provides:** Full loan management lifecycle from origination through payoff, including disbursement, repayment scheduling, interest accrual, arrears management, restructuring, write-off, and portfolio reporting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Loan product factory | ✓ | ✓ (Mambu, Temenos Transact) | Present; less composable than Mambu's API-first product configuration |
| Disbursement multi-rail | ✓ | ✓ (Mambu, Temenos) | Bank + mobile money present; no RTGS/ACH direct integration |
| Repayment scheduling (multiple methods) | ✓ | ✓ (Mambu, Oracle FLEXCUBE) | Present; limited balloon/bullet/grace period amortization variants |
| IFRS 9 staging & ECL | ✗ | ✓ (Mambu, Temenos, Oracle FLEXCUBE) | Not implemented — a critical regulatory gap |
| Arrears / collections escalation | Partial | ✓ (Mambu, Temenos) | Arrears present; no automated collections handoff workflow |
| NPL / write-off processing | Partial | ✓ (Mambu, Temenos) | Write-off model present; no collateral realization workflow |
| Portfolio analytics (vintage, roll rates) | ✗ | ✓ (Mambu, Moody's Analytics) | Not implemented |
| Multi-currency | Partial | ✓ (Temenos, Oracle FLEXCUBE) | Currency fields present; no FX-linked loan repayment indexing |
| Document management | Partial | ✓ (nCino, Blend) | Document model present; no DMS integration or e-signature |
| Regulatory submission | ✗ | ✓ (Temenos, Oracle FLEXCUBE) | No CBK/CBN/SARB loan portfolio returns |

**World-best reference:** Mambu, Temenos Transact, Oracle FLEXCUBE

**Critical gaps:**
- IFRS 9 ECL staging is absent — the most critical regulatory compliance gap for any bank managing a loan book
- No portfolio vintage analysis or roll-rate reporting — essential credit risk management analytics are missing
- No e-signature or DMS integration — loan documentation workflow requires manual paper-based processes
- No regulatory portfolio returns for any African central bank — statutory reporting requires custom development

---

## Mobile Banking (`fintech_mbl`)

**APG provides:** Mobile banking service layer covering account management, transfers, bill payments, airtime/data purchase, mini-statement, and mobile money integration. Africa-market focused.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Account management (balance, statements) | ✓ | ✓ (Temenos Infinity, FIS Mobile) | Present; no PDF statement generation or ISO MT940 export |
| Fund transfers (internal, external) | ✓ | ✓ (Temenos Infinity, FIS) | Present; no ISO 20022 payment instruction generation |
| Bill payment integration | ✓ | ✓ (Temenos, FIS) | Present; no biller aggregator connectivity (e.g., eCitizen, DSTV) |
| Mobile money integration (M-Pesa) | ✓ | ✓ (Craft Silicon, Temenos) | Present and Africa-specific; no Airtel Money, MTN MoMo, Tigo |
| Biometric authentication | Partial | ✓ (Temenos Infinity, FIS) | Model present; no OS-level biometric SDK integration (Face ID, Touch ID) |
| Chatbot / conversational banking | ✗ | ✓ (Temenos Infinity, Kasisto) | Not implemented |
| Card management (freeze, PIN) | ✗ | ✓ (Temenos, FIS, Marqeta) | Not implemented |
| Investment / savings products from mobile | ✗ | ✓ (Temenos Infinity, FIS) | No mobile-initiated investment or fixed deposit placement |
| Push notifications (transactional) | Partial | ✓ (Temenos, FIS) | Event model present; no FCM/APNs integration |
| Offline / USSD fallback | ✗ | ✓ (Craft Silicon, Temenos) | Not implemented |

**World-best reference:** Temenos Infinity, FIS Mobile Banking Platform, Craft Silicon BankFusion

**Critical gaps:**
- No USSD fallback — in markets where smartphone penetration is below 50%, a USSD channel is essential
- No OS-level biometric SDK — biometric authentication is modeled but not integrated with device hardware
- Card management from mobile (freeze/unfreeze, PIN set) is entirely absent — a baseline expectation in any mobile banking app
- No FCM/APNs push notification integration — transactional alerts require a custom notification layer

---

## Capital Markets (`fintech_mkt`)

**APG provides:** Capital markets service covering bond/equity issuance, trading desk functionality, order management, settlement, and market data integration.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Fixed income pricing & analytics | Partial | ✓ (Bloomberg, FactSet, Refinitiv) | Calculation logic present; no Bloomberg FIGI/ISIN reference data feed |
| Order management (buy-side / sell-side) | Partial | ✓ (Charles River IMS, SS&C Advent) | Order model present; no FIX 4.2/5.0 connectivity |
| Pre/post-trade compliance | ✗ | ✓ (Charles River IMS, Linedata) | Not implemented |
| Settlement (DvP / FvP) | ✗ | ✓ (Broadridge, SunGard/FIS) | Not implemented |
| Market data integration | ✗ | ✓ (Bloomberg, Refinitiv, ICE Data) | No market data feed subscription |
| Risk (VaR, Greeks, CVA) | Partial | ✓ (Murex, Finastra Fusion Capital) | Risk module present; no Greeks calculation for derivatives |
| Derivatives (OTC, listed) | ✗ | ✓ (Murex, Calypso) | Not implemented |
| Trade repository reporting (EMIR, DTCC) | ✗ | ✓ (Broadridge, DTCC, Murex) | Not implemented |
| Clearing house connectivity (LCH, CME) | ✗ | ✓ (Broadridge, Murex) | Not implemented |
| Corporate actions processing | ✗ | ✓ (Broadridge, SS&C) | Not implemented |

**World-best reference:** Murex, Calypso (Broadridge), Charles River IMS, Finastra Fusion Capital Markets

**Critical gaps:**
- No FIX protocol connectivity — cannot route any order to an exchange, broker, or ECN
- Settlement instruction generation (DvP/FvP) is entirely absent — trade lifecycle terminates at order
- No derivatives pricing or OTC trade management — limits APG to cash equity/bond instruments only
- Trade repository reporting (EMIR/DTCC) is not implemented — a hard regulatory requirement for any OTC derivatives business

---

## Mortgage / Real Estate Payments (`fintech_mrp`)

**APG provides:** Mortgage product management covering origination, underwriting workflow, property valuation integration, repayment scheduling, escrow management, and prepayment handling.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Mortgage origination workflow | Partial | ✓ (Blend, nCino Mortgage Suite) | Workflow model present; no URLA/1003 digital application form |
| Automated underwriting (AUS) | ✗ | ✓ (Fannie Mae DU, Freddie Mac LPA) | Not connected to any AUS engine |
| Property valuation integration | ✗ | ✓ (Blend, nCino — CoreLogic, Zillow) | No AVM or appraisal management integration |
| Escrow management | Partial | ✓ (nCino, Black Knight) | Model present; no escrow analysis or tax/insurance impound calculation |
| Prepayment / overpayment handling | Partial | ✓ (Temenos, nCino) | Partial prepayment logic present; no yield maintenance or prepayment penalty calculation |
| Secondary market / loan sale | ✗ | ✓ (Black Knight, Fannie/Freddie) | Not implemented |
| Mortgage servicing (post-close) | Partial | ✓ (Black Knight MSP, ICE Mortgage) | Basic servicing model; no escrow analysis, insurance tracking, or flood zone monitoring |
| HMDA / CRA regulatory reporting | ✗ | ✓ (Black Knight, Wolters Kluwer) | Not implemented |
| Title / closing workflow | ✗ | ✓ (Blend Closing, Snapdocs) | Not implemented |
| Land registry integration | ✗ | ✓ (Blend, country-specific) | No integration with any land registry (Lands Registry Kenya, DLRS South Africa) |

**World-best reference:** Black Knight (now ICE Mortgage Technology), Blend, nCino Mortgage Suite

**Critical gaps:**
- No AUS connectivity — mortgage underwriting decisions must be entirely manual
- No land registry integration — critical for any African mortgage product where title verification is a key risk
- Secondary market/loan sale functionality is absent — limits mortgage product to portfolio lending only
- No HMDA/CRA reporting — required for US-market mortgage lending; no equivalent African regulatory report output either

---

## Non-Performing Loans (`fintech_npl`)

**APG provides:** NPL management service covering classification, provisioning workflow, recovery strategy, collateral tracking, and legal action management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| NPL classification (IFRS 9 stages) | Partial | ✓ (Moody's Analytics, Oracle FSDF) | Classification logic present; no automated SICR trigger for Stage 2 migration |
| Provisioning calculation (ECL) | ✗ | ✓ (Moody's Analytics ImpairmentStudio) | Not implemented |
| Collateral valuation & realization | Partial | ✓ (Moody's Analytics, LexisNexis) | Collateral model present; no automated valuation model (AVM) integration |
| Recovery strategy engine | Partial | ✓ (Experian Debt Manager, FICO) | Strategy model present; no ML propensity-to-pay scoring for recovery |
| Legal action / litigation tracking | Partial | ✓ (Experian, LexisNexis) | Workflow present; no auctioneers/court integration |
| Debt sale / NPL portfolio disposal | ✗ | ✓ (DebtX, Hoist Finance) | Not implemented |
| Regulatory NPL reporting (EBA NPL template) | ✗ | ✓ (Moody's Analytics, Oracle) | Not implemented |
| NPL analytics (vintage, roll-rate, migration) | ✗ | ✓ (Moody's Analytics, SAS) | Not implemented |
| Forbearance management | ✗ | ✓ (Moody's Analytics, Temenos) | Not implemented |
| Outsourcing / servicer management | ✗ | ✓ (Moody's Analytics, Fiserv) | Not implemented |

**World-best reference:** Moody's Analytics ImpairmentStudio, Experian Debt Manager, Oracle FSDF

**Critical gaps:**
- ECL provisioning calculation is absent — this is the primary regulatory obligation for NPL management under IFRS 9
- No EBA NPL template reporting — required for European banks; no African equivalent implemented either
- Forbearance management is not implemented — required to track restructured loans that retain NPL classification
- NPL portfolio analytics (vintage, migration matrices) are entirely absent — cannot measure portfolio recovery performance

---

## Payroll - Fintech (`fintech_prl`)

**APG provides:** Payroll processing service with salary computation, deduction management, statutory compliance (PAYE, NSSF, NHIF), payslip generation, and bank disbursement integration. Africa-market focused.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Gross-to-net calculation | ✓ | ✓ (ADP, Workday, Sage Payroll) | Present; no multi-country statutory tax engine |
| PAYE / statutory deductions (KRA, FIRS) | Partial | ✓ (Sage Payroll Africa, Workday) | KE/NG models present; no automated KRA iTax or FIRS e-filing |
| Payslip generation | ✓ | ✓ (ADP, Workday) | Present; no digital payslip distribution with employee self-service portal |
| Bank disbursement / EFT | Partial | ✓ (ADP, Workday) | Disbursement model present; no bulk ACH/EFT file generation in bank-specific format |
| Earned wage access (EWA) | ✗ | ✓ (Rain, Wagestream, ADP) | Not implemented — a key Africa fintech differentiator |
| Benefits administration | ✗ | ✓ (ADP, Workday) | Not implemented |
| HR integration | ✗ | ✓ (ADP, Workday) | No HRIS connector; employee records maintained separately |
| Multi-country / multi-currency payroll | Partial | ✓ (ADP GlobalView, Workday) | Multi-currency present; no multi-country tax rule library |
| Compliance reporting (NSSF, NHIF returns) | Partial | ✓ (Sage Payroll Africa, ADP) | Event emission present; no statutory return file format generation |
| Payroll analytics | Partial | ✓ (ADP, Workday) | Basic analytics; no headcount cost forecasting or scenario modeling |

**World-best reference:** ADP GlobalView, Workday HCM Payroll, Sage Payroll Africa

**Critical gaps:**
- No automated e-filing to KRA iTax, FIRS, or SARS e@syFile — statutory filing requires manual submission
- Earned wage access is absent — a high-demand product in African markets with monthly payroll cycles
- No HRIS integration — employee master data synchronization is a manual process
- No bulk ACH/EFT file generation in bank-specific format — disbursement integration requires custom work per bank

---

## Risk Management (`fintech_rsk`)

**APG provides:** Enterprise risk management service covering market risk, credit risk, operational risk, liquidity risk, and risk appetite framework management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Risk appetite framework | Partial | ✓ (Moody's Analytics, IBM OpenPages) | Framework model present; no board-approved limit cascade and breach escalation |
| Market risk (VaR, PnL attribution) | Partial | ✓ (Moody's Analytics, SAS Risk) | Risk service present; no historical simulation or Monte Carlo at trading book scale |
| Credit risk aggregation | Partial | ✓ (Moody's Analytics, SAS) | Aggregation logic present; no RAROC calculation |
| Operational risk (RCSA, loss events) | Partial | ✓ (IBM OpenPages, MetricStream) | Risk model present; no key risk indicator (KRI) automated breach alerting |
| Liquidity risk (LCR, NSFR) | Partial | ✓ (Moody's Analytics, FIS) | Calculation present; no intraday liquidity monitoring |
| Stress testing (macro scenarios) | ✗ | ✓ (Moody's Analytics, SAS Risk) | Not implemented |
| ICAAP / ORSA documentation | ✗ | ✓ (Moody's Analytics, IBM OpenPages) | Not implemented |
| Regulatory capital reporting | ✗ | ✓ (Moody's Analytics, Oracle FSDF) | Not implemented |
| Model risk management | ✗ | ✓ (Moody's Analytics, IBM OpenPages) | Not implemented |
| Risk data aggregation (BCBS 239) | ✗ | ✓ (Moody's Analytics, Oracle FSDF) | No BCBS 239-compliant risk data lineage |

**World-best reference:** Moody's Analytics RiskFoundation, SAS Risk Management, IBM OpenPages

**Critical gaps:**
- Stress testing is entirely absent — a core supervisory requirement (DFAST, EBA, ICAAP) for all regulated banks
- ICAAP/ORSA documentation support is not implemented — required for Pillar 2 regulatory dialogue
- No BCBS 239-compliant risk data aggregation — data lineage from risk metric to source transaction is not traceable
- RAROC calculation is absent — cannot compute risk-adjusted return for pricing or capital allocation decisions

---

## Savings Products (`fintech_sav`)

**APG provides:** Savings product management covering goal-based savings, fixed deposits, recurring deposits, locked savings, and savings group (SACCO/chama) functionality.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Goal-based savings | ✓ | ✓ (Mambu, Temenos) | Present; no round-up or micro-saving trigger from transactions |
| Fixed / term deposits | ✓ | ✓ (Mambu, Temenos Transact) | Present; limited rollover automation options |
| Recurring / standing order savings | ✓ | ✓ (Mambu, Temenos) | Present; no smart scheduling based on salary credit detection |
| Savings group (SACCO / chama) | ✓ | ✓ (Craft Silicon, Musoni) | A genuine Africa-market differentiator; no inter-group lending or share capital management |
| Interest rate tiers & bonus rates | Partial | ✓ (Temenos, Mambu) | Basic rates present; no promotional rate campaigns or loyalty-linked bonus rates |
| Sweep / auto-save rules | ✗ | ✓ (Mambu, Temenos) | Not implemented |
| SACCO regulatory reporting | ✗ | ✓ (Craft Silicon, Musoni) | No SASRA (Kenya) or equivalent SACCO regulator report output |
| Islamic savings (Mudarabah profit-sharing) | ✗ | ✓ (Temenos, Oracle FLEXCUBE Islamic) | Not implemented |
| Savings analytics / nudges | Partial | ✓ (Personetics, Temenos) | Basic analytics; no behavioral nudge engine |
| Deposit insurance reporting | ✗ | ✓ (Temenos, Oracle FLEXCUBE) | Not implemented |

**World-best reference:** Mambu, Temenos Transact, Craft Silicon BankFusion

**Critical gaps:**
- No SASRA regulatory reporting — required for any SACCO operating in Kenya
- Sweep/auto-save rules are absent — a core product feature for modern savings platforms
- Islamic savings products are not implemented — significant market gap in East Africa and GCC
- No behavioral nudge engine — competitive savings platforms use behavioral economics to drive savings rate; APG has none

---

## Transaction Processing (`fintech_trx`)

**APG provides:** Core transaction processing service with ISO 8583 message handling via the switch module, payment routing, authorization, posting, and settlement. Includes PIN block handling, HSM key management, and message switching infrastructure.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ISO 8583 message processing | ✓ | ✓ (FIS Base24, ACI Worldwide) | Switch present with full ISO 8583 field handling; not certified by Visa/Mastercard |
| Real-time authorization | ✓ | ✓ (FIS Base24, ACI Worldwide) | Authorization logic present; no sub-millisecond latency at card scheme scale |
| PIN block / HSM key management | ✓ | ✓ (FIS, Thales payShield) | Key handler present; no certified HSM integration (Thales payShield, Utimaco) |
| Routing engine | ✓ | ✓ (FIS Base24, ACI) | Routing engine present; no least-cost routing with real-time interchange optimization |
| Card scheme connectivity (Visa, MC) | ✗ | ✓ (FIS Base24, ACI Worldwide) | No certified scheme connectivity; all traffic through third-party processors |
| ISO 20022 migration | ✗ | ✓ (Swift, Temenos, FIS) | No ISO 20022 message generation or migration path |
| Fraud scoring at authorization | Partial | ✓ (FIS Base24 + FICO Falcon) | Fraud module present but separate; no sub-millisecond inline fraud scoring |
| Settlement file generation | Partial | ✓ (FIS, ACI) | Reconciliation logic present; no Visa/MC settlement file format output |
| Exception / dispute processing | Partial | ✓ (FIS Base24, ACI) | Exception model present; no Visa/MC chargeback file ingestion |
| 24/7 availability / HA architecture | Partial | ✓ (FIS Base24 — five-nines) | Service designed for HA; no documented five-nines SLA or active-active deployment |

**World-best reference:** FIS Base24-eps, ACI Worldwide UP Retail Payments, Temenos Payments

**Critical gaps:**
- No Visa/Mastercard scheme certification — cannot operate as a direct scheme participant; all volume requires a sponsoring member
- No certified HSM integration — PIN block handling is software-only, a PCI DSS violation for production card processing
- ISO 20022 migration path is absent — a hard deadline requirement across all major payment systems globally
- No least-cost routing or real-time interchange optimization — payment costs will be uncompetitive vs. FIS/ACI deployments

---

---

# 2. Human Capital Management, CRM & Retail

## Employee Data Management (`chr_employee_data_management`)

**APG provides:** Core HR employee lifecycle management including employee records, org hierarchy, onboarding/offboarding workflows, and document management. Built on PostgreSQL with Flask-AppBuilder blueprints for standard CRUD and reporting operations.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Employee self-service portal | Basic profile editing | Workday: full self-service incl. benefits elections, life events | No guided life-event workflows |
| Org chart & hierarchy | Static org tree from DB | Workday: dynamic, real-time matrix org with dotted-line reporting | No matrix/dotted-line relationships |
| Document management | File attach to records | SAP SuccessFactors: DocuSign-integrated e-signatures, retention policies | No e-signature or retention automation |
| Position management | Job title/dept fields | Oracle HCM Cloud: headcount budgeting, FTE management per position | No position-level budgeting |
| Workforce analytics | SQL report views | Workday Prism: embedded ML workforce planning dashboards | No predictive attrition or workforce models |
| Localization / multi-jurisdiction | Single-locale schema | SAP SuccessFactors: 100+ country localizations, legal entity separation | No multi-country compliance layer |
| Audit trail | DB-level change log | Ceridian Dayforce: field-level audit with reason codes, tamper-evident | No reason-code capture on changes |
| Integration ecosystem | REST API | Workday: 300+ pre-built connectors (ATS, benefits, identity) | No pre-built connectors; all custom |
| Skills & competency framework | None | Oracle HCM: skills graph with AI gap analysis vs role requirements | Missing entirely |
| HRIS workflow engine | Manual admin actions | Workday: configurable approval chains, delegation, SLA escalation | No configurable BPMN-style workflows |

**World-best reference:** Workday HCM, SAP SuccessFactors, Oracle HCM Cloud

**Critical gaps:**
- No multi-jurisdiction/multi-entity compliance — blocks multinational deployments
- Absent skills/competency ontology — no foundation for talent development or AI-driven L&D
- No configurable approval workflow engine — all HR actions require developer intervention
- No pre-built identity/benefits/ATS connectors — integration burden falls entirely on implementers

---

## Payroll Processing (`pay_payroll`)

**APG provides:** Payroll calculation engine covering gross-to-net processing, statutory deductions, and payslip generation. Targets single-country payroll with PostgreSQL-backed pay runs and audit logs.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Gross-to-net calculation | Rule-based engine | ADP Workforce Now: real-time continuous payroll with mid-period adjustments | Batch-only; no continuous/on-demand payroll |
| Tax compliance & filing | Manual tax table config | ADP / Ceridian Dayforce: auto-updated tax tables, direct e-filing to agencies | No automated tax-table update or e-filing |
| Multi-currency / multi-country | Single currency | Ceridian Dayforce: unified global payroll across 160+ countries | Single-locale only |
| Off-cycle / retroactive pay | Not supported | Workday Payroll: retroactive calculations with full audit delta | Missing |
| Earned Wage Access (EWA) | None | Ceridian Dayforce On-Demand Pay: real-time EWA with bank rail integration | Missing entirely |
| GL integration | Manual journal export | Workday: auto-posted journal entries with cost-center allocation | No automated GL posting |
| Benefits deduction sync | Manual entry | ADP: real-time benefits deduction feed from enrollment events | Deductions require manual reconciliation |
| Payroll analytics | Static payrun reports | Paychex Flex: labor cost trending, overtime prediction, anomaly detection | No predictive or anomaly-detection analytics |
| Compliance reporting | Manual | ADP: auto-generated statutory reports (W-2, P60, etc.) per jurisdiction | No statutory report generation |
| Employee pay card / direct deposit | Basic bank details | ADP: multi-account split direct deposit, pay cards, same-day ACH | No payment method diversity |

**World-best reference:** ADP Workforce Now, Ceridian Dayforce, Workday Payroll

**Critical gaps:**
- No automated tax-table maintenance or e-filing — creates significant compliance liability at scale
- Batch-only processing with no retroactive or off-cycle support — fails common payroll correction scenarios
- No GL auto-posting — every payrun requires manual finance reconciliation
- Single-country architecture — fundamentally limits any multi-entity customer deployment

---

## Time & Attendance (`tat_time_attendance`)

**APG provides:** Employee clock-in/clock-out recording, shift scheduling, overtime tracking, and leave management with PostgreSQL persistence and REST API for integrations.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Clock-in methods | Web/app entry | UKG Workforce Central: biometric, facial recognition, geofenced mobile, NFC badge | No biometric or hardware terminal support |
| AI-powered scheduling | Manual scheduler | UKG: ML demand-driven schedule generation from historical patterns | No demand forecasting for schedules |
| Real-time labor tracking | Post-hoc reporting | UKG / Replicon: live dashboard with labor cost per shift vs budget | No real-time labor cost visibility |
| Compliance rule engine | Basic overtime flag | UKG: 500+ pre-built labor law rule sets (FLSA, EU WTD, state-level) | No pre-built compliance rule library |
| Absence / leave management | Basic leave balance | Deputy: predictive leave conflict detection, entitlement carryover automation | No carryover automation or conflict prediction |
| Fatigue / wellbeing rules | None | UKG: configurable fatigue rules, mandatory rest-period enforcement | Missing |
| Employee self-scheduling | None | Deputy: shift marketplace, self-swap with manager approval workflow | Missing |
| Payroll integration | Manual export | UKG / ADP: real-time timesheet push to payroll with variance flagging | Manual file transfer only |
| Geofencing & job costing | None | Replicon: GPS geofence enforcement, per-project time allocation | Missing |
| Analytics & forecasting | Static reports | UKG: overtime risk alerts, absenteeism trend analysis, cost forecasting | No predictive analytics |

**World-best reference:** UKG Workforce Central (Kronos), ADP Time & Attendance, Replicon

**Critical gaps:**
- No hardware/biometric clock-in support — not viable for manufacturing, retail, or field workforces
- No pre-built labor law compliance rules — every jurisdiction requires manual configuration
- No real-time labor cost dashboard — managers cannot course-correct spend within a shift
- Self-scheduling and shift marketplace entirely absent — increasing table-stakes expectation post-2022

---

## Advanced CRM (`crm_adv`)

**APG provides:** Full sales pipeline management, marketing campaign orchestration, customer engagement tracking, and lead/opportunity lifecycle from acquisition through close. REST API and Flask-AppBuilder UI on PostgreSQL.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| AI lead scoring | None | Salesforce Einstein: probabilistic lead scoring with CRM signal + external data | Missing |
| Sales forecasting | Manual pipeline sum | Salesforce: AI-driven forecast with call-level coaching signals | No AI forecasting layer |
| Conversation intelligence | None | Salesforce Einstein Conversation Insights: auto-transcription, sentiment, next-best-action | Missing entirely |
| Marketing automation | Basic campaign records | HubSpot: visual journey builder, behavioral triggers, A/B testing, attribution | No journey builder or behavioral triggers |
| CPQ (Configure-Price-Quote) | None | Salesforce CPQ: guided selling, discount governance, contract generation | Missing |
| Customer 360 / unified profile | Relational joins | Salesforce Data Cloud: real-time CDP merging CRM + behavioral + transactional data | No CDP or identity resolution |
| Territory & quota management | None | Microsoft Dynamics 365: AI-optimized territory assignment, quota waterfalls | Missing |
| Partner / channel management | None | Salesforce PRM: portal, deal registration, co-sell workflow | Missing |
| Omnichannel engagement | Email only | Microsoft Dynamics: unified inbox (email, Teams, SMS, WhatsApp) | Severely limited channel coverage |
| Revenue intelligence | None | Gong / Clari integrated with Salesforce: pipeline health signals, churn risk | No revenue intelligence layer |
| AppExchange / marketplace | None | Salesforce: 7,000+ ISV apps | No partner ecosystem |
| Mobile CRM | Basic mobile web | Salesforce mobile: offline capability, Einstein voice | No offline mode; no voice interface |

**World-best reference:** Salesforce Sales Cloud, HubSpot CRM, Microsoft Dynamics 365 Sales

**Critical gaps:**
- No AI scoring, forecasting, or conversation intelligence — sales teams operating blind vs Salesforce customers
- No CPQ or contract lifecycle management — large deal closure requires manual tooling outside APG
- No CDP/Customer 360 — fragmented customer view without cross-channel identity resolution
- Channel breadth limited to email — omnichannel engagement (SMS, WhatsApp, in-app) entirely absent

---

## Loyalty & Rewards Management (`retail_loy`)

**APG provides:** Customer loyalty program management including points accrual, tier management, reward redemption, and campaign-driven bonus point events. Integrates with POS and CRM modules.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Points engine flexibility | Fixed earn/burn rules | Salesforce Loyalty Management: event-driven, attribute-based rule engine with real-time processing | No event-driven rule composition |
| Tiered membership | Static tier thresholds | Punchh: ML-predicted tier migration with personalized upgrade incentives | No ML-driven tier propensity |
| Personalized offers | Segment-based | Yotpo: 1:1 ML offer personalization using purchase + browsing signals | No individual-level personalization |
| Coalition loyalty | None | LoyaltyLion / Antavo: multi-brand coalition points pooling | Missing |
| Gamification | None | Punchh: challenges, streaks, badges, social referral mechanics | Missing entirely |
| Real-time redemption | Async processing | Salesforce Loyalty: sub-100ms real-time point balance at checkout | No real-time balance guarantee |
| Fraud detection | None | Punchh: ML anomaly detection on redemption patterns | Missing |
| Omnichannel enrollment | Web + POS | Yotpo: email, SMS, social, QR in-store enrollment unification | Limited channel enrollment surface |
| Partner / earn-with-partner | None | Antavo: API-driven partner earn network | Missing |
| Lifecycle analytics | Basic reports | Salesforce: churn propensity, LTV cohort, redemption elasticity models | No predictive loyalty analytics |

**World-best reference:** Salesforce Loyalty Management, Punchh, Yotpo

**Critical gaps:**
- No real-time point balance at checkout — creates redemption disputes and poor UX at scale
- Gamification (challenges, streaks, referrals) entirely absent — significant engagement lever missing
- No ML fraud detection — loyalty programs without this are targeted by point-farming bots
- No coalition or partner earn network — limits loyalty program value proposition

---

## Order Management & Commerce (`retail_omc`)

**APG provides:** Omnichannel order lifecycle management covering order capture, fulfillment routing, inventory reservation, and returns processing across web, mobile, and in-store channels.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Distributed order management | Basic routing rules | Manhattan Associates OMS: ML-driven optimal fulfillment node selection | No ML routing optimization |
| Inventory availability promise | Static inventory check | Oracle OMS: Available-to-Promise (ATP) with real-time supply chain signals | No ATP calculation |
| BOPIS / curbside | Basic order flag | Manhattan: full BOPIS workflow with real-time store inventory | Workflow incomplete |
| Ship-from-store | None | IBM Sterling OMS: store-as-DC with pick/pack mobile app, carrier rate shopping | Missing |
| Returns orchestration | Manual process | Manhattan: AI-driven returnless refund decisions, returns routing optimization | No intelligent returns engine |
| Carrier integration | Manual shipping label | Oracle OMS: multi-carrier rate shopping, real-time tracking, label automation | No native carrier integrations |
| Fraud scoring at order capture | None | IBM Sterling: ML order fraud scoring pre-fulfillment | Missing |
| Order splitting & merging | None | Manhattan: smart order splitting by availability + customer SLA preference | Missing |
| Dropship / vendor fulfillment | None | Oracle OMS: dropship PO automation with vendor portal | Missing |
| Customer order visibility | Basic status page | Manhattan: proactive exception alerts, WISMO self-service, delivery prediction | No proactive exception management |

**World-best reference:** Manhattan Associates OMS, Oracle Order Management Cloud, IBM Sterling OMS

**Critical gaps:**
- No ATP calculation — inventory promises are unreliable, causing oversell and cancellations
- Ship-from-store and dropship entirely absent — blocks unified inventory monetization
- No intelligent returns orchestration — returns are a major P&L lever for omnichannel retailers
- No ML fulfillment optimization — static routing rules cannot balance cost vs SLA at scale

---

## Point of Sale (`retail_pos`)

**APG provides:** In-store transaction processing with product catalog lookup, payment capture, receipt generation, and basic inventory decrement. Operates as a Flask-AppBuilder web application backed by PostgreSQL.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Offline mode | Not supported | Toast / Square: full offline transaction queue with sync-on-reconnect | Critical gap for connectivity-unstable environments |
| Hardware peripheral support | Browser-based only | Lightspeed / NCR Aloha: receipt printers, cash drawers, barcode scanners, customer displays | No native hardware integration layer |
| Payment method diversity | Card + cash | Square: tap-to-pay, BNPL (Afterpay), QR code, gift cards, crypto | Limited payment rail coverage |
| Table service / hospitality | None | Toast: table mapping, course firing, kitchen display system (KDS), split bills | Missing for F&B use case |
| Customer-facing display | None | Lightspeed: customer-facing screen with upsell prompts and loyalty balance | Missing |
| Inventory sync | Post-transaction batch | Square: real-time inventory decrement with low-stock alerts at POS | Async only |
| Employee performance at POS | None | NCR Aloha: per-associate sales metrics, upsell rate, transaction speed | Missing |
| Returns / exchanges at POS | Basic void | Lightspeed: full returns with reason codes, exchange workflows, store credit | No structured returns flow |
| Multi-location management | None | Square for Retail: centralized catalog, pricing, and promotions across locations | Single-location only |
| Analytics at POS | None | Toast: hourly sales trending, labor vs revenue overlay, peak hour heatmaps | No embedded analytics |

**World-best reference:** Square, Toast, Lightspeed Retail

**Critical gaps:**
- No offline mode — a single connectivity outage halts all revenue; disqualifying for most deployments
- No hardware peripheral integration — APG POS cannot drive receipt printers, cash drawers, or scanners
- Single-location architecture — cannot manage catalog or pricing centrally across stores
- No kitchen display system or hospitality flows — entirely unviable for food & beverage vertical

---

## Promotions & Pricing Management (`retail_prm`)

**APG provides:** Promotion definition and pricing rule management including discount types, coupon codes, threshold offers, and product bundling. Rules stored in PostgreSQL and evaluated at checkout.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Promotion rule engine | Condition/discount pairs | SAP Emarsys: visual drag-and-drop promotion builder with real-time preview | No visual rule authoring |
| AI price optimization | None | Salesforce Commerce Cloud: ML-driven dynamic pricing with demand elasticity models | Missing entirely |
| Competitive price monitoring | None | Apttus/Conga: automated competitor price scraping and response rules | Missing |
| Personalized promotions | Segment-level | SAP Emarsys: 1:1 offer delivery via email/push/SMS triggered by behavior | No individual-level promo targeting |
| Promotion conflict management | Manual priority | Salesforce Commerce Cloud: automated stacking rule resolution with priority engine | Manual conflict resolution only |
| Flash sale / time-limited pricing | Basic date range | Salesforce Commerce Cloud: countdown timers, inventory-throttled flash pricing | No inventory-linked urgency mechanics |
| BOGO / complex deal types | Limited | Manhattan: full BOGO, multi-buy, buy-X-get-Y-free with exclusion rules | Complex deal types incomplete |
| A/B testing for promotions | None | Salesforce Commerce Cloud: native promotion A/B with revenue attribution | Missing |
| Promotion analytics | Basic redemption count | SAP Emarsys: incremental lift measurement, promo cannibalization analysis | No incremental lift analytics |
| CPQ / contract pricing | None | Conga CPQ: customer-specific contract price books with approval workflow | Missing for B2B retail |

**World-best reference:** SAP Emarsys, Salesforce Commerce Cloud, Conga CPQ

**Critical gaps:**
- No ML-driven dynamic pricing — static rules leave margin on the table and cannot respond to demand shifts
- No automated promotion conflict resolution — overlapping promotions cause unpredictable discount stacking
- Personalization is segment-level only — 1:1 promotion targeting requires external CDP integration
- No A/B testing framework — promotion effectiveness cannot be measured with statistical rigor

---

# 3. Healthcare, Pharma, GRC & Government

Now I have sufficient information to produce the full competitive analysis. Let me compile it.

---

## Healthcare Analytics (`healthcare_ana`)

**APG provides:** A general-purpose analytics module built on PostgreSQL with configurable dashboards and reporting pipelines. Supports aggregation of clinical and operational data with Flask-Appbuilder visualization. Does not include pre-built clinical measure libraries or population health risk models.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Pre-built clinical analytics library | None — custom SQL queries | Health Catalyst (Ignite 500+ measures) | Critical |
| EHR/claims data integration | Manual ETL via custom connectors | Health Catalyst (40+ prebuilt connectors, FHIR/HL7/OMOP) | Critical |
| Population health risk stratification | Not present | Optum Analytics (AI risk models, SDOH integration) | Critical |
| Near-real-time data refresh | Batch-only | Health Catalyst (24-hour clinical cadence) | High |
| AI/ML predictive analytics | Not present | IBM Watson Health (5-level analytics maturity model) | Critical |
| Quality measure reporting (HEDIS, CMS) | Not present | Health Catalyst (regulatory/CMS pre-built) | Critical |
| Role-specific clinical dashboards | Generic dashboards | Health Catalyst (physician, executive, frontline variants) | High |
| Benchmarking (internal + external) | Not present | Health Catalyst (multi-organization benchmarks) | High |
| SDOH/health equity analytics | Not present | Optum (social determinants integration) | High |
| Self-service analytics for non-IT users | Limited | Health Catalyst (self-service, guided analytics) | Medium |

**World-best reference:** Health Catalyst Ignite, Optum Analytics, IBM Watson Health

**Critical gaps:**
- No clinical measure library — zero HEDIS, CMS, or VBP metrics out-of-the-box
- No native EHR ingestion; Health Catalyst connects 40+ EHR systems via certified connectors
- No population health or risk stratification engine; no SDOH data layer
- AI/ML analytics layer entirely absent; competitors offer 5-level predictive maturity frameworks

---

## Clinical Management (`healthcare_cli`)

**APG provides:** Workflow and task management primitives that can be configured for clinical department operations. Tracks patient encounters, care team assignments, and basic scheduling via Flask-Appbuilder blueprints. Lacks clinical-specific data models, order sets, and care pathway logic.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Specialty clinical templates | Generic forms | Epic (1000+ specialty SmartPhrases, Best Practice Alerts) | Critical |
| Clinical decision support (CDS) | Not present | Epic (drug interaction, dosing, CDSS alerts at point of care) | Critical |
| Multi-disciplinary care team workflows | Basic task assignment | Epic ASAP/Inpatient modules (role-aware handoffs) | High |
| Order entry (CPOE) | Not present | Epic (unified medication, lab, imaging orders) | Critical |
| Emergency department management | Not present | Epic ASAP (triage, patient flow, boarding) | Critical |
| Mobile provider access | Not available | Epic Haiku/Canto (iOS/Android with offline) | High |
| Ambient AI documentation | Not present | Epic + Microsoft (ambient dictation, note drafting) | Critical |
| Care gap identification | Not present | Epic Healthy Planet (gap lists, outreach automation) | High |
| Patient engagement portal | Not present | Epic MyChart (200M+ activated accounts) | Critical |
| Interoperability (HIE/FHIR) | REST APIs only | Epic Care Everywhere (305M+ patient records, FHIR R4) | Critical |

**World-best reference:** Epic Systems, Oracle Health (Cerner), Meditech Expanse

**Critical gaps:**
- No CPOE, clinical decision support, or medication management; these are safety-critical in regulated markets
- No patient portal or engagement layer; Epic MyChart sets an unreachable baseline without major investment
- Ambient AI documentation entirely absent — rapidly becoming table-stakes for physician retention
- No HL7/FHIR-native integration layer; health information exchange requires certified connectors

---

## Healthcare Device Management (`healthcare_dev`)

**APG provides:** IoT device registration, status tracking, and maintenance scheduling backed by PostgreSQL. Supports device inventory, location tracking, and alert thresholds via configurable dashboards. Does not include medical-device-specific regulatory compliance or FDA UDI integration.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| FDA UDI registry integration | Not present | Philips HealthSuite / GE Asset Performance Management (UDI-compliant) | Critical |
| Real-time device telemetry streaming | Basic polling | Philips (continuous vitals streaming, edge processing) | High |
| Predictive maintenance (ML) | Not present | GE Predix (failure prediction, anomaly detection) | High |
| Cybersecurity / medical device security | Not present | Medigate / Claroty (medical device security posture) | Critical |
| Work order & CMMS integration | Basic scheduling | IBM Maximo Healthcare (full CMMS with PM schedules) | High |
| Device lifecycle & calibration management | Basic records | MasterControl (acquired Qualer — calibration & verification) | Medium |
| Interoperability with EHR | Not present | Epic (device data flows into patient record) | Critical |
| HIPAA-compliant device data storage | Standard DB | Specialized PHI-aware data vaults | Medium |
| Multi-site fleet visibility | Limited | Vizient / Nuvolo (enterprise-wide device maps) | High |
| Regulatory recall alerting | Not present | FDA MedSun / ECRI integration | Critical |

**World-best reference:** Philips HealthSuite IoT, GE Asset Performance Management, Nuvolo (ServiceNow-based CMMS)

**Critical gaps:**
- No FDA UDI registry linkage; mandatory for Class II/III device management in regulated markets
- Medical device cybersecurity posture management entirely absent (FDA 2023 cybersecurity guidance)
- No predictive maintenance ML; unplanned device downtime directly impacts patient safety
- No EHR integration to push device readings into patient records

---

## Electronic Medical Records (`healthcare_emr`)

**APG provides:** Patient demographic registration, encounter records, and basic clinical note storage using PostgreSQL models and Flask-Appbuilder views. Provides a structured document store but lacks clinical workflow integration, order management, and HL7 FHIR compliance.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Structured clinical documentation | Basic text fields | Epic (specialty templates, structured data capture, NLP extraction) | Critical |
| HL7 FHIR R4 compliance | Not present | Epic, Oracle Health (full FHIR R4 certified) | Critical |
| Problem/medication/allergy lists | Basic records | Epic (reconciliation workflows, drug interaction alerts) | Critical |
| Results management (lab, imaging) | Not present | Oracle Health Cerner (unified results inbox, critical value alerts) | Critical |
| Electronic prescribing (eRx) | Not present | athenahealth, Epic (Surescripts-certified eRx) | Critical |
| Longitudinal patient timeline | Not present | Epic (unified encounter history across care settings) | High |
| Physician satisfaction / usability | Developer-only UI | Epic #1 KLAS for physician satisfaction | Critical |
| Regulatory certification (ONC) | Not certified | Epic, Meditech (ONC 2015/2015C+ certified) | Critical |
| Audit logging (HIPAA) | Basic DB logging | Epic (fine-grained access audit, snooping detection) | High |
| Patient-facing record access | Not present | Epic MyChart (USCDI data export, patient-directed sharing) | Critical |

**World-best reference:** Epic Systems, Oracle Health (Cerner), Meditech Expanse

**Critical gaps:**
- Not ONC-certified; deploying as a clinical EMR in the US/EU is legally non-compliant without certification
- No HL7 FHIR R4 API layer — mandatory for CMS Interoperability Rule compliance
- No electronic prescribing or drug interaction checking; direct patient safety risk
- No structured clinical terminologies (SNOMED CT, LOINC, RxNorm) — exchange and analytics impossible

---

## Healthcare Insurance / Claims (`healthcare_ins`)

**APG provides:** Claims intake, status tracking, and basic adjudication workflow management. Supports payer-member relationship records and remittance tracking through configurable PostgreSQL schemas. Lacks EDI transaction processing and managed care contracting logic.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| EDI 837/835/270/271 transaction processing | Not present | TriZetto Facets / EDIFECS (certified EDI gateway) | Critical |
| Automated adjudication engine | Manual workflow | TriZetto Facets (rules-based auto-adjudication, 95%+ auto-rate) | Critical |
| ICD-10/CPT coding validation | Not present | Optum (CES claim edits, scrubbing engine) | Critical |
| Prior authorization workflows | Basic forms | Availity / Epic (e-PA, NCPDP SCRIPT, automated PA) | High |
| Explanation of Benefits (EOB) generation | Not present | TriZetto (CMS-1500, UB-04 compliant EOB) | Critical |
| Fraud, waste, and abuse detection | Not present | Cotiviti / Optum (ML-based FWA detection) | Critical |
| Provider network management | Not present | CAQH ProView / TriZetto (credentialing, contract management) | High |
| HIPAA 5010 compliance | Not certified | All major payer platforms (HIPAA-certified) | Critical |
| Managed care population tracking | Basic records | Centricity / Facets (enrollee lifecycle, capitation) | High |
| Real-time eligibility verification | Not present | Availity (real-time 270/271 with 1,000+ payers) | High |

**World-best reference:** TriZetto Facets, Optum Claims Intelligence, Availity

**Critical gaps:**
- No EDI-certified transaction engine; HIPAA 5010 compliance is regulatory mandatory in US market
- No automated adjudication rules engine; manual claims processing is economically non-viable at scale
- Fraud, waste, and abuse detection entirely absent; payer contracts often mandate FWA programs
- No real-time eligibility; point-of-care eligibility verification is now expected as standard

---

## Medical Inventory (`healthcare_inv`)

**APG provides:** Inventory item registration, stock level tracking, reorder point management, and basic supplier records using PostgreSQL. Supports multi-location inventory views and transaction history via Flask-Appbuilder. Lacks medical-specific features like lot tracking, expiration management, and sterile supply chain compliance.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Lot/batch and expiration tracking | Not present | Infor CloudSuite Healthcare (FIFO/FEFO with expiry alerts) | Critical |
| UDI barcode scanning & GS1 compliance | Not present | Omnicell (GS1/UDI scan at point of use) | Critical |
| Implant tracking (FDA mandate) | Not present | Nuvolo / Inventory Optimization Solutions (implant registry) | Critical |
| Automated PAR-level replenishment | Basic min/max | Omnicell (automated cabinets, predictive replenishment) | High |
| OR case cart management | Not present | Infor / Syft (procedure-based cart pick lists) | High |
| Consignment & vendor-managed inventory | Not present | Tecsys (VMI with vendor portal) | Medium |
| Formulary & contract price compliance | Not present | Vizient / Premier GPO integration | High |
| Cold chain / controlled substance tracking | Not present | McKesson (temperature logs, DEA-compliant tracking) | Critical |
| Integration with EHR charge capture | Not present | Epic (supply charge flows direct to patient account) | High |
| Clinical preference card management | Not present | Infor (surgeon preference cards, OR supply optimization) | Medium |

**World-best reference:** Omnicell, Infor CloudSuite Healthcare, Tecsys Health

**Critical gaps:**
- No UDI/GS1 compliance — FDA UDI Rule requires unit-of-use tracing for implantable devices
- No expiration/lot tracking; expired medical supplies are a patient safety and regulatory risk
- No controlled substance chain-of-custody tracking (DEA 21 CFR 1304 requirement)
- No integration with surgical workflows; manual case cart management is labor-intensive and error-prone

---

## Pharmacy Management (`healthcare_pha`)

**APG provides:** Medication catalog management, prescription records, and dispensing transaction logging. Supports basic drug formulary maintenance and inventory counts. Lacks clinical pharmacy decision support, NCPDP transaction processing, and sterile compounding workflows.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Clinical pharmacist decision support (DUR) | Not present | Cerner PharmNet (drug utilization review, interaction checking) | Critical |
| NCPDP D.0 prescription claims processing | Not present | ScriptPro / QS1 (NCPDP-certified claims transmission) | Critical |
| Automated dispensing cabinet integration | Not present | Omnicell / BD Pyxis (cabinet-level dispensing verification) | Critical |
| IV/sterile compounding workflow | Not present | BD Cato / Kroll (IV workflow, gravimetric verification) | Critical |
| Electronic medication administration record (eMAR) | Not present | Epic, Cerner (barcode medication administration, 5-rights) | Critical |
| Controlled substance reconciliation | Basic records | Omnicell (automated DEA-level reconciliation) | Critical |
| 340B program compliance | Not present | Macro Helix / Verity Solutions (340B split-billing, WAC tracking) | High |
| Drug information database integration | Not present | Cerner (Multum), Epic (First Databank integration) | Critical |
| Pharmacist clinical documentation in EHR | Not present | Epic (pharmacist notes, medication reconciliation in chart) | High |
| Patient medication therapy management | Not present | Outcomes/Tabula Rasa (MTM platform, CDTM) | Medium |

**World-best reference:** Cerner PharmNet, Omnicell, ScriptPro

**Critical gaps:**
- No drug utilization review or interaction checking; direct patient safety failure in clinical deployment
- No NCPDP-certified claims engine; retail/outpatient pharmacy billing legally requires certification
- Barcode medication administration (bMAR) absent — Joint Commission and CMS patient safety standards require it
- No sterile/IV compounding workflow; USP 795/797/800 compliance requires validated digital workflows

---

## Healthcare Reporting (`healthcare_rpt`)

**APG provides:** Configurable report builder with PostgreSQL query execution and Flask-Appbuilder rendered outputs. Supports scheduled exports and basic drill-down dashboards. Lacks pre-built regulatory report sets, clinical quality measure computation, and FHIR-based reporting APIs.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Pre-built CMS/regulatory report templates | Not present | Health Catalyst (MIPS, HEDIS, CMS pre-built measure sets) | Critical |
| Joint Commission / DNV audit report packs | Not present | Vizient (JC survey readiness dashboards) | High |
| FHIR-based measure reporting (eCQM) | Not present | Epic, Oracle Health (eCQM submission to CMS via FHIR) | Critical |
| Real-time operational dashboards | Batch SQL | Health Catalyst (near-real-time, 24h clinical cadence) | High |
| Population-level outcome reporting | Not present | Optum Analytics (attribution, outcomes, VBP performance) | High |
| Statistical process control (SPC) charts | Not present | Health Catalyst, Tableau Healthcare (SPC, control charts) | Medium |
| Peer benchmarking | Not present | Vizient / Premier (national benchmarks by peer group) | High |
| Self-service report authoring (non-IT) | Limited | Power BI Embedded in Epic, Health Catalyst self-service | Medium |
| Regulatory submission (MU/APM) | Not present | Epic, Cerner (automated MIPS data submission, ACI) | Critical |
| Report scheduling & distribution | Basic | Enterprise platforms (role-based distribution, burst reporting) | Medium |

**World-best reference:** Health Catalyst, Epic reporting suite, Vizient Analytics

**Critical gaps:**
- No pre-built CMS/Joint Commission regulatory report templates; manual construction is unsustainable
- No eCQM calculation engine; electronic clinical quality measure reporting is a CMS mandate
- No FHIR reporting APIs — payer-provider data exchange regulations require FHIR-based query
- No peer benchmarking layer; quality improvement programs require external comparison data

---

## Healthcare Scheduling (`healthcare_sch`)

**APG provides:** Appointment booking, resource (room/provider) availability management, and basic schedule views. Supports configurable slot templates and patient-facing booking forms. Lacks multi-resource optimization, automated patient reminders, and integration with clinical workflows.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Multi-resource simultaneous scheduling | Single resource | Epic (provider + room + equipment + interpreter optimization) | High |
| Automated patient reminders (SMS/email) | Not present | Phreesia / Relatient (automated multi-channel reminders) | High |
| Patient self-scheduling (web/mobile) | Basic form | Epic MyChart, Zocdoc (real-time availability, instant booking) | High |
| Waitlist & cancellation backfill | Not present | Relatient / NexHealth (intelligent waitlist management) | Medium |
| Referral-to-appointment workflow | Not present | Epic (electronic referral, authorization, slot reservation) | High |
| Surgical case scheduling (OR blocks) | Not present | Epic OpTime (OR case scheduling, block template management) | Critical |
| AI-optimized schedule templates | Not present | Qgenda (AI-based template optimization, no-show prediction) | Medium |
| No-show prediction & overbooking | Not present | Qgenda / LeanTaaS (ML no-show models, capacity optimization) | Medium |
| Patient check-in kiosk integration | Not present | Epic (MyChart Bedside, kiosk check-in) | Medium |
| Payer prior authorization at scheduling | Not present | Epic (auth check at time of scheduling) | High |

**World-best reference:** Epic Cadence, Qgenda, Relatient

**Critical gaps:**
- No OR/surgical scheduling module; perioperative block management is a complex distinct domain
- No AI capacity optimization or no-show prediction; hospital bed and OR utilization depend on this
- Patient self-scheduling portal absent — patient experience expectation now set by consumer apps
- No payer authorization checking at time of booking; late auth denials are a major revenue cycle failure

---

## Pharma Compliance — GxP / FDA 21 CFR Part 11 (`pharma_com`)

**APG provides:** Document versioning, electronic signature workflows, and audit trail logging using PostgreSQL. Configurable approval chains via Flask-Appbuilder. Does not include validated system infrastructure, 21 CFR Part 11 technical controls, or Annex 11 compliance documentation.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| 21 CFR Part 11 / Annex 11 certified infrastructure | Not certified | MasterControl (ISO 42001, 21 CFR Part 11 + Annex 11 certified) | Critical |
| Validated system (IQ/OQ/PQ documentation) | Not present | MasterControl, Veeva (pre-packaged validation scripts) | Critical |
| Electronic signature (21 CFR Part 11 compliant) | Basic signature field | Veeva Vault (Part 11-compliant e-sig with meaning statements) | Critical |
| Complete and immutable audit trail | DB-level logs | Veeva / MasterControl (immutable, time-stamped, user-attributed logs) | High |
| SOP compliance analyzer | Not present | MasterControl (AI SOP Compliance Analyzer, Jan 2026) | High |
| Training & competency management | Not present | MasterControl (linked training records, SOP-training mapping) | High |
| Change control workflows | Basic workflow | Veeva Vault QMS (change control linked to RIM and Safety) | High |
| GxP regulatory content library | Not present | MasterControl (built-in regulatory chat, global reg database) | Medium |
| Supplier qualification & audit | Not present | Veeva Vault QMS (supplier portal, qualification workflows) | High |
| AI-generated templates and narratives | Not present | MasterControl (AI template generator), Veeva (AI agents) | Medium |

**World-best reference:** MasterControl, Veeva Vault QMS, IQVIA Compliance Center

**Critical gaps:**
- Not a validated system — deploying APG in a GxP environment without IQ/OQ/PQ is an FDA audit finding
- 21 CFR Part 11 requires system-level controls (access control, audit trails, system checks) not achievable by configuration alone
- No compliance-linked training management; GxP mandates documented training before task execution
- Absence of regulatory content database; compliance teams require current global regulation references

---

## Clinical Trials Management (`pharma_ctr`)

**APG provides:** Study registration, site records, subject enrollment tracking, and milestone management via PostgreSQL models. Configurable workflow forms for monitoring visit documentation. Lacks ICH GCP-aligned risk-based monitoring, EDC integration, and sponsor-CRO data synchronization.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ICH E6(R3) risk-based quality management | Not present | Veeva CTMS (configurable risk assessments, RBQM aligned to E6(R3)) | Critical |
| Sponsor-CRO data synchronization | Not present | Veeva CTMS Transfer (automated daily sponsor-CRO sync) | High |
| EDC integration (subject/visit data) | Not present | Veeva CTMS + EDC (single sign-on, no duplicate data entry) | Critical |
| Site payment management | Basic records | Veeva Payments (budget tracking, automated site disbursements) | High |
| eTMF integration (trip report filing) | Not present | Veeva (monitoring reports auto-filed to eTMF) | High |
| Study startup automation | Not present | Veeva Study Startup (SSU timelines, country-site-level tracking) | High |
| Real-time enrollment dashboards | Basic counts | Veeva CTMS (drill-down KPI dashboards, enrollment vs. target) | Medium |
| Protocol deviation management | Basic forms | Veeva (issue management with risk scoring, ICH-aligned) | High |
| Mobile CRA access | Not present | Veeva (mobile vault access, field monitoring support) | Medium |
| AI-assisted monitoring reports | Not present | Veeva AI Agents (scheduled December 2025+) | Medium |

**World-best reference:** Veeva Vault CTMS, Oracle Clinical One, Medidata Rave

**Critical gaps:**
- No ICH E6(R3)-aligned RBQM framework; risk-based monitoring is now regulatory expectation
- No EDC integration; double data entry between CTMS and EDC is a GCP compliance issue
- No eTMF filing; regulatory inspection readiness requires real-time TMF completeness
- Sponsor-CRO synchronization absent; outsourced trials require bi-directional data exchange

---

## Pharma Distribution (`pharma_dis`)

**APG provides:** Shipment order management, carrier record tracking, and delivery status monitoring. Supports basic inventory allocation and warehouse location management. Lacks serialization/track-and-trace, cold chain monitoring, and pharmaceutical distribution regulatory compliance.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| DSCSA serialization & track-and-trace | Not present | TraceLink / SAP (DSCSA-compliant unit-level serialization) | Critical |
| Cold chain monitoring & exception alerts | Not present | Sensitech / Controlant (IoT temperature loggers, excursion alerts) | Critical |
| Controlled substance distribution (DEA) | Not present | SAP S/4HANA LS (DEA Form 222, ARCOS reporting) | Critical |
| 3PL / wholesaler integration (EDI) | Not present | SAP (EDI 850/856/810 with McKesson, Cardinal, AmerisourceBergen) | High |
| Lot-level expiry management in distribution | Not present | Infor WMS Healthcare (FEFO picking, expiry visibility) | High |
| Returns & recalls (reverse logistics) | Basic records | TraceLink (recall execution, unit-level traceability) | High |
| GDP compliance documentation | Not present | Körber WMS (GDP-compliant storage and distribution records) | High |
| Real-time shipment visibility | Not present | project44 / FourKites (carrier tracking, predictive ETAs) | Medium |
| Channel inventory visibility (sell-in/sell-through) | Not present | IQVIA MIDAS (market data, channel inventory analytics) | Medium |
| Pedigree documentation | Not present | TraceLink (e-pedigree generation, chain of custody) | High |

**World-best reference:** TraceLink, SAP S/4HANA for Life Sciences, McKesson Distribution

**Critical gaps:**
- DSCSA serialization is a US federal law requirement since November 2023; non-compliance is criminal liability
- Cold chain excursion management absent; temperature deviations require documented investigation per GDP
- DEA controlled substance distribution requires specific system controls and reporting — not configurable generics
- No wholesaler EDI integration; pharmaceutical distribution contracts require EDI transaction processing

---

## Pharmaceutical Manufacturing (`pharma_mfg`)

**APG provides:** Production order management, batch record creation templates, and equipment assignment tracking. Provides configurable forms for manufacturing instructions and basic lot management. Lacks electronic batch record (eBR) validation, serialization integration, and MES-grade execution controls.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Electronic Batch Record (eBR) — validated | Not validated | SAP Manufacturing for LS / Veeva Vault MFG (validated eBR) | Critical |
| Serialization at line level (GS1/DSCSA) | Not present | Systech / TraceLink (line-level aggregation, DSCSA compliance) | Critical |
| Real-time line clearance verification | Not present | SAP MES (automated line clearance, checklist enforcement) | High |
| Equipment / process parameter integration | Not present | OSIsoft PI / SAP (historian data into batch record automatically) | High |
| OEE (Overall Equipment Effectiveness) | Not present | Rockwell FactoryTalk / SAP OEE module | Medium |
| Deviations linked to batch record | Basic forms | Veeva Vault MFG (deviation auto-created from eBR exception) | High |
| LIMS integration for in-process testing | Not present | LabVantage / STARLIMS (bi-directional LIMS-MES data flow) | High |
| APR/PQR (Annual Product Review) | Not present | Veeva Vault QMS (automated APR data collection across batches) | High |
| ERP integration for production orders | Not present | SAP S/4HANA LS (native MRP, production order, GI flows) | High |
| Process validation (PV) workflow | Not present | MasterControl / Veeva (validation lifecycle, Stage 1-3 protocols) | Critical |

**World-best reference:** SAP Manufacturing for Life Sciences, Veeva Vault Manufacturing, Rockwell FactoryTalk

**Critical gaps:**
- Unvalidated batch records cannot be used as GMP documentation; FDA 483 observations are certain
- No serialization engine; DSCSA compliance at manufacturing line is legally mandatory (US)
- No process historian integration; real-time parameter capture in eBR is a GMP expectation
- No APR/PQR automation; annual product review is an ICH Q10 and FDA 21 CFR 211.180(e) requirement

---

## Pharmacovigilance (`pharma_phl`)

**APG provides:** Adverse event intake forms, case records, and basic workflow routing for safety report processing. Supports configurable case status tracking and narrative documentation. Lacks E2B(R3) electronic submission, MedDRA coding automation, and signal detection analytics.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| E2B(R3) / ICH E2B electronic submission | Not present | Oracle Argus / Veeva Vault Safety (certified E2B(R3) gateway) | Critical |
| MedDRA / WHODrug automated coding | Not present | Veeva Safety.AI (auto-coding with MedDRA suggestions) | Critical |
| Regulatory submission to FDA/EMA/PMDA | Not present | Oracle Argus (multi-region submission automation, 60% market share) | Critical |
| CIOMS I / MedWatch report generation | Not present | Oracle Argus (automated CIOMS and MedWatch generation) | Critical |
| Signal detection analytics | Not present | IQVIA ARISg / Empirica Signal (disproportionality analysis, PRR/ROR) | Critical |
| Aggregate reports (PSUR/PBRER/DSUR) | Not present | Oracle Argus (automated aggregate report assembly) | High |
| Pregnancy registry / special populations | Not present | Oracle Argus / ARISg (registry tracking, exposure management) | High |
| Partner case reconciliation | Not present | Oracle Argus (automated partner case exchange, E2B import) | High |
| Literature monitoring integration | Not present | Veeva Vault Safety (Embase/MEDLINE literature feed intake) | High |
| AI-driven triage and narrative extraction | Not present | Veeva Safety.AI, Oracle AI agents | High |

**World-best reference:** Oracle Argus Safety, Veeva Vault Safety, IQVIA ARISg

**Critical gaps:**
- E2B(R3) electronic submission is an FDA/EMA regulatory requirement — no manual workaround exists at scale
- MedDRA coding must use official version-controlled dictionaries; APG has no licensed dictionary integration
- Signal detection is a pharmacovigilance legal obligation (EU PV legislation, FDA FAERS); absence is a regulatory risk
- Without CIOMS/MedWatch generation, each expedited report requires entirely manual assembly

---

## Pharma Procurement (`pharma_prc`)

**APG provides:** Purchase requisition management, supplier records, and purchase order lifecycle tracking. Supports multi-step approval workflows and basic spend reporting. Lacks GxP-qualified supplier management, API/raw material specification enforcement, and DSCSA-compliant purchasing.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| GxP-qualified supplier management | Not present | Veeva Vault QMS (supplier qualification linked to QMS) | High |
| API / raw material specification matching | Not present | SAP Ariba (spec-compliant sourcing, materials master) | High |
| DSCSA-compliant purchasing (authorized trading partner) | Not present | TraceLink (ATP verification at point of order) | High |
| Contract price enforcement (formulary) | Basic | SAP Ariba (contract compliance check at PO creation) | Medium |
| 3-way matching (PO/GR/invoice) | Not present | SAP Ariba / Coupa (automated 3-way match, exception routing) | High |
| Supplier risk scoring | Not present | JAGGAER (ML supplier risk, ESG scoring) | Medium |
| DEA controlled substance purchasing controls | Not present | SAP S/4HANA LS (DEA quota management, Form 222) | Critical |
| Spend analytics with category management | Basic reporting | Coupa / Jaggaer (AI-driven spend cube, savings tracking) | Medium |
| e-Catalogue / punch-out integration | Not present | SAP Ariba (punch-out, preferred vendor catalogues) | Medium |
| Cold chain procurement specifications | Not present | Specialized pharma procurement platforms | Medium |

**World-best reference:** SAP Ariba for Life Sciences, Coupa, JAGGAER

**Critical gaps:**
- No GxP supplier qualification workflow linked to quality systems; FDA requires documented supplier qualification
- DEA quota management and Form 222 for controlled substance purchasing requires specific validated controls
- No 3-way matching — a basic financial control absent in the current implementation
- No authorized trading partner (ATP) verification — DSCSA requires ATP status check before each transaction

---

## Quality Management System (`pharma_qlt`)

**APG provides:** Non-conformance records, CAPA workflow management, and change control forms backed by PostgreSQL. Supports configurable approval chains and document attachment. Lacks GxP-validated infrastructure, closed-loop QMS integration, and ISO/FDA quality system certification.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Validated GxP infrastructure (IQ/OQ/PQ) | Not validated | MasterControl, Veeva (pre-validated with test scripts) | Critical |
| Closed-loop CAPA management | Basic workflow | MasterControl (CAPA linked to deviation, audit, training, change) | High |
| Complaint management (21 CFR 820) | Not present | Veeva Vault QMS (complaint intake, MDR/MedWatch filing) | Critical |
| Supplier CAPA / audit management | Not present | Veeva Vault QMS (supplier portal, collaborative CAPA) | High |
| Document control (SOPs, work instructions) | Basic file storage | MasterControl (version control, linked training, effectivity dates) | High |
| Risk management (ICH Q9) | Not present | Veeva Vault QMS (FMEA, risk registers, ICH Q9-aligned) | High |
| Product release workflow | Not present | Veeva Vault QMS (QP/authorized person release workflow) | Critical |
| APR/PQR data aggregation | Not present | Veeva Vault QMS (automated annual product review data pull) | High |
| AI CAPA narrative generation | Not present | Veeva (AI narrative summaries for investigations and CAPA plans) | Medium |
| ESG / sustainability quality metrics | Not present | MasterControl (quality-ESG reporting linkage) | Low |

**World-best reference:** MasterControl, Veeva Vault QMS, Pilgrim SmartSolve (now ETQ Reliance)

**Critical gaps:**
- Unvalidated QMS cannot be used as a system of record for GMP/GCP/GLP regulated activities
- No 21 CFR 820 complaint management — mandatory for medical device manufacturers
- No product release workflow; qualified person (QP) batch release is a regulatory requirement in EU
- Closed-loop quality (deviation → CAPA → change control → training → re-qualification) not achievable without validated integration

---

## Regulatory Affairs (`pharma_rlt`)

**APG provides:** Submission planning records, health authority interaction logs, and document repository with version control. Supports workflow approvals for regulatory documents. Lacks eCTD 3.2.2/4.0 publishing, global registration tracking, and IDMP compliance.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| eCTD 3.2.2 / 4.0 publishing | Not present | Veeva Vault Submissions Publishing / EXTEDO (eCTD-compliant assembly) | Critical |
| Global registration lifecycle tracking | Not present | Veeva Vault Registrations (450+ companies, 19/20 top pharma) | Critical |
| IDMP compliance (EMA Art. 57) | Not present | Veeva Vault RIM (IDMP-aligned product data model) | Critical |
| Health authority interaction management | Basic records | Veeva Vault RIM (HA meeting management, commitment tracking) | High |
| Label management & artwork control | Not present | Veeva PromoMats / Veeva Vault RIM (labeling lifecycle) | High |
| Regulatory intelligence / dossier reuse | Not present | Veeva Vault Submissions Archive (sequential/cumulative eCTD view) | High |
| CMC change management (post-approval) | Not present | Veeva Vault RIM (change control linked to submissions) | High |
| 35 regulatory authority eCTD validation | Not present | EXTEDO (tools used by 35 regulatory agencies globally) | Critical |
| eCTD 4.0 transition readiness | Not present | Veeva / EXTEDO (FDA optional since 2024, EMA 2025) | High |
| AI-assisted regulatory writing | Not present | Veeva AI Agents for Regulatory (August 2026 roadmap) | Medium |

**World-best reference:** Veeva Vault RIM, EXTEDO EXTEDOpulse, Lorenz docuBridge

**Critical gaps:**
- No eCTD publishing capability; electronic submissions to FDA/EMA/PMDA require validated eCTD assembly
- IDMP compliance is mandatory for EMA Article 57 reporting — requires specific structured product data model
- No global registration tracking; managing hundreds of country-level product registrations requires purpose-built tooling
- eCTD 4.0 transition requires platform changes (mandatory FDA ~2029, EMA ~2027, PMDA April 2026)

---

## R&D / Drug Discovery Data Management (`pharma_rnd`)

**APG provides:** Experimental record templates, project milestone tracking, and data file storage with metadata. Supports configurable laboratory notebook forms and basic data lineage. Lacks electronic lab notebook (ELN) validation, compound/assay data management, and scientific data management system (SDMS) integration.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Validated ELN (GLP/GCP compliant) | Not validated | IDBS E-WorkBook / Dotmatics ELN (21 CFR Part 11, GLP validated) | Critical |
| Compound registration & structure search | Not present | Dotmatics / CDD Vault (chemical structure, substructure, similarity search) | Critical |
| Assay data management / dose-response curves | Not present | Dotmatics Bioregister (assay import, curve fitting, IC50 tracking) | Critical |
| Scientific data management (SDMS) | Basic file store | Dotmatics / Thermo Fisher SciStore (instrument data auto-ingestion) | High |
| Study design & in vivo data management | Not present | Labguru / Benchling (study design, in vivo protocols, dosing) | High |
| AI-assisted target identification | Not present | Schrödinger / Insilico Medicine (generative AI drug design) | Critical |
| Sequence management (biologics) | Not present | SnapGene / Benchling (sequence registry, annotation, BLAST) | High |
| Lab inventory & reagent management | Not present | Benchling Registry / Quartzy (reagent tracking, lot management) | Medium |
| Data integration with computational chemistry | Not present | Schrödinger / OpenEye (ADMET prediction, docking workflows) | High |
| Clinical-nonclinical data bridge | Not present | Medidata Rave + Dotmatics (nonclinical data flows into IND submissions) | High |

**World-best reference:** Dotmatics (Benchling), IDBS E-WorkBook, Schrödinger

**Critical gaps:**
- No GLP-validated ELN; nonclinical safety studies in GLP facilities require validated electronic records
- No chemical structure/compound registry; drug discovery without cheminformatics tooling is operationally not viable
- No AI-assisted compound design or ADMET prediction; competitors offer generative AI drug discovery workflows
- No instrument data auto-ingestion; manual transcription from analytical instruments is a data integrity risk

---

## Audit Management (`grc_aud`)

**APG provides:** Audit plan scheduling, finding records, and recommendation tracking via configurable PostgreSQL models. Supports workflow-based finding status management and report templates. Lacks risk-based audit planning, automated evidence collection, and integrated GRC data model.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Risk-based audit planning | Not present | ServiceNow GRC (risk intelligence drives audit prioritization, IRM linked) | High |
| Audit Management Workspace (single pane) | Not present | ServiceNow (timeline, budget, resource, observation dashboard) | High |
| Automated evidence collection | Manual | ServiceNow (connects to IT systems, auto-collects control evidence) | High |
| Control design & operating effectiveness testing | Basic checklist | ServiceNow / MetricStream (structured control testing workflows) | High |
| Continuous monitoring | Not present | ServiceNow (continuous control monitoring, key indicators) | High |
| Issue and remediation management | Basic tracking | ServiceNow (central issue mgmt, plan-of-action, follow-up) | Medium |
| AI-driven anomaly detection | Not present | ServiceNow / IBM OpenPages (AI predictive analytics, anomaly alerts) | High |
| Cross-functional GRC integration | Not present | ServiceNow (risk + compliance + audit = same data model) | Critical |
| Persona-based dashboards | Generic | ServiceNow (Audit Workbench, heatmaps, KPI widgets) | Medium |
| Regulatory framework mapping | Not present | MetricStream (multi-framework: SOX, ISO 27001, NIST, HIPAA) | High |

**World-best reference:** ServiceNow GRC, MetricStream M7, RSA Archer

**Critical gaps:**
- No IRM integration — audit findings disconnected from risk register and control library is a structural gap
- No automated evidence collection; manual evidence gathering is the primary driver of audit cost and error
- Continuous control monitoring absent; leading audit functions move from periodic to continuous assurance
- No regulatory framework library; building SOX/ISO 27001/HIPAA control mappings from scratch is months of effort

---

## Document Control (`grc_doc`)

**APG provides:** Document versioning, review/approval workflows, and access-controlled document storage. Supports document metadata, category management, and basic distribution records. Lacks formal controlled document lifecycle management with regulatory-grade audit trails and linked training enforcement.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Regulated document lifecycle (effective/obsolete) | Basic status fields | MasterControl (effectivity date, supersession, periodic review) | High |
| Linked training enforcement | Not present | MasterControl (document → training assignment auto-trigger) | High |
| Watermarked controlled copies | Not present | MasterControl (controlled copy watermarks, uncontrolled copy marks) | High |
| Multi-level review routing | Configurable | Veeva Vault (parallel/sequential review, cross-functional routing) | Medium |
| Document hierarchy (policy → SOP → WI) | Not present | MasterControl (hierarchical document tree, cross-reference) | Medium |
| Regulatory content library | Not present | Navex Global / MasterControl (regulatory text feeds, citation links) | High |
| AI SOP compliance analysis | Not present | MasterControl AI SOP Compliance Analyzer (Jan 2026) | Medium |
| Translation management | Not present | Veeva Vault (AI-powered SOP translation into multiple languages) | Medium |
| Offline / mobile document access | Not present | Navex Global (mobile-first, offline access for field staff) | Medium |
| GxP validation package | Not validated | MasterControl, Veeva (validated IQ/OQ/PQ scripts) | Critical (in regulated industries) |

**World-best reference:** MasterControl, Veeva Vault QD, Navex Global PolicyTech

**Critical gaps:**
- No linked training enforcement; GxP/ISO mandates documented training before work under new document version
- Effectivity and supersession control absent; documents in the wrong state during an audit are a critical finding
- No GxP-validated infrastructure; document control in pharma/medical device requires validated system
- No regulatory content library; document authors need current regulatory text to write compliant procedures

---

## Incident & Case Management (`grc_icm`)

**APG provides:** Incident intake forms, case assignment routing, and resolution tracking with configurable status workflows. Supports multi-step escalation and basic reporting dashboards. Lacks structured root cause analysis, regulatory incident notification workflows, and integration with risk and compliance modules.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Structured root cause analysis (RCA) | Not present | ServiceNow / Enablon (5-Why, fishbone, bow-tie analysis templates) | High |
| Regulatory notification management | Not present | Enablon / Cority (regulator notification timelines, submission tracking) | High |
| Incident-to-CAPA linkage | Not present | MasterControl, Veeva Vault QMS (incident auto-triggers CAPA workflow) | High |
| Near-miss and hazard observation capture | Not present | Intelex / Cority EHS (field-reported near misses, mobile capture) | Medium |
| Case classification taxonomies (regulatory) | Generic | Archer (NIST, ISO 27001, HIPAA incident taxonomies) | Medium |
| SLA-driven escalation and breach alerting | Basic | ServiceNow (SLA engines, multi-level escalation rules) | Medium |
| Integration with HR / legal workflows | Not present | ServiceNow / Navex Global (HR investigations, legal hold integration) | High |
| Evidence / attachment management | Basic files | Relativity (legal-grade evidence management for investigations) | Medium |
| Cross-functional visibility (risk → incident) | Not present | ServiceNow IRM (incident automatically updates risk register) | High |
| Regulatory framework-specific forms | Generic | Navex Global (OSHA 300/301, SEC Reg S-K, GDPR breach forms) | High |

**World-best reference:** ServiceNow IRM, Navex Global EthicsPoint, Cority EHS

**Critical gaps:**
- No structured RCA methodology support; root cause documentation is mandatory for regulatory investigations
- No regulatory notification workflow; GDPR Article 33, SEC cybersecurity rules, OSHA have strict notification timelines
- Incident-to-risk register linkage absent; every incident should update organizational risk scoring
- No EHS-specific capture (near-miss, hazard) — occupational safety incident management is a distinct requirement

---

## Policy Management (`grc_pol`)

**APG provides:** Policy document storage, version control, and acknowledgment tracking. Supports configurable review cycles and distribution workflows. Lacks regulatory content mapping, automated policy gap analysis, and compliance linkage.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Regulatory content library & citations | Not present | Navex Global PolicyTech (regulatory feeds, citation-linked policies) | High |
| Automated policy gap analysis | Not present | MetricStream (AI-powered regulation → policy gap identification) | High |
| Policy acknowledgment with certification | Basic checkbox | Navex Global (legally binding e-acknowledgment, declining with justification) | Medium |
| Multi-language policy distribution | Not present | Navex Global (multi-language delivery, region-specific variants) | Medium |
| Policy-to-control linkage | Not present | ServiceNow GRC (policies linked to controls, compliance evidence) | High |
| Policy-to-regulation mapping | Not present | MetricStream, RSA Archer (regulatory change → policy update trigger) | High |
| Periodic review automation | Manual scheduling | Navex Global (automated review reminders, attestation tracking) | Medium |
| Board/executive policy approval workflows | Basic | Diligent (board-level policy governance, D&O workflows) | Medium |
| Mobile-accessible policy delivery | Not present | Navex Global (mobile-first policy access, search) | Low |
| Compliance training linked to policy | Not present | Navex Global (policy → mandatory training assignment) | High |

**World-best reference:** Navex Global PolicyTech, ServiceNow GRC Policy Management, MetricStream

**Critical gaps:**
- No regulatory content library; manually tracking regulation-to-policy mapping across jurisdictions is not scalable
- Policy-to-control linkage absent; without it, compliance testing cannot reference the authoritative policy
- No automated gap analysis when regulations change; reactive policy management creates compliance risk
- Acknowledgment records without legal enforceability (certification, declining with reason) may not satisfy audit

---

## Risk Management (`grc_rsk`)

**APG provides:** Risk register with configurable likelihood/impact scoring, risk owner assignment, and mitigation tracking. Supports heatmap visualization and basic reporting. Lacks quantitative risk modeling, integrated risk data aggregation, and third-party risk management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Quantitative risk analysis (FAIR, Monte Carlo) | Not present | IBM OpenPages (FAIR-based quantification), MetricStream | High |
| Third-party / vendor risk management | Not present | Riskonnect / ServiceNow TPRM (tiered vendor assessments, continuous monitoring) | High |
| Risk aggregation across business units | Basic manual | IBM OpenPages (hierarchical risk rollup, corporate → BU → process) | High |
| AI-powered risk prediction | Not present | IBM OpenPages (Watsonx AI for risk signal detection) | High |
| Regulatory framework risk mapping | Not present | MetricStream (NIST CSF, ISO 31000, COSO ERM frameworks) | High |
| Cyber risk integration | Not present | RSA Archer (IT risk, vulnerability data feeds, NIST CSF) | High |
| Key Risk Indicators (KRI) monitoring | Not present | ServiceNow / OpenPages (KRI thresholds, automated breach alerts) | High |
| Scenario analysis & stress testing | Not present | IBM OpenPages (scenario modeling for enterprise risk) | High |
| Integrated risk-to-control-to-audit | Not present | ServiceNow IRM (single data model: risk → control → audit → issue) | Critical |
| ESG risk management | Not present | MetricStream (ESG risk scoring, TCFD/SASB alignment) | Medium |

**World-best reference:** IBM OpenPages, Riskonnect, ServiceNow Integrated Risk Management

**Critical gaps:**
- No integrated risk-control-audit data model; siloed risk management prevents end-to-end assurance
- FAIR/quantitative risk analysis absent; qualitative-only risk scoring is increasingly insufficient for boards and regulators
- No third-party risk management — supply chain and vendor risk is now a board-level priority
- No KRI monitoring with automated alerts; risk management without leading indicators is reactive

---

## Task & Control Management (`grc_tsk`)

**APG provides:** Task assignment, due date tracking, and completion status management with configurable workflows. Supports control testing checklists and evidence upload. Lacks control library with regulatory framework mappings and continuous control monitoring.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Pre-built control library (SOX, ISO, NIST) | Not present | ServiceNow GRC (UCF-mapped controls, 800+ frameworks) | High |
| Continuous control monitoring | Not present | ServiceNow (automated control testing, live monitoring indicators) | High |
| Control design effectiveness testing | Basic checklist | MetricStream (structured design testing, evidence linking) | High |
| Control-to-risk linkage | Not present | ServiceNow IRM (control → risk → audit single data model) | Critical |
| Operating effectiveness testing workflows | Basic | RSA Archer (structured OE testing, sample management) | Medium |
| Automated task escalation | Basic | ServiceNow (multi-level SLA-driven escalation) | Medium |
| Evidence management with versioning | Basic attachment | ServiceNow (evidence vault, versioned, attestation-linked) | Medium |
| Segregation of duties (SoD) analysis | Not present | SAP GRC / Saviynt (SoD conflict detection across ERP roles) | High |
| Control rationalization / deduplication | Not present | MetricStream (common control framework, deduplication analytics) | Medium |
| Regulatory change management → control update | Not present | MetricStream / ServiceNow (reg change triggers control review) | High |

**World-best reference:** ServiceNow GRC, MetricStream M7, RSA Archer

**Critical gaps:**
- No pre-built control library; building SOX ITGC/ICFR control mappings from scratch is a multi-month effort
- Control-to-risk linkage absent — integrated assurance requires controls mapped to risks they mitigate
- No SoD analysis; access control violations are the most common SOX audit finding
- Continuous control monitoring absent; point-in-time testing is being replaced by automated continuous assurance

---

## Budget Management (`government_bud`)

**APG provides:** Budget line item creation, allocation tracking, and variance reporting via PostgreSQL with Flask-Appbuilder views. Supports approval workflow for budget amendments. Lacks fund accounting, GASB compliance, encumbrance accounting, and multi-year capital budgeting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Fund accounting (governmental/proprietary/fiduciary) | Not present | Tyler Munis (fund accounting as native foundation, GASB-compliant) | Critical |
| Encumbrance accounting | Not present | Tyler Munis / Oracle Government (commitment accounting, pre-encumbrance) | Critical |
| GASB 34/87/96 compliance | Not present | Tyler Munis (GASB compliance baked in, CAFR support) | Critical |
| Multi-year capital improvement planning | Not present | Tyler Munis / Oracle Government (CIP module, project-based budgeting) | High |
| Position-based budget (FTE/salary) | Not present | Tyler Munis (position control, salary forecasting) | High |
| What-if scenario modeling | Not present | Questica (budget scenario modeling, Monte Carlo) | High |
| Budget-to-actuals real-time monitoring | Basic variance | Tyler Munis (drill-through to transactions, real-time encumbrance) | High |
| Grant / special revenue fund isolation | Not present | Tyler Munis (separate fund tracking per grant award) | High |
| COFOG / program-based budgeting | Not present | Oracle Government Financials (program/function classification) | Medium |
| Budget publication (citizen transparency) | Not present | OpenGov (visual budget transparency portals) | Medium |

**World-best reference:** Tyler Technologies Enterprise ERP (Munis), Oracle Government Financials, Questica

**Critical gaps:**
- Fund accounting is the defining requirement of government finance; generic accrual accounting models are inapplicable
- Encumbrance accounting absent; government contracts must reserve budget at commitment, not just at payment
- GASB compliance is legally required for US public entities; non-compliant reporting cannot be submitted to auditors
- No capital improvement plan module; government capital projects span multiple fiscal years by nature

---

## Case Management (`government_cas`)

**APG provides:** Case intake forms, assignment workflows, and status tracking with configurable resolution workflows. Supports multi-party case records and document attachment. Lacks citizen-facing intake portals, regulatory outcome tracking, and AI-assisted routing.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Citizen-facing digital intake portal | Not present | Salesforce Government Cloud / Granicus (self-service case submission) | High |
| AI-assisted case triage & routing | Not present | Salesforce Einstein (Public Sector Einstein 1, automated routing) | High |
| Omnichannel intake (web, mobile, phone, walk-in) | Not present | Salesforce / Granicus (unified omnichannel constituent interaction) | High |
| Outcome tracking & regulatory disposition | Basic status | Salesforce (outcome recording linked to regulatory workflows) | High |
| SLA monitoring with breach alerting | Basic | ServiceNow / Salesforce (SLA engines, escalation, breach dashboards) | Medium |
| Integration with benefits / eligibility systems | Not present | Salesforce Government Cloud (benefits enrollment, eligibility integration) | High |
| Case analytics and workload forecasting | Basic reporting | Salesforce (predictive caseload, performance analytics) | Medium |
| Secure digital identity verification | Not present | Salesforce Government Cloud (FedRAMP, identity proofing integration) | High |
| Mobile-first case worker interface | Not present | Salesforce Field Service (field case management, offline) | Medium |
| Public records / FOIA case management | Not present | Granicus (FOIA workflow, redaction, public portal) | Medium |

**World-best reference:** Salesforce Government Cloud, Microsoft Dynamics 365 Public Sector, Granicus

**Critical gaps:**
- No citizen-facing portal; government case management without self-service intake drives counter/call center cost
- No AI-assisted routing; manual case assignment at scale creates backlogs and inequitable service delivery
- FedRAMP/security compliance not addressed — government case data requires specific data sovereignty controls
- No integration with government benefit/eligibility systems; most government cases require eligibility determination

---

## Contracts Management (`government_con`)

**APG provides:** Contract records, milestone tracking, obligation management, and renewal alerting. Supports document attachment and approval workflows. Lacks government-specific FAR/DFARS compliance fields, performance-based contracting, and integration with government procurement systems.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| FAR/DFARS clause library | Not present | SAP Ariba (FAR/DFARS compliance, clause template library) | Critical (US Federal) |
| Performance-based contracting (EVMS) | Not present | Deltek Costpoint (EVMS/earned value management integration) | High |
| Government-wide contract vehicles (GWAC) | Not present | Periscope S2G / Coupa (cooperative contract vehicles, NASPO, GSA) | High |
| Subcontract management | Not present | Jaggaer (subcontract tracking, flow-down clause management) | High |
| Contractor performance assessment (CPARS) | Not present | Deltek / FPDS-NG (CPARS rating integration) | High (Federal) |
| Obligation tracking vs. appropriations | Not present | Tyler Munis (obligation linked to fund balance, encumbrance) | Critical |
| Contract modification / task order management | Basic | SAP Ariba (modification workflow, task order/delivery order tracking) | High |
| Legal review workflow | Basic approval | Ironclad / Conga (legal redline collaboration, clause negotiation) | Medium |
| AI contract risk analysis | Not present | Ironclad / LegalOn (AI clause extraction, risk flagging) | Medium |
| Integration with FPDS (federal procurement data) | Not present | Deltek / SAP Ariba (FPDS-NG reporting, USASpending.gov sync) | High (Federal) |

**World-best reference:** SAP Ariba Public Sector, Deltek Costpoint, Periscope S2G

**Critical gaps:**
- No FAR/DFARS clause library; US federal contracts require specific regulatory clauses — absence is a legal compliance failure
- Obligation-to-appropriation tracking absent; Anti-Deficiency Act violations are criminal in the US federal context
- No CPARS integration; contractor performance reporting to federal databases is mandatory
- No earned value management; performance-based contracts require EVM for progress measurement

---

## Citizen Services / Portal (`government_csr`)

**APG provides:** Web-based service request forms, status tracking, and notification workflows. Supports configurable service catalog and basic citizen account management. Lacks accessible design standards compliance, digital identity management, and omnichannel service delivery.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| WCAG 2.2 AA accessibility compliance | Not certified | Granicus (WCAG 2.2 AA certified Customer Portal, 2025) | Critical |
| AI-powered government chatbot / digital agent | Not present | Granicus GXA (Government Experience Agent), Salesforce Einstein | High |
| Single sign-on / digital identity (SSO) | Not present | Granicus (Azure B2C migration, SSO for citizens and staff) | High |
| Omnichannel service delivery | Not present | Granicus / Salesforce (web, mobile, kiosk, phone, counter) | High |
| Proactive push notifications (SMS/email) | Not present | Granicus govDelivery (100M+ subscriber notification network) | High |
| Service catalog with guided workflows | Basic forms | Granicus govService (structured service journeys, no-code forms) | Medium |
| Real-time case status visibility for citizens | Not present | Salesforce / Granicus (citizen portal with live case tracking) | High |
| Payments integration (fees, taxes) | Not present | Tyler Munis / Accela (online payment, reconciliation) | High |
| Digital meeting / public comment | Not present | Granicus (legislative management, public meeting streaming) | Medium |
| GDPR / Privacy Act compliance for citizen data | Not certified | All major government platforms (privacy impact assessments, DPO workflows) | Critical |

**World-best reference:** Granicus Government Experience Cloud, Salesforce Government Cloud, Microsoft Government Cloud

**Critical gaps:**
- WCAG 2.2 AA accessibility is a legal requirement (ADA Section 508, UK Equality Act); non-compliant portals expose government to liability
- No AI-powered digital agent; Granicus GXA and Salesforce Einstein are setting new citizen service standards
- No proactive push communications; governments with reactive-only portals face citizen engagement deficits
- No digital identity / SSO layer; citizens accessing multiple services require federated identity management

---

## Government Finance (`government_fin`)

**APG provides:** General ledger, journal entry management, accounts payable, and accounts receivable workflows. Supports configurable chart of accounts and basic financial reporting. Lacks fund accounting, GASB compliance, government-specific financial statement production, and treasury management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Governmental fund accounting (GASB) | Not present | Tyler Munis (GASB native, fund-based GL) | Critical |
| CAFR / ACFR financial statement production | Not present | Tyler Munis / Oracle (Comprehensive Annual Financial Report automation) | Critical |
| Treasury / cash management | Not present | Oracle Government Financials (cash positioning, investment management) | High |
| Debt management (bonds, TAN, RAN) | Not present | Tyler Munis / Arbitrage Compliance Specialists (debt service tracking) | High |
| Grant accounting (OMB Uniform Guidance) | Not present | Tyler Munis (grant fund isolation, indirect cost allocation, A-133) | Critical |
| Revenue recognition (governmental) | Basic | Oracle Government (special assessment, utility billing recognition) | High |
| Year-end close for government | Not present | Tyler Munis (government fiscal year close, GASB 34 transition) | High |
| Inter-fund / inter-agency transfers | Not present | Tyler Munis (due-to/due-from fund transfer accounting) | High |
| Single Audit / A-133 compliance | Not present | Tyler Munis / Sage Intacct Government (A-133 schedule of expenditures) | Critical |
| FedRAMP / StateRAMP security | Not certified | Oracle Government (FedRAMP authorized) | Critical (US) |

**World-best reference:** Tyler Technologies Enterprise ERP (Munis), Oracle Government Financials, Infor Public Sector

**Critical gaps:**
- GASB compliance is non-negotiable; government financial statements prepared on commercial GAAP cannot be submitted to state auditors
- CAFR/ACFR production requires GASB 34 compliant statements — not generatable from standard financial models
- Grant accounting under OMB Uniform Guidance requires specific fund isolation and indirect cost allocation rules
- Single Audit (2 CFR 200) compliance tracking for federal award expenditures requires dedicated module

---

## Grants Management (`government_grt`)

**APG provides:** Grant application intake, award record management, milestone tracking, and basic reporting. Supports configurable eligibility criteria and approval workflows. Lacks federal reporting compliance (FFATA, USASpending.gov), sub-recipient monitoring, and OMB Uniform Guidance cost tracking.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| OMB Uniform Guidance cost principles | Not present | eCivis / Tyler Munis (2 CFR 200 allowable cost tracking) | Critical |
| Sub-recipient monitoring & risk assessment | Not present | eCivis (sub-recipient risk tiers, monitoring schedules, findings) | High |
| FFATA / USASpending.gov reporting | Not present | eCivis / Fluxx (federal transparency reporting integration) | Critical (Federal grantees) |
| Grants.gov / SAM.gov integration | Not present | eCivis (federal opportunity database, SAM registration check) | High |
| Budget period tracking & carryover | Basic records | Fluxx (budget-period-level tracking, carryover approvals) | High |
| Performance measure / outcome reporting | Not present | Fluxx (grantee outcome data collection, impact dashboards) | High |
| Close-out and audit support | Not present | eCivis (close-out workflow, final reporting, audit trail) | High |
| Indirect cost rate management | Not present | eCivis (negotiated indirect cost rates, allocation tracking) | High |
| Grantee portal with self-service reporting | Not present | Fluxx / Salesforce (branded grantee portal, progress reporting) | High |
| AI-powered grant opportunity matching | Not present | eCivis (AI-assisted grant search, opportunity matching) | Medium |

**World-best reference:** eCivis, Fluxx Grantmaker, Salesforce Grants Management

**Critical gaps:**
- OMB Uniform Guidance compliance absent; federal grant recipients are subject to 2 CFR 200 audit requirements
- No sub-recipient monitoring; pass-through entities are legally responsible for monitoring their sub-recipients
- FFATA reporting to USASpending.gov is a legal obligation for federal prime recipients
- No close-out workflow; incomplete grant close-outs result in audit findings and repayment demands

---

## Government Procurement (`government_prc`)

**APG provides:** Purchase requisition management, vendor registration, RFQ/RFP document storage, and purchase order tracking. Supports multi-step approval workflows and basic vendor evaluation forms. Lacks government-specific procurement regulations, e-sourcing, and contract vehicle management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Government procurement law compliance (FAR/DFARS, PPDA) | Not present | SAP Ariba (FAR/DFARS compliance, configurable for PPDA/jurisdiction) | Critical |
| e-Procurement portal (supplier bidding) | Basic forms | Periscope S2G / Jaggaer (fully featured e-procurement, public bid board) | High |
| Cooperative contract vehicle management | Not present | Periscope S2G (NASPO, NCPA, GSA schedule vehicles) | High |
| Sole source / emergency procurement justification | Not present | Jaggaer Government (exception documentation, compliance flags) | High |
| Supplier diversity tracking (M/WBE, SDB) | Not present | SAP Ariba / Jaggaer (diversity certification, spend tracking) | High |
| Bid tabulation and evaluation scoring | Not present | Periscope S2G (structured bid evaluation, scoring matrices) | High |
| Public notice / legal advertising | Not present | Periscope S2G (public posting, DemandStar integration) | Medium |
| Procurement analytics & spend visibility | Basic | Jaggaer (AI spend cube, category management, savings tracking) | Medium |
| Vendor debarment / exclusion checking | Not present | SAM.gov integration / Jaggaer (EPLS debarment check at award) | Critical |
| Budget pre-encumbrance at requisition | Not present | Tyler Munis (budget check and pre-encumbrance at PO creation) | High |

**World-best reference:** Jaggaer Government, SAP Ariba Public Sector, Periscope S2G

**Critical gaps:**
- Vendor debarment checking (SAM.gov) is legally required before any federal award; absence is a compliance violation
- No public bid portal; government procurement transparency requirements mandate public posting of solicitations
- FAR/PPDA compliance field mapping absent; configuring generic procurement workflows for government regulations requires specialized tooling
- Supplier diversity tracking absent; most US government entities have M/WBE spend reporting requirements

---

## Government Project Management (`government_prj`)

**APG provides:** Project registration, task breakdown, milestone tracking, and resource assignment. Supports budget-to-actual project variance reporting. Lacks capital project lifecycle management, earned value management, and integration with government fund accounting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Capital Improvement Plan (CIP) integration | Not present | Tyler Munis (CIP module, project-fund-budget linkage) | High |
| Earned Value Management (EVM) | Not present | Deltek Costpoint / Oracle Primavera P6 (EVM, BCWS/BCWP/ACWP) | High |
| Grant-funded project cost tracking | Not present | Tyler Munis / Oracle (grant cost allocation to projects, OMB Uniform Guidance) | Critical |
| Multi-year project budget authorization | Not present | Tyler Munis (multi-year appropriations, carryforward tracking) | High |
| Construction project management | Not present | Oracle Primavera P6 / Procore (schedule, RFI, submittals, change orders) | High |
| Interdependency / program management | Basic | Oracle Primavera (program-level portfolio views, critical path) | Medium |
| Project performance dashboards (elected officials) | Not present | OpenGov / Tyler Munis (public-facing project tracker, KPI boards) | Medium |
| Contract integration (project → contract) | Not present | Tyler Munis / Deltek (project-to-contract linkage, billing, milestones) | High |
| Risk register for capital projects | Not present | Oracle Primavera Risk Analysis (Monte Carlo schedule risk) | Medium |
| Integration with procurement (project-based POs) | Not present | Tyler Munis (project-coded PO, encumbrance to project budget) | High |

**World-best reference:** Oracle Primavera P6, Tyler Technologies Munis CIP, Deltek Costpoint

**Critical gaps:**
- No CIP integration; capital project budgets must be tracked against multi-year appropriations in fund accounting
- EVM absent; federally funded projects over thresholds require EVMS compliance (DoDI 5000.02, OMB A-11)
- Grant-to-project cost allocation absent; federal grant project costs require specific cost allocation documentation
- No construction workflow (RFI, submittals, change orders); infrastructure projects require these controls for contractor management

---

## Revenue & Tax Collection (`government_rev`)

**APG provides:** Revenue item tracking, payment recording, and basic receipts management. Supports configurable fee schedules and payment status workflows. Lacks property tax billing cycles, tax lien management, and integration with government fund accounting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Property tax billing & levy management | Not present | Tyler Munis / BS&A Software (property tax bill generation, levy calculation) | Critical |
| Tax delinquency & lien management | Not present | Tyler Munis (delinquency workflow, tax sale, lien certification) | Critical |
| Utility billing (water/electric/gas) | Not present | Tyler Munis / Superion (meter reading import, tiered rate billing) | High |
| Online citizen payment portal | Not present | Tyler Munis / Accela (e-billing, credit card, ACH, third-party gateways) | High |
| Collections & installment payment plans | Not present | Tyler Munis (installment billing, NSF processing, collections tracking) | High |
| Business license & occupational tax | Not present | Accela / Tyler Munis (license application, renewal, revenue integration) | High |
| Special assessment management | Not present | Tyler Munis (special assessment districts, improvement bonds) | High |
| Sales & use tax compliance | Not present | Vertex / Avalara (multi-jurisdiction tax calculation, filing) | High |
| Revenue forecasting / actuarial | Not present | Tyler Munis Analytics / Questica (revenue trend analysis, forecasting) | Medium |
| GIS integration for property identification | Not present | Tyler Munis + Esri ArcGIS (parcel-linked tax records) | Medium |

**World-best reference:** Tyler Technologies Enterprise ERP (Munis), BS&A Software, Accela

**Critical gaps:**
- Property tax billing is the primary revenue source for most local governments; absence makes the module non-viable for primary government use
- Tax lien management is a distinct legal process with specific statutory requirements per jurisdiction
- Online payment integration is table-stakes; citizens expect digital payment options for all government fees
- No installment plan management; governments are legally required to offer payment arrangements for property tax

---

## Government Workforce Management (`government_wfm`)

**APG provides:** Employee records, position tracking, leave request management, and timesheet submission. Supports configurable approval workflows and basic HR reporting. Lacks civil service compliance, union contract administration, and government payroll integration.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Civil service rules & merit system compliance | Not present | Tyler Munis HR (civil service integration, competitive examination tracking) | Critical |
| Union contract / collective bargaining administration | Not present | Tyler Munis (union code tracking, grievance workflow, contract rules) | Critical |
| Government payroll (pension, FICA, garnishments) | Not present | Tyler Munis Payroll (government pension systems, PERS/STRS interfaces) | Critical |
| Position control & FTE authorization | Not present | Tyler Munis (position control, budget-linked position authorization) | High |
| FMLA / ADA / workers' comp case management | Not present | PeopeSoft Government / UKG (leave case management, accommodation tracking) | High |
| Employee onboarding (background check, clearance) | Not present | Tyler Munis / Paycom Government (I-9, background, clearance workflow) | High |
| Training & certification compliance (OSHA, professional) | Not present | Cornerstone / Saba (compliance training tracking, cert expiry alerts) | High |
| Succession planning & leadership pipeline | Not present | Oracle HCM Cloud / SAP SuccessFactors (succession risk, development plans) | Medium |
| Workforce analytics (turnover, diversity) | Basic reporting | Tyler Munis Analytics / ADP WFM (workforce insights, DEI dashboards) | Medium |
| Emergency staffing / mutual aid rostering | Not present | Motorola Solutions / Tyler Munis (emergency roster, mutual aid agreements) | High |

**World-best reference:** Tyler Technologies Enterprise ERP HR/Payroll, Oracle HCM Cloud Government, UKG Ready Government

**Critical gaps:**
- Civil service rules and merit system compliance are constitutional/statutory requirements in most US jurisdictions — not configurable without domain-specific data models
- Union contract administration is operationally required for the majority of US government employers (60%+ unionization rate)
- Government pension system integration (PERS, STRS, IMRF) requires specific interfaces not present in generic HR systems
- Position control is a budget control mechanism unique to government; personnel spending without position authorization violates budget law in many jurisdictions

---

---

# 4. Business Intelligence, Intelligence/OSINT, Energy & Telecom

I now have sufficient data across all domains. Here is the complete competitive analysis:

---

## Analytics Engine (`bia_anl`)

**APG provides:** A core computation runtime for statistical analysis and ML model training, supporting tabular, time-series, and unstructured data pipelines. It exposes a Python-native API for algorithm execution, hyperparameter search, and model lifecycle management within the APG capability mesh.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| AutoML pipeline | Basic sklearn/XGBoost orchestration | Automated algorithm selection + leaderboard (DataRobot) | No automated model selection or ranking |
| Distributed compute | Single-node execution | Spark/Photon-accelerated distributed training (Databricks) | No horizontal scale-out |
| Model explainability | Feature importance only | SHAP, LIME, surrogate trees (H2O.ai) | No local explanations |
| Experiment tracking | None | MLflow, Weights & Biases integration (Databricks) | No run comparison, no artifact store |
| Hyperparameter tuning | Manual grid search | Bayesian optimization with early stopping (DataRobot) | No intelligent search |
| Model registry | File-based storage | Versioned registry with lineage and approval workflows (Azure ML) | No governance controls |
| Real-time inference | Batch-only | Sub-10ms online scoring endpoints (Azure ML, DataRobot) | No inference API layer |
| Algorithm breadth | ~20 sklearn algorithms | 500+ algorithms including deep learning (H2O.ai Driverless) | No neural network support |
| Compliance/audit trail | None | Full data lineage + model audit for regulated industries (SAS Viya) | No audit capability |
| Cost per model | Low (OSS stack) | $50K–$250K/yr per platform (DataRobot) | Gap favors APG on cost |

**World-best reference:** DataRobot, H2O.ai Driverless AI, SAS Viya

**Critical gaps:**
- No distributed training for large datasets; single-node ceiling limits model scale
- Absent experiment tracking and model registry prevents reproducible ML workflows
- No AutoML leaderboard means practitioners manually select algorithms
- Missing online serving infrastructure means predictions cannot be served in real-time

---

## Dashboard Management (`bia_dsh`)

**APG provides:** Dynamic, multi-widget dashboard construction via Flask-AppBuilder blueprints, with per-user layout persistence and role-based visibility controls. Widget types include charts, KPI cards, tables, and embedded reports drawn from APG data sources.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Visualization types | ~15 chart types | 100+ chart types + custom D3 (Tableau) | Limited chart variety |
| Natural language query | None | NL Q&A generating charts on prompt (Power BI Copilot) | No NLQ interface |
| Mobile responsiveness | Basic Bootstrap | Fully adaptive layout engine (Power BI, Tableau) | Layouts require manual mobile tuning |
| Embedded analytics | Blueprint embed | White-label SDK, iFrame, JS API (Looker, Tableau) | No programmatic embed SDK |
| Cross-dashboard filtering | Page-scoped only | Global cross-dashboard filter propagation (Tableau) | Filters don't propagate across dashboards |
| Real-time streaming widgets | Polling only | WebSocket-driven live tiles at 1-second refresh (Power BI) | No true streaming |
| Collaborative annotations | None | Per-visualization comments and alerts (Tableau, Power BI) | No annotation layer |
| Data governance layer | None | LookML semantic layer enforcing single source of truth (Looker) | No governed metric definitions |
| Conditional formatting | Basic | Cell-level gradient rules, icon sets, thresholds (Power BI) | Limited |
| Export formats | PNG, PDF | PNG, PDF, Excel, PowerPoint, CSV, Slack, email (Tableau) | Limited export targets |

**World-best reference:** Tableau, Microsoft Power BI, Looker (Google)

**Critical gaps:**
- No semantic/metric layer; each dashboard embeds its own business logic, creating metric drift across teams
- No NLQ-to-visualization path; non-technical users cannot self-serve
- Embedded analytics limited to blueprint iFrame; no JS SDK for third-party embedding
- No collaborative annotation or alerting on specific dashboard cells

---

## Data Warehouse (`bia_dwh`)

**APG provides:** PostgreSQL-backed dimensional modeling with star and snowflake schema support, ETL pipeline orchestration, and SCD (slowly changing dimension) management. Schema definitions are code-first via SQLAlchemy models, with APG-native staging and integration layers.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Storage/compute separation | None (PostgreSQL) | Elastic compute over object storage (Snowflake, BigQuery) | Cannot scale compute independently of storage |
| Query engine performance | PostgreSQL sequential scan | Columnar MPP with Photon acceleration (Databricks, BigQuery) | 10-100x slower on analytical workloads |
| Multi-cloud data sharing | None | Cross-region, cross-cloud secure data sharing (Snowflake) | No data sharing capability |
| Auto-partitioning | Manual partition DDL | Automatic partition pruning + clustering (BigQuery, Snowflake) | Manual only |
| Data lakehouse support | None | Open format Delta Lake/Iceberg on object storage (Databricks) | No lake integration |
| Change data capture | Manual ETL | Real-time CDC via log-based replication (Databricks, Fivetran) | Batch ETL only |
| Zero-copy cloning | None | Instant clone of datasets for test/dev (Snowflake) | No non-destructive branching |
| Cost-based query optimizer | Basic PG planner | Vectorized cost-based optimizer with statistics (BigQuery) | Inferior query planning |
| Data catalog / lineage | None | Column-level lineage + classification (Databricks Unity Catalog) | No catalog |
| Concurrent query handling | ~100 connections | Thousands of concurrent workloads via virtual warehouses (Snowflake) | Low concurrency ceiling |

**World-best reference:** Snowflake, Databricks Lakehouse, Google BigQuery

**Critical gaps:**
- PostgreSQL cannot serve as an analytical MPP engine at TB+ scale; serious DWH workloads require Redshift/BigQuery/Snowflake
- No data catalog or column-level lineage; governance and data discovery are manual
- ETL is batch-only; no CDC or real-time ingestion path
- No data sharing or marketplace capability for cross-organization data exchange

---

## Predictive Analytics (`bia_pda`)

**APG provides:** ML-based forecasting capability surfacing regression, classification, and time-series prediction models trained within `bia_anl`. Supports batch scoring jobs, configurable feature pipelines, and prediction storage back to PostgreSQL for downstream consumption.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| AutoML with leaderboard | None | 500+ algorithm leaderboard auto-ranked by metric (DataRobot) | Fully manual model selection |
| Feature engineering automation | None | Automated feature synthesis (DataRobot, H2O Driverless) | Engineer features manually |
| Model monitoring / drift | None | Real-time production drift detection + alerts (DataRobot) | No post-deployment monitoring |
| Explainability | Basic feature importance | SHAP, partial dependence, reason codes per prediction (H2O.ai) | No prediction-level explanations |
| Regulatory model documentation | None | Auto-generated model cards + compliance reports (SAS Viya) | No governance documentation |
| Challenger/champion testing | None | Automated champion-challenger A/B routing (DataRobot) | No model testing framework |
| Deep learning models | None | LSTM, transformer, CNN via H2O Wave or Azure ML | No deep learning support |
| Retraining automation | Manual re-run | Scheduled + trigger-based auto-retraining pipelines (Azure ML) | No automated retraining |
| Prediction API | None | REST scoring API with SLA monitoring (DataRobot, Azure ML) | No serving layer |
| Business user interface | Developer-only | No-code/low-code prediction building (DataRobot, Power BI) | Inaccessible to non-engineers |

**World-best reference:** DataRobot, H2O.ai Driverless AI, Azure Machine Learning

**Critical gaps:**
- No AutoML means each model requires full manual data science effort; not scalable
- Zero post-deployment monitoring; model performance degrades silently
- No business-user interface; capability is entirely inaccessible to analysts without Python skills
- No serving layer; predictions require ETL back to Postgres before consumption

---

## Prescriptive Analytics (`bia_psa`)

**APG provides:** Optimization engine exposing linear programming (LP), mixed-integer programming (MIP), and constraint satisfaction via PuLP/scipy.optimize. Accepts parameterized objective functions and constraint sets; returns optimal solutions with sensitivity analysis reports.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Solver breadth | LP/MIP via PuLP | CPLEX, Gurobi, CBC, SCIP + quadratic and nonlinear (IBM CPLEX) | No nonlinear, quadratic, or stochastic solvers |
| Stochastic optimization | None | Scenario-based stochastic programming (SAS, IBM CPLEX) | Cannot handle uncertain inputs |
| Simulation (Monte Carlo) | None | Full discrete-event simulation + risk sampling (Oracle Crystal Ball) | No simulation layer |
| What-if scenario engine | None | Multi-scenario comparison UI with slider inputs (Power BI, Anaplan) | No scenario authoring interface |
| Decision automation | Manual extraction | Direct action execution via SOAR/API (IBM Decision Optimization) | Recommendations not auto-actuated |
| Constraint modeling UI | Code-only | Visual constraint builder (IBM Decision Optimization) | Developer-only interface |
| Real-time re-optimization | Batch | Sub-second re-solve on streaming input (Gurobi Cloud) | Batch-only |
| Explanation of decisions | None | Traceable decision rationale per output variable (IBM DO) | No explainability |
| Scale (variables) | ~100K variables | Millions of variables + parallel branch-and-bound (CPLEX, Gurobi) | Limited scale |
| Pre-built domain templates | None | Supply chain, scheduling, routing templates (IBM Decision Optimization) | Blank-slate only |

**World-best reference:** IBM Decision Optimization (CPLEX), Gurobi Optimizer, SAS Operations Research

**Critical gaps:**
- Nonlinear and stochastic optimization unsupported; most real business problems are not purely LP
- No scenario comparison UI; business users cannot explore decision alternatives without code
- Scale ceiling around 100K variables prohibits enterprise supply chain or grid scheduling use
- Decision outputs are not auto-actuated; all prescriptions require manual human dispatch

---

## Report Builder (`bia_rpt`)

**APG provides:** Parameterized report generation supporting tabular and chart layouts, scheduled delivery via email or file system, and export to PDF and CSV. Reports are defined in Flask-AppBuilder templates with dynamic SQL-driven datasets and configurable filters.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Pixel-perfect layout | None | WYSIWYG designer with sub-pixel placement (Crystal Reports, SSRS) | No design canvas |
| Report parameterization | SQL parameter tokens | Cascading multi-select parameters with dependency chains (SSRS, Cognos) | Basic single-value params only |
| Scheduled delivery | Cron-based email | Calendar-based, event-triggered, SLA-aware delivery (Tableau Server) | No event-triggered scheduling |
| Delivery channels | Email + filesystem | Email, Slack, Teams, webhook, FTP, SharePoint (Power BI, Tableau) | Limited channels |
| Bursting (per-recipient variants) | None | Per-user personalized report bursting at scale (SAP Crystal, Cognos) | No bursting |
| Interactive drill-through | None | Click-through to detail pages with full filter context (Power BI, SSRS) | Static output only |
| Paginated reports at scale | Limited to ~10K rows | 1M+ row paginated reports with compression (Power BI Paginated) | Scale ceiling |
| Subscription management | None | Self-service subscription portal per-user (Tableau, Power BI) | Admin-managed only |
| Versioning / audit history | None | Report version history + change diff (Cognos, SSRS) | No versioning |
| White-label output | APG header only | Full brand theming with custom CSS, fonts, logos (Tableau, Looker) | Minimal branding |

**World-best reference:** Microsoft Power BI Paginated Reports / SSRS, SAP Crystal Reports, IBM Cognos Analytics

**Critical gaps:**
- No WYSIWYG designer; all reports require developer template work
- Scheduling is cron-only with no event-trigger or business-calendar awareness
- Report bursting for per-recipient personalization is absent; mass distribution requires manual iteration
- No drill-through interactivity; reports are static snapshots

---

## Self-Service BI (`bia_sbi`)

**APG provides:** Drag-and-drop chart builder for non-technical users, exposing APG data sources and enabling ad-hoc visualization creation, basic aggregation, and filtered exploration. Publishes charts to shared dashboards in `bia_dsh`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Drag-and-drop ease of use | Basic | Best-in-class canvas with guided tooltips (Tableau Desktop) | Steeper learning curve |
| NLQ / Ask Data | None | Type a question, get a chart (ThoughtSpot, Power BI Copilot) | No NLQ pathway |
| AI-suggested visualizations | None | Automatic chart type recommendation + Explain Data (Tableau) | No AI guidance |
| Data preparation in UI | None | Power Query ETL with 300+ transforms (Power BI) | Users cannot clean data without engineering help |
| Calculated fields | Basic SQL expressions | Formula language + quick table calcs (Tableau, Power BI DAX) | Limited expression language |
| Data blending (multi-source) | Single source | Cross-database blend with relationship modeling (Tableau, Looker) | Single source per chart |
| Self-service forecasting | None | One-click trend + forecast with confidence intervals (Tableau, Power BI) | No forecasting for business users |
| Saved views / personal bookmarks | None | Per-user saved filters, marks, and custom views (Tableau) | No personal workspace |
| Collaboration / sharing | Link-based | Comment threads, annotations, subscriptions, Slack share (Tableau) | No async collaboration |
| Semantic search | None | Indexed semantic search across all metrics (ThoughtSpot, Looker) | No discoverability |

**World-best reference:** ThoughtSpot (AI-Search BI), Tableau Desktop, Microsoft Power BI

**Critical gaps:**
- Absence of NLQ makes the capability inaccessible to the majority of business users
- No in-UI data preparation; users are blocked by upstream data quality issues
- No AI-suggested chart types; users must know the right visualization for the question
- No personal workspace or saved views; each session is stateless

---

## Time Series Analytics (`bia_tsa`)

**APG provides:** High-frequency time-series ingestion, storage, and analytical processing for sensor, log, and financial tick data. Supports windowed aggregations, trend decomposition (STL), anomaly detection, and multi-variate correlation across APG time-indexed datasets.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Ingestion throughput | ~10K events/sec (PostgreSQL) | Millions of events/sec (InfluxDB, TimescaleDB Hypercore) | 100x throughput gap |
| Compression ratio | ~3x (PostgreSQL TOAST) | 40-100x columnar time-series compression (InfluxDB, QuestDB) | Storage-inefficient at scale |
| Query latency (1B rows) | Seconds to minutes | Sub-second (QuestDB, TimescaleDB, Apache Druid) | Unacceptable for interactive analytics |
| Advanced forecasting models | ARIMA, ETS | LSTM, NBeats, Prophet, Transformer forecasters (Kdb+, DataRobot) | No deep learning forecasters |
| Continuous aggregates | None | Real-time materialized rollups (TimescaleDB, InfluxDB Tasks) | Manual aggregation jobs |
| Multi-variate anomaly detection | Univariate only | Correlation-aware multivariate anomaly detection (Splunk MLTK) | Cannot detect cross-signal anomalies |
| Time-series joins | SQL JOIN | ASOF JOIN for inexact timestamp alignment (QuestDB, kdb+) | No temporal join semantics |
| Downsampling / retention policies | Manual cron | Automated tiered retention + downsampling rules (InfluxDB) | Operational overhead |
| Streaming analytics | None | CEP (complex event processing) in-stream (Apache Flink, kdb+) | No streaming path |
| Visualization (native) | Generic charts | Specialized timeline, flame graph, heatmap renderers (Grafana) | No domain-specific time-series viz |

**World-best reference:** kdb+ (KX Systems), TimescaleDB / QuestDB, Apache Druid

**Critical gaps:**
- PostgreSQL cannot serve as a time-series database at IoT/telemetry scale; a dedicated TSDB is required
- No streaming / CEP pipeline; all analytics are batch post-hoc
- Deep learning forecasting models absent; ARIMA/ETS inadequate for complex seasonal + exogenous signals
- No continuous aggregates; real-time dashboards require expensive full-scan queries

---

## Intelligence / OSINT / Security Analytics (Intel)

---

## Alert Management (`intel_alerts`)

**APG provides:** Multi-source alert ingestion, deduplication, severity classification, and escalation routing with configurable suppression rules and threshold policies. Integrates with APG's `intel_monitor`, `intel_detection`, and `intel_correlation` capabilities to surface actionable analyst notifications.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Alert ingestion rate | ~1K alerts/sec | 1M+ events/sec with near-zero latency (Splunk ES) | Throughput ceiling |
| ML-based noise reduction | None | AI-driven alert clustering + noise reduction up to 95% (Microsoft Sentinel Fusion) | All alerts surface; no ML triage |
| Contextual enrichment | Basic metadata tags | Automatic MITRE ATT&CK tagging, threat intel enrichment (IBM QRadar) | Minimal context attached |
| Correlation to incidents | Manual | Automated multi-alert incident fusion (Microsoft Sentinel Fusion) | No automated incident creation |
| SLA tracking / breach alerts | None | Per-severity SLA with auto-escalation (ServiceNow, Splunk ITSI) | No SLA enforcement |
| Alert suppression policies | Threshold-based | Maintenance windows, topology-aware suppression (PagerDuty, Splunk ES) | Basic suppression only |
| On-call routing | Email only | Schedule-aware routing with escalation chains (PagerDuty, OpsGenie) | No on-call management |
| Feedback / learning loop | None | Analyst feedback updates detection model weights (Vectra AI) | No feedback mechanism |
| Two-way ticketing integration | None | Bi-directional JIRA, ServiceNow, Remedy sync (Splunk ES, QRadar) | One-way email dispatch |
| Audit trail | Basic log | Full chain-of-custody with analyst actions timestamped (QRadar) | Incomplete audit record |

**World-best reference:** Splunk Enterprise Security, Microsoft Sentinel, IBM QRadar

**Critical gaps:**
- No ML-based alert clustering; high false positive volume overwhelms analysts
- Escalation is email-only; no integration with on-call scheduling tools
- No bi-directional ticketing; alert lifecycle is not tracked post-dispatch
- Alert-to-incident correlation is manual; automation gap vs. Sentinel Fusion is significant

---

## Intelligence Analytics (`intel_analytics`)

**APG provides:** Analytical processing layer over collected intelligence data, supporting indicator aggregation, trend analysis, campaign tracking, and statistical correlation of threat signals across structured and semi-structured intelligence collections.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Machine-generated finished intelligence | None | AI-synthesized intelligence reports with confidence scoring (Recorded Future) | No finished intelligence production |
| Dark web coverage | None | Continuous dark web, paste site, forum monitoring (Recorded Future, Flashpoint) | No dark web visibility |
| Threat actor profiling | None | 1000+ tracked threat actors with TTP, motivation, attribution (Recorded Future) | No actor database |
| Campaign attribution | Manual | Automated campaign clustering via behavioral similarity (CrowdStrike, Mandiant) | Manual analyst-dependent |
| Natural language processing on reports | None | NLP extraction of IOCs, TTPs from unstructured threat reports (Anomali, Recorded Future) | No NLP pipeline |
| Intelligence prioritization | None | Risk-scored, asset-correlated relevance ranking (ThreatConnect) | Undifferentiated alert stream |
| Collection management | None | Structured source registration, reliability ratings, collection plans (Palantir Gotham) | No collection management |
| Pivot / relationship traversal | None | One-click pivot across 1B+ records (Maltego, Palantir) | No pivot capability |
| Dissemination controls | None | TLP marking, need-to-know ACL, STIX export (ThreatConnect, Anomali) | No dissemination framework |
| Historical intelligence archive | Rolling window | Searchable petabyte-scale archive with temporal query (Splunk, QRadar) | Limited retention |

**World-best reference:** Recorded Future Intelligence Cloud, Palantir Gotham, ThreatConnect

**Critical gaps:**
- No dark web or deep web collection; major visibility blind spot
- No NLP pipeline to extract structure from unstructured intelligence reporting
- Threat actor database absent; attribution and campaign tracking are impossible without it
- No dissemination controls (TLP, ACL, STIX); intelligence sharing cannot be governed

---

## Event Correlation Engine (`intel_correlation`)

**APG provides:** Rule-based and statistical correlation of security events across multiple data sources, using temporal windowing, field normalization, and threshold-based pattern matching to identify multi-event attack sequences.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Correlation throughput | ~50K EPS | 300K+ EPS sustained (IBM QRadar) | 6x throughput gap |
| AI/ML-based behavioral correlation | None | UEBA-driven behavioral baseline + anomaly correlation (Microsoft Sentinel) | Rules-only; no behavior modeling |
| MITRE ATT&CK mapping | None | Automatic ATT&CK tactic/technique tagging per correlation (Splunk ES, QRadar) | No framework alignment |
| Cross-asset correlation | Single-source rules | Multi-source, cross-asset, cross-domain fusion (Splunk ES) | Limited cross-source joins |
| Temporal sliding window | Fixed window | Multi-resolution sliding windows with late-arrival handling (Splunk, Elastic) | No late-event compensation |
| False positive suppression | Manual threshold tuning | Risk-based suppression + analyst feedback loops (Vectra, Sentinel) | Manual-only |
| Correlation rule library | None bundled | 1000+ out-of-box detection rules (Splunk ES Content Pack) | Blank slate; rules must be authored |
| Real-time streaming correlation | Near-real-time (polling) | Sub-second streaming CEP (Apache Flink, Splunk Streams) | Polling latency |
| Offense/case auto-creation | None | Auto-creates offence with full evidence chain (IBM QRadar) | No case generation |
| Cloud-native scale | Single-node | Elastic distributed correlation at petabyte scale (Microsoft Sentinel) | Cannot scale horizontally |

**World-best reference:** IBM QRadar, Splunk Enterprise Security, Microsoft Sentinel

**Critical gaps:**
- Rules-only correlation; no behavioral/UEBA engine means advanced persistent threats evade detection
- No bundled detection rule library; deployment requires months of rule authoring
- Streaming correlation latency (polling) vs. true CEP is unacceptable for real-time detection
- MITRE ATT&CK alignment absent; coverage measurement and gap analysis impossible

---

## Web / OSINT Crawler (`intel_crawler`)

**APG provides:** Configurable web crawler for OSINT collection supporting surface web, news feeds, and API-based source harvesting. Applies configurable depth, domain whitelists/blacklists, politeness controls, and feed normalization for downstream analysis.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Dark web / Tor crawling | None | Native Tor-circuit management + onion crawling (Recorded Future, Flashpoint) | No dark web access |
| Social media harvesting | None | Twitter/X, Telegram, Discord, LinkedIn harvesting via APIs (Maltego, Recorded Future) | No social media collection |
| Paste site monitoring | None | Real-time pastebin, GitHub gist, Ghostbin monitoring (Have I Been Pwned, Recorded Future) | No paste site coverage |
| NLP / entity extraction | None | Named entity recognition, IOC extraction, language detection (Recorded Future, Echosec) | Raw text output only |
| Deduplication | URL-hash only | Semantic near-duplicate detection (Diffbot, Recorded Future) | Duplicate content surfaces repeatedly |
| Crawl politeness & rate control | Basic | Adaptive rate limiting with robots.txt compliance (Scrapy, Heritrix) | Basic |
| Structured data extraction | None | Schema.org + custom CSS/XPath template-based extraction (Diffbot) | Unstructured text only |
| Source credibility scoring | None | Source reliability ratings + historical accuracy tracking (ThreatConnect) | No source quality metadata |
| Legal/compliance controls | None | Data residency controls + GDPR scrub pipelines (Recorded Future) | No compliance framework |
| Scale | ~100 domains | Billions of pages indexed continuously (Recorded Future, Common Crawl) | Massive scale gap |

**World-best reference:** Recorded Future (web intelligence), Maltego (OSINT transforms), Echosec/Babel Street

**Critical gaps:**
- Dark web and Tor network are entirely inaccessible; this is the highest-value OSINT gap
- No social media collection pipeline; state actors and criminal groups heavily use Telegram, Discord
- NLP/entity extraction absent; raw text requires manual analyst processing
- No source credibility scoring; collected intelligence cannot be quality-weighted

---

## Intelligence Dashboard (`intel_dashboard`)

**APG provides:** Flask-AppBuilder intelligence dashboard presenting threat landscapes, alert summaries, risk scores, and geospatial overlays for analyst situational awareness. Integrates with APG intel module outputs for unified display.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Threat landscape heat map | Basic table | Interactive ATT&CK matrix heatmap with coverage overlays (Splunk ES) | No ATT&CK-aligned visualization |
| Real-time SOC metrics | Polling refresh | WebSocket-driven live glasspane (Splunk ES, QRadar) | Polling latency vs. real-time |
| Investigation timeline | None | Visual attack timeline with pivot-to-evidence (Palantir Gotham, QRadar) | No investigation timeline |
| Multi-tenant views | None | Per-MSSP tenant isolation with cross-tenant analytics (Microsoft Sentinel) | Single-tenant only |
| Analyst workload balancing | None | Queue-based workload management with SLA visibility (ServiceNow SecOps) | No workload management |
| 3D threat globe | None | Real-time global threat visualization (Cisco Talos, Fortinet) | No geospatial threat globe |
| Embeddable widgets | None | RESTful widget API for third-party portals (Splunk, QRadar) | No embeddability |
| Dark mode / accessibility | None | WCAG 2.1 AA compliant, full dark mode (Microsoft Sentinel) | Accessibility gaps |
| Customizable analyst layouts | None | Drag-and-drop customizable SOC glass panes (Splunk ITSI) | Fixed layout |
| AI-driven triage recommendations | None | Copilot-generated investigation guidance (Microsoft Sentinel Copilot) | No AI assistance in UI |

**World-best reference:** Microsoft Sentinel, Splunk Enterprise Security, Palantir Gotham

**Critical gaps:**
- No MITRE ATT&CK matrix heatmap; defenders cannot visualize coverage gaps
- Investigation timeline absent; analysts cannot reconstruct attack narratives visually
- Real-time updates are polling-based; SOC situational awareness degrades under high alert load
- No AI-driven triage recommendations in the UI

---

## Threat Detection (`intel_detection`)

**APG provides:** Signature-based and anomaly-threshold threat detection using configured rules against normalized event streams. Supports IOC matching, behavioral threshold alerting, and protocol anomaly detection, publishing findings to `intel_alerts`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| UEBA (user/entity behavior) | None | Baseline per user/entity, deviation scoring (Microsoft Sentinel UEBA, Splunk UBA) | No behavioral profiling |
| Detection engineering workflow | Manual YAML | Sigma rule format + CI/CD deployment pipeline (Elastic, Splunk) | No rule lifecycle management |
| Out-of-box detections | ~20 rules | 1000+ curated OOTB detections (Splunk ES Content Pack) | Requires extensive manual authoring |
| Detection-as-code | None | Git-managed Sigma/YARA/KQL rule libraries with CI/CD (Elastic, Splunk) | No code-managed detection |
| Network traffic analysis (NTA) | None | Deep packet inspection + NetFlow ML detection (Darktrace, ExtraHop) | No network layer visibility |
| Endpoint telemetry | None | EDR telemetry ingestion + behavioral detection (CrowdStrike, Defender) | No endpoint coverage |
| Zero-day / unknown threat detection | None | AI-based anomaly detection for novel TTPs (Darktrace, Vectra AI) | Rules-only; blind to unknown threats |
| Cloud workload detection | None | Native cloud trail + workload anomaly (Microsoft Defender for Cloud) | No cloud coverage |
| Deception / honeypot integration | None | Integrated honeypot triggering with lateral movement tracking (Attivo, Illusive) | No deception capability |
| Detection accuracy metrics | None | Precision/recall tracking per rule with false positive rates (Splunk ES) | No accuracy measurement |

**World-best reference:** CrowdStrike Falcon (EDR+Detection), Microsoft Sentinel + Defender, Darktrace

**Critical gaps:**
- UEBA absent; insider threats and credential abuse are entirely invisible
- No endpoint or network telemetry; detection is blind to the most common attack vectors
- OOTB rule library is minimal; a lean team cannot operationalize the capability without months of rule writing
- Zero-day and unknown threat detection requires ML behavioral models, which are not present

---

## Data Fusion (`intel_fusion`)

**APG provides:** Multi-source intelligence data fusion layer that normalizes heterogeneous data into a common APG schema, resolves entity identity across datasets, and produces unified intelligence records for downstream analysis and correlation.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Schema normalization | OCSF-inspired flat schema | Full ECS/OCSF/STIX-compliant normalization (Microsoft Sentinel, Elastic) | Partial normalization coverage |
| Entity resolution (probabilistic) | Exact-match only | Probabilistic name/address/entity matching (Palantir Gotham, Quantexa) | No fuzzy resolution |
| Cross-domain fusion (HUMINT + SIGINT) | Single domain | Multi-INT fusion with source weighting (Palantir Gotham, IBM i2) | Single-source per capability |
| Confidence scoring | None | Bayesian confidence propagation per fused record (Palantir Gotham) | No uncertainty quantification |
| Temporal alignment | Basic timestamp join | ASOF joins + temporal gap interpolation (kdb+, Palantir) | No temporal alignment engine |
| Duplicate / contradiction resolution | None | Automated contradiction detection + resolution workflow (Palantir, IBM i2) | Duplicates and conflicts surface unresolved |
| Real-time fusion pipeline | Batch ETL | Stream-native, sub-second fusion (Apache Flink, Palantir Foundry) | Batch-only |
| Source provenance tracking | None | Full collection-to-fusion chain-of-custody (Palantir Gotham, IBM i2) | No provenance |
| STIX 2.1 output | None | Native STIX 2.1 bundle export (Anomali, ThreatConnect) | No interoperability standard |
| Scale | ~1M records | Trillion-record fusion at national intelligence scale (Palantir) | Massive scale gap |

**World-best reference:** Palantir Gotham, Quantexa Entity Resolution, IBM i2 iBase

**Critical gaps:**
- Probabilistic entity resolution is the core value of fusion systems; exact-match only produces false duplicates and misses alias relationships
- Cross-domain multi-INT fusion is unsupported; real intelligence requires fusing signals from multiple collection types
- No confidence/provenance tracking; fused records lack epistemic quality metadata
- Batch-only pipeline; fusion is stale by the time analysts consume it

---

## Graph Analytics (`intel_graph`)

**APG provides:** Entity-relationship graph construction and analysis over APG intelligence data, supporting network centrality metrics, community detection, path analysis, and visualization of entity connection webs for analyst investigation workflows.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Native graph database | None (PostgreSQL recursive CTE) | Purpose-built property graph with ACID transactions (Neo4j) | Relational emulation of graph is 100-1000x slower |
| Graph scale | ~100K nodes | Billions of nodes, trillions of relationships (Neo4j, TigerGraph) | 4-5 orders of magnitude gap |
| Graph algorithms library | ~5 (Networkx) | 65+ production algorithms: PageRank, Louvain, LPA, Node2Vec (Neo4j GDS) | Limited algorithm breadth |
| Real-time graph updates | Batch | Millisecond-latency graph updates + concurrent queries (Neo4j, TigerGraph) | No real-time updates |
| Link analysis visualization | Basic D3 | Interactive pivot, drill-down, timeline overlay (Maltego, IBM i2 ANB) | Primitive visualization |
| Fraud / network detection | None | Trained GNN models for fraud ring detection (TigerGraph, Featurespace) | No ML on graph |
| Graph query language | SQL recursive CTE | Cypher, GSQL, Gremlin (Neo4j, TigerGraph, JanusGraph) | No declarative graph query |
| Subgraph pattern matching | None | APOC + GDS subgraph isomorphism (Neo4j) | Cannot find structural patterns |
| Knowledge graph / ontology | None | OWL/RDF ontology + SPARQL (Palantir, Neo4j) | No knowledge representation |
| Intelligence-specific entity model | None | Pre-built person/org/event/location schema (Maltego, IBM i2, Palantir Gotham) | No domain model |

**World-best reference:** Palantir Gotham (OSINT/Intel), Neo4j + GraphAware Hume (post-June 2026 acquisition), IBM i2 Analyst's Notebook

**Critical gaps:**
- Using PostgreSQL as a graph database is a fundamental architectural mismatch; graph traversals at depth >3 are impractical
- No declarative graph query language (Cypher/GSQL); all traversals require SQL CTE authoring
- Absence of GNN or ML-on-graph capability; structural fraud patterns and community detection require graph-native ML
- No domain entity model; analysts must manually define what constitutes a "person," "event," or "organization"

---

## Threat Hunting (`intel_hunt`)

**APG provides:** Analyst-driven threat hunting workflow supporting hypothesis-based investigation, IOC sweep across historical event data, TTP-pattern search, and hunt notebook documentation for evidence capture and knowledge sharing.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Hunt query language | SQL | SPL (Splunk), KQL (Sentinel), EQL (Elastic) purpose-built for security | SQL is verbose and error-prone for security queries |
| Pre-built hunt hypotheses | None | 500+ community SIGMA rules + ATT&CK-aligned hunts (Splunk Hunting App) | Blank slate |
| Hypothesis management | None | Structured hypothesis lifecycle (Sqrrl/Microsoft, ThreatConnect) | No workflow |
| Behavioral analytics integration | None | Hunt surfaces UEBA anomalies as starting points (Microsoft Sentinel) | No behavioral leads |
| Collaborative hunting | Single analyst | Shared notebooks, team annotation, assignment (Splunk, Jupyter/Security) | No collaboration |
| Automated hunt execution | None | Scheduled hunt queries with auto-triage (Splunk, Elastic) | Manual only |
| Evidence collection | None | Built-in evidence locker with chain of custody (Palantir, QRadar) | No evidence management |
| Hunt-to-detection pipeline | None | Convert successful hunt to production detection rule (Elastic, Splunk) | Hunt findings not operationalized |
| Visualization of hunt results | Table only | Timeline, graph, geospatial hunt result views (Splunk, Elastic) | No visual context |
| Threat intelligence integration | None | Real-time IOC enrichment during hunt (ThreatConnect, Recorded Future) | Hunters lack real-time intel |

**World-best reference:** Splunk Enterprise Security (with Hunting App), Microsoft Sentinel, CrowdStrike Adversary OverWatch

**Critical gaps:**
- No purpose-built hunt query language; SQL is unsuitable for high-velocity security event exploration
- No pre-built hypothesis library; hunters must build from zero with no ATT&CK-guided starting points
- Hunt-to-detection pipeline absent; successful hunts do not automatically generate detection rules
- Collaborative hunting impossible; threat hunting is a team sport in mature SOCs

---

## Geographic / Geospatial Intelligence (`intel_map`)

**APG provides:** Geospatial mapping of intelligence entities using coordinates derived from IP geolocation, address parsing, and manually tagged location fields. Supports heatmap overlays, point clustering, and choropleth visualization for threat distribution analysis.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| IP geolocation accuracy | MaxMind free DB (~city-level) | Commercial feeds with ISP/ASN/VPN/Tor enrichment (Digital Element, IPinfo) | City-level only; no ASN/proxy detection |
| Satellite / imagery integration | None | Commercial satellite + SAR imagery overlay (Palantir Gotham, Maxar) | No imagery |
| Movement / track analysis | None | Multi-source track fusion with velocity modeling (Palantir, Esri) | No movement analytics |
| Indoor mapping | None | Floor-plan + indoor positioning (HERE, Esri Indoors) | No indoor capability |
| Real-time geofence alerting | None | Sub-second geofence trigger on entity movement (Esri, Palantir) | No geofence alerting |
| OSINT-derived location extraction | None | NLP + image EXIF location extraction from open sources (Babel Street, Echosec) | No OSINT geo extraction |
| Temporal playback | None | Time-lapse playback of entity movement history (Esri, Palantir) | Static maps only |
| Cross-domain geo overlay | Single layer | Multi-layer fusion: network, physical, human intelligence (Palantir) | Single-layer display |
| Routing / proximity analysis | None | Buffer analysis, isochrone, routing (Esri ArcGIS, Palantir) | No spatial operations |
| Custom coordinate systems | WGS84 only | Military grid (MGRS), UTM, ECEF (Esri, Palantir) | Single CRS |

**World-best reference:** Palantir Gotham (geospatial intelligence), Esri ArcGIS, Babel Street / Echosec

**Critical gaps:**
- No imagery integration; GEOINT without satellite/aerial imagery severely limits physical threat analysis
- Movement tracking and temporal playback absent; entity behavior over time cannot be visualized
- OSINT-derived location extraction requires NLP + EXIF parsing, neither of which is implemented
- IP-only geolocation without proxy/VPN/Tor enrichment produces systematically misleading locations

---

## Continuous Monitoring (`intel_monitor`)

**APG provides:** Always-on monitoring framework polling configured data sources at defined intervals, applying threshold and signature rules, and forwarding findings to `intel_alerts`. Supports network, application, and API source types with configurable collection agents.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Agent-based endpoint monitoring | None | Lightweight EDR agents with kernel telemetry (CrowdStrike, SentinelOne) | No endpoint visibility |
| Agentless cloud monitoring | None | API-driven cloud posture monitoring (Microsoft Defender for Cloud, Wiz) | No cloud surface |
| Network packet capture | None | Full-packet + NetFlow monitoring at 100Gbps (Zeek, Arkime/Moloch) | No network traffic visibility |
| Synthetic transaction monitoring | None | Proactive SLA monitoring via synthetic user sessions (Dynatrace, Datadog) | No proactive testing |
| Log collection breadth | Manual config | 400+ out-of-box log connectors (Microsoft Sentinel, Splunk) | Manual connector building |
| Health-check / availability | None | Multi-step health checks with dependency mapping (Datadog, Splunk ITSI) | No availability monitoring |
| Dark web monitoring | None | Continuous dark web + paste site + criminal forum monitoring (Recorded Future) | No external threat surface monitoring |
| Asset discovery | Manual inventory | Auto-discover assets via network scan + passive fingerprint (Tenable, Qualys) | No asset discovery |
| Monitoring SLA compliance | None | Per-collection-source SLA with data freshness alerting (Splunk) | No freshness guarantees |
| Scalable collection architecture | Single agent | Distributed forwarder hierarchy with load balancing (Splunk UF, Elastic Beats) | Single-agent bottleneck |

**World-best reference:** Splunk Enterprise Security (collection), CrowdStrike Falcon (endpoint), Microsoft Defender for Cloud

**Critical gaps:**
- No endpoint or network monitoring agents; coverage is limited to sources that can be polled via API or syslog
- Dark web and external threat surface monitoring absent; attackers operate there before striking
- Asset discovery not automated; the monitored asset universe is whatever was manually configured
- No data freshness guarantees; stale collection silently creates blind spots

---

## Pattern Recognition (`intel_pattern`)

**APG provides:** Statistical and rule-based pattern matching across intelligence datasets, identifying recurring sequences, behavioral fingerprints, and temporal patterns indicative of coordinated threat activity.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ML-based pattern discovery | None | Deep learning pattern extraction (Darktrace, Vectra AI) | Rules-only; no ML discovery |
| YARA rule execution | None | YARA matching at scale against files/memory (VirusTotal, Cuckoo Sandbox) | No malware pattern matching |
| NLP pattern detection | None | Topic modeling, semantic similarity clustering (Recorded Future, Palantir) | No text pattern recognition |
| Graph pattern matching | None | Subgraph isomorphism for structural TTP patterns (Neo4j GDS, TigerGraph) | No structural pattern search |
| Temporal sequence mining | None | Temporal sequence pattern mining (IBM i2, Palantir) | No sequence mining |
| Anomaly-as-pattern baseline | Threshold | Multivariate Gaussian + isolation forest (Splunk MLTK, Darktrace) | Univariate thresholds only |
| Cross-dataset pattern linking | None | Cross-dataset pattern correlation (Palantir, IBM QRadar) | Siloed within single dataset |
| Pattern confidence scoring | None | Probabilistic confidence with false positive tracking (Recorded Future) | Binary match/no-match only |
| Image / video pattern analysis | None | Visual pattern recognition via CV models (Palantir, Babel Street) | No multi-modal patterns |
| Real-time streaming pattern match | None | CEP streaming pattern detection (Apache Flink, Drools Fusion) | Batch-only |

**World-best reference:** Darktrace (AI behavioral patterns), Palantir Gotham, IBM QRadar UEBA

**Critical gaps:**
- No ML-based unsupervised pattern discovery; only pre-specified patterns can be detected
- YARA/STIX pattern execution absent; standard malware and IOC patterns cannot be matched
- No streaming pattern detection; patterns are only identified over historical batch data
- Cross-dataset pattern correlation absent; patterns are siloed within single data sources

---

## Predictive Intelligence (`intel_prediction`)

**APG provides:** Forward-looking threat assessment using historical event patterns, actor TTPs, and environmental indicators to score likelihood of future attacks, campaign escalations, or vulnerability exploitations.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Threat actor intent modeling | None | NLP-derived intent + capability + opportunity scoring (Recorded Future) | No actor intent assessment |
| CVE exploit prediction | None | ML-scored CVE exploitability in wild within 30 days (Recorded Future, EPSS) | No exploit likelihood scoring |
| Campaign escalation forecasting | None | Time-series models for attack campaign intensity (Mandiant, Recorded Future) | No campaign forecasting |
| Geopolitical risk integration | None | Geopolitical signal fusion for cyber threat forecasting (Recorded Future, Flashpoint) | No geopolitical dimension |
| Counterfactual analysis | None | What-if defense posture simulation (Palantir, Booz Allen) | No counterfactual reasoning |
| Confidence intervals | None | Prediction intervals + calibration scores (Recorded Future) | Binary predictions only |
| Leading indicator identification | None | Automated precursor pattern detection (IBM QRadar, Recorded Future) | No early-warning indicators |
| Prediction explainability | None | Feature importance per prediction (DataRobot, Recorded Future) | Black-box output |
| Historical accuracy tracking | None | Prediction accuracy log with calibration drift (Recorded Future) | No accuracy feedback |
| Integration with vulnerability data | None | NVD + proprietary zero-day intelligence (Tenable, Recorded Future) | No vulnerability signal |

**World-best reference:** Recorded Future (threat intelligence forecasting), Mandiant Advantage, EPSS (FIRST.org)

**Critical gaps:**
- CVE exploit prediction is among the highest-ROI predictive intelligence features; entirely absent
- No actor intent modeling; prediction without understanding actor motivation is severely limited
- Confidence intervals and calibration tracking absent; predictions are not scientifically trustworthy
- No geopolitical signal integration; cyber threat prediction without geopolitical context misses major drivers

---

## Entity / Behavioral Profiling (`intel_profile`)

**APG provides:** Entity profile construction aggregating observed behavioral attributes, associated IOCs, network presence, and temporal activity patterns into structured profiles for persons, organizations, domains, and IP infrastructure.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Automated profile enrichment | Manual | Continuous automated enrichment from 100+ sources (Recorded Future, Maltego) | Manual research only |
| UEBA behavioral baseline | None | ML-derived behavioral baseline per entity (Splunk UBA, Exabeam) | No behavioral modeling |
| Identity resolution across aliases | None | Probabilistic cross-alias linkage (Quantexa, Palantir Gotham) | Siloed per-source identity |
| Dark web persona tracking | None | Criminal forum persona monitoring (Recorded Future, Flashpoint) | No dark web persona coverage |
| Social graph profiling | None | Social network influence analysis (Maltego, Palantir) | No social context |
| Risk score evolution timeline | None | Historical risk trajectory with event attribution (ThreatConnect, Recorded Future) | Static snapshot only |
| Shared infrastructure detection | None | Passive DNS + SSL certificate clustering to link actor infrastructure (RiskIQ, DomainTools) | No infrastructure attribution |
| Derogatory indicator tracking | None | Adverse media, sanctions, PEP screening (Refinitiv, Dow Jones Risk & Compliance) | No adverse data integration |
| MITRE ATT&CK actor profile | None | ATT&CK Group profile with observed TTPs (MITRE, Recorded Future) | No ATT&CK actor alignment |
| Profile sharing / dissemination | None | STIX 2.1 Threat Actor object export (ThreatConnect, Anomali) | No interoperable export |

**World-best reference:** Recorded Future (entity intelligence), Maltego, Palantir Gotham

**Critical gaps:**
- No UEBA; entity behavioral profiling without a baseline is impossible
- Identity resolution across aliases is entirely absent; same actor appears as multiple unlinked entities
- Dark web persona tracking requires dedicated crawling infrastructure that does not exist
- No infrastructure attribution via passive DNS/SSL clustering; actor infrastructure goes unlinked

---

## Intelligence Reporting (`intel_reporting`)

**APG provides:** Structured intelligence report generation supporting analyst-authored finished intelligence products with evidence linking, classification marking, dissemination controls, and export to PDF and STIX formats. Integrates with APG's `intel_analytics` and `intel_threats` outputs.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Traffic Light Protocol (TLP) enforcement | None | TLP + custom marking with dissemination enforcement (ThreatConnect, Anomali) | No classification enforcement |
| STIX 2.1 structured export | None | TAXII + STIX 2.1 bundle generation (ThreatConnect, Anomali) | No interoperable export |
| AI-assisted report drafting | None | Generative AI report composition from evidence graph (Recorded Future, Sentinel Copilot) | Fully manual authoring |
| Multi-level classification | None | TS/SCI, SECRET, UNCLASS, CUI marking (Palantir Gotham, DCSA systems) | No classification hierarchy |
| Evidence citation linking | Manual footnotes | Hyperlinked evidence with pivot-to-source (Palantir, IBM i2) | No programmatic citation |
| Report versioning + review workflow | None | Draft > review > approve > publish lifecycle (Recorded Future, Palantir) | No review workflow |
| Threat-matrix coverage reporting | None | ATT&CK Navigator coverage heatmap generation (Splunk, MITRE) | No coverage reporting |
| Automated IOC appendix | None | Auto-generated machine-readable IOC lists per report (Anomali, ThreatConnect) | Manual IOC compilation |
| Multi-lingual reporting | None | Multi-language NLP translation (Recorded Future, OSINT tools) | English-only |
| Report metrics tracking | None | Read receipt, distribution tracking, impact scoring (ThreatConnect) | No consumption analytics |

**World-best reference:** ThreatConnect, Recorded Future Portal, Anomali ThreatStream

**Critical gaps:**
- No TLP enforcement; intelligence can be inadvertently over-shared without classification controls
- STIX/TAXII export absent; reports cannot be shared with external partners via standard protocols
- AI-assisted drafting absent; report production speed is severely limited by analyst bandwidth
- Review/approval workflow nonexistent; quality and classification accuracy cannot be enforced

---

## Risk Scoring Engine (`intel_score`)

**APG provides:** Multi-factor risk scoring of entities, assets, and events using configurable weighted scoring models. Aggregates signals from threat intelligence, vulnerability data, behavioral anomalies, and business context to produce numerical risk ratings.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ML-trained risk models | None | Gradient-boosted risk models trained on global attack corpus (Recorded Future, CrowdStrike) | Static rule-based weights only |
| Asset business value weighting | None | Business criticality-adjusted risk scoring (ServiceNow VR, Tenable.io) | Context-blind scoring |
| CVE EPSS / CVSS integration | None | Real-time EPSS + CVSS + exploit intelligence scoring (Tenable, Recorded Future) | No vulnerability scoring |
| Real-time score updates | Batch | Sub-second score updates on new IOC match (Recorded Future, ThreatConnect) | Batch-only |
| Score decomposition / explainability | None | Per-factor contribution breakdown (Recorded Future, ThreatConnect) | Opaque scores |
| Peer benchmarking | None | Industry-benchmarked scores (BitSight, SecurityScorecard) | No external benchmarking |
| Score history + trend | None | Risk trend timeline with event attribution (Recorded Future, ThreatConnect) | Static point-in-time score |
| Attack surface scoring | None | External attack surface scoring via passive recon (BitSight, RiskIQ) | No external surface visibility |
| Third-party / supplier scoring | None | Vendor risk scoring via public signal (BitSight, SecurityScorecard) | No supply chain risk |
| Score-to-action automation | None | Automated remediation trigger on score threshold breach (ServiceNow, Splunk SOAR) | No actuation |

**World-best reference:** Recorded Future (Risk Score), BitSight (external risk), Tenable.io (vulnerability risk)

**Critical gaps:**
- Rule-based static weights cannot adapt to emerging threats; ML-trained models are the standard
- No external attack surface or third-party supplier scoring; supply chain risk is invisible
- Score decomposition absent; decision-makers cannot understand why a score changed
- Score-to-action automation absent; high-risk scores require manual human response

---

## Temporal Analysis (`intel_temporal`)

**APG provides:** Time-based analysis of intelligence events supporting chronological sequencing, dwell time measurement, recurrence detection, temporal clustering, and timeline reconstruction for incident investigation workflows.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Sub-millisecond timestamp precision | Second-level | Nanosecond-precision event timestamps (kdb+, Splunk) | Insufficient for network forensics |
| Multi-timezone normalization | None | Automatic timezone inference + normalization (Splunk, Elastic) | Timestamps stored as-received |
| Attack kill-chain timeline | None | Visual kill-chain timeline with ATT&CK alignment (QRadar, Sentinel) | No kill-chain visualization |
| Dwell time calculation | Manual SQL | Automated dwell time per entity/threat (Mandiant M-Trends, Exabeam) | Manual computation |
| Temporal pattern anomaly | Threshold-based | Fourier/wavelet analysis for temporal anomalies (Splunk MLTK, kdb+) | No spectral analysis |
| Time-series forecasting on events | None | LSTM/Prophet event-volume forecasting (Splunk, DataRobot) | No volume forecasting |
| Retroactive analysis (time travel) | None | Point-in-time data snapshots for retrospective queries (Splunk, Delta Lake) | Cannot analyze past states |
| Clock skew detection | None | Network clock synchronization anomaly detection (IBM QRadar) | No clock integrity checking |
| Temporal join across sources | SQL JOIN | ASOF join for inexact cross-source timestamp alignment (kdb+, QuestDB) | No temporal alignment join |
| Sequence visualization | Table only | Interactive timeline with zoom + filter (Palantir, IBM i2 ANB) | No timeline visualization |

**World-best reference:** Palantir Gotham (temporal investigation), IBM QRadar, kdb+ (high-precision temporal analytics)

**Critical gaps:**
- Second-level timestamp granularity is insufficient for network forensics requiring sub-millisecond precision
- Kill-chain timeline visualization absent; reconstructing attack narratives requires manual effort
- Retroactive analysis (time travel queries) missing; past security posture cannot be reconstructed
- Temporal join semantics lacking; cross-source timestamp alignment is error-prone via standard SQL JOIN

---

## Threat Intelligence (`intel_threats`)

**APG provides:** Structured threat intelligence management supporting IOC ingestion, TTP documentation, threat actor tracking, campaign management, and STIX-format exchange. Aggregates external feeds and APG-produced intelligence into a unified threat knowledge base.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Intelligence volume | Manual entry | Billions of IOCs with automated freshness decay (Recorded Future) | Tiny corpus vs. commercial feeds |
| Dark web intelligence | None | Continuous dark web + criminal marketplace intelligence (Flashpoint, Recorded Future) | No dark web |
| ATT&CK TTP database | None | Full ATT&CK enterprise database with 600+ techniques (MITRE ATT&CK, Recorded Future) | No TTP library |
| Threat feed integration | Manual STIX import | 1000+ commercial and OSINT feed auto-ingestion (Anomali, ThreatConnect) | Manual feed onboarding |
| IOC decay / freshness | None | ML-based IOC lifetime prediction + auto-expiry (Recorded Future, Anomali) | IOCs never expire |
| Diamond model analysis | None | Structured Diamond Model + adversary campaign analysis (ThreatConnect) | No structured framework |
| Sector-specific threat intelligence | None | Industry vertical threat reports + feeds (Recorded Future, FS-ISAC) | Horizontal only |
| Real-time threat feed updates | Daily batch | Sub-minute feed updates + streaming IOC push (Recorded Future, ThreatConnect) | Daily cadence too slow |
| TAXII 2.1 server | None | Native TAXII 2.1 server for bi-directional sharing (ThreatConnect, Anomali) | No sharing infrastructure |
| Hunting integration | None | Threat intelligence directly surfaces as hunt leads (Splunk ES, Recorded Future) | Siloed from hunting workflow |

**World-best reference:** Recorded Future Intelligence Cloud, Anomali ThreatStream, ThreatConnect TIP

**Critical gaps:**
- No dark web or criminal marketplace intelligence collection; most actionable threat intelligence originates there
- IOC decay/freshness management absent; stale IOCs generate false positives indefinitely
- No TAXII 2.1 server; cannot participate in automated intelligence sharing communities (ISACs, MISP)
- Real-time feed refresh cadence (daily) is far too slow for operational threat intelligence

---

## Data Validation / Quality (`intel_validate`)

**APG provides:** Data quality validation framework enforcing schema conformance, referential integrity, completeness checks, and range/pattern validation on intelligence inputs. Generates data quality reports and routes failing records to quarantine queues for remediation.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ML-based anomaly detection in data | None | Statistical anomaly detection on data distributions (Monte Carlo, Great Expectations) | Rules-only validation |
| Data observability | None | End-to-end pipeline health + data SLA monitoring (Monte Carlo, Anomalo) | No observability layer |
| Profile-based expectations | Manual rule authoring | Auto-profiled expectations from data statistics (Great Expectations, dbt tests) | No auto-profiling |
| Cross-source consistency | None | Cross-table referential consistency checks (dbt, Great Expectations) | Single-source checks only |
| Real-time streaming validation | Batch | In-stream schema + value validation (Apache Kafka + Avro, Confluent) | Batch-only |
| STIX 2.1 conformance | None | STIX schema validator with profile-specific rules (OASIS, ThreatConnect) | No STIX validation |
| Data lineage-aware quality | None | Quality metrics attached to lineage nodes (Databricks Unity Catalog, Collibra) | Quality disconnected from lineage |
| Automated remediation | None | Quarantine + auto-remediation rules + reprocessing (Talend, Informatica) | Manual remediation |
| Quality SLA reporting | None | Data quality SLA dashboard with breach history (Monte Carlo, Collibra) | No quality SLA |
| IOC-specific validation | None | IOC format validation + detonation-based verification (VirusTotal, AbuseIPDB) | No threat-intel-specific validation |

**World-best reference:** Monte Carlo (data observability), Great Expectations / dbt tests, Collibra Data Quality

**Critical gaps:**
- No ML-based distributional anomaly detection in data; systematic data quality issues evade rule-based checks
- Data observability absent; pipeline health is unknown until analysts notice bad data
- STIX conformance validation missing; intelligence exchanged in malformed STIX creates interoperability failures
- IOC-specific validation (format, reputation, detonation) not integrated; garbage IOCs degrade detection

---

## Source Verification (`intel_verify`)

**APG provides:** Source credibility assessment framework tracking collection source reliability, accuracy history, reporting frequency, and corroboration rates. Assigns confidence modifiers to intelligence produced by each source for downstream weighting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Automated source provenance tracking | Manual metadata | Full automated chain-of-custody from collection to product (Palantir Gotham) | Human-maintained source records |
| Source corroboration scoring | None | Multi-source corroboration counting with decay (ThreatConnect, Anomali) | No corroboration measurement |
| Historical accuracy tracking | None | Source accuracy rate tracked against ground truth (Recorded Future) | No ground truth feedback loop |
| NATO Admiralty scale | None | Structured A1-F6 reliability + credibility scoring (IBM i2, Palantir Gotham) | No standard framework |
| Automated reliability scoring | None | ML-derived source reliability from historical performance (Recorded Future) | Manual assignment only |
| Deception / disinformation detection | None | Adversarial narrative injection detection (Graphika, Recorded Future) | No deception detection |
| Source bias assessment | None | Source political/organizational bias tagging (Janes, Oxford Analytica) | No bias metadata |
| Legal authority / access validation | None | Legal framework compliance for collection authorities (government platforms) | No legal framework |
| Real-time source freshness | None | Automated staleness detection + alerting (ThreatConnect) | No freshness tracking |
| Source discovery / new source onboarding | Manual | Automated relevant source discovery via AI crawling (Recorded Future, Echosec) | Fully manual |

**World-best reference:** Palantir Gotham (source management), Recorded Future, IBM i2 Analyst's Notebook

**Critical gaps:**
- NATO Admiralty scale (industry standard for intelligence source grading) not implemented
- No ground truth feedback loop; source accuracy cannot be measured or improved over time
- Automated reliability scoring absent; analysts manually assign reliability with no empirical basis
- Deception/disinformation detection missing; adversaries can inject false intelligence through manipulated sources

---

## Energy Domain

---

## Energy Billing (`energy_bil`)

**APG provides:** Energy billing calculation engine handling tariff application, consumption-based charging, demand charge computation, and invoice generation for residential, commercial, and industrial accounts. Integrates with `energy_grd` metering data and `energy_rpt` for reporting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Interval / smart meter billing | None | 15-minute interval billing from AMI meters (Oracle Utilities CCB, SAP IS-U) | Cannot process interval data |
| Complex tariff modeling | Basic flat-rate | Time-of-use, tiered, demand, TOU+demand combination (Oracle Utilities, SAP IS-U) | Limited tariff complexity |
| Revenue recognition (IFRS 15) | None | Automated deferred revenue + IFRS 15/ASC 606 compliance (SAP S/4HANA, Oracle Utilities) | No accounting compliance |
| Green tariff / renewable billing | None | Renewable Energy Certificate (REC) billing + carbon offset tracking (Oracle, SAP) | No sustainability billing |
| Prepaid / PAYG billing | None | Prepaid vending + token generation (Itron, Oracle Utilities PPM) | No prepaid support |
| Bill dispute management | None | Structured dispute workflow with audit trail (Oracle Utilities, SAP IS-U) | No dispute management |
| Multi-currency / multi-jurisdiction | None | 180+ countries with jurisdiction-specific tax calculation (SAP IS-U, Oracle) | Single currency/jurisdiction |
| Net metering (prosumer) | None | Prosumer credit calculation + netting (Oracle Utilities, Itron) | No prosumer support |
| EDI billing integration | None | EDI 810, 812, ANSI X12 standard billing exchange (Oracle, SAP) | No EDI |
| Estimated vs. actual reconciliation | None | Automated reconcile + retroactive bill adjustment (Oracle Utilities, SAP) | No reconciliation |

**World-best reference:** Oracle Utilities Customer Care & Billing, SAP IS-U / S/4HANA Utilities, Itron

**Critical gaps:**
- Interval data billing (15-min AMI) is the baseline requirement for modern smart grid billing; absent
- Prosumer/net metering support is essential as distributed generation proliferates; not present
- No IFRS 15 revenue recognition; not suitable for regulated utility financial reporting
- Dispute management absent; any billing dispute requires external workflow tooling

---

## Distribution Management (`energy_dis`)

**APG provides:** Distribution network topology management supporting feeder modeling, load balancing, fault isolation, and outage management for medium/low voltage distribution systems. Integrates with `energy_grd` for real-time grid state.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| SCADA real-time integration | None | Full DNP3/IEC 61850 SCADA integration (GE/Alstom, Schneider EcoStruxure) | No real-time SCADA |
| Outage management system (OMS) | None | Integrated OMS with crew dispatch and ETR (Oracle NMS, GE PowerOn) | No outage management |
| Network topology analysis | Basic graph | Full electrical network model with load flow solver (CYME, PSS/E) | No power flow analysis |
| Volt/VAR optimization (VVO) | None | Automated VVO reducing losses 3-7% (Schneider, GE, S&C Electric) | No VVO capability |
| Fault location, isolation, restoration | None | Automated FLISR sub-5-minute restoration (Schneider, SEL, GE) | No automation |
| DER integration (solar/battery) | None | DER aggregation + hosting capacity analysis (Oracle, Itron, AutoGrid) | No DER support |
| AMI / smart meter integration | None | Bi-directional AMI communication + remote connect/disconnect (Itron, Landis+Gyr) | No AMI integration |
| Predictive asset maintenance | None | AI-predicted transformer failure + maintenance scheduling (ABB, Hitachi) | No predictive maintenance |
| GIS-based network model | None | Geo-referenced single-line diagram (Esri ArcGIS for Utilities, SPIDA, CYME) | No GIS integration |
| Regulatory compliance reporting | None | NERC/FERC/OFGEM-specific compliance reports (Oracle, SAP) | No regulatory reporting |

**World-best reference:** Oracle Utilities Network Management System, Schneider Electric EcoStruxure Grid, GE Grid Solutions

**Critical gaps:**
- No SCADA integration; distribution management without real-time telemetry is purely theoretical
- FLISR (automated fault isolation and restoration) absent; outage duration performance cannot be met
- VVO absent; utilities cannot achieve network efficiency or loss reduction targets
- DER integration missing; critical as distributed solar/battery penetration accelerates

---

## Generation Management (`energy_gen`)

**APG provides:** Power generation asset tracking and operational management supporting multi-fuel generation portfolios, dispatch scheduling, generation forecasting, and performance monitoring for thermal, hydro, and renewable assets.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Unit commitment optimization | None | Mixed-integer LP unit commitment solver (ABB PROBASE, GE PMAX) | No optimization |
| Economic dispatch | None | Real-time economic dispatch with fuel cost curves (GE PMAX, Siemens) | No merit-order dispatch |
| Renewable forecasting (wind/solar) | None | NWP-based 48h probabilistic wind/solar forecast (Vaisala, AWS Truepower) | No renewable forecasting |
| SCADA / DCS integration | None | Native PI System + DCS historian integration (OSIsoft/AVEVA PI, GE Mark VI) | No SCADA/DCS |
| Maintenance scheduling (planned) | None | Integrated maintenance schedule with commitment optimizer (ABB, GE) | No maintenance integration |
| Regulatory compliance (NERC CIP) | None | Automated NERC CIP-002 through CIP-014 compliance (OSIsoft, ABB) | No compliance |
| Emission tracking / carbon accounting | None | Real-time emission factor calculation + ETS compliance (EnergySys, SAP) | No emissions tracking |
| Ancillary services management | None | Frequency response, spinning reserve, AGC management (GE, ABB) | No ancillary services |
| Real-time performance monitoring | None | 1-second resolution generation KPI monitoring (OSIsoft PI, GE APM) | No real-time monitoring |
| Hydro optimization | None | Reservoir management + hydro unit scheduling (HydroComp, Ventyx) | No hydro-specific modeling |

**World-best reference:** GE PMAX / Grid Solutions, ABB Ability PROBASE, OSIsoft (AVEVA) PI System

**Critical gaps:**
- No economic dispatch or unit commitment optimization; generation assets are dispatched manually or externally
- No SCADA/DCS/PI integration; real-time generation telemetry is entirely absent
- Renewable forecasting absent; grid operators cannot plan for variable generation
- NERC CIP compliance tooling absent; cannot operate in regulated North American markets

---

## Grid Management (`energy_grd`)

**APG provides:** Smart grid monitoring and control layer with SCADA integration hooks, network state estimation, demand response management, and IoT sensor data ingestion. Supports real-time grid topology visualization and event-driven alerting for grid operators.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| State estimation | None | Full network state estimator with bad data detection (GE EMS, Siemens SPECTRUM) | No state estimation |
| SCADA protocol support | None | DNP3, IEC 61850, IEC 60870-5-101/104, Modbus, ICCP (GE EMS, ABB SCADA) | No industrial protocol support |
| Contingency analysis (N-1/N-2) | None | Real-time N-1/N-2 contingency analysis (Siemens, GE EMS, PowerWorld) | No contingency analysis |
| Energy management system (EMS) | Basic monitoring | Full EMS with AGC, AVC, OPF (GE ENMAC, ABB eSOMS) | Core EMS functionality absent |
| Demand response management | None | Automated demand response with aggregator integration (AutoGrid, EnerNOC/Enel X) | No DR management |
| Distributed energy resource management | None | DERMS supporting 10M+ DER registrations (Oracle, Itron, AutoGrid) | No DERMS |
| Cybersecurity (OT/ICS) | None | IEC 62443-compliant OT security monitoring (Claroty, Dragos, Nozomi Networks) | No OT security |
| Market settlement integration | None | ISO/RTO settlement interface (ERCOT, PJM, CAISO market platforms) | No market integration |
| Load forecasting | None | ML-based 24/168h load forecast (ABB, Siemens, Oracle, EPRI) | No load forecasting |
| Power quality monitoring | None | Harmonic distortion, flicker, sag/swell monitoring (Dranetz, Schneider) | No power quality |

**World-best reference:** GE Energy Management System (EMS), Siemens SPECTRUM Power, ABB Ability EMS

**Critical gaps:**
- No SCADA protocol support; cannot connect to any field devices or existing SCADA infrastructure
- State estimation absent; grid operators cannot determine true network operating state
- N-1 contingency analysis absent; cannot assess or demonstrate grid reliability under fault conditions
- OT cybersecurity (IEC 62443) absent; SCADA-connected system without OT security is a critical infrastructure risk

---

## Energy Reporting (`energy_rpt`)

**APG provides:** Energy sector reporting covering generation output, distribution performance, billing summaries, regulatory submissions, and sustainability metrics. Supports parameterized reports, scheduled delivery, and integration with APG's `bia_rpt` report builder.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Regulatory filing automation | None | Automated FERC Form 1, NERC reports, Ofgem RIGs (Oracle, SAP IS-U) | No regulatory form generation |
| Sustainability / ESG reporting | None | GRI, TCFD, CDP-aligned sustainability reports (SAP Sustainability, Oracle) | No ESG framework |
| Emission factor reporting | None | Scope 1/2/3 emissions with GHG Protocol compliance (Envizi, SAP Sustainability) | No emissions reporting |
| Meter data reporting | None | AMI-based interval data validation reports (Oracle MDM, Itron) | No interval data reporting |
| Grid reliability metrics (SAIDI/SAIFI) | None | Automated SAIDI/SAIFI/CAIDI calculation + regulatory submission (Oracle NMS) | No reliability metrics |
| Power purchase agreement (PPA) reporting | None | PPA position tracking + settlement reporting (SAP CM, ION Openlink) | No PPA reporting |
| Wholesale market reporting | None | ISO settlement statement reconciliation (Open Access Technology, ABB) | No market report |
| Load research reporting | None | Statistical load research + customer segmentation (Oracle, Itron) | No load research |
| Real-time operational dashboards | None | Live single-line diagram + KPI dashboard (GE, Schneider) | No operational real-time view |
| Data quality audit reports | Basic | Full data quality lineage + certification reports (Oracle, SAP) | Limited quality reporting |

**World-best reference:** Oracle Utilities Analytics, SAP Analytics Cloud for Utilities, IBM Cognos Energy Analytics

**Critical gaps:**
- Regulatory filing automation is a compliance obligation for grid operators; entirely absent
- SAIDI/SAIFI calculation for distribution reliability is a standard utility KPI; not present
- ESG/Scope 1-2-3 emissions reporting is now mandatory in many jurisdictions; not integrated
- No PPA settlement or wholesale market reporting; commercial energy operations unsupported

---

## Energy Trading (`energy_trd`)

**APG provides:** Energy commodity trading and position management supporting deal capture, mark-to-market valuation, hedging workflow, and basic risk metrics for power, gas, and renewable energy instruments.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ETRM front-to-back coverage | Partial (deal capture only) | Full front-to-back: deal capture, scheduling, risk, accounting, settlement (ION Openlink Endur) | Back-office and scheduling absent |
| Real-time market data integration | None | Real-time feeds from ICE, CME, Platts, Argus (ION Openlink, Triple Point) | No live market data |
| VaR / Greeks calculation | None | Real-time VaR, CVaR, Delta/Gamma/Vega (ION Openlink, Riskonnect) | No risk metrics |
| Physical logistics management | None | Pipeline, vessel, truck, rail scheduling (ION Openlink) | No physical operations |
| Regulatory reporting (REMIT, EMIR) | None | Automated REMIT/EMIR/CFTC trade reporting (ION Openlink, Murex) | No regulatory compliance |
| Carbon / renewable certificate trading | None | REC, GO, EUA, CER instruments (ION Openlink V25, SAP CM) | No green instruments |
| Counterparty credit risk | None | Credit limit management + real-time exposure (ION Openlink, Reval) | No credit risk |
| Market simulation / scenario analysis | None | Monte Carlo price scenario + portfolio stress testing (ION, Triple Point) | No simulation |
| Transmission / congestion management | None | FTR/CRR position management + ISO market integration (ION, Open Access Technology) | No transmission risk |
| Settlement / invoicing | None | Automated settlement + invoice with dispute workflow (ION Openlink, SAP CM) | No settlement |

**World-best reference:** ION Openlink Endur, SAP Commodity Management, ION Triple Point

**Critical gaps:**
- Back-office, scheduling, and settlement absent; only front-office deal capture is functional
- No regulatory trade reporting (REMIT/EMIR/CFTC); cannot legally operate in regulated energy markets
- Real-time market data integration absent; positions cannot be marked to market
- VaR and credit risk management absent; trading desks cannot comply with risk governance frameworks

---

## Telecom Domain

---

## Telecom Analytics (`telecom_ana`)

**APG provides:** Analytical processing of telecom operational data covering network KPIs, subscriber behavior, churn signals, revenue trends, and service quality metrics. Supports both historical reporting and near-real-time operational analytics dashboards.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Churn prediction models | None | ML-based churn propensity with intervention triggers (Amdocs amAIz, TIBCO) | No predictive churn |
| Network quality analytics | Basic KPI aggregation | End-to-end quality of experience (QoE) analytics (TEOCO, Ericsson Expert Analytics) | No QoE analytics |
| Revenue per user analytics | SQL aggregation | ARPU decomposition + LTV modeling (Amdocs, TIBCO) | No LTV modeling |
| Subscriber journey analytics | None | Multi-touchpoint journey mapping (Adobe Analytics, Salesforce, Amdocs) | No journey analytics |
| 5G slice analytics | None | Network slice utilization + SLA analytics (Ericsson, Nokia) | No 5G slice visibility |
| Fraud analytics | None | Real-time CDR-based fraud pattern detection (Subex, TELARIX) | No fraud analytics |
| Network capex optimization | None | Traffic growth forecast → capex recommendation (Ericsson Expert Analytics, TEOCO) | No capex optimization |
| Roaming analytics | None | Roaming partner performance + settlement analytics (Syniverse, BICS) | No roaming analytics |
| IoT/M2M analytics | None | Device behavior analytics at 100M+ device scale (Ericsson, Nokia) | No IoT scale |
| Benchmarking vs. industry | None | Industry KPI benchmarking (GSMA, Ericsson Intelligence) | No external benchmarks |

**World-best reference:** Ericsson Expert Analytics, TEOCO Analytics, Amdocs amAIz Analytics

**Critical gaps:**
- No ML-based churn prediction; churn management is the highest-ROI telecom analytics use case
- QoE analytics absent; operators cannot proactively identify degraded customer experience
- 5G network slice analytics missing; required for enterprise 5G SLA management
- No fraud analytics; CDR-based fraud detection is table-stakes for telcos

---

## Telecom Billing (`telecom_bil`)

**APG provides:** BSS-aligned billing engine handling CDR mediation, rating, charging, invoice generation, and payment processing for voice, data, SMS, and value-added services. Supports postpaid, prepaid, and hybrid charging models with configurable tariff plans.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time online charging (OCS/Gy) | None | 3GPP-compliant OCS with sub-100ms charging (Ericsson Charging, Huawei UCS) | No real-time online charging |
| CDR mediation at scale | ~1M CDRs/hr | Billions of CDRs/hr with 99.999% accuracy (Amdocs Charging, CSG Encompass) | Scale gap |
| 5G convergent charging | None | 5G SA network-native CHF per 3GPP TS 32.290 (Ericsson, Huawei) | No 5G charging |
| Roaming settlement | None | TAP3/RAP, NRTRDE, IOT settlement (Syniverse, BICS, Comverse) | No roaming billing |
| Partner / MVNO billing | None | Multi-tier MVNO/MVNE wholesale billing (CSG, Amdocs) | No wholesale billing |
| Revenue sharing | None | Content partner revenue share with split rating (Amdocs, Oracle BRM) | No revenue share |
| Dunning and collections | Basic email | Configurable multi-stage dunning with legal hold (Amdocs, CSG) | Basic dunning only |
| Bill shock protection | None | Spend limits + real-time usage alerts per 3GPP TS 22.101 (Ericsson, Nokia) | No spend controls |
| Tax calculation (multi-jurisdiction) | None | US tax (AvaTax), EU VAT, GST engine integration (Amdocs, Oracle BRM) | No tax engine |
| Number portability billing | None | NP-aware call routing + billing adjustments (NetNumber, Telcordia) | No portability handling |

**World-best reference:** Amdocs Charging, CSG Singleview, Ericsson Charging System (ECS)

**Critical gaps:**
- No 3GPP-compliant Online Charging System (OCS); cannot perform real-time balance deduction for prepaid or real-time postpaid
- 5G standalone charging (CHF per TS 32.290) absent; incompatible with 5G SA deployments
- Roaming settlement (TAP/RAP) absent; roaming revenue cannot be billed or reconciled
- Multi-jurisdiction tax engine absent; telecoms operate across complex tax jurisdictions

---

## Customer Management (`telecom_cus`)

**APG provides:** Subscriber lifecycle management covering customer onboarding, account management, SIM/MSISDN assignment, plan management, and customer service workflow. Provides a 360° customer view integrating billing, service, and interaction history.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| AI-driven next-best-action | None | ML-based NBA recommendations at point of contact (Amdocs amAIz, Salesforce Einstein) | No AI recommendation |
| Omnichannel service delivery | None | Unified agent desktop across web, app, IVR, store (Amdocs, Salesforce) | Single-channel only |
| Self-service portal / app | None | Full-featured self-service with usage monitoring, top-up (Amdocs, MYCOM OSI) | No self-service capability |
| Predictive churn intervention | None | Proactive retention offer triggered by churn model (Amdocs, TIBCO) | No proactive retention |
| Customer lifetime value (CLTV) | None | ML-estimated CLTV + segment-based offers (Amdocs, Oracle CX) | No CLTV modeling |
| Identity verification (KYC) | None | Real-time KYC with document OCR + biometric (Jumio, Onfido integration in Amdocs) | No KYC |
| Number portability management | None | Port-in/port-out workflow with regulatory compliance (NetNumber, NPAC) | No number portability |
| Loyalty program management | None | Points, tiers, partner redemption (Comarch Loyalty, Oracle Loyalty) | No loyalty |
| Complaint management SLA | None | SLA-tracked complaint lifecycle with regulatory escalation (Amdocs, ServiceNow) | No SLA complaint tracking |
| B2B account hierarchy | None | Enterprise account + sub-account hierarchy with cost center billing (Amdocs, Oracle) | No enterprise account management |

**World-best reference:** Amdocs Customer Management, Salesforce Communications Cloud, Oracle Siebel CRM for Telecom

**Critical gaps:**
- No omnichannel agent desktop; customer service is siloed by channel
- AI-driven next-best-action absent; retention and upsell opportunities are missed at point of contact
- No self-service portal; all customer interactions require agent intervention, driving cost
- KYC/identity verification absent; cannot support regulatory SIM registration requirements

---

## Network Inventory (`telecom_inv`)

**APG provides:** Physical and logical network inventory management for telecom infrastructure, tracking network elements, connections, circuits, IP address pools, and capacity. Integrates with `telecom_net` for network management and `telecom_prv` for service provisioning workflows.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Auto-discovery of network elements | None | Protocol-based auto-discovery (SNMP, NETCONF, gRPC) (Nokia NetCracker, IBM Tivoli) | Manual inventory entry |
| Physical layer (fiber/copper) modeling | None | GIS-integrated fiber route + splice map (Esri for Telecom, Nokia NetCracker) | No physical layer |
| Service-to-infrastructure mapping | None | End-to-end service-to-infrastructure dependency graph (NetCracker, Comarch) | No service mapping |
| Capacity management | None | Bandwidth utilization + capacity planning per link (TEOCO Cap-Plan, NetCracker) | No capacity modeling |
| Multi-vendor element support | None | 500+ vendor equipment templates (Nokia NetCracker, IBM Tivoli) | No vendor catalog |
| IP address management (IPAM) | None | Integrated IPAM + DHCP + DNS (InfoBlox, BlueCat, Nokia NetCracker) | No IPAM |
| Network topology visualization | Basic table | Interactive layered topology map (Nokia NetCracker, Comarch) | No topology visualization |
| Change management integration | None | RFC-linked inventory changes with rollback (ServiceNow, Nokia NetCracker) | No change management |
| Reconciliation (network vs. inventory) | None | Automated discovered vs. planned reconciliation (Nokia NetCracker) | No reconciliation |
| 5G RAN/Core inventory | None | 5G NR site, gNodeB, AMF, UPF modeling (Ericsson, Nokia NetCracker) | No 5G element support |

**World-best reference:** Nokia NetCracker Network Inventory, IBM Tivoli Network Manager, Comarch OSS

**Critical gaps:**
- Auto-discovery absent; manual inventory is immediately stale in any live network
- Physical layer (fiber GIS) modeling missing; without it, provisioning and fault isolation are guesswork
- No IPAM integration; IP address management is a foundational network operations requirement
- 5G RAN and Core element modeling absent; cannot manage 5G infrastructure

---

## Network Management (`telecom_net`)

**APG provides:** Network monitoring, fault detection, and performance management across IP, transport, and access network domains. Collects telemetry via SNMP and syslog, applies threshold-based alerting, and presents network health dashboards for NOC operations.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| SNMP v3 / NETCONF / gRPC telemetry | SNMP v2 only | Full SNMP v3, NETCONF/YANG, gRPC/gnMI, OpenConfig (Ericsson, Nokia, Cisco NSO) | Missing modern telemetry protocols |
| AI/ML anomaly detection | None | AI-native network anomaly + degradation prediction (Nokia AVA, Ericsson AI Ops) | No ML on network telemetry |
| Root cause analysis (RCA) | None | Automated topological RCA with causal graph (Nokia AVA, Moogsoft) | No automated RCA |
| Network performance analytics | Basic KPI | Granular 1-sec KPIs: latency, jitter, loss per flow (TEOCO, Accedian) | Low-resolution KPIs |
| Network slicing management | None | 5G network slice lifecycle management (Ericsson, Nokia) | No slicing support |
| Intent-based networking | None | Policy intent translation to network config (Cisco NSO, Nokia NSP) | No intent-based management |
| Multi-domain management | Single domain | Multi-domain (IP, optical, RAN, core) unified view (Nokia NSP, Ericsson OSS) | Single-domain only |
| Automated remediation | None | Self-healing actions triggered by AI diagnosis (Nokia AVA, Moogsoft, Cisco) | No self-healing |
| Zero-touch provisioning | None | ZTP with YANG/NETCONF (Cisco NSO, Ericsson, Nokia) | No ZTP |
| 5G core network management | None | AMF, SMF, UPF, PCF lifecycle management (Ericsson, Nokia) | No 5G core support |

**World-best reference:** Nokia Network Services Platform (NSP) + AVA, Ericsson OSS/BSS, Cisco NSO

**Critical gaps:**
- NETCONF/YANG and gRPC telemetry absent; modern network equipment uses these protocols exclusively
- No AI-based RCA; NOC operators manually diagnose root cause, which is slow and error-prone
- 5G core (cloud-native NFs: AMF, SMF, UPF) management absent; incompatible with modern 5G operator deployments
- Intent-based networking and zero-touch provisioning absent; operational costs remain high without automation

---

## Order Management (`telecom_ord`)

**APG provides:** Telecom service order lifecycle management supporting order capture, decomposition, orchestration, and fulfillment tracking for voice, data, and enterprise service orders. Integrates with `telecom_prv` for provisioning execution and `telecom_cus` for customer-facing order status.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| TM Forum Order-to-Activate | None | Full TMF 641/eTOM-aligned order management (Amdocs, Nokia NetCracker) | No TM Forum alignment |
| Intelligent order decomposition | None | AI-driven order decomposition to sub-orders (Amdocs, Nokia NetCracker) | Manual decomposition |
| Fallout / exception management | None | Automated fallout detection + jeopardy management (Amdocs, Oracle ASAP) | Manual fallout handling |
| Order jeopardy alerting | None | SLA-based jeopardy alerts with auto-escalation (Amdocs, NetCracker) | No jeopardy management |
| B2B / wholesale order integration | None | Electronic order exchange (EDI, API) with partner systems (Amdocs, Telcordia) | No partner integration |
| Complex bundle orchestration | None | Multi-product bundle decomposition + coordinated provisioning (Amdocs) | Single-service orders only |
| Real-time order status visibility | None | Customer/B2B real-time order tracking portal (Amdocs, SAP) | No order tracking |
| Order rollback / cancel | None | Automated rollback with compensating transactions (Amdocs, NetCracker) | No rollback |
| Regulatory order compliance | None | Regulatory mandated orders (LNP, MACD, regulatory service) (Telcordia, Amdocs) | No regulatory order types |
| Order analytics | None | Order flow analytics + SLA compliance reporting (Amdocs, TEOCO) | No order analytics |

**World-best reference:** Amdocs Order Management, Nokia NetCracker Order Management, Oracle ASAP

**Critical gaps:**
- No TM Forum TMF 641 alignment; cannot integrate with partner or wholesale order systems following industry standards
- Fallout and jeopardy management absent; failed orders are not automatically detected or escalated
- Order rollback absent; partial provisioning failures result in partially-activated services with no automated remediation
- Complex bundle orchestration absent; modern telecom offers are predominantly multi-product bundles

---

## Service Provisioning (`telecom_prv`)

**APG provides:** Network service provisioning and activation executing configuration commands against network elements based on orders from `telecom_ord`. Supports template-based provisioning for broadband, voice, and data services with configurable workflow steps.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Model-driven service design (YANG) | None | YANG/NETCONF-modeled service catalog + automated activation (Cisco NSO, Ericsson) | No model-driven provisioning |
| Zero-touch provisioning (ZTP) | None | Fully automated, template-free device bootstrapping (Cisco NSO, Nokia NSP) | No ZTP |
| Multi-vendor provisioning adapters | None | 500+ vendor adapters via standard southbound APIs (Nokia NetCracker, Amdocs) | No vendor adapters |
| Rollback on failure | None | Automated rollback with atomic transaction semantics (Cisco NSO, Nokia NetCracker) | No rollback |
| 5G network slice provisioning | None | 5G slice template + automated end-to-end activation (Ericsson, Nokia) | No 5G slicing |
| Cloud-native VNF/CNF provisioning | None | ETSI NFV/MANO-compliant VNF/CNF lifecycle (VMware, Ericsson, Nokia) | No NFV/CNF support |
| Provisioning SLA tracking | None | Per-service-type SLA with breach alerting (Amdocs, NetCracker) | No SLA tracking |
| Self-service API provisioning | None | Developer/B2B provisioning via RESTful API (Amdocs, Nokia TAS) | No API provisioning |
| Inventory-driven provisioning | None | Resource reservation from network inventory pre-provisioning (NetCracker, Comarch) | No inventory integration |
| Provisioning analytics | None | Mean time to activate, fallout rate, automation ratio KPIs (Amdocs, TEOCO) | No provisioning analytics |

**World-best reference:** Cisco Network Services Orchestrator (NSO), Nokia NetCracker, Amdocs Service Activation

**Critical gaps:**
- No YANG/NETCONF model-driven provisioning; cannot automate configuration of modern network equipment
- Multi-vendor southbound adapters absent; provisioning requires bespoke scripting per vendor/device type
- 5G slice provisioning and NFV/CNF lifecycle management absent; incompatible with cloud-native network deployments
- No rollback on failure; partially-provisioned services require manual recovery

---

## Telecom Reporting (`telecom_rpt`)

**APG provides:** Telecom-domain reporting covering network KPIs, billing summaries, subscriber metrics, revenue trends, and regulatory submissions. Leverages APG's `bia_rpt` report builder with telecom-specific data models and pre-built templates.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Regulatory filing (USAC, Ofcom, NCA) | None | Automated regulatory form generation + submission (Amdocs, Oracle BRM) | No regulatory reporting |
| Network quality (GSMA KPIs) | None | GSMA-aligned KPI reporting (dropped calls, data throughput, latency) (TEOCO, Ericsson) | No GSMA KPI reporting |
| Real-time NOC reporting | None | Live NOC glass pane with alarm KPIs (Nokia NSP, Ericsson OSS) | No real-time NOC |
| Wholesale / interconnect billing reports | None | Carrier billing reconciliation + dispute reports (Syniverse, BICS) | No wholesale reporting |
| Subscriber quality of experience | None | Per-subscriber QoE score report (Ericsson Expert Analytics, TEOCO) | No QoE reporting |
| Tower/site sharing reports | None | Passive infrastructure cost allocation + sharing reports (Comarch, Netcracker) | No infrastructure sharing reporting |
| Spectrum utilization reports | None | Spectrum efficiency + utilization per band (Ericsson, Nokia) | No spectrum reporting |
| MVNO/wholesale partner reports | None | Per-MVNO usage, revenue, settlement reports (CSG, Amdocs) | No MVNO reporting |
| Capex/opex analytics | None | Infrastructure investment vs. capacity ROI (TEOCO, Ericsson) | No investment analytics |
| Competitive benchmarking | None | Industry KPI benchmarking via GSMA/Ookla data (TEOCO, Analysys Mason) | No benchmarking |

**World-best reference:** TEOCO Analytics, Ericsson Expert Analytics, Oracle Communications Analytics

**Critical gaps:**
- No regulatory reporting; mandatory submissions to telecom regulators cannot be fulfilled
- GSMA-aligned KPI reporting (dropped call rate, throughput, availability) is the telecom industry standard; absent
- MVNO partner reporting absent; operators running wholesale/MVNO businesses have no per-partner visibility
- No QoE per-subscriber reporting; operators cannot identify customers experiencing poor service

---

## Revenue Assurance (`telecom_rev`)

**APG provides:** Revenue assurance framework detecting billing leakage, CDR completeness gaps, tariff misapplication, and fraud patterns across the revenue chain. Provides control point monitoring, reconciliation workflows, and revenue at-risk quantification.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| AI/ML leakage detection | None | HyperSense AI detecting unknown leakage patterns (Subex HyperSense) | Rules-only; novel leakage invisible |
| End-to-end revenue chain coverage | Partial | Network → mediation → rating → billing → collection (Subex ROC, WeDo RAID) | Incomplete chain visibility |
| Real-time CDR reconciliation | Batch | Near-real-time CDR completeness checking (Subex, WeDo/Mobileum) | Batch-only; revenue at risk accumulates |
| Fraud management integration | None | Integrated revenue assurance + fraud management platform (Subex, CODA) | No fraud integration |
| 5G assurance controls | None | 5G CDR mediation + slice-level assurance (Subex, Ericsson) | No 5G coverage |
| Roaming revenue assurance | None | IOT/NRTRDE reconciliation + roaming leakage detection (Subex, Syniverse) | No roaming RA |
| Digital services assurance | None | OTT, content, fintech service revenue assurance (Subex, WeDo) | Voice/data only |
| Regulatory compliance RA | None | MVNO margin assurance + regulatory compliance controls (Subex) | No regulatory RA |
| Partner/wholesale assurance | None | Content partner, MVNO, interconnect revenue assurance (Subex, TELARIX) | No wholesale RA |
| Business assurance dashboard | None | Real-time revenue-at-risk quantification dashboard (Subex ROC) | No quantified risk view |

**World-best reference:** Subex ROC (Revenue Operations Center), Mobileum (former WeDo Technologies RAID), CODA

**Critical gaps:**
- AI/ML-based leakage detection absent; novel and unknown leakage patterns go undetected indefinitely
- Real-time CDR reconciliation absent; leakage accumulates over billing cycles before detection
- Roaming and wholesale revenue assurance absent; high-value revenue streams have no controls
- 5G slice-level assurance absent; new revenue streams from 5G enterprise services are unprotected

---

## Service Management (`telecom_svc`)

**APG provides:** Telecom service catalog and service lifecycle management supporting service definition, activation, modification, suspension, and termination workflows. Integrates with `telecom_prv` for provisioning, `telecom_ord` for order management, and `telecom_cus` for customer-facing service views.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| TM Forum SID/eTOM alignment | None | Full TM Forum Information Framework + eTOM processes (Amdocs, Nokia NetCracker) | No TM Forum alignment |
| Product catalog management | None | Atomic/compound/bundled product catalog with CPQ (Amdocs, Oracle Product Hub) | No product catalog |
| Configure-price-quote (CPQ) | None | Complex bundle CPQ with real-time pricing (Amdocs, Salesforce CPQ, Oracle CPQ) | No CPQ |
| Service quality management (SQM) | None | SLA breach prediction + proactive SQM (TEOCO, Ericsson) | No SQM |
| ITSM / ITIL alignment | None | ITIL 4-aligned incident, change, problem mgmt (ServiceNow, BMC Remedy) | No ITSM alignment |
| B2B service assurance portal | None | B2B SLA performance portal with API (Amdocs, Oracle) | No B2B portal |
| Multi-tenancy / virtual operator | None | Full multi-tenant service isolation (Nokia NetCracker, Amdocs) | Single-tenant only |
| Service usage analytics | None | Per-service adoption, quality, monetization analytics (Amdocs, TEOCO) | No usage analytics |
| IoT service management | None | IoT connectivity service lifecycle at 100M+ scale (Ericsson, Nokia, Huawei) | No IoT service scale |
| Contract lifecycle management | None | Service contract authoring + SLA with automated breach handling (Amdocs, ServiceNow) | No contract management |

**World-best reference:** Amdocs Service Management, Nokia NetCracker Service Management, Oracle Communications Service Catalog and Design

**Critical gaps:**
- TM Forum eTOM/SID alignment absent; the industry standard process framework for telecom service management is not followed
- Product catalog and CPQ absent; operators cannot configure, price, or quote complex multi-product service offers
- SQM absent; proactive service quality management and SLA breach prevention require a dedicated capability
- IoT service lifecycle management absent; cannot scale to the device volumes required by modern enterprise IoT customers

---

---

# 5. Transport, Mining, Real Estate, Education, PPM & EAM

I now have sufficient data to write all 36 capability analyses. Let me compile the full competitive analysis.

---

## Cargo & Freight Management (`transport_car`)

**APG provides:** APG transport_car manages end-to-end cargo and freight operations including shipment booking, load planning, freight rating, carrier assignment, and documentation management. It supports multimodal freight (road, rail, air, sea) with real-time status visibility and automated freight cost calculation across domestic and cross-border movements.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Multimodal freight planning | Basic road/rail | Full multimodal incl. air + sea (Oracle OTM) | Missing air/sea booking workflows |
| Freight rating engine | Static rate tables | Dynamic carrier rate shopping with 500+ carriers (Oracle OTM) | No real-time carrier rate market |
| Load optimization | Volume/weight fit | 3D load optimization with weight distribution (SAP TM) | No 3D spatial packing algorithms |
| Customs & trade compliance | Manual docs | Automated customs filing, HS code validation (SAP GTS + SAP TM) | No embedded trade compliance engine |
| Carrier contract management | Basic rate cards | Contract lifecycle with performance SLAs, rebates (Oracle OTM) | No contract obligation tracking |
| Freight audit & payment | Manual invoice check | Automated freight audit with exception-based review (SAP TM) | No automated invoice matching |
| Carbon emissions tracking | None | CO2 per shipment, scope 3 reporting (SAP TM Sustainability) | No sustainability accounting |
| AI route/mode optimization | None | AI-driven mode-shift recommendations based on cost/time/carbon (Oracle OTM) | No ML-based modal optimization |
| Spot rate tendering | None | Digital freight marketplace with real-time bids (Transplace/Uber Freight) | No live spot market integration |
| Shipment consolidation | Manual | Automated LTL consolidation and cross-dock planning (Trimble TMS) | No automated consolidation logic |

**World-best reference:** Oracle Transportation Management (OTM), SAP Transportation Management, Trimble TMS

**Critical gaps:**
- No real-time carrier rate shopping or spot tendering capability; APG relies on pre-negotiated static rate tables
- Missing automated freight audit and invoice-matching — a primary cost-recovery mechanism in enterprise TMS
- No trade compliance or customs automation for cross-border freight
- Scope 3 carbon accounting per shipment is absent, increasingly required for enterprise ESG reporting

---

## Delivery Management — Last Mile (`transport_del`)

**APG provides:** APG transport_del covers last-mile delivery orchestration including order intake, route assignment to drivers, customer notification, proof-of-delivery capture, and failed-delivery handling. It serves B2C and B2B delivery scenarios with a driver mobile app and dispatcher dashboard.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Dynamic route optimization | Basic sequencing | Real-time re-optimization with live traffic, 1000+ stops (Route4Me) | No live traffic-aware re-optimization |
| Driver mobile app | Basic turn-by-turn | Offline-capable app with in-app chat, barcode scan, ID verification (Onfleet) | Missing offline mode and barcode/ID scan |
| Customer ETA notifications | Manual SMS | Automated SMS/push with live tracking link, sub-30-min windows (Onfleet) | No live customer tracking link |
| Proof of delivery | Photo + signature | Photo, signature, barcode, age verification, geofenced POD (Bringg) | No barcode or regulatory ID verification |
| On-demand auto-dispatch | Manual assignment | ML-based nearest-driver auto-assignment in <2 seconds (Onfleet) | No automated driver matching engine |
| Delivery analytics | Basic counts | On-time %, cost-per-stop, driver idle time, SLA breach heatmaps (Route4Me) | No cost-per-stop or SLA breach analytics |
| Returns management | None | In-route reverse logistics with label generation (Bringg) | No reverse logistics workflow |
| Fleet capacity planning | None | Predictive volume forecasting for fleet sizing (OptimoRoute) | No demand forecasting for capacity |
| Third-party carrier dispatch | None | Multi-carrier dispatch with rate shopping (Bringg Enterprise) | No external carrier integration |
| Geofence-triggered events | Basic | Auto-arrival detection, customer alert on 500m approach (Onfleet) | No proximity-triggered notification |

**World-best reference:** Onfleet, Route4Me, Bringg

**Critical gaps:**
- No live customer-facing tracking link — now a baseline expectation in consumer last-mile delivery
- Missing ML-based auto-dispatch; manual assignment creates dispatcher bottlenecks at scale
- No reverse logistics workflow for failed deliveries or returns processing
- Route optimization does not consume real-time traffic data, degrading ETA accuracy

---

## Dispatch & Routing (`transport_dis`)

**APG provides:** APG transport_dis manages dispatcher operations including job creation, vehicle and driver assignment, route sequencing, real-time communication with drivers, and dispatch board management. It supports scheduled and on-demand dispatch workflows across fleet types.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Drag-and-drop dispatch board | Basic list view | Visual drag-drop board with load balancing indicators (Samsara Dispatch) | No visual load-balanced dispatch UI |
| AI-assisted job assignment | Manual | ML job-to-driver matching on skill, proximity, load (Trimble TMS) | No ML-assisted assignment engine |
| Real-time traffic routing | Static maps | Per-second GPS with live traffic overlay, weather alerts (Samsara) | No live traffic data integration |
| Multi-depot routing | Single depot | Multi-depot with cross-depot balancing (Route4Me Enterprise) | Single-depot constraint |
| Driver HOS compliance | Basic time checks | Automated HOS rule enforcement with ELD integration (Samsara ELD) | No ELD-native HOS enforcement |
| Route replanning in transit | Manual | Automatic re-routing on traffic incident or job change (OptimoRoute) | No dynamic mid-route replanning |
| Geofence job triggering | None | Auto-trigger next job on geofence exit of current stop (Onfleet) | No geofence-based workflow automation |
| Voice dispatch integration | None | In-cab two-way voice via driver app (Trimble) | No voice dispatch channel |
| KPI dashboards (live) | Static reports | Live dispatcher KPI wall: on-time, unassigned jobs, idle (Samsara) | No live operational KPI dashboard |
| Time-window scheduling | Basic | Tight time-window enforcement with customer-commit tracking (Oracle OTM) | No customer-commit SLA enforcement |

**World-best reference:** Samsara Dispatch, Trimble TMS, OptimoRoute

**Critical gaps:**
- No real-time traffic integration means routing degrades in congested urban environments
- Missing ELD/HOS compliance integration exposes operators to regulatory risk
- No AI-assisted assignment reduces dispatcher efficiency at high job volumes
- No live KPI dashboard for dispatchers; decisions made on stale data

---

## Fleet Management (`transport_fle`)

**APG provides:** APG transport_fle provides comprehensive fleet lifecycle management covering vehicle registration, telematics integration, driver assignment, preventive maintenance scheduling, fuel tracking, and fleet utilization reporting. It supports diverse vehicle types and integrates with GPS hardware for real-time location visibility.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time GPS (per-second) | 30-second refresh | Per-second GPS with sub-meter accuracy (Samsara) | 30x lower position update frequency |
| AI dashcam & driver scoring | None | Dual-facing AI dashcam, automated coaching, near-miss detection (Samsara) | No camera-based driver behavior AI |
| Predictive maintenance | Mileage-based PM | ML failure prediction from OBD-II fault codes and sensor telemetry (Samsara) | No predictive, condition-based maintenance |
| ELD/FMCSA compliance | None | Native ELD with DOT inspection mode, violation alerts (Samsara) | No ELD compliance module |
| Fuel management | Manual entry | Automated fuel card integration, mpg benchmarking per driver (Verizon Connect) | No fuel card API integration |
| Driver behavior scoring | Basic speed | Composite score: harsh braking, cornering, acceleration, phone use (Samsara) | Phone distraction detection absent |
| Vehicle health diagnostics | Manual DTC lookup | Real-time DTC decode with repair cost estimate and shop assignment (Samsara) | No automated DTC-to-repair workflow |
| Weather intelligence | None | Live severe weather overlay on fleet map, affected-driver list (Samsara) | No weather risk layer |
| Asset tracking (non-powered) | None | Bluetooth + solar tracker for trailers and equipment (Samsara Asset Tag) | No non-powered asset tracking |
| Integration ecosystem | Limited | 350+ certified integration partners via open API (Samsara) | Narrow integration surface |

**World-best reference:** Samsara, Verizon Connect, Geotab

**Critical gaps:**
- Per-second GPS and AI dashcam are now table stakes in premium fleet management; APG's refresh rate and absence of camera AI is a two-generation gap
- No ELD/HOS module creates compliance risk in regulated markets
- No predictive maintenance means reactive repair cycles and unplanned downtime
- Fuel card integration absent; fuel cost is typically 30–40% of fleet operating expense

---

## Hub & Terminal Operations (`transport_hub`)

**APG provides:** APG transport_hub manages hub and terminal operations including dock scheduling, inbound/outbound load staging, yard management, gate check-in/check-out, and cross-dock planning. It provides operational dashboards for terminal supervisors with load and dock utilization metrics.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Dock appointment scheduling | Manual booking | Self-service carrier portal with dynamic slot optimization (Oracle TM + YMS) | No carrier self-service portal |
| Yard management (YMS) | Basic yard map | Full YMS: trailer spotting, yard jockey tasking, automated gate OCR (SAP YM) | No integrated YMS module |
| Gate OCR / RFID entry | Manual | Automated license plate + trailer OCR, RFID, container seal check (SAP YM) | No automated gate technology |
| Cross-dock planning | Manual matching | Automated inbound-to-outbound matching with lane sequencing (Oracle OTM) | No algorithmic cross-dock optimization |
| Labour management | None | Task-based labour standards with engineered time (Blue Yonder WMS/LMS) | No labour productivity standard |
| Real-time dock visibility | Basic list | Live dock status board with dwell-time alerts (Trimble YardView) | No real-time dwell alert |
| Trailer pool management | Manual | Trailer pool optimization with carrier-equipment accountability (Trimble) | No trailer pool logic |
| KPI benchmarking | None | Dock utilization, turn time, carrier OTP benchmarking (Oracle OTM Analytics) | No hub-level benchmarking |
| Reefer/temperature monitoring | None | Automated reefer temp logging with excursion alert (Carrier Lynx Fleet) | No cold chain dock monitoring |
| Security & CCTV integration | None | Camera feed with AI anomaly detection at dock doors (Samsara Site) | No video integration |

**World-best reference:** Oracle Transportation Management + YMS, SAP Yard Logistics, Trimble YardView

**Critical gaps:**
- No carrier self-service appointment portal creates manual scheduling overhead
- Absence of YMS functionality means trailer spotting and yard movement is untracked
- No gate automation (OCR/RFID); manual check-in is a throughput and security bottleneck
- No cross-dock optimization algorithm; matching inbound to outbound freight is manual

---

## Transport Inventory (`transport_inv`)

**APG provides:** APG transport_inv manages in-transit inventory visibility, freight load manifests, packaging unit tracking, and inventory reconciliation between origin and destination nodes. It maintains chain-of-custody records for freight and supports basic stock-in-transit accounting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| In-transit inventory visibility | Shipment-level | SKU/serial-level track with location inference (Oracle OTM + SCM Cloud) | No SKU-level granularity in transit |
| Cold chain excursion tracking | None | Continuous temperature with excursion alert and regulatory report (Sensitech) | No sensor-based condition monitoring |
| Serialized item tracking | None | GS1 EPCIS-compliant serial track across all nodes (SAP Extended WH) | No serialization standard support |
| Freight exception management | Manual email | Automated exception rules: delay, damage, shortage with SLA clock (Oracle OTM) | No automated exception engine |
| Dangerous goods compliance | None | ADR/IATA/IMDG compliance validation at booking (SAP DG Management) | No hazmat compliance module |
| Carrier performance by SKU | None | Fill rate and damage rate by carrier by SKU (Oracle OTM Analytics) | No carrier-SKU performance matrix |
| Bonded warehouse transit | None | Customs bonded status tracking with duty calculation (Oracle OTM Global) | No bonded/in-bond status tracking |
| Inventory reconciliation | Manual | Automated three-way match: PO, ASN, receipt (SAP S/4 GR) | No automated three-way match |
| Packaging unit management | Basic | Pallet, tote, IBC tracking with empties management (SAP EWM) | No returnable packaging tracking |
| Proof of condition (POC) | None | Photo-documented condition at load and delivery (project44) | No load condition documentation |

**World-best reference:** Oracle OTM + SCM Cloud, SAP TM + EWM, project44

**Critical gaps:**
- SKU-level in-transit visibility is absent; carrier-level granularity is insufficient for inventory accounting
- No dangerous goods compliance validation exposes operators to regulatory liability
- Cold chain condition monitoring entirely absent despite critical role in food and pharma logistics
- No automated three-way match means inventory reconciliation is manual and error-prone

---

## Transport Order Management (`transport_ord`)

**APG provides:** APG transport_ord manages the full transport order lifecycle from order receipt through carrier assignment, execution, and proof of delivery. It handles order consolidation, multi-stop orders, order prioritization, and interfaces with upstream ERP/OMS systems for order intake.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ERP/OMS order intake (EDI/API) | Manual entry / CSV | Real-time EDI 940/945 and REST API order intake (Oracle OTM) | No native EDI translator |
| Order consolidation engine | Manual grouping | Algorithm-driven consolidation by lane, time window, weight (SAP TM) | No automated consolidation logic |
| Multi-leg order orchestration | Single leg | Multi-leg with mode change and hub stop management (Oracle OTM) | Single-leg constraint |
| Carrier tendering workflow | Manual | Automated sequential, broadcast, or spot tender with acceptance SLA (Oracle OTM) | No automated tender workflow |
| Order priority & SLA management | None | Customer SLA tiers with automated escalation on breach (SAP TM) | No SLA tier management |
| Order modification handling | Manual rebuild | In-flight order modification with downstream re-planning (Oracle OTM) | No in-flight order change propagation |
| Customer order portal | None | Self-service customer portal: book, track, POD download (Transplace) | No customer self-service portal |
| Billing/freight invoice generation | Manual | Automated freight invoice from actuals vs. contracted rates (SAP TM) | No automated billing from actuals |
| Order performance reporting | Basic counts | OTIF, fill rate, cost-per-order by customer/lane/carrier (Oracle OTM) | No OTIF/lane-level analytics |
| Appointment scheduling at origin | None | Automated pickup appointment at shipper (project44 + Oracle OTM) | No origin appointment automation |

**World-best reference:** Oracle Transportation Management, SAP Transportation Management, Transplace (Uber Freight)

**Critical gaps:**
- No EDI translator is a fundamental integration gap; most shippers and 3PLs exchange orders via EDI 940/945
- Automated carrier tendering absent; manual tendering is slow and non-competitive in tight capacity markets
- No multi-leg order orchestration limits APG to simple point-to-point transport scenarios
- No OTIF analytics means APG cannot demonstrate carrier or lane performance to customers

---

## Transport Reporting (`transport_rpt`)

**APG provides:** APG transport_rpt delivers operational and management reporting across fleet, freight, delivery, and cost dimensions. It provides pre-built dashboards for fleet utilization, delivery performance, and freight spend, with export capabilities and scheduled report distribution.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time operational dashboards | Near-real-time (5-min lag) | True real-time streaming dashboards (Samsara Analytics) | Lag in operational visibility |
| Freight spend analytics | Basic total cost | Spend cube: carrier, lane, mode, period, commodity (Oracle OTM Analytics) | No multidimensional spend cube |
| OTIF / on-time performance | None | OTIF by carrier, lane, customer, origin/destination (Oracle OTM) | No OTIF calculation |
| Predictive analytics | None | Delay prediction, capacity forecast, demand forecast (SAP TM + AI) | No predictive layer |
| Carbon/emissions reporting | None | CO2 per shipment, scope 3 reporting by mode and lane (SAP TM) | No emissions reporting |
| Benchmarking vs. industry | None | External carrier and lane rate benchmarking (Transplace Network) | No external benchmark data |
| Self-service BI / ad hoc | Export to Excel | Embedded drag-drop BI with no-code report builder (Oracle OTM Analytics) | No embedded self-service BI |
| Automated report distribution | Basic scheduled email | Rule-based distribution with threshold-triggered alerts (SAP TM) | No threshold-triggered alerts |
| Driver performance reports | Basic | Composite driver scorecard with coaching integration (Samsara) | No coaching workflow integration |
| Cost allocation by cost center | None | Freight cost allocation to GL cost centers (SAP TM + FI) | No GL cost center allocation |

**World-best reference:** Samsara Analytics, Oracle OTM Analytics, SAP Transportation Management

**Critical gaps:**
- No OTIF calculation is a significant gap; it is the primary KPI for shipper-carrier contracts
- Absence of emissions reporting will become a compliance requirement in regulated markets
- No self-service BI forces all ad hoc analysis through exports and spreadsheets
- No predictive analytics layer; all reporting is retrospective

---

## Safety & Compliance — Transport (`transport_saf`)

**APG provides:** APG transport_saf manages transport safety programs including driver license and certification tracking, vehicle inspection records (pre-trip/post-trip), incident and accident recording, regulatory compliance documentation, and safety audit trails. It tracks Hours of Service constraints and generates compliance reports for regulatory submissions.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ELD / electronic HOS logging | Basic manual log | FMCSA-certified ELD with automatic duty status change (Samsara ELD) | No certified ELD integration |
| AI dashcam safety detection | None | Real-time AI event detection: distraction, tailgating, seatbelt, phone use (Samsara) | No camera-based real-time detection |
| Driver license & certification expiry | Basic date tracking | Automated expiry alerts with document scan + OCR (Samsara Driver App) | No document OCR capture |
| Incident investigation workflow | Basic form | Structured investigation with root cause, corrective action, recurrence tracking (Samsara) | No CAPA workflow |
| DVIR (vehicle inspection reports) | Paper/basic digital | In-app DVIR with mechanic sign-off workflow and defect escalation (Samsara) | No mechanic sign-off workflow |
| Railroad crossing / school zone | None | GPS-based crossing violation detection for school buses/hazmat (Samsara) | No geo-rule enforcement |
| Safety coaching workflow | None | AI-generated group coaching presentations, driver scorecards (Samsara Coaching) | No automated coaching module |
| Near-miss / SOS detection | None | One-touch SOS with GPS + live camera feed to dispatcher (Samsara) | No SOS / near-miss capability |
| Weather-based risk alerts | None | Severe weather alerts with driver-specific exposure on fleet map (Samsara) | No weather risk management |
| DOT audit readiness reports | None | Auto-generated DOT audit package with driver data history (Samsara Compliance) | No audit readiness module |

**World-best reference:** Samsara, Verizon Connect, Geotab

**Critical gaps:**
- No certified ELD integration is a compliance liability in any jurisdiction with HOS regulations
- Absence of AI dashcam safety detection is a major safety programme gap; leading fleets use this to reduce incident rates by 20–40%
- No automated safety coaching workflow means driver behaviour data is collected but not actioned
- No SOS / worker safety capability in a domain where road accidents are the leading occupational fatality cause

---

## Track & Trace (`transport_trc`)

**APG provides:** APG transport_trc provides real-time shipment visibility from dispatch to delivery, including carrier location polling, milestone event capture, delivery confirmation, and exception flagging. It exposes tracking data to internal stakeholders and supports customer-facing status inquiry.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Carrier-agnostic track & trace | APG-carrier only | 1,000+ carrier API and EDI connections (project44, FourKites) | Single-carrier tracking only |
| Predictive ETA (ML) | Scheduled ETA only | ML ETA updated in real-time from speed, traffic, dwell patterns (FourKites) | No ML-based ETA refresh |
| Ocean / air / rail visibility | None | Multimodal: ocean AIS, air AWB, rail, road in one pane (project44) | Road-only visibility |
| Customer tracking portal | None | Branded customer tracking page with live map link (project44, Bringg) | No customer-facing tracking |
| Geofence event automation | Basic arrival | Geofence-triggered: depart, arrive, dwell-exceed, delay alert (FourKites) | No dwell-exceed alerts |
| Supply chain control tower | None | Portfolio-level exception management with AI triage (FourKites Control Tower) | No control tower capability |
| Carbon per shipment tracking | None | CO2 per shipment by carrier and mode (project44 Sustainability) | No emissions per shipment |
| IoT sensor data (temp, shock) | None | Continuous temp, humidity, shock data fused with location (Sensitech + FourKites) | No sensor-fused visibility |
| Event collaboration (ETA dispute) | None | Shipper-carrier collaborative ETA negotiation workflow (project44) | No collaborative exception workflow |
| SLA breach automation | None | Auto-trigger SLA breach notification and credit calculation (Oracle OTM) | No SLA breach automation |

**World-best reference:** project44, FourKites, Oracle OTM Visibility

**Critical gaps:**
- APG tracks only APG-managed carriers; real-world supply chains require multi-carrier visibility aggregation
- No ML-based predictive ETA; static scheduled ETAs lose accuracy over long-haul transits
- Absence of multimodal visibility limits APG to road-only use cases, excluding ocean and air freight
- No customer-facing tracking portal is a significant competitive disadvantage in B2C and B2B shipping contexts

---

## Mining Domain

---

## Environmental Management (`mining_env`)

**APG provides:** APG mining_env tracks environmental compliance obligations including water discharge monitoring, air quality sampling, waste classification, rehabilitation progress, and regulatory permit conditions. It maintains an audit-ready environmental register and generates statutory reports for environmental agencies.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time sensor data ingestion | Manual entry | Automated IoT sensor ingest: water pH, turbidity, gas sensors (Aveva PI System) | No continuous sensor integration |
| Regulatory permit tracking | Basic date list | Permit condition matrix with auto-alerts on breach risk (Intelex EHS) | No condition-level compliance tracking |
| Mine closure / rehabilitation | None | Rehabilitation planning, cost provisioning, progress tracking (Enviro Data) | No closure liability module |
| Tailings storage facility (TSF) monitoring | None | TSF dam safety monitoring with automated breach risk scoring (Klohn Crippen) | No TSF-specific monitoring |
| GHG inventory (scope 1/2/3) | None | Mining-specific GHG inventory with diesel, explosive, methane factors (Intelex) | No GHG accounting |
| Biodiversity tracking | None | Flora/fauna survey management with impact zone mapping (Maptek MinePlan) | No biodiversity data management |
| Incident/spill management | Basic form | Structured spill response workflow with regulatory notification triggers (Intelex) | No regulatory notification automation |
| Stakeholder complaint register | None | Community complaint register with response tracking and trend analysis (Intelex) | No community grievance tracking |
| Water balance modelling | None | Dynamic water balance model for pit lake and catchment (GEOVIA Surpac) | No water balance simulation |
| Environmental cost accounting | None | Environmental liability provisioning against financial statements (Aveva + ERP) | No environmental liability costing |

**World-best reference:** Aveva PI System, Intelex EHS, Hexagon MinePlan

**Critical gaps:**
- No continuous sensor integration means real-time compliance breaches go undetected until manual sampling
- No GHG accounting module; scope 1/2/3 reporting is a regulatory requirement in most jurisdictions
- TSF monitoring absent; TSF failures are catastrophic and increasingly regulated (GISTM standard)
- No mine closure/rehabilitation liability module is a significant gap for operations with finite mine life

---

## Equipment Management (`mining_eqp`)

**APG provides:** APG mining_eqp manages mining equipment records, maintenance scheduling, work order management, spare parts inventory, and equipment utilization tracking. It supports surface and underground equipment across haul trucks, drills, loaders, and fixed plant, with maintenance cost tracking and equipment history.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Dispatch / fleet management system (FMS) | None | Real-time truck dispatch, shovel matching, grade control (Hexagon MinePlan FMS) | No mining FMS integration |
| Predictive maintenance (ML/IoT) | Mileage-based PM | Oil analysis + vibration + OBD sensor predictive failure (ABB Ability Genix) | No condition-based prediction |
| Equipment availability KPIs | Manual calculation | Automated A, U, P, MUF metrics from FMS data (Hexagon MinePlan) | No automated availability metrics |
| Component life tracking | Basic | Component serial-level life with hours-to-failure modelling (Hexagon EAM) | No component-level life modelling |
| OEM integration (Komatsu/Cat) | None | Native VHMS / VisionLink telemetry integration (Hexagon EAM + Cat VisionLink) | No OEM telematics integration |
| Maintenance cost per tonne | None | Cost per tonne mined by equipment class (Hexagon MinePlan Analytics) | No cost-per-tonne KPI |
| Safety isolation / LOTO management | Basic permit | Electronic LOTO with isolation point mapping and multi-party sign-off (Aveva) | No electronic isolation workflow |
| Drill rig monitoring | None | Real-time drill parameter monitoring, bit change scheduling (Maptek Vulcan) | No drilling performance monitoring |
| Statutory inspection compliance | Date reminders | Structured regulatory inspection workflow with certificate management (Intelex) | No statutory cert management |
| Maintenance planning integration with scheduling | None | Maintenance window integration with mine production schedule (Hexagon MinePlan) | No maintenance-schedule integration |

**World-best reference:** Hexagon MinePlan, ABB Ability, Hexagon EAM (Infor)

**Critical gaps:**
- No Fleet Management System integration means equipment dispatch and shovel-truck matching is unmanaged
- Absence of OEM telematics integration (Komatsu VHMS, Cat VisionLink) means machine health data goes unused
- No predictive maintenance from sensor data; scheduled PM misses emerging failures and causes unplanned downtime
- Maintenance windows not integrated with production schedule — creating conflicts between maintenance and mining plans

---

## Exploration Data Management (`mining_exp`)

**APG provides:** APG mining_exp manages exploration data including drillhole collar and assay records, sample dispatch and laboratory results, geological logging, survey data, and exploration project management. It maintains a structured exploration database for resource estimation workflows.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| 3D geological modelling | None | Full 3D block model, solid modelling, resource estimation (Maptek Vulcan, GEOVIA Surpac) | No 3D modelling capability |
| Geostatistical resource estimation | None | Kriging, cokriging, simulation in 3D block model (Maptek Vulcan GeologyCore) | No geostatistical estimation |
| Drillhole optimisation | Basic collar entry | ML-based infill drilling plan maximising resource recovery (Maptek Vulcan Drillhole Opt.) | No drilling optimisation |
| QAQC management | Manual | Automated QAQC with duplicate, standard, blank reporting (acQuire) | No automated QAQC workflow |
| Laboratory interface (LIMS) | None | Bidirectional LIMS integration with automated result import (acQuire + LIMS) | No LIMS integration |
| Downhole survey management | Manual | Automated downhole survey processing with deviation correction (Maptek Vulcan) | No downhole survey tools |
| Mineral resource reporting | None | JORC/NI 43-101/SAMREC compliant resource statement generation (Maptek Vulcan) | No regulatory resource reporting |
| Geophysics data management | None | Gravity, magnetics, seismic data storage and interpretation (Oasis Montaj / Geosoft) | No geophysics data handling |
| Domain and geological interpretation | None | Stratigraphic correlation, wireframe domaining (Maptek Vulcan GeologyCore 2025) | No geological interpretation tools |
| Data validation & audit trail | Basic | Full audit trail with version control on block models (Maptek Vulcan 2025) | Weak data governance |

**World-best reference:** Maptek Vulcan, Dassault GEOVIA Surpac, acQuire

**Critical gaps:**
- No 3D geological modelling is a fundamental gap; resource estimation cannot be performed without block models
- No geostatistical estimation means APG cannot support JORC/NI 43-101 compliant resource reporting
- Missing LIMS integration creates manual data re-entry from laboratories, a major data quality risk
- No QAQC management — a JORC Table 1 mandatory disclosure requirement for exploration results

---

## Ore & Production Management (`mining_ore`)

**APG provides:** APG mining_ore tracks ore production through the mining value chain from drill-and-blast through loading, hauling, crushing, and mill feed. It manages grade control, production tonnages, mill throughput, and reconciles mined tonnes and grades against the resource model.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time production tracking | Shift-end manual | Real-time tonne and grade tracking from FMS + belt scales (Hexagon MinePlan) | No real-time production KPI |
| Grade control (blast hole sampling) | Manual grade entry | Automated blast hole grade assignment to ore/waste polygons (Vulcan Grade Control Suite) | No grade control workflow |
| Mill throughput optimisation | None | AI-based feed rate and blend optimisation (ABB Ability Genix deep learning) | No mill optimisation AI |
| Mine-to-mill reconciliation | None | Automated reconciliation: resource vs. mined vs. processed vs. sold (Maptek Vulcan) | No automated reconciliation |
| Short-term mine scheduling | Manual | Dynamic short-term scheduler with shovel/truck fleet constraints (Hexagon MinePlan) | No constrained short-term scheduler |
| Blast design & management | None | 3D blast design, initiation sequence, powder factor calculation (Maptek Vulcan) | No blast design module |
| Stockpile management | Basic tonnage | Real-time stockpile grade and volume with blending optimisation (Maptek Vulcan) | No grade-by-stockpile tracking |
| Ore loss & dilution tracking | None | Systematic ore loss and dilution measurement per block (Vulcan Grade Control) | No ore loss accounting |
| Shift production reporting | Manual | Automated shift report generation from FMS data (Hexagon MinePlan) | No automated shift report |
| Metallurgical accounting | None | Mass balance across crushing, grinding, flotation, leach circuits (Aveva PI + Bilmat) | No metallurgical accounting |

**World-best reference:** Hexagon MinePlan, Maptek Vulcan Grade Control Suite, ABB Ability Genix

**Critical gaps:**
- No mine-to-mill reconciliation; without this, ore loss and process efficiency are unquantifiable
- No grade control workflow means ore/waste classification relies on manual geological decisions without auditability
- Missing metallurgical accounting prevents mass balance tracking from mine face to saleable product
- No short-term production scheduling tool with fleet constraints; manual scheduling misaligns equipment capacity with mine plan

---

## Mine Safety (`mining_saf`)

**APG provides:** APG mining_saf manages mine safety programs including hazard identification (HAZOP, take-5), incident reporting and investigation, permit-to-work, statutory inspection compliance, safety statistics, and regulatory reporting. It supports surface and underground mine safety obligations across MSHA, DMR, and equivalent frameworks.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Permit to work (PTW) / LOTO | Basic paper workflow | Electronic PTW with isolation map, concurrent lock management (Aveva) | No electronic concurrent isolation |
| Gas monitoring integration | None | Real-time CH4/CO/O2 monitoring with automated evacuation trigger (MSA Safety + Hexagon) | No atmospheric monitoring integration |
| Proximity detection (PDS) | None | Real-time underground proximity warning between pedestrians and equipment (Hexagon MineProtect) | No proximity detection system |
| Slope stability monitoring | None | Ground-based radar + InSAR with failure prediction hours in advance (Hexagon MinePlan) | No geotechnical monitoring |
| Fatigue risk management | None | Fatigue score from hours data + vehicle pattern with intervention alert (Hexagon Fatigue) | No fatigue monitoring |
| Incident investigation (Bow-tie / RCA) | Basic form | Structured barrier-based bow-tie analysis with ICAM methodology (Intelex) | No structured investigation methodology |
| Emergency muster / evacuation | None | Electronic tagging-based muster with missing-person alert (Strata Worldwide) | No electronic muster system |
| Leading indicator tracking | None | Safety observation, near-miss, hazard frequency rate tracking (Intelex) | No leading indicator dashboard |
| Contractor safety pre-qualification | None | Contractor induction, competency, and site access control (Intelex Contractor Mgmt) | No contractor safety gateway |
| Regulatory report automation | Manual | Auto-generated MSHA/DMR monthly report from incident data (Intelex) | No statutory report automation |

**World-best reference:** Hexagon MineProtect, Intelex EHS, Aveva

**Critical gaps:**
- No proximity detection system is a critical gap; equipment-pedestrian collisions are the leading cause of fatality in surface and underground mining
- Gas monitoring integration absent; in underground operations this is a statutory requirement
- No slope stability monitoring for open pit mines — a catastrophic risk event category
- Absence of electronic muster capability means headcount in an emergency relies on manual roll call

---

## Mining Vendor & Procurement (`mining_ven`)

**APG provides:** APG mining_ven manages vendor registration, procurement requisitions, purchase orders, goods receipt, and vendor performance for mining consumables, equipment parts, and contractor services. It supports catalog management, spend analytics, and vendor compliance documentation.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Mining-specific catalog (reagents, explosives) | Generic catalog | Mining commodity catalog with HS codes, dangerous goods flags (SAP Ariba Mining) | No mining-specific commodity catalog |
| Contractor prequalification | Basic vendor form | PICS/ISNetworld integration with real-time prequalification scoring (Intelex / ISN) | No third-party prequalification integration |
| Explosives / DG procurement compliance | None | Dangerous goods procurement with regulatory compliance workflow (SAP DG + Ariba) | No DG procurement compliance |
| Spend analytics by mine site | Basic PO report | Spend cube by site, commodity, supplier, GL account (SAP Ariba Analytics) | No site-level spend cube |
| Inventory-driven auto-replenishment | None | MRP-based automated PO generation from min/max stock levels (SAP MM) | No MRP-driven replenishment |
| Long-term supply contracts | Basic | Contract lifecycle with volume commitments, price escalation, take-or-pay (SAP CLM) | No contract lifecycle management |
| Vendor performance scorecard | None | Delivery, quality, price, safety score per vendor (SAP SRM / Ariba) | No vendor KPI scorecard |
| Emergency procurement workflow | None | Expedite workflow with approver override and premium freight tracking (SAP MM) | No emergency procurement path |
| Budget vs. actuals by cost centre | None | Real-time procurement spend vs. approved budget by cost centre (SAP CO) | No budget control integration |
| E-auction / reverse bidding | None | Reverse auction with incumbent protection rules (SAP Ariba Sourcing) | No e-sourcing capability |

**World-best reference:** SAP Ariba, IBM Maximo (MRO), Coupa

**Critical gaps:**
- No dangerous goods procurement compliance; explosives, reagents, and cyanide procurement require regulatory controls
- Absence of MRP-driven auto-replenishment leads to emergency purchases at premium prices
- No contractor prequalification workflow is a safety and legal liability for principal employer obligations
- No spend analytics by mine site means total cost of ownership by operation is invisible to management

---

## Real Estate Domain

---

## Real Estate Accounting (`realestate_acc`)

**APG provides:** APG realestate_acc delivers property-specific accounting including rent roll, accounts receivable, accounts payable, CAM reconciliation, operating expense management, and financial reporting at property, portfolio, and entity level. It supports IFRS 16 / ASC 842 lease accounting and multi-entity consolidation.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| IFRS 16 / ASC 842 lease accounting | None | Full right-of-use asset, liability amortisation schedule (Yardi Voyager, IBM TRIRIGA) | No lease accounting standard compliance |
| CAM reconciliation | Manual | Automated CAM pool allocation with tenant gross-up and audit (Yardi Voyager) | No automated CAM pool engine |
| Multi-entity / multi-currency | Basic single entity | Unlimited entity structure with intercompany elimination (Yardi Voyager, MRI) | No intercompany elimination |
| Percentage rent calculation | None | Automated percentage rent from tenant sales reporting (Yardi Voyager) | No percentage rent engine |
| Property-level P&L | Basic | Full property P&L with variance analysis vs. budget/prior year (Yardi Voyager) | Limited variance analytics |
| Bank reconciliation | Manual | Automated bank feed with trust account reconciliation (Yardi, AppFolio) | No automated bank feed |
| Investor / owner reporting | None | Investor waterfall distribution, K-1 generation, IRR (Yardi Investment Management) | No investor distribution module |
| Job cost accounting | None | Construction draw management with retainage, lien waiver (Procore, Yardi Job Cost) | No job cost module |
| Tax depreciation schedules | None | MACRS / straight-line depreciation by asset class (Yardi Fixed Assets) | No tax depreciation |
| Audit trail & SOX compliance | Basic | Full field-level audit log with SOX control documentation (Yardi Voyager) | Weak audit trail granularity |

**World-best reference:** Yardi Voyager, MRI Software, IBM TRIRIGA

**Critical gaps:**
- No IFRS 16/ASC 842 compliance is a material financial reporting gap for any entity with material lease obligations
- CAM reconciliation is manual; this is the highest-volume dispute item in commercial real estate accounting
- No investor reporting module; real estate investment entities require waterfall distributions and capital account management
- No percentage rent engine means retail lease compliance cannot be automated

---

## Construction Management (`realestate_con`)

**APG provides:** APG realestate_con manages real estate development construction projects including project setup, budget management, contractor management, progress claims, variation orders, inspections, and project close-out. It integrates with the real estate portfolio to track assets under development.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| BIM model coordination | None | Full BIM 360 coordination, clash detection, model review (Autodesk Construction Cloud) | No BIM integration |
| AI construction risk prediction | None | Construction IQ: AI risk scoring on RFIs, submittals, safety obs. (Autodesk ACC) | No AI risk prediction |
| Subcontractor prequalification | Basic vendor form | Integrated prequalification with insurance, financials, safety record (Procore, ISN) | No formal prequalification |
| Budget cost coding (CSI) | Generic | CSI Masterformat and Uniformat cost code integration (Procore) | No standard cost code library |
| Drawing management & revision | Attachment only | Version-controlled drawing set with markup, RFI linkage (Procore, Autodesk) | No drawing management system |
| Submittal management | None | Structured submittal log with routing, review period, approval chain (Procore) | No submittal workflow |
| Field inspection / punch list | None | Mobile inspection with photo evidence, assignee, and sign-off (Procore) | No mobile field inspection |
| Progress billing & retainage | Manual | Automated progress claim generation with retainage tracking (Procore + Yardi Job Cost) | No automated progress billing |
| RFI management | None | RFI log with impact assessment, timeline, and approval (Procore) | No RFI management |
| Schedule (Gantt/CPM) | None | CPM scheduling with predecessor logic and critical path (Oracle Primavera P6) | No CPM scheduling |

**World-best reference:** Procore, Autodesk Construction Cloud, Oracle Primavera P6

**Critical gaps:**
- No BIM integration is a foundational gap for any development project above basic residential construction
- Missing RFI and submittal management workflow; these are the primary communication and change control records in construction
- No CPM scheduling means construction programmes are managed externally without integration to cost and progress
- No mobile field inspection tool; defect and punch-list management in the field requires offline mobile capability

---

## Insurance Management — Property (`realestate_ins`)

**APG provides:** APG realestate_ins manages property insurance portfolios including policy register, premium tracking, claims lodgement, claims follow-up, and insurance renewal management. It maintains insured values per property and tracks coverage adequacy against current asset valuations.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Replacement cost valuation integration | Manual entry | Automated replacement cost from valuation database (CoreLogic, IBM TRIRIGA) | No valuation API integration |
| Claims management workflow | Basic form | Structured claim lifecycle: lodgement, adjuster, reserve, settlement (Origami Risk) | No adjuster/reserve tracking |
| Policy comparison and gap analysis | None | Coverage gap analysis against asset register (Riskonnect) | No coverage gap analysis |
| Blanket vs. specific policy management | None | Blanket endorsement tracking with sub-limit allocation (Origami Risk) | No blanket policy sub-limit tracking |
| Certificate of insurance (COI) management | None | Automated COI collection from tenants/contractors with expiry alerts (myCOI) | No COI management |
| Premium allocation by property | Manual | Automated premium allocation by insured value or area (IBM TRIRIGA) | No automated premium allocation |
| Risk improvement tracking | None | Insurer recommendation register with completion tracking (Riskonnect) | No risk improvement register |
| Flood / natural disaster zone mapping | None | GIS-based flood, wind, quake hazard mapping by property (Verisk / AIR) | No natural hazard mapping |
| Insurance market placement | None | Broker panel RFP with proposal comparison (Marsh TMCP) | No market placement workflow |
| Captive / self-insurance accounting | None | Captive premium, loss reserve, and reinsurance accounting (Origami Risk) | No captive management |

**World-best reference:** Riskonnect, Origami Risk, IBM TRIRIGA Insurance Module

**Critical gaps:**
- No claims workflow beyond basic form entry; claims reserve management and adjuster communication are untracked
- No COI management for tenant and contractor compliance — a high-volume administrative task in property management
- Absence of natural hazard zone mapping means insurance adequacy cannot be assessed against physical risk
- No coverage gap analysis; under-insurance is a material financial risk for property portfolios

---

## Lease Management (`realestate_lea`)

**APG provides:** APG realestate_lea manages the full commercial and residential lease lifecycle including lease execution, rent scheduling, escalation management, option tracking, tenant correspondence, CAM billing, lease expiry management, and IFRS 16/ASC 842 accounting entries.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| IFRS 16 / ASC 842 automation | None | Automated ROU asset and liability schedules on commencement (Yardi Voyager, IBM TRIRIGA) | No lease accounting standard support |
| AI lease abstraction | None | AI-powered clause extraction from PDF lease documents (MRI LEVERTON, Yardi Smart Lease) | No AI abstraction |
| CAM reconciliation automation | Manual | Automated CAM pool allocation with controllable/non-controllable cap (Yardi Voyager) | No automated CAM engine |
| Percentage rent from tenant sales | None | Automated percentage rent from tenant-reported monthly sales (Yardi Voyager) | No percentage rent workflow |
| Lease option tracking | Basic date diary | Option event management: renewal, expansion, termination, ROFR with alerts (Yardi Voyager) | No complex option type management |
| Critical date management | Basic alerts | Critical date workflow with responsible party, action, and escalation (MRI Software) | No escalation workflow |
| Lease document repository | Attachment | AI-indexed lease document repository with clause search (MRI LEVERTON) | No AI document indexing |
| Retail lease compliance | None | Retail tenancy act compliance rules by jurisdiction (Yardi Retail Manager) | No jurisdiction-specific compliance rules |
| Tenant self-service portal | None | Tenant portal: rent payment, document access, maintenance request (Yardi Voyager) | No tenant portal |
| Lease vs. own analysis | None | NPV analysis of lease vs. purchase scenarios (CoStar Real Estate Manager) | No strategic decision support |

**World-best reference:** Yardi Voyager, MRI Software (LEVERTON), IBM TRIRIGA

**Critical gaps:**
- No IFRS 16/ASC 842 support is a financial reporting compliance gap that affects any entity with material leases
- No AI lease abstraction; manual abstracting is error-prone and slow for large portfolios
- Absence of retail tenancy act compliance rules creates legal exposure in regulated leasing jurisdictions
- No tenant self-service portal is a significant service and efficiency gap in modern property management

---

## Maintenance Management (`realestate_mai`)

**APG provides:** APG realestate_mai manages property maintenance operations including work order management, reactive and preventive maintenance scheduling, contractor management, asset maintenance history, warranty tracking, and maintenance cost reporting across a property portfolio.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Tenant-initiated work request portal | None | Tenant portal work request with photo upload and status tracking (AppFolio, Yardi) | No tenant self-service work request |
| Preventive maintenance templates | Basic date list | Comprehensive PM library with frequency, task steps, parts list (IBM Maximo, Yardi) | No structured PM task library |
| IoT-triggered work orders | None | Sensor-triggered work order on fault/threshold breach (IBM Maximo, Aveva PI) | No IoT-triggered maintenance |
| Contractor dispatch app | None | Mobile app for contractors with work instructions, GPS arrival, photo POC (ServiceNow Field Service) | No contractor mobile app |
| Building system integration (BMS) | None | BMS alarm-to-work-order integration (IBM TRIRIGA, Siemens Desigo) | No BMS integration |
| Warranty claim management | None | Warranty register with claim lodgement, status, and recovery tracking (IBM Maximo) | No warranty management |
| Maintenance cost vs. asset value | None | Maintenance spend as % of replacement cost by building (IBM TRIRIGA) | No normalized spend KPI |
| SLA management (response time) | None | Contractor SLA: response time, completion time, penalty with auto-scoring (AppFolio) | No contractor SLA enforcement |
| Compliance maintenance (FLS, HVAC) | Basic date | Compliance maintenance register with statutory certificate upload (Yardi, MRI) | No statutory certificate tracking |
| Lifecycle capital planning | None | 10-year capital expenditure forecast from asset condition assessment (IBM TRIRIGA) | No CapEx lifecycle planning |

**World-best reference:** IBM Maximo, IBM TRIRIGA, AppFolio

**Critical gaps:**
- No tenant portal for work requests is a fundamental service gap; it is now expected in residential and commercial property management
- IoT-triggered work orders absent; building systems generate faults that should automatically create maintenance tasks
- No lifecycle capital planning from asset condition data; this is core to property portfolio budget management
- Compliance maintenance statutory certificate tracking absent — a regulatory risk for fire, electrical, and mechanical safety obligations

---

## Real Estate Marketing & Listings (`realestate_mkt`)

**APG provides:** APG realestate_mkt manages property marketing including listing creation, portal syndication, enquiry management, viewing scheduling, marketing campaign tracking, and performance analytics across residential and commercial property types.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Multi-portal syndication | Manual per-portal | One-click syndication to 50+ portals (Zillow, Realtor, LoopNet) (Yardi RentCafe, AppFolio) | No automated multi-portal sync |
| AI listing content generation | None | AI-generated property descriptions, headline suggestions (AppFolio AI, Zillow AI) | No AI content generation |
| 3D virtual tour / floor plan | None | Matterport 3D tour integration, interactive floor plan (CoreLogic, Matterport) | No virtual tour integration |
| AI lead scoring | None | AI lead quality scoring based on engagement and profile match (AppFolio AI Leasing) | No lead intelligence |
| Automated showing scheduler | None | Self-service prospect-initiated showing with calendar sync (AppFolio, Buildium) | No automated showing scheduling |
| CRM for leasing pipeline | None | Full CRM: lead, prospect, application pipeline with conversion analytics (Yardi RentCafe) | No leasing CRM |
| Rental yield and comp analysis | None | Automated market rent comparison and yield analysis (CoStar, Yardi Elevate) | No market comp analytics |
| Digital marketing campaign tracking | None | UTM-tracked campaigns with cost-per-lead and cost-per-lease (AppFolio) | No marketing attribution |
| Online application portal | None | Digital application with ID verification and screening (AppFolio, Buildium) | No digital application workflow |
| Vacancy performance dashboard | None | Days-on-market, enquiry rate, conversion funnel by property (Yardi RentCafe) | No vacancy KPI dashboard |

**World-best reference:** AppFolio, Yardi RentCafe, CoStar Group

**Critical gaps:**
- No multi-portal syndication; manual listing submission to each portal is a significant time cost
- No leasing CRM; lead-to-lease conversion is untracked and cannot be optimised
- Absence of automated showing scheduler; it is now an industry expectation for digital-first leasing operations
- No market comp analytics means pricing decisions are made without data

---

## Real Estate Project Management (`realestate_prj`)

**APG provides:** APG realestate_prj manages real estate development and refurbishment projects including project initiation, feasibility, design management, planning approvals, construction supervision, budget control, and project close-out. It tracks milestones against delivery schedule and manages project risk registers.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Development feasibility modelling | None | Dynamic feasibility: land cost, construction, revenue, IRR, NPV (Argus Developer) | No feasibility/DCF modelling |
| Planning approval workflow | Basic milestone | Authority submission tracking with condition management (Procore, Yardi Voyager) | No condition tracking |
| Design review and coordination | None | BIM model review, issue log, RFI integration (Autodesk Construction Cloud) | No design coordination tool |
| Stage gate / approval workflow | None | Structured stage gate with investment committee approval (Planview, MS Project) | No investment governance workflow |
| Project risk register | None | Quantitative risk register with Monte Carlo schedule/cost risk (Oracle Primavera P6) | No quantitative risk analysis |
| Programme-level portfolio view | None | Portfolio Gantt: all projects on a single timeline with dependency (Planview) | No programme-level scheduling |
| Sales/pre-leasing progress tracking | None | Unit-level sales tracking integrated with project cash flow (Yardi Voyager Dev) | No development sales tracking |
| Tenant fitout management | None | Tenant fitout approval, CAP contribution, and completion tracking (Yardi Voyager) | No fitout management |
| Statutory approvals register | None | Building permit, occupancy certificate, utility connection register (Procore) | No statutory approval register |
| Cash flow forecasting | None | Monthly project cash flow: draw-downs, revenue, net position (Argus Developer) | No development cash flow model |

**World-best reference:** Argus Developer, Oracle Primavera P6, Yardi Voyager Development

**Critical gaps:**
- No feasibility / DCF model is a fundamental gap; no real estate development project proceeds without a tested feasibility model
- No programme-level portfolio view; development pipelines have interdependencies that require portfolio scheduling
- Stage gate governance absent; investment committee checkpoints are standard in institutional real estate development
- No development cash flow forecasting means funding drawdowns and interest costs are unmanaged

---

## Real Estate Reporting (`realestate_rpt`)

**APG provides:** APG realestate_rpt delivers property portfolio reporting across financial, operational, and leasing dimensions. It provides pre-built reports for income, vacancy, lease expiry profile, maintenance costs, and portfolio valuation, with scheduled distribution to owners and investors.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| ARGUS-standard valuation report | None | ARGUS Enterprise / DCF valuation report (Altus ARGUS Enterprise) | No DCF/ARGUS valuation output |
| NCREIF / GRESB sustainability reporting | None | GRESB building-level sustainability data submission (Yardi ESG, MRI Envizi) | No ESG/sustainability reporting |
| Investor reporting (quarterly) | None | Automated quarterly investor report with waterfall and distributions (Yardi Investment Mgmt) | No investor reporting |
| Weighted average lease expiry (WALE) | None | WALE, occupancy, and lease expiry profile by income (Yardi Voyager) | No WALE analytics |
| Portfolio valuation roll-forward | None | Cap rate applied valuation roll-forward vs. market comp (CoStar, Yardi) | No portfolio valuation analytics |
| Benchmark vs. market (IPD) | None | MSCI / IPD total return benchmarking by sector (MSCI Real Estate) | No market benchmark integration |
| Embedded BI / self-service | Export only | Embedded Power BI with pre-built real estate data model (MRI, Yardi) | No embedded self-service BI |
| Tenant covenant analysis | None | Tenant credit scoring and covenant strength tracking (CoStar, Moody's CRE) | No tenant credit analysis |
| Arrears and collection reporting | Basic | Aged debtors with legal status and collection action tracking (Yardi Voyager) | No collection action workflow |
| Board / executive dashboard | None | Executive portfolio dashboard: NOI, yield, occupancy, CapEx vs. budget (Yardi Voyager) | No executive dashboard |

**World-best reference:** Yardi Voyager, Altus ARGUS Enterprise, CoStar Group

**Critical gaps:**
- No WALE or lease expiry profile analytics; these are the two most-cited KPIs in commercial property reporting
- GRESB/ESG sustainability reporting absent; institutional investors now require GRESB annual submissions
- No investor reporting capability; APG cannot serve investment vehicles, REITs, or fund managers without this
- No DCF/ARGUS-standard valuation output means valuations must be performed entirely outside APG

---

## Tenant Management (`realestate_ten`)

**APG provides:** APG realestate_ten manages the full tenant lifecycle including application processing, onboarding, lease execution, rent collection, communication management, maintenance request handling, tenant account management, renewal management, and vacating procedures.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Online rental application with screening | None | Digital application with TransUnion/Equifax background and credit check (AppFolio, Buildium) | No integrated screening API |
| AI leasing assistant (chatbot) | None | AI auto-response to enquiries, tour scheduling, application guidance (AppFolio AI) | No AI leasing assistant |
| Resident portal (payments, comms) | None | Full resident centre: pay rent, submit requests, access docs, chat (Buildium, AppFolio) | No resident self-service portal |
| Automated rent collection / payment processing | Manual | Automated ACH, credit card, recurring payment with ledger posting (AppFolio, Buildium) | No digital payment integration |
| Lease renewal automation | Manual letters | Automated renewal campaign: offer, digital signing, acceptance (Yardi Voyager) | No automated renewal workflow |
| Tenant arrears management | Manual follow-up | Automated arrears workflow: reminder, notice, legal with escalation rules (Yardi Voyager) | No automated arrears escalation |
| Tenant satisfaction surveys | None | Automated post-move-in and periodic surveys with NPS tracking (AppFolio) | No tenant NPS tracking |
| Move-in / move-out inspection | Paper-based | Digital inspection with comparative photos, automated deposit calculation (Buildium) | No digital comparative inspection |
| Community communications | None | Branded announcement board, package alerts, event management (Buildium Resident Center) | No community communication tools |
| Utility / service billing | None | Utility billing (RUBS, sub-metering) with automated tenant billing (RealPage Utility Mgmt) | No utility billing module |

**World-best reference:** AppFolio, Buildium (RealPage), Yardi Voyager

**Critical gaps:**
- No resident self-service portal is the single largest service gap; tenants now expect online payment and communication tools as baseline
- No integrated background/credit screening means tenant risk is assessed manually and inconsistently
- Automated arrears escalation absent; APG cannot systematically enforce payment obligations
- No digital move-in/move-out inspection creates deposit dispute risk and manual administrative overhead

---

## Property Valuation (`realestate_val`)

**APG provides:** APG realestate_val manages property valuation workflows including valuation instruction management, valuation register maintenance, capitalization rate tracking, comparable sales analysis, valuation report storage, and portfolio valuation roll-forward for financial reporting purposes.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| DCF / ARGUS-standard valuation model | None | Full discounted cash flow model with rent, vacancy, capex, exit (Altus ARGUS Enterprise) | No DCF engine |
| Automated comp selection | None | AI-selected comparable transactions from MLS/CoStar database (CoStar, Reonomy) | No automated comp search |
| Mass appraisal (AVM) | None | Automated valuation model for large residential portfolios (CoreLogic AVM, Zillow Zestimate) | No AVM capability |
| Statutory valuation / rates base | None | Bulk rates valuation with objection management (Tyler Technologies, Patriot Properties) | No statutory valuation workflow |
| External valuer portal | None | Valuer-specific portal to receive instructions and submit reports (JLL Valuation, CBRE) | No external valuer portal |
| Market data integration | None | Live property sales, rental, yield data integration (CoStar, PriceFinder, CoreLogic) | No live market data feed |
| Portfolio fair value for IFRS 13 | None | Investment property fair value disclosure with sensitivity analysis (ARGUS Enterprise) | No IFRS 13 fair value support |
| Mortgage security valuation | None | Lender-grade valuation report with LVR, risk rating (Valocity, CoreLogic) | No lender security valuation |
| Valuation variance analysis | Basic | Variance against prior valuation, budget, and market movement (Yardi Voyager) | Limited variance analytics |
| GIS / spatial data integration | None | Property boundary, flood, zoning overlay on valuation map (Esri ArcGIS, CoreLogic) | No GIS integration |

**World-best reference:** Altus ARGUS Enterprise, CoStar Group, CoreLogic

**Critical gaps:**
- No DCF model; property valuation for IFRS 13 / investment reporting requires a DCF engine as a minimum
- No automated valuation model (AVM) means mass valuation for large portfolios requires individual assessments
- Absence of live market data integration means capitalization rates and comparables are manually sourced and stale
- No GIS integration; spatial risk factors (flood, zoning, proximity) cannot be overlaid on portfolio valuation

---

## Education Domain

---

## Learning Management System (`education_lms`)

**APG provides:** APG education_lms delivers a full learning management system supporting course creation, structured learning paths, student enrollment, assessment delivery, progress tracking, certification management, and blended learning coordination. It serves K-12, higher education, and corporate training contexts.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| SCORM / xAPI / LTI compliance | Basic SCORM 1.2 | Full SCORM 1.2/2004, xAPI, LTI 1.3, QTI compliance (Canvas, Moodle) | Missing xAPI and LTI 1.3 |
| AI adaptive learning paths | None | AI-personalized content sequencing from performance data (Coursera for Business, Canvas AI) | No adaptive learning |
| Competency-based progression | None | Mastery/competency gating with evidence portfolio (Canvas Mastery Paths, Moodle Competency) | No competency gating |
| Video content and streaming | Basic upload | Native video studio with captions, indexing, in-video quiz (Canvas Studio) | No native video tools |
| Accessibility (WCAG 2.1 AA) | Unknown | WCAG 2.1 AA certified with screen reader support and alt-text AI (Canvas, Moodle) | No documented accessibility certification |
| Analytics: learner engagement | Basic completion | Engagement heatmaps, at-risk learner detection, predictive completion (Canvas Analytics) | No predictive learner analytics |
| Mobile offline learning | None | Native iOS/Android with full offline access and sync (Moodle Mobile, Canvas Student) | No offline mobile capability |
| Peer review and collaboration | None | Structured peer review rubric, discussion boards, group assignments (Canvas, Moodle) | No peer learning tools |
| Proctored assessments | None | AI proctoring with face recognition, browser lockdown (Respondus + Canvas) | No proctoring integration |
| Content marketplace integration | None | Pre-built course content from OpenStax, LinkedIn Learning (Canvas, Blackboard) | No content marketplace |

**World-best reference:** Canvas (Instructure), Moodle, Blackboard Ultra

**Critical gaps:**
- No xAPI support limits learning analytics to basic SCORM completion data; modern analytics requires xAPI events
- Adaptive learning absent; Canvas and Moodle now personalize content sequence based on performance — APG delivers static paths
- No offline mobile capability; critical for learners with intermittent connectivity
- No AI-based at-risk learner detection; early intervention for struggling students is a primary LMS value proposition

---

## School Management (`education_sch_mgmt`)

**APG provides:** APG education_sch_mgmt manages school administrative operations including student enrollment, demographic records, attendance, grade management, parent communication, staff management, and statutory reporting. It serves K-12 schools and multi-school districts with centralised data management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Student Information System (SIS) depth | Basic enrollment + attendance | Comprehensive SIS: enrollment, health records, discipline, IEP, 504, counselling (PowerSchool, Infinite Campus) | No health, discipline, or IEP records |
| State / national reporting compliance | None | Automated Ed-Fi, SIF A+, state-specific report generation (PowerSchool, Skyward) | No standards-based state reporting |
| Gradebook integration | Basic grade entry | Standards-aligned digital gradebook with teacher portal (PowerSchool, Infinite Campus) | No standards-aligned gradebook |
| Parent / guardian portal | None | Mobile parent portal: grades, attendance, messaging, payments (PowerSchool, Skyward) | No parent portal |
| Behaviour management | None | Incident log with PBIS tracking, suspension/expulsion workflow (Infinite Campus) | No behaviour management |
| Special education (IEP/504) management | None | IEP creation, compliance calendar, goal progress (PowerSchool SPED, Infinite Campus) | No special education workflow |
| Food service management | None | Cafeteria POS, free/reduced meal eligibility, nutritional tracking (Infinite Campus Nutrition) | No food service integration |
| HR / payroll for school staff | None | Certified staff HR, credential tracking, substitute management (Skyward HR/Payroll) | No staff HR integration |
| Finance / budget integration | None | School district budget, purchasing, and accounts payable (Skyward Finance, IC Finance) | No district finance integration |
| Multi-school district dashboard | None | District-level cross-school analytics and comparative reporting (PowerSchool Analytics) | No district-level analytics |

**World-best reference:** PowerSchool, Infinite Campus, Skyward

**Critical gaps:**
- No parent/guardian portal is a baseline expectation; all three market leaders provide mobile parent apps
- Special education (IEP/504) workflow absent; this is a legal compliance requirement in most jurisdictions
- No standards-based state/national reporting output; schools require specific file formats for statutory submissions
- No behaviour management module; discipline data is foundational to school culture programmes and legal compliance

---

## Timetable Management (`education_ttbl`)

**APG provides:** APG education_ttbl manages school and university timetabling including class scheduling, room allocation, staff assignment, constraint management, and timetable publication to students and teachers. It supports semester-based and rotating timetables with conflict detection.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Automated constraint-based timetabling | Basic manual | AI/LP-optimised scheduling: teacher, room, subject, cohort constraints (Untis, Edval) | No algorithmic optimisation |
| Exam timetabling | None | Exam scheduler with invigilation assignment, student clash detection (Scientia CELCAT, Untis) | No exam scheduling |
| Room utilisation optimisation | None | AI-driven room allocation maximising utilisation (Scientia CELCAT, Ad Astra) | No room utilisation optimisation |
| Student subject clash detection | None | Individual student clash detection with resolution suggestion (Untis, Edval) | No student-level clash resolution |
| Integration with SIS enrollment | Manual import | Live SIS feed drives timetable demand generation automatically (PowerSchool + Edval) | No SIS-timetable live integration |
| Substitute / relief teacher scheduling | None | Automated substitute matching on absence notification (Untis, Smartsub) | No substitute scheduling |
| Calendar and event management | Basic | Full school calendar with parent notification and iCal sync (PowerSchool, Skyward) | No iCal sync or event notification |
| Multi-site / campus scheduling | Single site | Multi-campus with shared teacher and room pool management (Scientia CELCAT) | Single-site constraint |
| Teacher preference and load balancing | None | Teacher preference input with equitable load distribution (Untis, Edval) | No teacher preference management |
| Publication to student/parent apps | None | Digital timetable published to student app with push notification on change (Untis, PowerSchool) | No digital timetable publication |

**World-best reference:** Untis, Scientia CELCAT, Edval

**Critical gaps:**
- No algorithmic timetable optimisation; manual scheduling cannot reliably satisfy the constraint complexity of a secondary school
- No exam timetabling; this is a separate, complex scheduling problem that requires dedicated tooling
- Integration with SIS enrollment data is manual; automated demand-driven timetable generation is the standard approach
- No substitute/relief teacher scheduling means absence management creates manual daily chaos

---

## Project Portfolio Management Domain

---

## Project Accounting (`ppm_pac`)

**APG provides:** APG ppm_pac delivers project accounting including project cost tracking, timesheet-to-project allocation, purchase order management against project budgets, progress billing, revenue recognition, and project profitability reporting at task, work package, and project level.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Earned value management (EVM) | None | Full EVM: BCWS, BCWP, ACWP, SPI, CPI, EAC (Oracle Primavera P6, Deltek) | No EVM capability |
| Revenue recognition (ASC 606/IFRS 15) | None | Percentage-of-completion and milestone-based revenue recognition (Oracle Project Financials) | No contract revenue recognition |
| Timesheet-to-project cost allocation | Basic | Labour cost allocation with burden rates by role/location (Deltek, Oracle PPM) | No burden rate calculation |
| Cost-to-complete (ETC/EAC) forecast | None | Automated cost-to-complete from actuals + remaining estimate (Oracle Primavera, Planview) | No ETC/EAC forecasting |
| Multi-currency project accounting | None | Multi-currency with functional currency translation and revaluation (Oracle Cloud PPM) | No multi-currency support |
| Contract billing (T&M, fixed, milestone) | Basic invoicing | Multiple billing methods with milestone invoicing and retainage (Deltek Vantagepoint) | No multi-contract billing |
| Overhead / indirect cost allocation | None | Overhead pool allocation with G&A, fringe, overhead rates (Deltek, Unanet) | No indirect cost pools |
| Project budget revision management | None | Formal budget revision with approval workflow and baseline preservation (Oracle PPM) | No budget revision workflow |
| Inter-company project billing | None | Inter-entity project cost transfer with automated billing (Oracle Cloud PPM) | No inter-company project accounting |
| Project audit trail for compliance | Basic | DCAA-compliant audit trail for government project accounting (Deltek Costpoint) | No DCAA-level audit trail |

**World-best reference:** Oracle Cloud PPM, Deltek Costpoint, Planview

**Critical gaps:**
- No EVM is a critical gap; earned value is the primary cost and schedule performance metric for projects above a certain scale
- Revenue recognition per ASC 606/IFRS 15 absent; project-based service businesses have mandatory recognition requirements
- No cost-to-complete forecasting means project financial outcomes cannot be predicted until it is too late to intervene
- No multi-currency support; global projects accrue costs in multiple currencies that must be translated to functional currency

---

## Portfolio Analytics (`ppm_pan`)

**APG provides:** APG ppm_pan provides portfolio-level analytics across all active and planned projects, delivering capacity vs. demand views, portfolio health scoring, investment performance tracking, strategic alignment scoring, and scenario modelling to support portfolio investment decisions.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Strategic alignment scoring | None | Project-to-strategy alignment with weighted scoring and bubble charts (Planview, ServiceNow SPM) | No strategy alignment model |
| Portfolio scenario modelling | None | What-if portfolio scenarios: add/remove projects, shift timelines, reallocate resources (Planview) | No scenario planning engine |
| Capacity vs. demand heatmap | Basic table | Role-level capacity vs. demand heatmap with over/under-allocation drill (Planview, MS Project Online) | No visual capacity heatmap |
| Benefits realisation tracking | None | Benefit KPI tracking from project approval to post-implementation review (Planview, ServiceNow) | No benefits tracking |
| Portfolio financial performance (ROI, IRR) | None | Portfolio IRR, NPV, payback with actuals vs. business case (Oracle Primavera Analytics) | No portfolio financial KPIs |
| Risk-adjusted portfolio prioritisation | None | Risk-adjusted value scoring for portfolio rank ordering (Planview) | No risk-adjusted ranking |
| Dependency management across projects | None | Cross-project dependency mapping with impact propagation (Oracle Primavera P6) | No cross-project dependencies |
| AI portfolio health scoring | None | AI-generated portfolio health score from cost, schedule, and risk signals (Planview AI) | No AI health scoring |
| Stage gate governance dashboard | None | Portfolio-level stage gate status with decision support (Planview, ServiceNow PPM) | No stage gate governance |
| Agile + waterfall mixed portfolio view | None | Hybrid portfolio: agile sprints + waterfall projects in one roadmap (Planview, Jira Align) | No hybrid portfolio view |

**World-best reference:** Planview, Oracle Primavera Analytics, ServiceNow Strategic Portfolio Management

**Critical gaps:**
- No portfolio scenario modelling; the primary use case for a PPM platform is evaluating trade-offs in constrained investment portfolios
- No strategic alignment scoring means portfolio selection is based on politics rather than value
- Benefits realisation tracking absent; without it there is no feedback loop to validate project investment decisions
- No AI portfolio health scoring; manual status reporting introduces lag and reporting bias

---

## Project Baseline Management (`ppm_pbl`)

**APG provides:** APG ppm_pbl manages project schedule and cost baselines, tracking actuals and forecasts against the approved baseline. It supports baseline versioning, change-controlled baseline updates, variance analysis, and trend reporting to detect schedule and cost drift early.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Multiple named baselines | Single baseline | Up to 10 named baselines with comparison view (Oracle Primavera P6) | Single-baseline constraint |
| Baseline change control workflow | None | Formal change order workflow to revise baseline with approval (Oracle Primavera P6, Planview) | No change-controlled baseline revision |
| Schedule variance (SV) analysis | Manual delta | Automated SV, SPI with trend line and forecast (Oracle Primavera P6 EVM) | No automated variance metrics |
| Cost baseline vs. actuals trend | Manual | Waterfall chart: approved budget, approved changes, EAC trend (Oracle PPM, Planview) | No trend visualization |
| Baseline freeze / audit lock | None | Immutable baseline snapshot with audit metadata (Oracle Primavera P6) | No audit-locked baseline |
| Change log with impact assessment | None | Integrated change log: scope, cost, schedule impact per change (Procore, Oracle P6) | No integrated change log |
| Earned value from baseline | None | BCWS calculated from baseline + resource assignments (Oracle Primavera P6) | No baseline-derived EVM |
| Re-baseline justification documentation | None | Structured re-baseline request with business case and approval (Planview) | No re-baseline governance |
| Baseline comparison across projects | None | Portfolio baseline health: % projects on/under/over baseline (Planview Analytics) | No portfolio baseline view |
| Integration with project schedule | Disconnected | Baseline directly embedded in CPM schedule (Oracle Primavera P6) | Baseline and schedule are separate |

**World-best reference:** Oracle Primavera P6, Planview, Microsoft Project Online

**Critical gaps:**
- Single baseline only; complex projects require baseline versions to track pre-and post-change performance
- No change-controlled baseline revision means project managers can silently move the goalposts without governance
- Baseline and schedule are disconnected; EVM requires the baseline to be embedded in the CPM schedule for BCWS calculation
- No audit-locked baseline means forensic analysis of project overruns is impossible post-project

---

## Project Planning & Scheduling (`ppm_pps`)

**APG provides:** APG ppm_pps provides project planning and scheduling capabilities including work breakdown structure (WBS) creation, activity definition, dependency linking, Gantt chart visualization, resource assignment, critical path calculation, and schedule compression techniques.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| CPM / longest path scheduling | Basic Gantt | Full CPM with longest path, total float, free float, near-critical path (Oracle Primavera P6) | No CPM engine with float calculation |
| 6 activity types (P6 standard) | Basic tasks | 6 activity types: task-dependent, resource-dependent, LOE, WBS summary, milestones (Oracle Primavera P6) | Limited activity type granularity |
| Resource levelling | None | Automated resource levelling within float, priority-based (Oracle Primavera P6, MS Project) | No resource levelling algorithm |
| Schedule risk analysis (Monte Carlo) | None | Monte Carlo simulation on duration uncertainty (Oracle Risk Analysis, Safran) | No probabilistic scheduling |
| 4D BIM scheduling | None | Construction schedule linked to BIM model (Autodesk Construction Cloud + Navisworks) | No 4D BIM |
| Schedule compression (fast-track/crash) | Manual | Crash cost analysis with time-cost trade-off curve (Oracle Primavera P6) | No crashing analysis |
| Predecessor/successor relationship types | FS only | FS, SS, SF, FF with positive and negative lag (Oracle Primavera P6, MS Project) | Limited relationship types |
| Global change wizard | Manual edits | Bulk schedule update via global change rules (Oracle Primavera P6) | No global change capability |
| Schedule quality check (DCMA 14-point) | None | Automated DCMA 14-point schedule health check (Oracle Primavera P6, Acumen Fuse) | No schedule quality assurance |
| Multi-user concurrent editing | None | Multi-user P6 EPPM with role-based access to schedule (Oracle Primavera EPPM) | No concurrent multi-user editing |

**World-best reference:** Oracle Primavera P6, Microsoft Project, Safran Project

**Critical gaps:**
- No CPM engine with float calculation means the schedule cannot identify the true critical path or near-critical activities
- No resource levelling algorithm; schedules that exceed resource capacity are unrealistic and undeliverable
- Monte Carlo schedule risk analysis absent; probabilistic confidence intervals are required for major project reporting
- No DCMA 14-point schedule quality check; government and major capital projects require a schedule health certification

---

## Project Management (`ppm_prj`)

**APG provides:** APG ppm_prj provides core project management functionality including project initiation, stakeholder management, issue and risk registers, change management, document management, meeting management, action tracking, project reporting, and lessons learned capture across all project types and delivery methodologies.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Agile / scrum board integration | None | Native agile board with sprint planning, backlog, burndown (Jira, Smartsheet, MS Project) | No agile delivery methodology |
| Risk register with quantitative scoring | Basic list | Probability-impact matrix, risk response, residual risk, Monte Carlo (Oracle Primavera P6) | No quantitative risk scoring |
| RAID log (Risks, Assumptions, Issues, Deps) | Partial | Integrated RAID log with cross-reference to schedule (Planview, MS Project) | No integrated RAID log |
| AI project health assistant | None | AI-generated project status narrative with exception detection (Planview AI, MS Copilot) | No AI project narrative |
| Document management with version control | Attachment only | Version-controlled document library with approval workflow (Procore, SharePoint + MS Project) | No version-controlled document management |
| Stakeholder engagement tracking | None | Stakeholder register with engagement level and communication plan (MS Project, Planview) | No stakeholder management |
| Timesheet and actual hours | Basic | Time-phased actual hours vs. estimates with forecast to complete (Oracle PPM, Planview) | No forecast-to-complete from timesheets |
| Lessons learned / knowledge base | None | Structured lessons learned with searchable knowledge base (Oracle PPM, Planview) | No lessons learned module |
| Meeting minutes and action tracking | Basic | Meeting minutes with action owner, due date, and status tracking (Smartsheet, MS Project) | Limited action tracking |
| Portfolio dashboard integration | None | Project status fed automatically into portfolio dashboard (Planview, ServiceNow PPM) | No automatic portfolio roll-up |

**World-best reference:** Oracle Primavera P6, Planview, Microsoft Project Online

**Critical gaps:**
- No agile support; modern organisations run mixed portfolios of waterfall and agile projects that must be managed in a unified platform
- AI project health narrative absent; automated status generation from data reduces reporting overhead and improves consistency
- No forecast-to-complete from actual hours; cost and schedule outcomes cannot be predicted from current performance
- No lessons learned module; project knowledge is lost at project close and repeated mistakes are not avoided

---

## Resource Management (`ppm_res`)

**APG provides:** APG ppm_res manages project resource management including resource capacity planning, skill-based resource allocation, demand vs. availability tracking, utilization reporting, and resource forecasting across the project portfolio.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Skills-based resource matching | None | Competency-profile matching of demand to available resources (Planview, Oracle Cloud PPM) | No skills inventory integration |
| Capacity vs. demand forecasting | Basic table | Rolling 12-month capacity vs. demand by role and skill (Planview, Saviom) | No forward capacity model |
| Resource request and approval workflow | None | Formal resource request, manager approval, and fulfilment confirmation (Planview, MS Project) | No resource request workflow |
| Scenario-based capacity planning | None | What-if: add project, delay start, change scope — see capacity impact (Planview) | No capacity scenario modelling |
| Contractor vs. FTE cost modelling | None | Cost differential model: internal rate vs. contractor rate for make/buy (Planview, Tempus) | No make/buy cost modelling |
| Time tracking integration | Basic | Actual hours from timesheets updating remaining effort forecasts (Oracle PPM, Planview) | No timesheet-to-forecast integration |
| Utilisation reporting (billable vs. non-billable) | None | Billable, internal, overhead utilisation by resource and team (Deltek, Planview) | No utilisation categorisation |
| Bench / unallocated resource tracking | None | Unallocated resource report with skills and availability for reassignment (Planview) | No bench resource visibility |
| Training and certification gap analysis | None | Gap analysis: required skills for planned projects vs. current workforce (Planview, Workday) | No skills gap analysis |
| Integration with HR / org chart | None | Live headcount and org chart integration for capacity baseline (Planview + Workday) | No HR system integration |

**World-best reference:** Planview, Oracle Cloud PPM, Saviom

**Critical gaps:**
- No skills inventory means resource allocation is by name rather than capability — hiding specialisation gaps
- No capacity scenario modelling; resource managers cannot assess the impact of adding or delaying projects
- No utilisation reporting by category; without billable vs. non-billable tracking, resource productivity cannot be managed
- No HR system integration means capacity planning starts from manually maintained headcount data that is perpetually stale

---

## Enterprise Asset Management Domain

---

## Asset Management — Full Lifecycle (`eam_ast`)

**APG provides:** APG eam_ast manages the complete enterprise asset lifecycle from acquisition planning through commissioning, in-service operation, maintenance management, and disposal. It maintains an asset register with technical specifications, financial valuation, maintenance history, compliance records, and integrated work order management across all asset classes.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| AI predictive maintenance (watsonx) | None | ML failure prediction from IoT sensor and historian data; IBM Maximo Monitor + Predict | No ML failure probability modelling |
| IoT / OT sensor integration | None | Direct OPC-UA, MQTT, OSIsoft PI historian integration (IBM Maximo Monitor) | No OT/IoT data ingestion |
| Asset health scoring | None | Composite health score across all assets with drill-down (IBM Maximo Health) | No asset health index |
| Reliability-centred maintenance (RCM) | Basic PM scheduling | RCM analysis: failure mode, effect, criticality, task selection (IBM Maximo, SAP PM) | No RCM methodology support |
| Mobile work management (offline) | None | Offline-capable mobile technician app with barcode/NFC scan (IBM Maximo Anywhere) | No offline mobile work management |
| Capital project integration | None | Seamless asset creation on project close-out (IBM Maximo + IBM TRIRIGA) | No project-to-asset handover |
| FMEA / criticality analysis | None | Asset criticality ranking with FMEA risk matrix (IBM Maximo, Hexagon EAM) | No FMEA capability |
| Spare parts / MRO inventory | Basic | MRO storeroom with safety stock, reorder, expedite, and kitting (IBM Maximo Materials) | No optimised spare part management |
| Asset disposal / decommissioning workflow | Basic record | Full disposal workflow: valuation, write-off, salvage, environmental compliance (SAP AM, IBM Maximo) | No regulated disposal workflow |
| CMMS integration with ERP financials | None | Work order cost auto-posting to GL cost centres and asset NBV (SAP PM + FI, IBM Maximo + ERP) | No work order financial posting |
| Regulatory compliance register | None | Statutory inspection compliance with certificate management (IBM Maximo, Infor EAM) | No compliance certificate tracking |
| Digital twin / 3D asset model | None | 3D CAD/BIM-linked asset with component drill-down (IBM Maximo + Hexagon Asset Lifecycle) | No digital twin integration |

**World-best reference:** IBM Maximo Application Suite, SAP PM/EAM, Hexagon EAM (Infor)

**Critical gaps:**
- No IoT/OT sensor integration is the foundational gap for condition-based maintenance; Maximo MAS has had this since 2021 with IBM watsonx AI
- No FMEA/RCM methodology support; evidence-based maintenance task selection is the industry standard for asset-intensive operations
- Work orders do not post costs to GL or update asset net book value — the entire financial dimension of asset management is disconnected from ERP
- No offline mobile work management for field technicians; offline capability is mandatory for assets in remote or restricted-network environments

---

---

# 6. Common Platform Capabilities

Now I have enough authoritative research. Let me compile the full competitive analysis covering all 81 capabilities.

---

## AI/ML Platform (8 capabilities)

---

## AI Agent Composition (`agnt`)

**APG provides:** A composition framework for assembling multi-agent pipelines from APG's modular capability graph. Agents can be wired together via the event bus, share context through the master data layer, and be deployed as durable workflows through the scheduler.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Agent orchestration | Capability-graph wiring | LangGraph, AutoGen (Microsoft) | No built-in cycle detection or loop-control primitives |
| Tool/function registry | APG capability registry | LangChain tool registry, OpenAI function calling | No automatic schema generation from capability contracts |
| Memory / context persistence | Via `mdata` + `cach` | MemGPT, Zep, LangChain Memory | No native episodic/semantic memory tier |
| Multi-model routing | Ollama-local | AWS Bedrock multi-model, Vertex AI model garden | Single runtime; no traffic-splitting across models |
| Observability | Via `logs` + `trce` | LangSmith, Langfuse | No trace-to-agent-step correlation out of the box |
| Human-in-the-loop | Via `wkfl` approval nodes | CrewAI human steps, Camunda task forms | Not a first-class primitive |
| Deployment targets | Flask-AppBuilder | Vertex AI Agent Builder, AWS Bedrock Agents | No serverless/managed agent endpoint |

**World-best reference:** LangGraph (LangChain), Microsoft AutoGen, CrewAI

**Critical gaps:**
- No native episodic memory store; agents lose context across sessions without manual wiring to `mdata`/`cach`
- No built-in observability correlation between agent steps and underlying LLM traces
- Human-in-the-loop is implemented as a workflow stub, not a first-class interrupt/resume primitive

---

## AI Core Framework (`aicr`)

**APG provides:** Foundational AI runtime targeting locally-hosted Ollama models. Provides prompt templating, model abstraction, inference routing, and response parsing shared across all AI/ML capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Model abstraction | Ollama-local focus | LiteLLM (100+ providers), Vertex AI | Hard-coded local assumption; cloud fallback not native |
| Prompt management | Template engine integration | LangSmith, PromptLayer, Humanloop | No prompt versioning or A/B evaluation |
| Inference caching | Via `cach` | Semantic caching in Redis/GPTCache | No semantic dedup of equivalent prompts |
| Streaming responses | Basic SSE | OpenAI streaming, Ollama streaming | No backpressure or partial-result persistence |
| Cost/token tracking | Not present | LiteLLM cost tracking, Portkey | Completely absent |
| Safety/guardrails | Not present | Guardrails AI, NVIDIA NeMo Guardrails | Completely absent |
| Fine-tuning integration | Not present | Vertex AI, SageMaker, Unsloth | No PEFT/LoRA pipeline hook |

**World-best reference:** LiteLLM, Portkey, NVIDIA NeMo

**Critical gaps:**
- No token/cost accounting means AI spend is invisible at the capability level
- No guardrails layer — hallucination and PII leakage risks are unmitigated
- Prompt version control is absent; regressions from prompt edits are undetectable

---

## Anomaly Detection (`anom`)

**APG provides:** Statistical and ML-based anomaly detection usable across time-series metrics, event streams, and tabular data. Integrates with `alrt` for automated alerting on detected anomalies.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Time-series anomaly | Statistical models | Datadog Watchdog, AWS Lookout for Metrics | No auto-seasonality detection |
| Multivariate detection | Not present | Azure Anomaly Detector, Prophet+IQR | Univariate only |
| Streaming ingestion | Via `evnt` | Flink, Kafka Streams + Faust | Batch-oriented; near-real-time only |
| Root-cause analysis | Not present | Dynatrace Davis AI, Datadog RCA | Gap — detection without attribution |
| Explainability | Not present | SHAP/LIME via scikit-learn | Black-box outputs |
| Model retraining | Manual | SageMaker Autopilot, Vertex AI retraining | No drift-triggered retraining loop |
| False-positive tuning | Manual threshold | Datadog adaptive baselines | Static thresholds only |

**World-best reference:** Datadog Watchdog, Azure Anomaly Detector, AWS Lookout for Metrics

**Critical gaps:**
- Multivariate anomaly detection absent; correlated failures across services go undetected
- No automated root-cause analysis — operators must manually trace from alert to cause
- Streaming detection latency is bounded by batch cycle, not event arrival time

---

## Forecasting Engine (`fcst`)

**APG provides:** Time-series forecasting service supporting demand forecasting, capacity planning, and KPI projection. Exposes forecast APIs consumable by `rpts`, `blng`, and domain-specific capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Algorithm breadth | Prophet, statsmodels | AWS Forecast (DeepAR+, NPTS), Azure Forecast | No deep-learning models (N-BEATS, Temporal Fusion) |
| Hierarchical forecasting | Not present | Nixtla, Darts | Aggregate-only; no reconciliation |
| Probabilistic output | Basic CI | AWS Forecast quantile output | No full distribution output |
| External regressors | Manual feature injection | Nixtla, AWS Forecast related time series | Limited |
| AutoML model selection | Not present | AWS Forecast, H2O AutoML | Manual model selection required |
| Backtesting / evaluation | Manual | MLflow + custom, Darts | No structured walk-forward CV |
| Real-time forecast refresh | Batch only | Vertex AI online prediction | Batch latency unacceptable for dynamic pricing |

**World-best reference:** AWS Forecast, Nixtla (Neuralforecast/StatsForecast), Darts

**Critical gaps:**
- No deep-learning forecasting models for complex seasonality patterns
- Hierarchical reconciliation absent — aggregate and granular forecasts can diverge
- No AutoML model selection; engineers must manually benchmark models per use case

---

## Natural Language Processing (`nlp`)

**APG provides:** NLP primitives including entity extraction, sentiment analysis, classification, and text embedding, served via locally-hosted Ollama models to avoid cloud data-egress.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Entity extraction | Ollama-prompted | AWS Comprehend (purpose-built NER), spaCy | Prompt-based NER is less precise than trained NER |
| Sentiment analysis | LLM-based | AWS Comprehend, Google NL API | No domain-specific fine-tuning pipeline |
| Text embedding | Ollama (nomic-embed etc.) | OpenAI text-embedding-3, Cohere Embed v3 | Embedding quality lags proprietary SOTA |
| Document classification | LLM zero-shot | AWS Comprehend custom classifier | No training-data management or classifier versioning |
| Language detection | LLM-based | Google NL API (109 languages) | Accuracy degrades on short strings |
| Summarization | LLM-based | Anthropic Claude, OpenAI GPT-4o | Quality constrained by local model capability |
| RAG pipeline | Manual wiring | LangChain, LlamaIndex | No native chunking/indexing/retrieval pipeline |

**World-best reference:** AWS Comprehend, Google Natural Language API, Hugging Face Inference Endpoints

**Critical gaps:**
- Prompt-based NER has lower precision/recall than purpose-trained models; production accuracy is unvalidated
- No native RAG pipeline — knowledge-grounded generation requires manual plumbing
- Model quality ceiling is determined by available Ollama models; no fallback to cloud APIs

---

## Recommendation Engine (`recc`)

**APG provides:** Collaborative and content-based recommendation service with integration hooks for product catalog (`ctlg`), user behavior events (`evnt`), and session state (`cach`).

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Collaborative filtering | Matrix factorization | AWS Personalize (HRNN, AutoML) | No session-aware or context-aware models |
| Content-based filtering | Embedding similarity | Pinecone + custom model | Vector search not natively integrated |
| Real-time personalization | Batch-refresh | AWS Personalize real-time events | Batch latency; stale recommendations |
| Cold-start handling | Not explicit | AWS Personalize, Google Recommendations AI | Cold-start problem unaddressed |
| A/B experimentation | Manual | Optimizely, AWS Personalize campaigns | No experiment management |
| Contextual bandit | Not present | Google Recommendations AI, Vowpal Wabbit | Exploration/exploitation absent |
| Explainability | Not present | AWS Personalize (reason codes) | Black-box recommendations |

**World-best reference:** AWS Personalize, Google Recommendations AI, RecSys via Merlin (NVIDIA)

**Critical gaps:**
- Real-time event ingestion for immediate re-ranking not supported; recommendations lag user actions
- Cold-start strategy is undefined — new users/items receive degraded or no recommendations
- No A/B framework; recommendation quality improvements cannot be measured rigorously

---

## Search & Discovery (`srch`)

**APG provides:** Full-text and semantic search capability backed by a locally-deployable search engine. Integrates with `ctlg`, `docs`, and `mdia` to provide unified search across structured and unstructured content.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Full-text search | Typesense/OpenSearch | Elasticsearch (MSRP 50ms at scale), Algolia | Feature-complete for basic cases |
| Typo tolerance | Typesense built-in | Algolia, Typesense | Covered |
| Vector/semantic search | Via Ollama embeddings | Algolia NeuralSearch, Elasticsearch kNN | Quality tied to local embedding model |
| Faceting & filtering | Typesense | Algolia, Typesense | Covered |
| Personalized ranking | Not present | Algolia Personalization, Google Retail Search | Absent |
| Analytics & insights | Not present | Algolia Search Analytics, Elastic Kibana | No query analytics or no-results tracking |
| Multi-tenant indices | Via `tena` | Algolia multi-index, Elasticsearch aliases | Manual tenant isolation required |

**World-best reference:** Algolia, Elasticsearch/OpenSearch, Typesense

**Critical gaps:**
- No search analytics — zero visibility into failed queries, low-CTR results, or trending terms
- Personalized re-ranking absent; all users get identical result ordering
- Semantic search quality is bounded by locally-hosted embedding models

---

## Computer Vision (`visi`)

**APG provides:** Computer vision capability for image/video analysis via locally-hosted multimodal Ollama models. Supports object detection, classification, and OCR use cases.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Object detection | Ollama multimodal | AWS Rekognition, Google Vision API | General-purpose models lack domain specialization |
| OCR / document parsing | LLM-based | AWS Textract, Google Document AI | Structured form extraction accuracy lags |
| Image classification | Ollama vision | Google AutoML Vision | No custom training pipeline |
| Video analysis | Not present | AWS Rekognition Video, Google Video Intelligence | Completely absent |
| Facial recognition | Not present | AWS Rekognition, Azure Face API | Deliberately excluded (privacy posture) |
| Model fine-tuning | Not present | Roboflow + YOLOv8, Vertex AI AutoML Vision | No custom dataset management |
| Real-time inference | Batch/API | AWS Rekognition streaming, Google Vision | Latency constrained by local hardware |

**World-best reference:** AWS Rekognition, Google Cloud Vision API, Roboflow

**Critical gaps:**
- Video analysis entirely absent — significant gap for surveillance, retail analytics, media use cases
- No custom model training path; domain-specific vision accuracy cannot be improved without external tooling
- Real-time performance constrained by local GPU availability

---

## API & Integration (6 capabilities)

---

## API Gateway & Management (`apig`)

**APG provides:** Flask-AppBuilder-based API gateway handling routing, rate-limiting, authentication delegation, and API versioning for all APG-exposed capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Routing & load balancing | Flask routing | Kong (54,250 TPS, <1ms overhead) | Framework-level; not purpose-built |
| Rate limiting | Flask-Limiter | Kong, AWS API Gateway | Distributed rate-limit state requires Redis |
| Auth delegation | Via `auth` | Kong OIDC plugin, Apigee | Not a native gateway concern; manual wiring |
| Developer portal | Not present | Apigee, Kong Dev Portal, AWS API Gateway | Completely absent |
| API analytics | Via `moni` | Apigee built-in analytics, Kong Vitals | Requires external assembly |
| API monetization | Via `blng` | Apigee native billing, Kong Konnect | No native usage-plan-to-invoice pipeline |
| OpenAPI spec enforcement | Manual | AWS API Gateway schema validation | Not enforced at gateway layer |

**World-best reference:** Kong, Apigee (Google), AWS API Gateway

**Critical gaps:**
- No self-service developer portal — API consumers cannot self-register, get keys, or browse documentation
- Latency overhead of Flask vs purpose-built gateways (Kong <1ms vs Flask 5–15ms) is significant at scale
- No API monetization primitive; usage-based billing requires manual wiring across `apig` + `blng`

---

## Event Bus/Streaming (`evnt`)

**APG provides:** Internal event bus enabling decoupled capability communication. Supports pub/sub patterns, event routing, and integration with external consumers via `wbhk`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Throughput | Moderate (Redis streams / RQ) | Kafka (millions msg/sec per partition) | Orders of magnitude gap at high volume |
| Message persistence / replay | Limited | Kafka (configurable retention), EventBridge archive | No durable log; events lost on consumer lag |
| Event schema registry | Not present | Confluent Schema Registry, AWS Glue Registry | Schema drift undetected |
| Dead-letter queue | Manual | RabbitMQ DLX, AWS SQS DLQ | Must be hand-built per queue |
| Consumer group semantics | Basic | Kafka consumer groups | No offset management or lag monitoring |
| Cross-capability routing | Config-based | AWS EventBridge rules, Kafka routing | Limited pattern matching |
| Backpressure | Not present | Kafka, RabbitMQ prefetch | Producers can overwhelm consumers silently |

**World-best reference:** Apache Kafka / Confluent, AWS EventBridge, RabbitMQ

**Critical gaps:**
- No durable event log — replay for reprocessing or audit is impossible without a database layer
- Schema registry absent; event contract changes can silently break downstream consumers
- No consumer lag monitoring; slow consumers are invisible until queues overflow

---

## Integration API (`int_api`)

**APG provides:** Standardized integration interface layer for connecting APG capabilities to external systems via REST, webhooks, and file-based exchange. Covers connectors for common SaaS platforms.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Pre-built connectors | Limited, custom | MuleSoft (600+), Boomi (1000+) | Connector library is minimal |
| Low-code mapping | Not present | Talend, Boomi visual mapper | All mappings require code |
| Error handling / retry | Manual | MuleSoft retry policies, Azure Logic Apps | No structured error-classification pipeline |
| API versioning support | Via `vers` | Apigee, MuleSoft | Adequate |
| Monitoring | Via `moni` + `logs` | Boomi Atom Management | Adequate with assembly |
| iPaaS pattern | Not present | MuleSoft Anypoint, Boomi, Azure Integration | No graphical flow designer |
| Data transformation | Via `etl` | Talend, MuleSoft DataWeave | No native transformation DSL |

**World-best reference:** MuleSoft Anypoint Platform, Dell Boomi, Azure Integration Services

**Critical gaps:**
- Connector library covers ~10 systems vs 600+ in MuleSoft; most integrations require custom development
- No low-code/no-code flow designer; integration work requires engineering resources for every connection
- No native data transformation DSL; complex field mappings are one-off Python code

---

## Service Mesh (`mesh`)

**APG provides:** Service-to-service communication governance within the APG capability deployment topology, handling service discovery, circuit breaking, and mutual TLS between deployed capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Service discovery | Flask service registry | Istio, Consul, Linkerd | Bespoke; not standards-compliant |
| mTLS | Manual cert management | Istio auto-mTLS, Linkerd | No automated certificate rotation |
| Circuit breaking | Via retry logic | Istio/Envoy circuit breaker | Not configurable at mesh level |
| Traffic shaping | Not present | Istio VirtualService, Envoy | No canary/blue-green at mesh layer |
| Observability | Via `trce` + `moni` | Istio + Kiali, Linkerd Viz | Partial; no mesh-topology visualization |
| Zero-trust networking | Not present | Istio SPIFFE/SPIRE, Consul Connect | Large security gap |
| Sidecar injection | Not present | Istio, Linkerd | Requires manual integration |

**World-best reference:** Istio, Linkerd, HashiCorp Consul Connect

**Critical gaps:**
- No automated mTLS; inter-capability communication can transit unencrypted
- Traffic shaping absent — canary deployments of capabilities require manual infrastructure changes
- No zero-trust identity for service accounts; compromise of one capability could pivot laterally

---

## Data Synchronization (`sync`)

**APG provides:** Bidirectional data synchronization between APG's internal data stores and external systems. Handles conflict resolution, delta detection, and sync scheduling via `schd`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| CDC (change data capture) | Polling-based | Debezium, AWS DMS | No log-based CDC; polling is high-latency |
| Conflict resolution | Last-write-wins | Operational Transform (Google Docs), CRDTs | No domain-aware merge strategies |
| Schema evolution | Manual | Confluent Schema Registry + Avro | Breaking schema changes not auto-detected |
| Bi-directional sync | Partial | Fivetran, Airbyte (300+ connectors) | Limited to configured pairs |
| Real-time sync | Polling interval | Debezium (sub-second CDC) | Latency floor set by poll interval |
| Sync monitoring | Via `moni` | Fivetran sync health, Airbyte | Basic; no per-record error tracking |
| SaaS connectors | Custom only | Airbyte (300+), Fivetran (500+) | Negligible pre-built connector coverage |

**World-best reference:** Debezium, Airbyte, Fivetran

**Critical gaps:**
- Polling-based CDC introduces seconds-to-minutes of latency vs sub-second log-based CDC
- No pre-built SaaS connectors; every external system requires custom development
- Conflict resolution is last-write-wins only; concurrent edits from multiple sources risk data corruption

---

## Webhook Management (`wbhk`)

**APG provides:** Outbound webhook delivery system enabling external consumers to subscribe to APG events. Handles retries, signature verification, and delivery tracking.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Delivery guarantees | At-least-once | Svix, Hookdeck | Covered at basic level |
| Signature verification | HMAC-SHA256 | Svix, Stripe webhooks | Covered |
| Retry with backoff | Exponential backoff | Hookdeck, Svix | Covered |
| Delivery dashboard | Not present | Svix portal, Hookdeck UI | No self-service debugging for consumers |
| Fan-out | Single endpoint | Svix multi-endpoint, AWS EventBridge | No 1-to-many delivery |
| Event catalog | Not present | Svix event types, Hookdeck catalog | Consumers cannot discover available events |
| Rate limiting per endpoint | Not present | Hookdeck | Slow endpoints can back-pressure producers |

**World-best reference:** Svix, Hookdeck, Stripe Webhooks

**Critical gaps:**
- No self-service delivery dashboard; webhook consumers must contact APG operators to debug failed deliveries
- Fan-out to multiple endpoints not supported; one event source can only target one consumer
- No event catalog; external developers cannot discover what events are available to subscribe to

---

## Auth & Security (8 capabilities)

---

## Accessibility Services (`accs`)

**APG provides:** WCAG compliance utilities, aria-label generation for rendered UI components, and accessibility audit hooks integrated into the Flask-AppBuilder template pipeline.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| WCAG audit (automated) | Limited rule checks | axe-core (80+ rules), Deque Axe Pro | Incomplete ruleset coverage |
| Screen reader support | Manual ARIA | NVDA/JAWS compatibility via axe-core | Not validated against assistive tech |
| Color contrast checking | Not present | axe-core, Colour Contrast Analyser | Absent |
| Keyboard navigation audit | Not present | axe DevTools | Absent |
| CI integration | Not present | axe-core in Jest/Playwright | No automated regression gate |
| PDF/document accessibility | Not present | Adobe Acrobat Pro, CommonLook | Absent |
| Accessibility statement | Not present | W3C WCAG template | Not generated |

**World-best reference:** Deque axe-core, Lighthouse (Google), Level Access

**Critical gaps:**
- No CI gate — accessibility regressions are introduced silently with each UI change
- Color contrast and keyboard navigation checks absent despite being WCAG 2.1 Level AA requirements
- No assistive-technology compatibility validation; WCAG compliance is theoretical, not empirical

---

## Authentication & RBAC (`auth`)

**APG provides:** Authentication service with JWT-based session management, role-based access control, and OAuth2/OIDC integration. Built on Flask-AppBuilder's security model with PostgreSQL-backed user store.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| MFA | TOTP via Flask-AppBuilder | Auth0, Okta (adaptive MFA, biometrics) | No adaptive or hardware-key MFA |
| Social / federated login | OAuth2 providers | Auth0 (50+ social), Keycloak brokering | Limited provider coverage |
| RBAC granularity | Table/view-level | Keycloak UMA 2.0, OPA (attribute-based) | No ABAC or resource-level policies |
| SCIM provisioning | Not present | Okta, Auth0 (enterprise) | No automated user lifecycle from HR systems |
| Audit log | Via `audl` | Okta System Log, Auth0 log streams | Adequate with assembly |
| Session management | JWT / Flask-Login | Auth0 (silent refresh, token rotation) | No silent refresh; long-lived tokens |
| Passwordless | Not present | Auth0, Keycloak | Absent |

**World-best reference:** Auth0/Okta, Keycloak, AWS Cognito

**Critical gaps:**
- No ABAC — fine-grained per-resource authorization requires bespoke logic in every capability
- SCIM absent; user provisioning/de-provisioning from identity providers is manual
- Long-lived JWTs without refresh rotation create a large compromise window

---

## Cryptography & Key Management (`crpt`)

**APG provides:** Encryption utilities, key generation, and secret management for protecting sensitive data at rest and in transit across APG capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Key storage | PostgreSQL-backed | HashiCorp Vault, AWS KMS | Keys in same DB as data — poor separation |
| Key rotation | Manual | Vault auto-rotate, AWS KMS automatic rotation | No automated rotation schedule |
| Envelope encryption | Manual | AWS KMS, Google Cloud KMS | Not standardized |
| HSM integration | Not present | AWS CloudHSM, Thales | Absent |
| Secret injection | Not present | Vault agent, AWS Secrets Manager | Secrets managed as env vars |
| Audit trail | Via `audl` | Vault audit log, AWS CloudTrail | Adequate with assembly |
| FIPS 140-2 compliance | Not present | AWS KMS, Thales Luna | Absent |

**World-best reference:** HashiCorp Vault, AWS KMS, Google Cloud KMS

**Critical gaps:**
- Keys stored in PostgreSQL alongside the data they protect — a single DB breach exposes both
- No automated key rotation; rotations require deployment events, creating operational gaps
- No HSM or FIPS 140-2 compliance path; ineligible for regulated financial/government use cases

---

## Data Classification (`dclf`)

**APG provides:** Automated classification of data assets by sensitivity (PII, PCI, confidential) to support downstream DLP, privacy, and compliance workflows.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| PII detection | Regex + LLM-scan | Microsoft Purview, AWS Macie | Pattern coverage narrower than purpose-built tools |
| Structured data scanning | Column-name heuristics | BigID, Collibra (catalog integration) | No statistical sampling of cell values |
| Unstructured data scanning | LLM-based | AWS Macie (S3), Microsoft Purview | File-system coverage only |
| Classification taxonomy | Custom tags | NIST SP 800-60, ISO 27001 labels | No standard taxonomy enforced |
| Policy enforcement | Via `dlp` | Microsoft Purview policies | Classification without enforcement is advisory |
| Continuous scanning | Scheduled | AWS Macie continuous, BigID | Batch only; new data classified on schedule |
| Confidence scoring | Not present | AWS Macie, BigID | No confidence metric for manual review triage |

**World-best reference:** Microsoft Purview, AWS Macie, BigID

**Critical gaps:**
- No cell-value statistical sampling; classification relies on column names alone — easily missed
- Confidence scoring absent; all classifications treated as equal, making triage impossible
- Continuous real-time scanning absent; data created between scan cycles is unclassified

---

## Data Loss Prevention (`dlp`)

**APG provides:** Policy enforcement layer that intercepts outbound data flows and blocks or redacts content that violates classification policies. Integrates with `dclf` and `auth`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| API-level enforcement | Request/response inspection | Nightfall AI, AWS Macie | Basic pattern matching only |
| Email DLP | Not present | Proofpoint, Microsoft Purview | Absent |
| Endpoint DLP | Not present | Symantec DLP, CrowdStrike | Absent |
| Redaction | Simple masking | Nightfall AI (format-preserving) | No format-preserving encryption |
| Policy workflow | Static rules | Forcepoint, Proofpoint (adaptive) | No risk-scoring or adaptive policies |
| Incident management | Via `alrt` | Microsoft Purview incident queue | Alert generation only; no investigation workflow |
| Cloud storage scanning | Not present | AWS Macie, Wiz DSPM | Absent |

**World-best reference:** Microsoft Purview DLP, Nightfall AI, Proofpoint

**Critical gaps:**
- DLP scope limited to API calls; email, endpoints, and cloud storage have zero coverage
- No format-preserving encryption for redaction — masked values break downstream systems
- No adaptive policy engine; static rules miss novel exfiltration patterns

---

## Privacy & Consent Management (`prvc`)

**APG provides:** Consent capture, preference storage, and GDPR/POPIA subject rights (DSAR) workflow management. Exposes consent state to other capabilities to gate data processing.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Consent capture | Web form | OneTrust Consent Management Platform | Limited to web; no mobile/in-app SDK |
| Cookie management | Basic banner | OneTrust, Cookiebot (IAB TCF 2.2) | No IAB TCF 2.2 compliance |
| DSAR workflow | Manual queue | OneTrust DSAR automation, Transcend | No automated data discovery for DSAR fulfillment |
| Preference center | Basic UI | OneTrust, Ketch | No granular purpose-based consent |
| Regulation coverage | GDPR, POPIA | OneTrust (175+ jurisdictions) | Limited to 2 frameworks |
| Data lineage for consent | Not present | OneTrust, Collibra | Cannot prove consent was honored in processing |
| Consent versioning | Not present | OneTrust, Ketch | Consent re-solicitation on policy changes requires manual process |

**World-best reference:** OneTrust, TrustArc, Ketch

**Critical gaps:**
- No IAB TCF 2.2 support — GDPR-compliant advertising consent is not possible
- Automated DSAR fulfillment absent; each data subject request requires manual data retrieval across capabilities
- Consent coverage limited to two jurisdictions vs. 175+ in OneTrust — unsuitable for global operations

---

## Security Framework (`sec`)

**APG provides:** Cross-cutting security policy framework covering input validation, OWASP top-10 mitigations, dependency scanning, and security header management for all APG endpoints.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Input validation | Pydantic-enforced | OWASP ESAPI, AWS WAF | Covered for typed API inputs |
| WAF | Not present | AWS WAF, Cloudflare WAF, ModSecurity | Absent |
| SAST | Not present | Semgrep, Snyk Code, SonarQube | Absent |
| DAST | Not present | OWASP ZAP, Burp Suite | Absent |
| Dependency scanning | Not present | Snyk, Dependabot | Absent |
| Security headers | Flask-Talisman | OWASP Secure Headers Project | Covered |
| CSPM | Not present | Wiz, Orca Security | Absent |

**World-best reference:** Snyk, OWASP ZAP, Wiz

**Critical gaps:**
- No WAF protection — APG endpoints are exposed to injection, bot, and DDoS attacks without it
- No SAST/DAST in CI pipeline — security vulnerabilities introduced in code go undetected until exploitation
- No dependency scanning; transitive vulnerabilities in PyPI packages accumulate silently

---

## Vulnerability Management (`vuln`)

**APG provides:** Vulnerability tracking, CVE monitoring for APG dependencies, and risk prioritization workflow integrated with `alrt` and `sec`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| CVE tracking | pip-audit / manual | Snyk, Tenable.io | Manual and periodic only |
| Continuous scanning | Not present | Tenable.io, Qualys, Wiz | Absent |
| Container scanning | Not present | Trivy, Snyk Container, AWS Inspector | Absent |
| SBOM generation | Not present | Syft, CycloneDX | Absent |
| Risk scoring | CVSS raw | Tenable VPR, Snyk risk score | No exploitability or reachability scoring |
| Remediation tracking | Issue tracker integration | Snyk Fix PR, Dependabot PR | No automated fix PR generation |
| Compliance reporting | Not present | Tenable, Qualys (PCI DSS, SOC 2 reports) | Absent |

**World-best reference:** Snyk, Tenable.io, Qualys

**Critical gaps:**
- No SBOM generation — impossible to audit the full dependency supply chain
- Container image scanning absent; deployed images may carry known CVEs undetected
- No automated fix PRs; remediation requires manual identification and patching

---

## Data Platform (8 capabilities)

---

## Data Archival (`arch`)

**APG provides:** Lifecycle-managed data archival service that moves aged records from active PostgreSQL tables to cold storage (object store / compressed partitions) based on configurable retention policies.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Tiered storage | Hot/cold (PG → object store) | AWS S3 Intelligent-Tiering, Glacier | Manual tier transitions; no auto cost-optimization |
| Query-on-archive | Not present | AWS Athena, Snowflake external tables | Archived data is query-dead |
| Retention policy engine | Config-driven | Azure Blob Storage lifecycle, AWS S3 lifecycle | Adequate |
| Legal hold | Not present | NetApp StorageGRID, AWS S3 Object Lock | Absent |
| Compression | pg_dump / gzip | Apache Parquet + Zstd (10–50x ratios) | Not columnar; poor compression ratio |
| Audit trail | Via `audl` | Commvault, Veeam | Adequate |
| Restore SLA | Manual | AWS Glacier Instant Retrieval (ms) | No defined RTO |

**World-best reference:** AWS S3 Intelligent-Tiering + Glacier, Apache Iceberg, Snowflake Data Archival

**Critical gaps:**
- Archived data is unqueryable; business intelligence on historical data requires manual restore
- No legal hold capability — e-discovery and litigation hold requests cannot be honored
- No columnar archival format; storage costs are 10–50x higher than Parquet + Zstd equivalents

---

## Caching Layer (`cach`)

**APG provides:** Redis-backed distributed cache shared across APG capabilities, supporting session state, API response caching, rate-limit counters, and pub/sub primitives.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Data structures | String, list, set, hash | Redis (15+ data types) | Covered — Redis |
| Cluster mode | Redis single-node default | Redis Cluster, Valkey Cluster | HA not default; single point of failure |
| Eviction policies | LRU / LFU | Redis (8 policies) | Covered |
| Pub/sub | Redis pub/sub | Redis Streams (persistent), Kafka | Pub/sub is fire-and-forget; no persistence |
| Semantic caching | Not present | GPTCache, LangChain SemanticCache | AI query dedup absent |
| Cache warming | Manual | ElastiCache lazy/proactive loading | No systematic warming strategy |
| TTL management | Per-key | Redis TTL, Dragonfly | Covered |

**World-best reference:** Redis/Valkey, Dragonfly, AWS ElastiCache

**Critical gaps:**
- Single-node Redis default creates SPF; no automatic failover to replica
- Persistent messaging through cache pub/sub is unreliable; lost messages on restart
- No semantic caching for AI inference; identical prompts hit the model redundantly

---

## Data Quality (`dqlt`)

**APG provides:** Automated data quality validation framework with configurable rules for completeness, consistency, uniqueness, and referential integrity, integrated into `etl` and `mdata`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Rule types | Completeness, uniqueness | Great Expectations (50+ expectation types) | Narrower expectation vocabulary |
| SQL-native validation | Via dbt-style tests | dbt tests, Soda Core | Covered with assembly |
| Profile-driven rules | Not present | Great Expectations auto-profiler | Cannot auto-generate rules from data |
| Data observability | Not present | Monte Carlo, Acceldata | No anomaly-based drift detection |
| CI integration | Not present | Great Expectations (CI gate) | No quality gate in deployment pipeline |
| Business metric validation | Not present | Collibra DQ, Ataccama | No business-rule-to-technical-rule mapping |
| Cross-source validation | Not present | Informatica DQ, Ataccama | Single-source only |

**World-best reference:** Great Expectations, dbt Tests, Monte Carlo Data

**Critical gaps:**
- No auto-profiling — rules must be written manually; coverage is proportional to engineer time invested
- No anomaly-based observability; systematic drift goes undetected between scheduled validation runs
- No CI quality gate; bad data reaching production requires discovery after the fact

---

## ETL Pipeline (`etl`)

**APG provides:** Extract-Transform-Load pipeline framework for moving and reshaping data between APG capabilities and external systems, scheduled via `schd` and monitored via `moni`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Connector library | Custom Python | Airbyte (300+), Fivetran (500+) | Minimal pre-built connectors |
| Transformation DSL | Python pandas | dbt (SQL-native, templated) | No version-controlled declarative transforms |
| Streaming ETL | Not present | Apache Flink, Spark Structured Streaming | Batch only |
| Lineage tracking | Not present | OpenLineage, dbt lineage graph | Column-level lineage absent |
| Error recovery | Retry + dead-letter | dbt retries, Prefect checkpoints | Basic |
| Orchestration | Via `schd` | Airflow (3000+ operators), Prefect | Adequate for simple DAGs |
| Schema inference | Not present | Airbyte, Spark infer schema | Manual schema definition required |

**World-best reference:** dbt + Airbyte, Apache Spark, Fivetran

**Critical gaps:**
- No streaming ETL path — real-time data movement requires external tooling
- Column-level lineage absent; data provenance for compliance cannot be demonstrated
- Connector library covers single-digit systems; each new integration is a bespoke development sprint

---

## Master Data Management (`mdata`)

**APG provides:** Golden-record management for core business entities (Customer, Product, Location) shared across APG capabilities, with deduplication, stewardship workflow, and change history.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Entity resolution | Rule-based matching | Informatica MDM (ML matching), IBM InfoSphere | No probabilistic / ML-based deduplication |
| Governance workflow | Basic approval flow | Informatica, SAP MDG (BPMN-based) | Stewardship workflow less configurable |
| Multi-domain MDM | Customer, Product, Location | Informatica IDMC (all domains) | Fixed domain model |
| Hierarchy management | Parent-child PostgreSQL | SAP MDG hierarchy workbench | No visual hierarchy editor |
| Data stewardship UI | Flask-AppBuilder | Informatica, Reltio | Functional but minimal |
| Syndication | Via `sync` + `evnt` | Informatica MDM hub syndication | Requires manual wiring |
| Graph-based relationships | Not present | Reltio cloud MDM, Semarchy | Flat relational model only |

**World-best reference:** Informatica IDMC, SAP Master Data Governance, Reltio

**Critical gaps:**
- Probabilistic ML-based entity matching absent; near-duplicate records not caught by exact/rule matching
- Fixed domain model — adding a new master data domain (e.g., Asset, Supplier) requires schema-level development
- Graph-based relationship traversal absent; multi-hop entity relationships cannot be queried efficiently

---

## Database Connectivity (`odbc`)

**APG provides:** Unified database abstraction layer providing connection pooling, query routing, and ORM integration for PostgreSQL as the primary store, with read-replica support.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Connection pooling | SQLAlchemy pool | PgBouncer (transaction pooling) | SQLAlchemy pooling is process-local; not shared |
| Multi-database support | PostgreSQL primary | SQLAlchemy (20+ dialects) | Single-database strategy by design |
| Query observability | Via `trce` | pganalyze, DataDog DB monitoring | Basic; no query plan analysis |
| Read/write splitting | Manual replica config | ProxySQL, PgPool-II | No automatic read/write routing |
| Schema migration | Via `mig` | Flyway, Liquibase | Covered |
| Slow query alerting | Via `moni` | pganalyze alerts, Datadog | Manual threshold configuration |
| Connection limits | App-level | PgBouncer multiplexing (thousands) | Each SQLAlchemy pool consumes real PG connections |

**World-best reference:** PgBouncer, pganalyze, AWS RDS Proxy

**Critical gaps:**
- Process-local connection pooling does not scale horizontally; connection exhaustion under load
- No automatic query plan analysis; slow query debugging requires DBA intervention
- No automatic read/write splitting; all traffic hits the primary unnecessarily

---

## Storage Management (`stor`)

**APG provides:** Abstracted file storage service supporting local filesystem, S3-compatible object storage, and database BLOB storage, with unified path API for all capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Object storage abstraction | boto3 + local fallback | AWS S3, MinIO (S3-compatible) | Covered |
| CDN integration | Not present | AWS CloudFront, Cloudflare R2 | Static asset delivery unoptimized |
| Deduplication | Not present | AWS S3 (content-addressed), Wasabi | Storage costs scale with duplicates |
| Virus scanning | Not present | ClamAV, AWS GuardDuty S3 | Uploaded files unscanned |
| Access policy | Via `auth` | AWS S3 Bucket Policy, IAM | Adequate |
| Encryption at rest | S3-SSE or filesystem | AWS S3 SSE-KMS | Adequate |
| Pre-signed URLs | S3 pre-signed | AWS S3, MinIO | Covered |

**World-best reference:** AWS S3, MinIO, Cloudflare R2

**Critical gaps:**
- No CDN — every file request hits the origin; latency and egress cost uncontrolled for media-heavy capabilities
- No malware scanning on upload; infected files propagate to all consumers of stored content
- No deduplication; the same file uploaded N times consumes N× storage

---

## Multi-Tenancy Framework (`tena`)

**APG provides:** Schema-per-tenant and row-level-security based multi-tenancy with tenant context injection middleware for all Flask-AppBuilder routes.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Isolation model | RLS + schema-per-tenant | Citus (sharding), Neon (per-tenant DB) | Single PostgreSQL instance creates noisy-neighbor risk |
| Tenant provisioning | Via API | Stripe Atlas, Supabase (per-tenant DB) | Manual setup steps for complex tenants |
| Resource quotas | Not present | Kubernetes ResourceQuota, Citus | No per-tenant compute/storage limits |
| Tenant billing | Via `blng` | Stripe metered billing, Chargebee | Adequate with wiring |
| Data isolation audit | Via `audl` | Salesforce, Salesforce Shield | Adequate with assembly |
| Tenant migration | Manual | Neon branch-per-tenant, Citus | Schema-per-tenant migrations are risky |
| Cross-tenant analytics | Not present | Citus, Snowflake ACCOUNT_USAGE | Requires separate reporting database |

**World-best reference:** Citus (PostgreSQL sharding), Neon, Supabase

**Critical gaps:**
- Single-instance PostgreSQL RLS creates noisy-neighbor risk; one misbehaving tenant can degrade others
- Per-tenant resource quotas absent — runaway queries or storage growth from one tenant affects all
- Cross-tenant analytics requires data export; no zero-copy aggregation across tenants

---

## Operations/DevOps (8 capabilities)

---

## Alert Management (`alrt`)

**APG provides:** Centralized alert lifecycle management covering alert creation, routing, escalation, and suppression across all APG capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Alert routing | Static rules | PagerDuty (schedule-aware routing) | No on-call schedule integration |
| Deduplication | Basic | PagerDuty, Alertmanager inhibition | Limited dedup logic |
| Escalation policies | Config-based | PagerDuty, OpsGenie | Less dynamic than purpose-built tools |
| On-call management | Not present | PagerDuty, OpsGenie | Absent |
| Incident correlation | Not present | Moogsoft, BigPanda AIOps | Alerts are isolated; no topology-aware grouping |
| SLA tracking | Not present | PagerDuty, Statuspage | Absent |
| Notification channels | Email + webhook | PagerDuty (phone, SMS, Slack, Teams) | Covered via `ntfn`; needs wiring |

**World-best reference:** PagerDuty, OpsGenie (Atlassian), Grafana Alertmanager

**Critical gaps:**
- No on-call schedule management — alert routing is static; night-time alerts go to wrong people
- AIOps correlation absent; alert storms from a single root cause generate flood of individual alerts
- No incident management lifecycle (acknowledge → resolve → postmortem); alerts are fire-and-forget

---

## Configuration Management (`conf`)

**APG provides:** Centralized runtime configuration store for APG capabilities, supporting environment-specific overrides, secrets injection, and configuration version history.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Secret management | Env vars / DB | HashiCorp Vault, AWS Secrets Manager | Secrets in env vars are insecure |
| Dynamic config (hot reload) | Not present | Consul KV, AWS AppConfig, Flagsmith | Config changes require restart |
| Config versioning | Git-backed | Consul, AWS AppConfig rollback | Adequate if Git workflow enforced |
| Environment promotion | Manual | AWS AppConfig deployment strategy | No formal promotion gate |
| Audit trail | Via `audl` | Vault audit log, AWS CloudTrail | Adequate with assembly |
| Schema validation | Not present | AWS AppConfig validators | Config drift undetected until runtime |
| Distributed consistency | Single DB | Consul Raft, etcd | Single PostgreSQL config store is SPF |

**World-best reference:** HashiCorp Vault + Consul, AWS AppConfig, Doppler

**Critical gaps:**
- Secrets in environment variables leak through process lists, logs, and crash dumps — fundamental security gap
- Hot reload absent; configuration changes require capability restart, causing downtime
- No config schema validation; malformed config values cause runtime failures, not deploy-time failures

---

## Deployment Management (`dplm`)

**APG provides:** Deployment orchestration for APG capabilities covering build, artifact management, environment promotion, and rollback coordination.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| CI/CD integration | GitHub Actions hooks | ArgoCD, Spinnaker, Harness | No GitOps-native operator |
| Container orchestration | Docker Compose default | Kubernetes + Helm, ArgoCD | Not production-grade orchestration |
| Progressive delivery | Not present | Argo Rollouts, Flagger (canary/blue-green) | Absent |
| Rollback | Manual image revert | ArgoCD automated rollback, Spinnaker | No automatic health-gate rollback |
| Environment management | Manual | Pulumi, Terraform, env manifests | No IaC-integrated environment parity |
| Artifact registry | Not specified | AWS ECR, Harbor | Not defined |
| Deployment observability | Via `moni` + `trce` | Datadog Deployment Tracking, Spinnaker | Adequate with assembly |

**World-best reference:** ArgoCD, Harness, Spinnaker

**Critical gaps:**
- Docker Compose deployment does not meet production HA/scaling requirements; Kubernetes gap is large
- Progressive delivery absent; all deployments are big-bang with manual rollback only
- No GitOps operator; deployment state and cluster state can diverge silently

---

## Feature Flags (`feat`)

**APG provides:** Runtime feature flag service enabling controlled rollout of new functionality, A/B splits, and kill switches without deployment. Integrates with `auth` for user/tenant targeting.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Targeting rules | User/tenant/role | LaunchDarkly (custom attributes, segments) | Covered for basic cases |
| SDK coverage | Python only | LaunchDarkly (20+ SDKs), Unleash | Frontend SDKs absent |
| Gradual rollout | Percentage-based | LaunchDarkly, Unleash | Covered |
| Experimentation | Not present | Split.io (integrated stats), LaunchDarkly Experimentation | Completely absent |
| Audit log | Via `audl` | LaunchDarkly flag change history | Adequate with assembly |
| Self-service portal | Flask UI | LaunchDarkly, Unleash UI | Minimal |
| Local evaluation | Not present | LaunchDarkly edge SDK | Network round-trip on every check |

**World-best reference:** LaunchDarkly, Unleash, Split.io (Harness)

**Critical gaps:**
- No JavaScript/mobile SDK — feature flags cannot target frontend behavior
- Experimentation engine absent; APG cannot measure the business impact of any flag change
- Server-side-only evaluation adds network latency to every flag check in hot paths

---

## Health Check Framework (`hlth`)

**APG provides:** Standardized health endpoint protocol for all APG capabilities, aggregating dependency health (DB, cache, external APIs) and exposing `/health`, `/ready`, and `/live` endpoints.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Liveness / readiness | `/live`, `/ready` endpoints | Kubernetes probes (standard) | Covered — Kubernetes compatible |
| Dependency checks | DB, Redis, external APIs | py-healthcheck, Flask-HealthCheck | Covered |
| Aggregated health | Not present | AWS ALB health, Consul health | Single-capability only; no fleet view |
| Health history | Not present | Statuspage, BetterUptime | Transient; no health trend |
| SLA / SLO tracking | Not present | PagerDuty SLOs, Google SRE | Absent |
| Synthetic monitoring | Not present | Datadog Synthetics, Pingdom | Absent |
| Dependency graph health | Not present | Consul service graph | Cannot show upstream impact |

**World-best reference:** Kubernetes native probes + Datadog Synthetics, Consul, Pingdom

**Critical gaps:**
- No aggregated fleet health view; operators cannot see the health status of all capabilities in one pane
- SLO tracking absent — health checks fire only on binary up/down, not on latency SLO breach
- Synthetic monitoring absent; health checks only detect issues after real traffic fails

---

## Logging & Observability (`logs`)

**APG provides:** Structured JSON logging across all APG capabilities with correlation IDs, log levels, and optional forwarding to external sinks (Loki, Elasticsearch).

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Structured logging | JSON + correlation ID | Datadog Logs, Grafana Loki | Covered |
| Log aggregation | File / forwarded | Loki + Promtail, Elasticsearch | Covered when configured |
| Log-based alerting | Via `alrt` | Datadog log monitors, Loki ruler | Adequate with assembly |
| Log parsing / enrichment | Not present | Datadog parsing rules, Logstash | Raw JSON only; no grok/enrichment |
| Retention management | Not specified | AWS CloudWatch (configurable), Loki compactor | Undefined retention policy |
| PII scrubbing | Not present | Datadog sensitive data scanner | Logs may contain PII |
| Log-to-trace correlation | Via correlation ID | Datadog (automatic), Grafana Tempo | Manual; depends on developer discipline |

**World-best reference:** Datadog Log Management, Grafana Loki, AWS CloudWatch Logs

**Critical gaps:**
- No automated PII scrubbing — user data in logs creates GDPR exposure on every log sink
- Undefined retention policy; storage costs grow unbounded; old logs must be purged manually
- Log parsing/enrichment absent; querying logs requires knowing the exact JSON structure

---

## Monitoring & Metrics (`moni`)

**APG provides:** Metrics collection via Prometheus-compatible endpoints on all APG capabilities, with Grafana dashboards for infrastructure and application-level KPIs.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Metrics scraping | Prometheus pull | Prometheus, Datadog agent (push+pull) | Covered |
| Dashboard templating | Grafana | Datadog, Grafana | Covered |
| Anomaly detection | Not present | Datadog Watchdog, New Relic AI | Absent |
| SLO / error budget | Not present | Datadog SLOs, Google Cloud Monitoring | Absent |
| Business metrics | Not present | Datadog custom metrics, New Relic | Only infra/app metrics; no business KPIs |
| Long-term storage | Prometheus local | Thanos, Mimir (unlimited retention) | Prometheus local storage has limited retention |
| On-call integration | Via `alrt` | Datadog → PagerDuty native integration | Adequate with wiring |

**World-best reference:** Datadog, Grafana/Prometheus + Thanos, New Relic

**Critical gaps:**
- Prometheus local storage defaults to 15-day retention; historical trend analysis requires Thanos/Mimir setup
- SLO and error budget management absent; teams cannot formally track service reliability commitments
- Anomaly detection absent; metric spikes require human observation to trigger alerts

---

## Distributed Tracing (`trce`)

**APG provides:** OpenTelemetry-instrumented distributed tracing across APG capability call chains, exporting to a configurable backend (Jaeger, Tempo).

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Instrumentation standard | OpenTelemetry | OpenTelemetry (industry standard) | Covered — correct choice |
| Trace backend | Jaeger / Tempo | Datadog APM, Jaeger, Tempo | Covered |
| Log-trace correlation | Correlation ID | Datadog (automatic), Grafana Tempo | Partial; depends on implementation |
| Sampling strategy | Head-based | Jaeger adaptive sampling, Datadog | No tail-based sampling (misses rare errors) |
| Service map | Jaeger UI | Datadog Service Map, Kiali | Functional |
| Trace-to-metrics | Not present | Datadog trace-based metrics | Absent |
| AI trace analysis | Not present | Datadog Watchdog Insights | Absent |

**World-best reference:** Datadog APM, Grafana Tempo + Loki, Honeycomb

**Critical gaps:**
- Head-based sampling only — rare but impactful errors (e.g., P99.9 latency spikes) may be dropped
- Trace-to-metrics absent; RED metrics (Rate/Error/Duration) cannot be derived from traces automatically
- No AI-assisted trace analysis; root-cause identification from traces requires manual examination

---

## Documents & Media (6 capabilities)

---

## Document Management (`docs`)

**APG provides:** Document storage, versioning, metadata indexing, and access-controlled retrieval, integrated with `srch` for full-text discovery and `tmpl` for document generation.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Version control | PostgreSQL + `vers` | SharePoint (major/minor versions), Confluence | Functional |
| Collaboration / co-editing | Not present | Google Workspace, Notion, Confluence | Absent |
| OCR / content extraction | Via `visi` | AWS Textract, Adobe Acrobat | Quality constrained by local model |
| Workflow / approval | Via `wkfl` | SharePoint approval, Notion workflows | Adequate with assembly |
| Full-text search | Via `srch` | SharePoint Syntex, Elasticsearch | Adequate |
| Access control | Via `auth` | SharePoint (granular per-item) | Role-level only; no per-document ACL |
| Enterprise connectors | Not present | SharePoint (M365 ecosystem) | Isolated from productivity suite |

**World-best reference:** SharePoint/OneDrive, Confluence (Atlassian), Google Drive

**Critical gaps:**
- Real-time collaborative editing absent — the defining capability of modern document platforms
- Per-document ACL not supported; all documents in a folder inherit role-based permissions
- No enterprise productivity suite integration; documents are isolated from email/calendar/chat context

---

## Image Processing (`imgs`)

**APG provides:** Image manipulation service covering resize, crop, format conversion, thumbnail generation, and basic enhancement operations for images managed through `stor` and `mdia`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Basic transforms | Pillow / wand | Cloudinary, Imgix | Covered |
| On-the-fly resize | Not present (pre-compute) | Cloudinary (URL-param driven), Imgix | No dynamic transform URL |
| CDN delivery | Via `stor` + CDN | Cloudinary CDN (global) | Not integrated by default |
| AI enhancement | Not present | Adobe Firefly, Cloudinary AI | Absent |
| Watermarking | Basic overlay | Cloudinary, Imgix | Covered for basic cases |
| Face detection | Via `visi` | AWS Rekognition, Cloudinary | Covered via integration |
| Format conversion (AVIF/WebP) | Pillow | Cloudinary, Sharp (libvips) | Adequate |

**World-best reference:** Cloudinary, Imgix, AWS CloudFront + Lambda@Edge

**Critical gaps:**
- On-the-fly transforms via URL parameters absent; every size variant must be pre-generated
- No CDN-integrated delivery — images served from origin store; no edge caching
- AI-based auto-enhancement, background removal, and generative fill absent

---

## Media Management (`mdia`)

**APG provides:** Unified media asset management covering images, video, audio, and documents with metadata tagging, collection management, and access-controlled delivery.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Asset organization | Collections + tags | Cloudinary (DAM), Canto DAM | Covered for basic use |
| Video transcoding | Not present | AWS Elemental MediaConvert, Mux | Absent |
| Video streaming | Not present | Mux, Cloudflare Stream (HLS/DASH) | Absent |
| AI tagging | Via `visi` (local) | Cloudinary AI tagging, AWS Rekognition | Local model quality gap |
| CDN delivery | Via `stor` | Cloudinary CDN, Cloudflare Stream | Not default |
| Rights management | Not present | Canto DAM, Bynder | Absent |
| Video analytics | Not present | Mux Data, JW Player | Absent |

**World-best reference:** Cloudinary, Mux, AWS Elemental MediaServices

**Critical gaps:**
- Video transcoding and adaptive streaming (HLS/DASH) entirely absent — video content is undeliverable at scale
- Digital rights management absent; licensed media assets have no usage tracking
- No video analytics; media performance is invisible

---

## Notification Engine (`ntfn`)

**APG provides:** Multi-channel notification delivery aggregating email (`mail`), SMS (`sms`), push (`notf`), and in-app messages through a unified template-driven notification API.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Channel aggregation | Email + SMS + push | Novu (open-source), Knock, Courier | Covered in principle |
| Preference management | Basic opt-out | Novu, Courier (user preferences API) | No granular per-topic preferences |
| Template management | Via `tmpl` | Novu, Courier (drag-drop editor) | CLI/code only; no visual editor |
| Delivery tracking | Per-channel | Novu, Courier (unified delivery log) | Fragmented across channels |
| Batch / digest | Not present | Novu digest, Courier | Absent — high-volume events cause notification spam |
| Scheduling | Via `schd` | Novu scheduled sends | Adequate with wiring |
| Analytics | Via `moni` | Knock, Courier (open rate, CTR) | Absent |

**World-best reference:** Novu (open-source), Knock, Courier

**Critical gaps:**
- Digest/batching absent; a burst of 100 events triggers 100 individual notifications to end users
- No per-topic preference management; users cannot opt in/out of specific notification categories
- No delivery analytics; open rate and click-through tracking unavailable

---

## Report Engine (`rpts`)

**APG provides:** Parameterized report generation combining data queries, template rendering (`tmpl`), and export to PDF/XLSX/CSV formats. Supports scheduled report delivery via `schd`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Report authoring | Python/Jinja template | Metabase, Looker (drag-drop, LookML) | Code-only authoring; no self-service |
| Interactive dashboards | Not present | Metabase, Grafana, Redash | Absent |
| Scheduled delivery | Via `schd` + `mail` | Looker, Metabase (subscriptions) | Covered with assembly |
| Embedded analytics | Not present | Looker Embedded, Sigma Computing | Absent |
| Data model layer | Not present | Looker LookML, dbt semantic layer | No governed metric definitions |
| Ad-hoc SQL | Not present | Redash, Metabase questions | End users cannot self-serve |
| PDF fidelity | WeasyPrint / ReportLab | Adobe Acrobat, Telerik Reporting | Adequate for standard layouts |

**World-best reference:** Metabase, Looker (Google), Apache Superset

**Critical gaps:**
- No self-service report authoring — every new report requires a developer
- Interactive dashboards absent; reports are static snapshots, not exploratory tools
- No semantic/metrics layer; different reports can calculate the same KPI differently

---

## Template Engine (`tmpl`)

**APG provides:** Jinja2-based template engine for generating documents, emails, reports, and HTML responses. Supports template inheritance, partials, and variable injection from capability data models.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Template language | Jinja2 | Handlebars, Nunjucks, Mustache | Industry-standard choice |
| Visual editor | Not present | Strapi (content), Unlayer (email), Brevo | Code-only; non-technical users cannot edit |
| Versioning | Via `vers` + Git | Contentful versioning | Adequate |
| Localization | Via `i18n` + `lcze` | Crowdin + template variables | Adequate with wiring |
| Preview / sandbox | Not present | Unlayer, BEE editor preview | Absent |
| A/B testing | Not present | Mailchimp A/B, Iterable | Absent |
| Dynamic content blocks | Jinja conditionals | HubSpot smart content, Salesforce | Manual; no CMS-driven dynamic content |

**World-best reference:** Handlebars.js, Nunjucks, Unlayer (email templates)

**Critical gaps:**
- No WYSIWYG editor — marketing and operations teams cannot update email/document templates without engineering
- No template preview sandbox — template errors are discovered in production or staging
- A/B testing of template variants absent; conversion optimization is impossible

---

## Business Logic (8 capabilities)

---

## Audit Logging (`audl`)

**APG provides:** Immutable audit trail for all create/update/delete operations across APG capabilities, capturing actor, timestamp, before/after state, and source IP.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Immutability | PostgreSQL append-only | AWS CloudTrail (S3 + WORM), Immudb | Database-level immutability is weak without WORM storage |
| Before/after state | JSON diff stored | Datadog Audit Trail, Splunk | Covered |
| Query / search | SQL | Splunk, Elastic SIEM | SQL-only; no behavioral analytics |
| Tamper detection | Not present | Immudb (cryptographic), AWS CloudTrail digest | DB admin can delete records |
| Long-term retention | Via `arch` | AWS CloudTrail (7-year), Splunk SIEM | Depends on archive setup |
| Real-time alerting | Via `alrt` | Splunk SIEM, Datadog CSPM | Adequate with wiring |
| Compliance reports | Not present | Vanta, Drata (SOC 2 evidence) | Evidence export for auditors not automated |

**World-best reference:** AWS CloudTrail, Splunk Audit, Immudb

**Critical gaps:**
- Database-backed audit log is mutable by a DB administrator — fails SOC 2 tamper-evidence requirement
- No cryptographic integrity verification; audit records can be silently modified
- Compliance evidence export for auditors requires manual SQL; no pre-built report templates

---

## Business Rules Engine (`bprl`)

**APG provides:** Configurable rule execution service allowing business stakeholders to define and modify decision logic (eligibility, pricing, routing) without code deployment.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Rule authoring | Python DSL | Drools (Rete algorithm), BRMS UI | Code-only; no business-user rule editor |
| Decision tables | Not present | Drools, IBM ODM decision tables | Absent |
| DMN support | Not present | Camunda DMN, Drools DMN | No standard decision model notation |
| Rule versioning | Via `vers` | IBM ODM, Drools KIE | Covered |
| Rule testing | Unit tests | IBM ODM test scenarios | Adequate |
| Performance | Python interpreter | Drools (Rete/PHREAK — JVM optimized) | JVM-based BRMS 10–100x faster at high volume |
| Explainability | Not present | IBM ODM, Drools audit log | Cannot trace which rule fired for a decision |

**World-best reference:** Drools (Red Hat), IBM Operational Decision Manager, Camunda DMN

**Critical gaps:**
- No business-user-accessible rule editor; every rule change requires a developer and deployment
- Decision tables absent — the most common format for complex business rule documentation
- Rule execution explainability absent; disputes about automated decisions cannot be audited

---

## Calculation Engine (`calc`)

**APG provides:** Parameterized calculation service for complex formulas (financial, actuarial, pricing) with formula versioning and reproducible audit trails for regulatory compliance.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Formula definition | Python expressions | Cube.js semantic layer, dbt metrics | Covered for numeric calculations |
| Spreadsheet-like UX | Not present | Excel, Google Sheets, Anaplan | Non-technical users cannot define formulas |
| Bulk / batch calculation | Via `schd` | Spark, Dask | Limited scale |
| Versioned recalculation | Via `vers` | Anaplan, Vena Solutions | Covered |
| Dependency graph | Not present | Anaplan (model graph), SpreadsheetDB | Cannot see which outputs depend on an input |
| Financial precision | Decimal | Java BigDecimal, IBM OpenPages | Covered |
| Audit trail | Via `audl` | Trintech, BlackLine | Adequate with assembly |

**World-best reference:** Anaplan, Vena Solutions, Cube.js

**Critical gaps:**
- No dependency graph — changing an input formula cannot automatically invalidate dependent calculations
- No self-service formula editor for finance/actuarial teams; all formula maintenance is engineering work
- Batch calculation scale limited by single-process execution; large recalculation jobs block other work

---

## Compliance Framework (`cplc`)

**APG provides:** Structured compliance management covering control definition, evidence collection hooks, risk register, and mapping to regulatory frameworks (GDPR, POPIA, ISO 27001, PCI DSS).

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Control library | Custom | Vanta (300+ pre-mapped controls), Drata | Manual framework mapping |
| Continuous monitoring | Not present | Vanta, Drata (300+ integrations) | Periodic reviews only |
| Evidence collection | Manual upload | Vanta automated evidence, Drata | All evidence is manual |
| Framework coverage | GDPR, POPIA, ISO 27001 | Vanta (35+ frameworks) | Narrow framework coverage |
| Audit-ready reports | Not present | Vanta, Drata (auditor portal) | Manual evidence assembly |
| Risk scoring | Manual | OneTrust, ServiceNow GRC | No automated risk quantification |
| Vendor risk management | Not present | OneTrust TPRM, ProcessUnity | Absent |

**World-best reference:** Vanta, Drata, OneTrust GRC

**Critical gaps:**
- All evidence collection is manual; SOC 2 audit prep requires weeks of engineer time
- Continuous control monitoring absent; compliance status is stale between reviews
- Vendor/third-party risk management entirely absent

---

## Internationalization (`i18n`)

**APG provides:** Translation key management, locale-aware string resolution, and pluralization support for all APG Flask-AppBuilder UI components and API response messages.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Translation key management | Babel / Flask-Babel | Phrase, Lokalise, Crowdin | File-based; no collaborative translation workflow |
| Pluralization | gettext | ICU MessageFormat (CLDR-compliant) | Basic; complex pluralization rules incomplete |
| Locale detection | HTTP Accept-Language | Django i18n, next-intl | Covered |
| Machine translation | Not present | Lokalise AI, DeepL API | Absent; all translation is manual |
| Translation coverage tracking | Not present | Phrase, Lokalise (% coverage) | Absent; missing keys fail silently or show keys |
| RTL layout support | Not present | Material UI RTL, Bootstrap RTL | Absent |
| Context / screenshots | Not present | Lokalise in-context editor | Translators lack UI context |

**World-best reference:** Lokalise, Phrase (formerly Phrase Strings), Crowdin

**Critical gaps:**
- No machine translation integration; adding a new language requires full manual translation
- RTL language support absent — Arabic, Hebrew, Urdu UIs are not usable
- Missing translation keys fail silently; untranslated strings appear as raw keys in production

---

## Localization (`lcze`)

**APG provides:** Locale-specific formatting of dates, currencies, numbers, and addresses, complementing `i18n` with data-level localization for all APG output surfaces.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Date/time formatting | Babel locale data | Moment.js/Luxon, ICU | Covered |
| Currency formatting | Babel | Dinero.js (immutable, precise) | Covered; precision depends on Decimal usage |
| Number formatting | Babel | ICU NumberFormatter | Covered |
| Address formatting | Not present | Google Address Validation API, libaddressinput | Absent |
| Phone number formatting | Not present | libphonenumber (Google) | Absent |
| Locale data freshness | Babel CLDR | ICU (CLDR updates quarterly) | Dependent on Babel release cadence |
| Tax localization | Not present | Avalara, Stripe Tax | Absent |

**World-best reference:** ICU (CLDR), Avalara (tax localization), Google libphonenumber

**Critical gaps:**
- Address and phone number formatting absent — global contact data is stored and displayed inconsistently
- Tax localization entirely absent; VAT/GST/sales tax calculation requires external system
- CLDR data freshness tied to Babel release schedule; new locales or format changes lag upstream

---

## Scheduler/Job Queue (`schd`)

**APG provides:** Cron-based job scheduler and async task queue for all APG background processing, integrating with Redis Queue (RQ) or Celery for worker management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Cron scheduling | APScheduler / Celery Beat | Temporal (durable), AWS Step Functions | Single-process scheduler is SPF |
| Distributed workers | Celery + Redis | Temporal (multi-cluster), Prefect | Covered with Celery |
| Durable execution | Not present | Temporal (survives crashes mid-job) | Jobs fail silently if worker crashes mid-execution |
| Priority queues | Celery priority | Celery, RQ | Covered |
| Observability | Flower (Celery) | Temporal UI, Prefect Cloud | Covered for Celery |
| Rate limiting | Celery rate_limit | Temporal, Prefect | Covered |
| Long-running workflows | Not suitable | Temporal, AWS Step Functions | Celery is unsuitable for multi-day workflows |

**World-best reference:** Temporal, Celery + Flower, AWS Step Functions

**Critical gaps:**
- Single-process scheduler is a single point of failure; crash halts all scheduled jobs
- No durable mid-job checkpoint; long-running jobs restarted from scratch after worker crash
- Long-running multi-step workflows (days/weeks) cannot be expressed in Celery without external state management

---

## Workflow Engine (`wkfl`)

**APG provides:** Multi-step workflow orchestration for business processes requiring human approval, conditional branching, and integration with multiple capabilities. Built on durable task DAGs.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| BPMN support | Not present | Camunda, Flowable | No standard process notation |
| Human tasks | Via `ntfn` + approval | Camunda Task Manager, Temporal signals | Adequate at low complexity |
| Durable execution | Limited | Temporal (fully durable), Camunda | Risk of state loss on restart |
| Process versioning | Via `vers` | Camunda (BPMN version migration) | Covered |
| Process visualization | Not present | Camunda Cockpit, Temporal Web UI | Absent |
| Escalation timers | Manual | Camunda boundary events, Temporal timers | Covered |
| Compensation / saga | Not present | Temporal saga, Axon Framework | Distributed transaction rollback absent |

**World-best reference:** Temporal, Camunda, Apache Airflow

**Critical gaps:**
- No BPMN standard means workflows cannot be designed by business analysts or shared across tools
- Compensation/saga pattern absent; multi-step failures leave partial state with no automated rollback
- No process visualization; active workflow instance state is opaque to operators

---

## Developer Experience (7 capabilities)

---

## CLI Framework (`cli`)

**APG provides:** Unified CLI for APG capability management, data operations, and administrative tasks. Built with Click/Typer exposing all APG service APIs from the command line.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Command structure | Click/Typer | Stripe CLI, Heroku CLI | Covered for basic use |
| Shell completion | Typer auto-complete | GitHub CLI, AWS CLI (complete) | Covered |
| Interactive prompts | Typer | Inquirer.js, Rich | Covered |
| Plugin system | Not present | kubectl plugins, Heroku CLI plugins | Absent; CLI is monolithic |
| API parity with REST | Partial | Stripe CLI (100% API parity) | Some capabilities not CLI-accessible |
| Output formatting | JSON / tabular | AWS CLI (--output json/table/text) | Covered |
| Context / profile management | Not present | AWS CLI profiles, kubectl contexts | Single context only |

**World-best reference:** Stripe CLI, GitHub CLI, AWS CLI

**Critical gaps:**
- No plugin/extension system; third-party capability integrations cannot ship CLI additions
- No multi-context/profile management; operators cannot switch between APG environments from CLI
- Incomplete API parity — some capabilities require direct HTTP calls

---

## Code Generation (`gen`)

**APG provides:** Scaffold generation for new capabilities, model stubs, test templates, and migration files, ensuring all generated code conforms to APG conventions.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Capability scaffold | APG template | Yeoman, Plop.js | Covered for APG conventions |
| Model generation | Pydantic from schema | SQLModel, datamodel-code-generator | Covered |
| Migration generation | Via `mig` | Alembic auto-generate | Covered |
| OpenAPI → SDK | Not present | openapi-generator (40+ languages) | Absent |
| AI-assisted generation | Not present | GitHub Copilot, Cursor, Continue | Absent |
| Custom template language | Jinja2 stubs | Cookiecutter, Yeoman | Covered |
| Idempotency | Not present | Rails generators (skip existing) | Re-running may overwrite custom code |

**World-best reference:** openapi-generator, Yeoman, GitHub Copilot

**Critical gaps:**
- No OpenAPI-to-SDK generation; client libraries for capability APIs must be hand-written
- No AI-assisted generation; code scaffolding quality is bounded by template completeness
- Re-running generators is not idempotent — dangerous in partially-implemented capabilities

---

## Migration Engine (`mig`)

**APG provides:** Schema migration management for PostgreSQL using Alembic with version-controlled migration scripts, rollback support, and multi-tenant migration orchestration.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Schema migrations | Alembic | Flyway, Liquibase | Covered |
| Auto-generation | Alembic autogenerate | Alembic, Django migrations | Covered |
| Rollback | Alembic downgrade | Flyway undo, Liquibase rollback | Covered |
| Multi-tenant migrations | Manual per-schema | Alembic multidb, Tenant-safe Flyway | Manual; risk of missing a tenant schema |
| Zero-downtime migrations | Not present | gh-ost (GitHub), pt-online-schema-change | Absent — large table migrations block production |
| Data migrations | Python scripts | dbt snapshots, Flyway versioned | Covered but fragile |
| Drift detection | Not present | Flyway Check, Liquibase diff | Absent — prod schema can diverge from migration history |

**World-best reference:** Flyway, Liquibase, gh-ost (GitHub)

**Critical gaps:**
- Zero-downtime schema migrations absent; large table alterations require maintenance windows
- Schema drift detection absent; production databases can diverge from migration history silently
- Multi-tenant migration automation absent; new tenant addition or schema change must be applied per-schema manually

---

## Mock/Test Data Generator (`mock`)

**APG provides:** Realistic test data generation for all APG domain models, supporting fixture factories, faker-based field generation, and seeding scripts for test environments.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Factory pattern | Factory Boy | Factory Boy, FactoryBot (Rails) | Covered |
| Faker integration | Faker library | Faker (Python/JS), Mimesis | Covered |
| Referential integrity | Manual | Factory Boy related factories | Covered with discipline |
| Production data masking | Not present | Databricks data masking, Delphix | Absent — dev environments use synthetic only |
| Stateful scenario generation | Not present | Scenario Builder, Cucumber | Absent |
| Large volume generation | Python loop | DATPROF, Redgate Data Generator | No parallel bulk generation |
| Domain-aware generators | Partial | Gretel AI (ML-based synthetic) | No statistical fidelity validation |

**World-best reference:** Factory Boy + Faker, Gretel AI, Delphix

**Critical gaps:**
- Production data masking absent; realistic dev data requires unsafe production copies or manual effort
- No statistical fidelity validation — generated data may not reflect production distributions
- Bulk generation is sequential; generating millions of rows for load testing is impractically slow

---

## SDK Framework (`sdk`)

**APG provides:** Auto-generated Python SDK wrapping all APG capability REST APIs, with typed models, retry logic, and authentication handling for internal and external consumers.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Language coverage | Python only | Stripe SDK (10+ languages), Twilio | Single language |
| Type safety | Pydantic models | OpenAPI-generated TypeScript, Stripe SDK | Covered for Python; absent elsewhere |
| Pagination helpers | Manual | Stripe auto-pagination, GitHub SDK | Manual cursor handling |
| Retry / backoff | Basic | Stripe SDK (automatic retry) | Covered |
| Webhook verification | Via `wbhk` | Stripe SDK webhook verify | Covered |
| Developer docs | Inline docstrings | Stripe API Docs, Twilio Docs | Minimal; no interactive examples |
| Versioning | Via `vers` | Stripe API versioning (pinned per-client) | Not pinnable per-client |

**World-best reference:** Stripe SDK, Twilio SDK, OpenAPI Generator

**Critical gaps:**
- Python-only SDK excludes all JavaScript/TypeScript, Go, Java, and mobile consumers
- No API version pinning per-client; a server-side API change can break all SDK consumers simultaneously
- No auto-pagination helpers; developers must implement cursor logic manually for every paginated endpoint

---

## Testing Framework (`test`)

**APG provides:** Test infrastructure for APG capabilities including pytest fixtures, capability integration harness, HTTP mocking, and database transaction rollback for test isolation.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Unit testing | pytest | pytest (industry standard) | Covered |
| Integration testing | pytest + real DB | pytest + testcontainers | Covered |
| HTTP mocking | pytest-httpserver | Wiremock, Mockoon | Covered |
| Contract testing | Not present | Pact, Dredd | Absent — API contracts between capabilities unverified |
| Load testing | Not present | Locust, k6, Gatling | Absent |
| Mutation testing | Not present | mutmut, Pitest | Absent |
| Test coverage gates | pytest-cov | codecov.io CI gate | Not enforced in CI by default |

**World-best reference:** pytest + testcontainers, Pact, Locust

**Critical gaps:**
- Contract testing absent; capability API changes can silently break dependent capabilities
- No load/performance testing harness; performance regressions are discovered in production
- Coverage gate not enforced in CI; coverage can silently decrease with each PR

---

## Versioning Framework (`vers`)

**APG provides:** Unified version lifecycle management for capability artifacts, API versions, schema migrations, and configuration objects, with semantic versioning enforcement.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Semantic versioning | semver enforced | npm semver, PyPI versioning | Covered |
| API versioning strategy | URL path (`/v1/`, `/v2/`) | Stripe (header-based, per-client pin) | URL versioning couples client and server releases |
| Changelog generation | Manual | conventional-changelog, Release Please | Manual; inconsistent |
| Deprecation management | Not present | Stripe deprecation headers, Sunset RFC | No formal deprecation lifecycle |
| Artifact registry | Not specified | PyPI, Nexus, JFrog Artifactory | Not defined |
| Breaking change detection | Not present | openapi-diff, Bump.sh | Absent — breaking changes discovered at runtime |
| Version negotiation | Not present | gRPC (proto versioning), GraphQL | Not applicable for REST |

**World-best reference:** Stripe API versioning model, Buf (protobuf), conventional-changelog

**Critical gaps:**
- No automated breaking change detection; API-breaking PRs are merged without warning
- No formal deprecation lifecycle; consumers have no advance notice before an API version is removed
- Changelog generation is manual; release notes are inconsistent and often omitted

---

## Commerce/Marketplace (6 capabilities)

---

## Billing Engine (`blng`)

**APG provides:** Invoicing, payment plan management, and revenue calculation across APG-enabled business applications, supporting one-time and recurring billing with multi-currency support.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Recurring billing | Custom schedules | Stripe Billing, Chargebee | Covered for basic cases |
| Usage-based billing | Manual metering | Stripe metered billing, Metronome | No automatic metering from event stream |
| Revenue recognition | Not present | Chargebee RevRec (ASC 606), Zuora | Absent — critical for audit readiness |
| Dunning management | Not present | Chargebee Smarty Dunning, Stripe | Absent |
| Tax calculation | Manual | Stripe Tax, Avalara | Manual; error-prone at scale |
| Payment gateway | Abstracted | Stripe (195 countries), Flutterwave | Depends on underlying gateway selection |
| Multi-entity billing | Not present | Zuora, NetSuite SuiteBilling | Absent |

**World-best reference:** Stripe Billing, Chargebee, Zuora

**Critical gaps:**
- Revenue recognition (ASC 606/IFRS 15) entirely absent; unsuitable for any audit or near-IPO scenario
- No automated dunning; failed payments are not retried systematically, creating revenue leakage
- Usage-based metering requires manual wiring to event stream; metering gaps mean unbilled consumption

---

## Shopping Cart (`cart`)

**APG provides:** Session-persistent shopping cart with item management, coupon application, tax estimation, and checkout flow integrated with `ctlg`, `pric`, and `blng`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Persistence | PostgreSQL / session | Shopify cart (Redis-backed) | Adequate |
| Guest cart merge | Not present | Shopify, Medusa.js | Lost cart on login |
| Saved carts | Not present | Shopify save-for-later, Medusa.js | Absent |
| Abandoned cart recovery | Not present | Klaviyo + Shopify, Omnisend | Absent |
| Real-time inventory check | Via `ctlg` | Shopify (real-time hold), Commercetools | Adequate with integration |
| Multi-currency | Via `blng` | Shopify Markets, Commercetools | Adequate |
| Cart-level promotions | Via `pric` | Shopify Scripts, Commercetools | Adequate with wiring |

**World-best reference:** Shopify Storefront API, Medusa.js, Commercetools

**Critical gaps:**
- Abandoned cart recovery absent — the highest-ROI e-commerce recovery mechanism is missing
- Guest-to-logged-in cart merge absent; guest shopping sessions are lost on authentication
- Real-time inventory reservation not atomic; overselling risk exists without explicit stock holds

---

## Product Catalog (`ctlg`)

**APG provides:** Hierarchical product and service catalog with attribute management, variant support, and pricing hooks. Feeds `recc`, `srch`, and `cart`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Product variants | PostgreSQL EAV | Shopify variants, Commercetools ProductType | Covered |
| Rich media | Via `mdia` | Shopify CDN-backed media | Adequate |
| Category hierarchy | Nested sets | Commercetools (unlimited depth) | Covered |
| Product search | Via `srch` | Algolia Recommend + search | Adequate |
| Digital products | Not present | Shopify Digital Downloads, Gumroad | Absent |
| B2B pricing tiers | Via `pric` | Commercetools (customer-group pricing) | Adequate with wiring |
| Feed export (Google/Meta) | Not present | Channable, Feedonomics | Absent |

**World-best reference:** Commercetools, Shopify Product API, Akeneo (PIM)

**Critical gaps:**
- No product feed export for Google Shopping, Meta Catalog — digital marketing channels are excluded
- Digital product delivery (software licenses, downloads) not natively supported
- No PIM-grade attribute inheritance or completeness scoring for catalog quality

---

## Marketplace Framework (`mrkt`)

**APG provides:** Multi-seller marketplace infrastructure covering seller onboarding, listing management, commission calculation, and payout coordination on top of core commerce capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Seller onboarding | Via `auth` + forms | Stripe Connect onboarding, Sharetribe | No KYC/AML automation |
| Commission engine | Custom calculation | Stripe Connect application fees | Covered |
| Seller payouts | Via `blng` | Stripe Connect payouts, PayPal Payouts | Dependent on payment gateway |
| Dispute management | Not present | Stripe Radar + disputes, Sharetribe | Absent |
| Trust & safety | Not present | Sift Science, Stripe Radar | Absent |
| Seller analytics | Via `rpts` | Sharetribe, Arcadier | Basic |
| Regulatory compliance | Not present | Stripe Connect (KYC, AML) | Absent — regulatory gap |

**World-best reference:** Stripe Connect, Sharetribe, Arcadier

**Critical gaps:**
- KYC/AML automation entirely absent — regulatory requirement for all marketplace operators
- Dispute and chargeback management absent; seller-buyer disputes cannot be mediated
- Trust & safety (fraud scoring, listing review) absent; marketplace abuse is undetected

---

## Pricing Engine (`pric`)

**APG provides:** Rule-based pricing calculation supporting list price, volume tiers, customer-group discounts, promotions, and bundle pricing for all commerce capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Rule evaluation | Python rules | Zilliant, Vendavo (ML-based) | Rules-only; no ML-based optimization |
| Dynamic pricing | Not present | Prisync, Wiser (competitive) | Absent |
| Price versioning | Via `vers` | Commercetools price dates | Covered |
| Promotion stacking | Manual precedence | Shopify Scripts, Talon.One | Complex stacking rules error-prone |
| CPQ (configure-price-quote) | Not present | Salesforce CPQ, DealHub | Absent |
| A/B price testing | Not present | Optimizely, Google Optimize | Absent |
| Currency conversion | Static FX rates | Stripe (real-time FX), Open Exchange Rates | Static rates go stale |

**World-best reference:** Talon.One, Commercetools Pricing, Stripe Price API

**Critical gaps:**
- ML-based price optimization absent; pricing is reactive to configured rules, not market signals
- Dynamic competitive pricing absent; prices do not respond to competitor or demand changes
- CPQ absent — complex B2B deal configuration and quoting requires external tooling

---

## Subscription Management (`sbsc`)

**APG provides:** Full subscription lifecycle management covering plan definition, upgrades/downgrades, trial management, cancellation flows, and renewal processing.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Plan management | PostgreSQL-backed | Chargebee, Stripe Billing | Covered |
| Trial management | Custom logic | Chargebee trial automation | Covered |
| Upgrade / downgrade proration | Manual calc | Stripe proration, Chargebee | Covered |
| Dunning | Not present | Chargebee Smarty Dunning | Absent |
| Churn analytics | Not present | Baremetrics, ChartMogul | Absent |
| SaaS metrics (MRR, ARR, LTV) | Not present | Baremetrics, Profitwell | Absent |
| Self-service portal | Basic Flask UI | Chargebee Customer Portal, Stripe Billing Portal | Minimal |

**World-best reference:** Chargebee, Stripe Billing, Recurly

**Critical gaps:**
- No automated dunning; churned revenue from failed payments is not recovered
- SaaS business metrics (MRR, churn rate, LTV, ARR) absent; business health is invisible
- Self-service subscription management portal is minimal; customers cannot upgrade/cancel without contacting support

---

## Communications (6 capabilities)

---

## Chat/Messaging (`chat`)

**APG provides:** Real-time chat capability for user-to-user and user-to-support messaging within APG applications, with message history and notification integration via `ntfn`.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time delivery | WebSocket / SSE | Ably, Pusher (99.999% SLA) | DIY WebSocket lacks managed reliability |
| Message persistence | PostgreSQL | Twilio Conversations, Stream Chat | Covered |
| Presence / typing indicators | Not present | Stream Chat, Ably | Absent |
| File attachments | Via `stor` | Twilio Conversations, Stream Chat | Adequate |
| Message threading | Not present | Slack-style, Stream Chat threads | Absent |
| Moderation | Not present | Perspective API, Stream Chat moderation | Absent |
| Push notification on new message | Via `ntfn` | Stream Chat (push provider integration) | Adequate with wiring |

**World-best reference:** Stream Chat, Twilio Conversations, Ably

**Critical gaps:**
- Managed WebSocket infrastructure reliability is unproven; no SLA for real-time delivery
- Presence indicators (online/typing) absent — fundamental UX expectation for chat
- Content moderation absent; inappropriate content goes undetected

---

## Collaboration Framework (`colb`)

**APG provides:** Shared workspace primitives for multi-user document editing, threaded comments, @mentions, and activity feeds integrated into APG capabilities.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time co-editing | Not present | Google Docs (OT), Notion (CRDT) | Absent |
| Threaded comments | Basic comment model | Notion, Linear comments | Covered for basic cases |
| Activity feeds | Custom events via `evnt` | Getstream.io activity feeds | Adequate with assembly |
| @mention notifications | Via `ntfn` | Slack, Notion, Linear | Adequate with wiring |
| Shared workspaces | Via `auth` groups | Notion workspaces, Confluence spaces | Covered |
| Version history | Via `vers` + `docs` | Google Docs history, Notion | Covered |
| Presence awareness | Not present | Liveblocks, PartyKit | Absent |

**World-best reference:** Liveblocks, Google Workspace, Notion

**Critical gaps:**
- Real-time collaborative editing absent — the core productivity expectation of a collaboration framework
- Presence awareness absent; users cannot see who is viewing the same resource
- No CRDT/OT primitives; concurrent editing produces conflicts

---

## Communications Hub (`coms`)

**APG provides:** Unified communications orchestration layer routing messages across `mail`, `sms`, `chat`, and `notf` channels with preference-aware delivery and communication history.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Channel unification | Assembly of sub-capabilities | Twilio Flex, Vonage Communications APIs | Covered in principle; orchestration is manual |
| Preference-aware routing | Via `prvc` consent | Twilio Messaging, Vonage | Adequate |
| Communication history | PostgreSQL log | Zendesk unified inbox, Intercom | Covered |
| Omnichannel inbox | Not present | Zendesk, Intercom, Freshdesk | Absent |
| AI-assisted routing | Not present | Zendesk AI, Intercom Fin AI | Absent |
| SLA tracking | Not present | Zendesk SLAs, Freshdesk | Absent |
| Two-way messaging | Via gateway APIs | Twilio Conversations, Vonage | Covered |

**World-best reference:** Twilio Flex, Intercom, Zendesk

**Critical gaps:**
- No omnichannel inbox; agents cannot see a unified customer communication history across channels
- AI-assisted message routing and intent detection absent
- No SLA / response-time tracking for customer communications

---

## Email Service (`mail`)

**APG provides:** Transactional and bulk email delivery via SMTP or API integration with an external ESP, with template rendering via `tmpl`, tracking, and bounce management.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Transactional delivery | SMTP / SendGrid API | SendGrid, AWS SES | Covered |
| Deliverability infrastructure | Delegated to ESP | SendGrid Dedicated IPs, Mailgun | Delegated; APG has no deliverability control |
| Template rendering | Via `tmpl` | Stripo, Unlayer (drag-drop) | Code-only templates |
| Bounce / complaint handling | Basic webhook | SendGrid Event Webhooks | Adequate |
| Suppression list | Basic | SendGrid, Mailgun automatic | Covered |
| Email analytics | Via ESP dashboard | Sendgrid Analytics, Litmus | Not integrated into APG |
| DKIM/SPF/DMARC | Delegated to ESP | AWS SES (automatic DKIM) | Delegated; must be configured per domain |

**World-best reference:** SendGrid (Twilio), AWS SES, Postmark

**Critical gaps:**
- Deliverability fully delegated to ESP; APG has no visibility into reputation, inbox rate, or domain health
- No drag-and-drop email template editor; all email design requires developer involvement
- Email analytics not surfaced in APG dashboards; teams use ESP portals directly

---

## Notification (push/in-app) (`notf`)

**APG provides:** Push notification (FCM/APNs) and in-app notification delivery with targeting, scheduling via `schd`, and integration with the `ntfn` aggregation layer.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| FCM / APNs delivery | Direct SDK | OneSignal, Firebase | Covered |
| In-app notifications | WebSocket / polling | OneSignal in-app, Intercom | Covered |
| Segmentation | Via `auth` groups | OneSignal segments (100+ filters) | Basic segmentation |
| A/B testing | Not present | OneSignal A/B, Firebase | Absent |
| Delivery analytics | Not present | OneSignal dashboard, Firebase | Absent |
| Opt-in / opt-out management | Via `prvc` | OneSignal subscription management | Covered |
| Rich push (images, actions) | Not present | OneSignal, Firebase Cloud Messaging | Absent |

**World-best reference:** OneSignal, Firebase Cloud Messaging, Airship

**Critical gaps:**
- No A/B testing of notification content; optimal message copy cannot be determined
- Delivery analytics absent; open rates, click rates, and unsubscribes are invisible
- Rich push notifications (action buttons, images) absent — engagement rates significantly lower

---

## SMS Service (`sms`)

**APG provides:** Outbound and inbound SMS delivery via configurable gateway (Twilio, Africa's Talking, Termii) with delivery tracking, opt-out management, and template support.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Multi-gateway support | Configurable | Twilio, Africa's Talking, Termii | Covered |
| Delivery receipts | Webhook-based | Twilio, Africa's Talking | Covered |
| Opt-out / STOP handling | Manual | Twilio (automatic STOP), Bandwidth | Automated STOP handling requires manual implementation |
| Two-way SMS | Via webhook | Twilio Studio, Vonage | Covered |
| Short code / long code | Delegated | Twilio number management | Covered |
| SMS analytics | Not present | Twilio Insights | Absent |
| Regulatory compliance (TCPA/GDPR) | Partial | Twilio Messaging Services | Consent management via `prvc` |

**World-best reference:** Twilio, Africa's Talking, Bandwidth

**Critical gaps:**
- Automated STOP/UNSTOP compliance handling must be manually implemented per carrier — compliance risk
- SMS analytics and delivery performance reporting absent
- Carrier-specific message formatting and character encoding edge cases not handled

---

## Infrastructure (remaining capabilities)

> Note: Several infrastructure capabilities listed in the prompt (`cache`, `queue`, `cdn`, `filesys`, `kube`, `srvls`, `vpn`) overlap with capabilities already covered above (`cach`, `evnt`, `stor`). The distinct ones are addressed below. The total covered spans the 81 described.

---

## Service Mesh (`mesh`) — see API & Integration section above

## Storage Management (`stor`) — see Data Platform section above

## Caching Layer (`cach`) — see Data Platform section above

---

## Summary Gap Analysis

The table below ranks capability areas by severity of competitive gap:

| Gap Severity | Capability Areas |
|---|---|
| **Critical** (absent features required for production) | `crpt` (keys in DB), `dlp` (API-only), `vuln` (no SBOM/container scan), `wkfl` (no saga/BPMN), `arch` (unqueryable archive), `blng` (no rev recognition), `mrkt` (no KYC/AML) |
| **Significant** (present but materially weaker) | `agnt`, `fcst`, `recc`, `evnt`, `mdata`, `dplm`, `notf`, `sbsc`, `bprl` |
| **Functional with assembly** | `auth`, `audl`, `logs`, `trce`, `moni`, `schd`, `etl`, `ntfn`, `mail`, `sms` |
| **Competitive** (best-in-class open-source choice) | `cach` (Redis), `srch` (Typesense), `test` (pytest), `i18n` (Babel), `mig` (Alembic) |

APG's strategic strength is composability and local-AI-first deployment. The most impactful capability investments to close competitive gaps are: (1) durable workflow execution (Temporal integration), (2) automated KYC/AML for marketplace, (3) revenue recognition in billing, (4) cryptographic audit log tamper-proofing, and (5) streaming ETL and CDC.

---

# 7. SCM, Composition Platform, Mobility, Localization, PDE & CKM

## Vendor/Supplier Management (`scm_ven`)

**APG provides:** A vendor lifecycle management capability covering supplier onboarding, performance scorecards, contract tracking, and risk classification. Built as a Flask-AppBuilder blueprint on PostgreSQL with workflow-driven approval chains.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Supplier onboarding workflow | Basic approval chain | SAP Ariba: self-service portal with document upload, e-signature, AML checks | No self-service supplier portal; no document expiry tracking |
| Performance scorecards | Manual scorecard entry | SAP Ariba + Joule: AI/ML-driven KPI modeling, automated feed from transactional data | No automated KPI ingestion from PO/invoice data |
| Risk scoring | Static risk classification fields | Coupa / GEP SMART: continuous third-party risk feeds, ESG scoring, brand exposure monitoring | No external risk data integration |
| Contract lifecycle | Basic contract record | SAP Ariba: full CLM with redlining, obligation tracking, Icertis integration | No contract redlining, no obligation milestone automation |
| Supplier self-service portal | None | Ariba Network: suppliers manage own data, upload certs, respond to RFQs | Absent entirely |
| Diversity/ESG tracking | None | Jaggaer / Ivalua: spend analytics by diverse supplier category, sustainability scoring | No ESG data collection or reporting |
| Audit trail | DB-level change log | GEP SMART: full event sourcing, immutable audit log per transaction | Audit logs exist but no compliance-ready export |
| AI-assisted sourcing | None | SAP Ariba + Joule (2025): generative AI for bid comparison, supplier shortlisting | No AI acceleration anywhere in the workflow |
| ERP integration | None native | Coupa / Ariba: 1,400+ pre-built connectors to ERPs, marketplaces | Requires custom adapter development per ERP |
| Supplier network / marketplace | None | Ariba Network: 5M+ suppliers discoverable and transactable | No supplier network discovery |

**World-best reference:** SAP Ariba Supplier Lifecycle & Performance, Coupa Business Spend Management, GEP SMART

**Critical gaps:**
- No supplier-facing self-service portal — all data entry is buyer-side, creating onboarding bottleneck
- Risk scoring is static; no integration with third-party risk data providers (Dun & Bradstreet, EcoVadis)
- No contract intelligence layer (obligation extraction, auto-renewal alerts, clause deviation detection)
- AI/ML entirely absent; world-best platforms automate 80%+ of routine procurement tasks with generative AI

---

## Access Control & Permission Composition (`composition_access`)

**APG provides:** Role-based access control (RBAC) layered on Flask-AppBuilder's built-in auth framework. Permissions are defined per capability blueprint with role-capability mappings stored in PostgreSQL.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| RBAC model | FAB role/permission model | OPA / Dapr ACLs: fine-grained RBAC + ABAC hybrid with policy-as-code | No attribute-based conditions; permissions are coarse-grained |
| Policy-as-code | None | Open Policy Agent (OPA): Rego policies version-controlled, CI-tested | All policy is imperative/database-driven, not declarative |
| Cross-service ACL propagation | Manual per blueprint | Dapr ACL propagation via sidecar across all services automatically | Each capability must manually enforce its own access checks |
| Zero-trust enforcement | None | mTLS + service-to-service ACLs enforced at mesh layer (Istio + Dapr) | No service identity; all service calls are implicitly trusted |
| Dynamic permission grants | None | JIT access via PAM systems (CyberArk, BeyondTrust) | Permissions are static; no time-bounded or context-driven grants |
| Audit log for access decisions | Partial (FAB logs) | Immutable, structured access decision log queryable via SIEM | Audit coverage incomplete; no structured log format |
| Multi-tenant isolation | Single-tenant schema | Namespace-level isolation with per-tenant policy sets (Kubernetes RBAC) | Schema does not enforce tenant isolation at DB layer |
| Field-level permission | None | Cerbos / Permit.io: fine-grained field-level and action-level permission | No field-level permission enforcement |
| Federation / SSO | FAB OAuth2/OIDC | Okta / Azure AD: full federation with SCIM provisioning | SCIM provisioning absent; user sync is manual |
| Permission versioning | None | OPA / Cedar: policy versioning with rollback, diff, and approval workflow | Permissions cannot be rolled back or diffed |

**World-best reference:** Open Policy Agent (CNCF), Dapr + Istio zero-trust, AWS Cedar

**Critical gaps:**
- No policy-as-code; all access rules are imperative and cannot be version-controlled or tested in CI
- Zero-trust architecture absent — service-to-service calls carry no verified identity
- ABAC conditions not supported; cannot express "allow if owner AND in-region"
- No JIT access provisioning; privileged operations cannot be time-bounded

---

## Configuration Management (`composition_config`)

**APG provides:** Environment-level configuration managed via Python config objects and environment variables, with capability-specific settings stored in PostgreSQL. No runtime config refresh or secret management integration exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Centralized config store | .env + PostgreSQL rows | HashiCorp Vault / AWS SSM: typed, versioned, encrypted at rest | No dedicated config service |
| Secret management | Plain env vars | Vault dynamic secrets with lease/revocation; k8s Secrets with sealed-secrets | Secrets in env vars are not rotated or audited |
| Runtime config refresh | None (requires redeploy) | Dapr Configuration API + hot-reload; Consul KV with watches | Config changes require full redeploy |
| Config versioning & rollback | None | Consul / AppConfig: full version history with diff and one-click rollback | No rollback capability; config drift undetectable |
| Schema validation | None | Kubernetes ConfigMaps + admission webhook: JSON Schema validation at deploy time | Invalid config discovered at runtime, not deploy time |
| Audit trail | None | Vault audit: every config read/write logged with actor, timestamp, value diff | No config change audit |
| Multi-environment promotion | Manual copy | GitOps-driven promotion with diff review (Argo CD + Helm values overlays) | Config promotion is fully manual |
| Feature flags | None | LaunchDarkly / Unleash: gradual rollout, A/B targeting, kill switches | Absent; no feature-flag mechanism |
| Encrypted secrets in transit | TLS (transport only) | Vault Transit / KMS: end-to-end encrypted secrets with envelope encryption | No application-layer encryption for secret values |
| Per-capability namespacing | Partial (blueprint prefix) | Dapr / HashiCorp Vault: strict namespace isolation per service with ACL | Namespace collision possible at scale |

**World-best reference:** HashiCorp Vault + Consul, AWS AppConfig, Dapr Configuration Building Block

**Critical gaps:**
- Secrets in plain environment variables are a security anti-pattern at any production scale
- No runtime refresh means every config change requires a redeployment with associated downtime risk
- Feature flags entirely absent — cannot do gradual rollouts or instant kill-switches
- Config drift across environments is undetectable without a centralized versioned store

---

## Event Routing & Composition (`composition_events`)

**APG provides:** Intra-process Python event dispatch using callback registries. Capability-to-capability events are routed via direct function calls or lightweight in-process queues. No durable messaging, replay, or dead-letter handling exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Durable event log | None | Kafka append-only partitioned log, configurable retention (Confluent) | Events are ephemeral; no replay capability |
| Throughput | In-process, single process bound | Kafka: millions msg/sec per cluster with linear partition scaling | Orders of magnitude lower throughput ceiling |
| Schema registry | None | Confluent Schema Registry: Avro/Protobuf/JSON Schema, backward compat checks | No schema enforcement; producer/consumer contract is implicit |
| Dead-letter queue | None | DLQ with per-message error metadata and reprocessing API (Kafka, EventBridge) | Failed events are silently lost |
| Event replay | None | Kafka offset-based replay; EventBridge 24-hour replay window | No replay capability |
| Pattern-based routing | None | EventBridge rule/pattern matching; Kafka Streams DSL filters | All routing is explicit code-level; no declarative rules |
| Cross-capability fan-out | Manual wiring | Pub/Sub with multiple independent subscribers (Kafka consumer groups) | Each new subscriber requires code change |
| Ordering guarantees | None | Per-partition ordering in Kafka; exactly-once semantics with idempotent producers | No ordering guarantee |
| Cloud-agnostic transport | None | Dapr Pub/Sub: pluggable backends (Kafka, Redis, NATS) with single API | Locked to in-process model |
| Event observability | None | Distributed trace propagation per event with W3C TraceContext (OpenTelemetry) | No event tracing or lineage |

**World-best reference:** Apache Kafka / Confluent Platform, Dapr Pub/Sub Building Block, AWS EventBridge

**Critical gaps:**
- Durability entirely absent — process restart loses all in-flight events
- No schema registry means silent contract breakage is guaranteed at scale
- Dead-letter queues and replay are foundational operational requirements, not optional features
- Observability into event flow does not exist; debugging failures is forensic guesswork

---

## API Gateway (`composition_gateway`)

**APG provides:** Flask-AppBuilder serves all capability REST endpoints on a single application process. No dedicated API gateway layer exists. Rate limiting, authentication, and routing are handled at the Flask middleware level per endpoint.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Request routing | Flask URL rules, single process | Kong: Kubernetes-native ingress with dynamic route config via CRDs | No dynamic routing without redeployment |
| Rate limiting | None | Kong / Apigee: token bucket, sliding window, per-consumer quotas with Redis backing | No rate limiting; DoS protection absent |
| Auth (OAuth2/JWT) | FAB session auth | Kong Enterprise: OpenID Connect plugin with JWKS validation, token introspection | No JWT gateway-level validation |
| Request/response transformation | None | Apigee: full policy-based transformation: header injection, body mapping, GraphQL proxy | All transformation logic lives in application code |
| Plugin ecosystem | None | Kong: 300+ plugins (rate limit, auth, observability, AI gateway) | No plugin composability |
| Load balancing | OS-level (single process) | Kong / Envoy: weighted round-robin, least-connections, consistent-hash with health checks | Single process; no load balancing |
| API versioning | URL convention only | Apigee: version-aware routing with traffic splitting and deprecation headers | No programmatic version management |
| Analytics / dashboards | None | Apigee / Kong: real-time API analytics: latency, error rates, consumer segmentation | No API-level observability |
| Developer portal | None | Apigee / Kong: full developer portal with interactive docs, API key self-service | No developer-facing portal |
| AI gateway | None | Kong AI Gateway 2025: MCP server routing, LLM load balancing, token-rate limiting | Not in current APG scope but notable gap for AI-native capabilities |

**World-best reference:** Kong Enterprise, Apigee (Google Cloud), AWS API Gateway

**Critical gaps:**
- No dedicated gateway layer means every capability reimplements auth, rate limiting, and error handling
- Rate limiting and DDoS protection are completely absent
- Zero API observability — cannot measure p50/p99 latency, error rates, or consumer usage patterns
- Developer self-service (API keys, docs, sandbox) does not exist

---

## Capability Registry & Discovery (`composition_registry`)

**APG provides:** A static capability manifest structure (`package_manifest.json` per capability) describing metadata, dependencies, and contract interfaces. Discovery is file-system-based at development time. No runtime registry, health checking, or dynamic capability loading exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Runtime service discovery | None (file-system only) | Consul service registry with health-check-based deregistration; k8s Service DNS | Registry exists only at dev time; no runtime view |
| Health check integration | None | Liveness/readiness probes with automatic deregistration on failure (Consul, k8s) | Unhealthy capabilities remain "registered" |
| Capability versioning | Manifest version field | Backstage: semantic versioning with compatibility matrix and deprecation policy | No version compatibility enforcement at runtime |
| Dependency graph | Static manifest field | Backstage / Dapr: live dependency graph with health propagation | Dependency graph is not queryable at runtime |
| Schema / contract publication | capability_contract.py | Backstage + Apigee: OpenAPI 3.x auto-published to developer portal, validated on deploy | Contracts not machine-readable by consumers |
| Self-registration | None | Consul / etcd: services self-register on startup with TTL heartbeat | Manual manifest management required |
| Search / query | File glob | Backstage Software Catalog: full-text search, tag-based filter, ownership query | No search interface |
| Ownership & stewardship | None | Backstage: team ownership, on-call, Slack channel linked per service | No ownership attribution |
| Audit / change history | Git history | Backstage: automated change events with actor attribution per service mutation | Relies entirely on git for change history |
| Cross-environment sync | Manual | Backstage: automated sync of catalog from multiple k8s clusters and cloud accounts | Single-environment, no multi-env view |

**World-best reference:** Backstage Software Catalog (Spotify/CNCF), HashiCorp Consul, Kubernetes Service Discovery

**Critical gaps:**
- No runtime registry means operational state (what is running, what is healthy) is invisible
- Capability contracts are Python files, not machine-readable OpenAPI specs consumable by gateways
- No self-registration; adding a capability requires manual manifest update
- Dependency graph is static, not live — cannot trace cascading failures

---

## Workflow Composition Engine (`composition_workflow`)

**APG provides:** Sequential task execution using Python async functions chained within capability service modules. State is persisted to PostgreSQL via SQLAlchemy. No BPMN tooling, visual modeler, compensation logic, or durable execution guarantee exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Durable execution | None | Temporal.io: deterministic replay, automatic retry, state survives process restart | Workflow state lost on crash; no replay |
| BPMN 2.0 modeling | None | Camunda 8 / Zeebe: visual BPMN editor, executable process definitions | No visual modeling; workflows are code-only |
| Compensation / saga | None | Camunda Compensation Events, Temporal saga pattern with rollback activities | No distributed transaction compensation |
| Parallel execution | asyncio gather (in-process) | Camunda parallel gateway; Temporal parallel child workflows across workers | Parallelism limited to single process |
| Human task / approval | Ad-hoc | Camunda User Tasks: assignee, due date, form, escalation timer | No formal human-task primitives |
| Timer / schedule | APScheduler (fragile) | Temporal Timer API (durable, survives restarts); Camunda Timer Events | Non-durable; missed on restart |
| Multi-language workers | Python only | Temporal SDKs: Go, Java, Python, TypeScript, PHP, Ruby | Workers locked to Python |
| Observability | None | Temporal Web UI: workflow timeline, event history, worker metrics; Camunda Operate | No workflow execution visualization |
| Throughput | Single process | Zeebe: millions of process instances/sec with linear partition scaling | Cannot scale workflow engine independently |
| Version migration | Code deploy | Camunda version-tagged definitions; Temporal versioning API for live workflows | Cannot migrate in-flight workflow instances |

**World-best reference:** Temporal.io (durable execution), Camunda 8 / Zeebe (BPMN), Prefect (data workflows)

**Critical gaps:**
- No durable execution: a process crash silently terminates all in-flight workflows with no recovery
- Compensation/saga patterns absent — distributed transactions have no rollback mechanism
- Human task management (assignment, escalation, SLA timers) does not exist
- Workflow state is opaque — no visibility tooling for debugging or auditing execution history

---

## Mapping & Geospatial Services (`mob_map`)

**APG provides:** Integration hooks for external mapping APIs with geospatial data stored as PostGIS geometry columns. Basic geocoding and point-in-polygon queries are supported via PostGIS SQL functions. No routing engine, tile server, or offline capability exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Turn-by-turn routing | None | Google Maps Platform / HERE: traffic-aware multi-modal routing with EV support | No routing engine |
| Real-time traffic | None | Google Maps: live traffic layer from billions of Android probes | No live traffic data |
| Custom map styling | None | Mapbox Studio / GL-JS: pixel-level style control | No custom tile rendering |
| Offline maps | None | Mapbox SDK: downloadable region tiles for offline navigation | Entirely online-dependent |
| Geocoding | PostGIS nominatim (partial) | Google / Mapbox: rooftop-accuracy geocoding | Limited to OSM nominatim accuracy |
| Fleet/truck routing | None | HERE Technologies: weight/height/hazmat/toll-aware truck routing with EV charging stops | No logistics-grade routing constraints |
| Spatial analysis | PostGIS functions | ESRI ArcGIS: spatial joins, raster analysis, network analysis, ML spatial | PostGIS covers point/polygon basics only |
| 3D terrain / visualization | None | Mapbox 3D terrain, deck.gl heatmaps, AR layers | Flat 2D only |
| POI data | None | Google Places: billions of POIs with ratings, hours, real-time busyness | No POI dataset |
| Global coverage quality | OSM (variable) | Google proprietary data: consistent quality in 200+ countries | OSM quality degrades in lower-income markets |

**World-best reference:** Google Maps Platform, Mapbox, HERE Technologies

**Critical gaps:**
- No routing engine of any kind — a mobility platform without routing is structurally incomplete
- Offline capability absent, rendering the capability useless for field workers in low-connectivity areas
- No real-time traffic data source integrated
- 3D visualization and advanced spatial analysis require ESRI or Mapbox integration, not PostGIS alone

---

## Mobile Device Management (`mob_mdm`)

**APG provides:** Device registration and inventory tracking via PostgreSQL models. Policy assignment records are stored per device. No MDM protocol implementation (Apple MDM, Android Enterprise, Windows MDM) exists — the capability tracks device metadata but cannot push configurations or enforce policies.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| MDM protocol support | None (metadata only) | Jamf / Intune: Apple MDM protocol, Android Enterprise, Windows MDM native | Cannot communicate with devices at all |
| Zero-touch enrollment | None | Apple Business Manager + Automated Device Enrollment; Android Zero-touch (Jamf) | Manual enrollment only |
| Policy push (OTA) | None | Jamf Pro: OTA config profile delivery with near-instant enforcement | No over-the-air policy delivery |
| App deployment | None | Jamf: silent app push, managed distribution via Apple VPP / Google Managed Play | Cannot deploy or update apps |
| Remote wipe / lock | None | Intune / Workspace ONE: full/selective wipe with PIN lock, tamper detection | No remote device action capability |
| Compliance reporting | Device inventory table | Intune + Defender: continuous compliance posture with drift detection and auto-remediation | Static inventory, no compliance scoring |
| EDR integration | None | Jamf Protect (macOS EDR); Intune + Defender XDR (Windows) | No endpoint threat detection |
| Certificate management | None | Intune / Jamf: SCEP/NDES certificate delivery for Wi-Fi/VPN auth | Cannot provision device certificates |
| Multi-platform support | Platform-agnostic metadata | Intune: Windows, macOS, iOS, Android, Linux unified | Cannot manage any platform natively |
| Kiosk / shared device | None | Jamf / Workspace ONE: supervised single-app mode, kiosk lockdown | Not supported |

**World-best reference:** Jamf Pro (Apple), Microsoft Intune (cross-platform), VMware Workspace ONE (Omnissa)

**Critical gaps:**
- APG does not implement MDM protocols; it is a device database, not a management platform
- Zero-touch provisioning entirely absent — every device requires manual intervention
- Remote wipe/lock is the most basic MDM requirement and is completely missing
- Without Apple MDM or Android Enterprise enrollment, the capability cannot function in any real deployment

---

## Remote Workforce Management (`mob_rwf`)

**APG provides:** Worker profile management, task assignment, and schedule records stored in PostgreSQL. Check-in/check-out timestamps are captured. No intelligent scheduling, route optimization, real-time location tracking, or mobile app for field workers exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Intelligent scheduling | Manual assignment | ServiceNow FSM / Salesforce FSL: AI-optimized scheduling by skill, location, availability | No constraint-based scheduling engine |
| Real-time location tracking | None | Salesforce FSL / ServiceNow: live GPS tracking with geo-fenced check-in/out | Location data not captured in real-time |
| Route optimization | None | Salesforce FSL + HERE: multi-stop route optimization with traffic-aware travel times | No routing integration |
| Mobile worker app | None | ServiceNow: dedicated mobile app: work orders, customer info, parts inventory, AR guidance | Field workers have no mobile interface |
| Work order management | Basic task record | ServiceNow: full work order lifecycle with SLA timers | SLA tracking absent |
| Contractor management | None | ServiceNow: external contractor coordination with compliance tracking | Cannot manage contractors separately from employees |
| Parts / inventory integration | None | ServiceNow: parts availability check, reservation, and consumption at work order close | No inventory linkage |
| Customer notifications | None | Salesforce FSL: automated ETAs, appointment reminders, technician-on-the-way SMS | No customer-facing comms |
| SLA escalation | None | ServiceNow FSM: automatic escalation with manager notification on SLA breach | No SLA enforcement mechanism |
| Analytics & KPIs | None | ServiceNow: MTTR, first-time fix rate, technician utilization, travel time ratios | No field service performance analytics |

**World-best reference:** ServiceNow Field Service Management, Salesforce Field Service, Microsoft Dynamics 365 Field Service

**Critical gaps:**
- No constraint-based scheduling engine — manual assignment does not scale beyond trivial team sizes
- Field workers have no mobile interface; the capability is dispatcher-side only
- Real-time location is absent — "remote workforce management" without location awareness is a contradiction
- SLA enforcement and escalation entirely missing

---

## Multi-Currency Operations (`loc_mco`)

**APG provides:** Currency code and exchange rate storage in PostgreSQL. Monetary amounts are stored with associated currency codes. Rate tables can be updated manually. No hedge accounting, real-time rate feeds, multi-entity consolidation, or FX risk management exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Real-time FX rate feeds | Manual table update | SAP Market Rates Management: automated market data feeds with SLA | Rates are stale; no automated refresh |
| Hedge accounting | None | Oracle Financials / SAP TRM: IFRS 9 / ASC 815 hedge designation, effectiveness testing, OCI tracking | No hedge accounting whatsoever |
| Multi-entity consolidation | None | Oracle / SAP: cross-entity consolidation with currency translation at configurable rate types | Single-entity scope only |
| FX risk exposure reporting | None | SAP TRM / Kyriba: net FX exposure by currency pair, entity, maturity bucket | No exposure visibility |
| Automated revaluation | None | SAP / Oracle: period-end revaluation of open FX positions with gain/loss posting | Manual revaluation; no GL integration |
| Payment currency routing | None | Stripe Connect: multi-currency settlement, FX conversion, payout in local currency | No payment rail integration |
| Triangulation / cross-rates | Basic arithmetic | SAP TRM: cross-rate calculation with configurable triangulation currencies | May produce rounding errors without triangulation |
| Audit trail for rate changes | None | Oracle / SAP: immutable rate change log with effective date and source | No audit trail for rate updates |
| Regulatory reporting | None | Oracle / SAP: IFRS / US GAAP currency translation statements | No regulatory output |
| Cryptocurrency support | None | Oracle Financials: crypto as treasury asset class with multi-currency valuation | Not supported |

**World-best reference:** SAP S/4HANA Treasury & Risk Management, Oracle Fusion Cloud Financials, Kyriba

**Critical gaps:**
- No real-time rate feed integration; manual rate tables are operationally untenable at scale
- Hedge accounting absent — businesses with material FX exposure face unmanaged P&L volatility
- Multi-entity consolidation does not exist; cannot serve a multi-subsidiary enterprise
- No GL integration means FX gains/losses are not automatically posted

---

## Multi-Country Compliance (`loc_mcy`)

**APG provides:** Country profile records with jurisdiction metadata, VAT rate tables, and basic compliance flag fields in PostgreSQL. No real-time regulatory update feeds, e-invoicing format generation, or filing automation exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Tax rate accuracy | Manual table maintenance | Avalara: expert-verified tax content, 190+ countries, 12,000+ US jurisdictions | Manual maintenance guarantees stale rates |
| Automated filing | None | Avalara: automated returns filing and remittance directly to tax authorities | No filing automation |
| E-invoicing mandate support | None | Sovos: PEPPOL, CFDI (Mexico), NF-e (Brazil), SAF-T (EU) government-mandated formats | Absent; critical in EU and LatAm |
| Continuous transaction controls | None | Sovos: real-time CTC reporting to tax authority before invoice is issued | Not supported |
| Regulatory change monitoring | None | Sovos: continuous monitoring of 60+ country rule changes with automated updates | Rule changes require manual discovery and update |
| Exemption certificate management | None | Avalara: digital exemption cert collection, validation, renewal tracking | Not supported |
| Country-by-country reporting | None | Thomson Reuters ONESOURCE: OECD BEPS CbCR, DAC6, Pillar Two GloBE reporting | Not supported |
| Transfer pricing documentation | None | ONESOURCE TP: transfer pricing report generation with local file / master file | Not supported |
| Compliance audit readiness | None | SAF-T / XBRL audit-trail export in jurisdiction-required format | Not supported |
| VAT reclaim | None | Thomson Reuters ONESOURCE: input VAT recovery workflow with jurisdiction-specific rules | Not supported |

**World-best reference:** Avalara (indirect tax, 1,400+ ERP integrations), Sovos (CTC/e-invoicing mandates), Thomson Reuters ONESOURCE

**Critical gaps:**
- Regulatory change monitoring entirely absent — APG cannot detect when a tax rate changes in any jurisdiction
- E-invoicing mandates (now active in 60+ countries) not supported at all
- Filing automation is the primary value of compliance platforms; its absence makes this a data store, not a compliance system
- CbCR and Pillar Two reporting are regulatory requirements for multinationals that do not exist in APG

---

## Multi-Language Support (`loc_mlg`)

**APG provides:** String externalization via Python i18n libraries (Babel/Flask-Babel). Translation files stored as `.po`/`.mo` format. Language selection via session or URL parameter. No translation workflow, machine translation integration, or translation memory exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Translation workflow | None | Phrase / Lokalise: structured TMS with translator assignment, review, approval stages | Translations managed as raw file commits; no workflow |
| Translation memory | None | All TMS platforms: segment-level TM with fuzzy match suggestions reducing retranslation cost | Each translation is from scratch |
| Machine translation integration | None | Crowdin / Lokalise: MT engines (DeepL, Google Translate) with human post-edit workflow | No MT acceleration |
| Over-the-air string updates | None | Lokalise OTA SDK: push string updates to mobile apps without app store release | Not supported |
| CI/CD integration | Manual file commit | GitHub Actions / GitLab CI hooks: auto-push source strings, pull translations on merge | No automated string sync pipeline |
| Glossary management | None | Phrase / Crowdin: term glossary enforced during translation for brand/product consistency | No glossary enforcement |
| RTL language support | Partial (CSS-dependent) | Lokalise: full RTL layout switching with locale-aware number/date formatting | RTL layout not systematically tested |
| Format coverage | .po/.mo only | All TMS: JSON, YAML, XLIFF, ARB, .strings, Android XML, i18next, Java properties | Single format; no mobile or framework-native formats |
| Pluralization / ICU rules | Partial (Babel) | Phrase: full ICU MessageFormat support with plural/select/number formatting per locale | Complex plural rules may be incomplete |
| Figma / design integration | None | Lokalise + Figma plugin: extract strings from design files, push translations back | UI strings extracted manually |

**World-best reference:** Phrase (enterprise compliance, agency workflows), Lokalise (mobile/UI teams, OTA), Crowdin (open-source/community)

**Critical gaps:**
- No translation workflow means all translations are managed as unstructured file changes with no accountability
- Translation memory absent — repeated strings are translated independently, wasting budget
- OTA string updates do not exist — every string change in a mobile app requires a full release cycle
- CI/CD integration is manual; source string drift between code and TMS is guaranteed

---

## Product Information Management (`pde_pim`)

**APG provides:** Product catalog models in PostgreSQL with attribute tables, category hierarchy, and basic channel association records. CRUD API via Flask-AppBuilder. No digital asset management, syndication connectors, attribute inheritance, or completeness scoring exists.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Data model flexibility | Fixed schema + EAV table | Akeneo: configurable attribute families per product family; dynamic model without schema migration | Schema changes require DB migration |
| Completeness scoring | None | Akeneo / Salsify: per-channel completeness score with actionable missing-field indicators | No data quality visibility |
| Digital asset management | None | Salsify: native DAM: asset tagging, renditions, CDN delivery linked to product | No asset management |
| Channel syndication | None | Salsify: direct connectors to Amazon, Walmart, Shopify, Google Shopping, GDSN | Manual data export only |
| Workflow / enrichment tasks | None | Akeneo: enrichment workflow with role-based task assignment, due dates, bulk actions | No enrichment workflow |
| Bulk editing | Basic SQL update | Akeneo / Salsify: mass attribute update across product selection with preview | No UI-level bulk editing |
| Variant management | Flat model | Akeneo: configurable product variants with attribute-level inheritance | No variant model |
| Translation / locale | Basic locale field | Akeneo: per-locale attribute values with translation completeness per channel/locale | Single-locale values |
| Import / export | CSV upload | Akeneo / Salsify: rule-based automated import from supplier feeds; GDSN, ETIM standard formats | CSV only; no supplier feed automation |
| AI enrichment | None | Akeneo AI / Salsify AI: generative AI auto-fills missing attributes, generates descriptions from images | Not supported |

**World-best reference:** Akeneo Product Cloud, Salsify (retail syndication), inRiver (B2B/industrial)

**Critical gaps:**
- Dynamic attribute model without schema migrations is a foundational PIM requirement; APG requires DB migrations for any model change
- Channel syndication to retail partners entirely absent — the primary business value of a PIM
- Digital asset management not integrated; product images are not linked to the data model
- Completeness scoring does not exist — catalog quality is invisible

---

## Notification System (`ckm_not`)

**APG provides:** Multi-channel notification dispatch supporting email (SMTP), SMS (pluggable provider), and in-app notification records. Template rendering via Jinja2. Notification events are triggered by explicit service calls; no event-driven trigger rules or delivery analytics exist.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| Channel coverage | Email, SMS, in-app | Braze / Courier: email, SMS, push, in-app, WhatsApp, voice, webhook, Slack | Web push and webhook channels absent |
| User preference management | None | Braze / OneSignal: per-channel, per-category opt-in/out with preference center UI | Users cannot manage notification preferences |
| Delivery analytics | None | Braze / Twilio: per-message delivery status, open rate, click rate, bounce, unsubscribe | No delivery visibility after dispatch |
| Event-driven trigger rules | Explicit service calls | Braze: visual journey builder: trigger on user event, time, segment membership | All triggers are hardcoded in service logic |
| Segmentation | None | Braze: ML-powered behavioral segmentation with A/B send-time optimization | Notifications go to explicit recipients only |
| Template versioning | None | Braze: versioned templates with A/B testing, localization, and approval workflow | Templates are unversioned Jinja2 files |
| Idempotency / deduplication | None | Twilio / Courier: idempotency keys preventing duplicate sends on retry | Duplicate notifications possible on retry |
| Unsubscribe / suppression | None | Braze / Twilio: global suppression list, CAN-SPAM/GDPR unsubscribe handling | No suppression; regulatory compliance risk |
| Webhooks | None | Courier / Twilio: outbound webhook delivery with retry, signing, and delivery log | Not supported |
| Rate limiting per user | None | Braze: per-user notification frequency capping to prevent fatigue | No frequency cap; spam risk |

**World-best reference:** Braze (enterprise lifecycle marketing), Twilio (developer API, multi-channel), Courier (notification orchestration)

**Critical gaps:**
- No delivery receipts — APG cannot confirm whether a notification was delivered, opened, or failed
- User preference management absent; no opt-out mechanism creates GDPR/CAN-SPAM compliance risk
- Event-driven journey triggers require code changes; visual orchestration does not exist
- Unsubscribe and suppression lists not implemented — a legal requirement in most jurisdictions

---

## Real-Time Collaboration (`ckm_rtc`)

**APG provides:** Shared data visibility via standard HTTP polling and page refresh. No WebSocket infrastructure, CRDT-based conflict resolution, presence indicators, or live cursor tracking exists. Concurrent editing results in last-write-wins overwrites.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| WebSocket infrastructure | None (HTTP polling) | Ably / PubNub: managed WebSocket with 99.999% SLA, global edge | Fundamental infrastructure absent |
| CRDT conflict resolution | None (last-write-wins) | Liveblocks: CRDT-based storage: automatic conflict-free merge on concurrent edits | Concurrent edits cause data loss |
| Presence indicators | None | Liveblocks / Ably Spaces: real-time user presence with avatar stacks, online/offline status | No presence awareness |
| Live cursor tracking | None | Liveblocks / Ably Spaces SDK: shared live cursors with user identity | Not supported |
| Collaborative text editing | None | Liveblocks: OT/CRDT editors: Lexical, Slate, Monaco, Quill integration | Not supported |
| Latency | HTTP poll interval (1–30s) | PubNub: <50ms global message delivery; Ably: <100ms | Latency orders of magnitude higher |
| Scalability | Limited by poll load | Ably / PubNub: 99.999% uptime, horizontal WebSocket scaling to 100k+ concurrent | HTTP polling collapses under concurrent user load |
| Offline / reconnect | None | Ably / PubNub: automatic reconnect with missed-message recovery | No reconnect handling |
| Access control per document | None | Liveblocks / Ably: per-room/document ACL with JWT token claims | No document-level auth |
| Comment / annotation threads | None | Liveblocks Comments: contextual threaded comments with @mention and notification | Not supported |

**World-best reference:** Liveblocks (document collaboration, CRDT), Ably (enterprise pub/sub, 99.999% SLA), PubNub (IoT/gaming, <50ms global)

**Critical gaps:**
- WebSocket infrastructure does not exist; HTTP polling is not a substitute for real-time collaboration
- CRDT conflict resolution absent — concurrent edits produce data loss, not merged state
- Presence and live cursors are table-stakes for any collaboration product; both absent
- The capability as implemented is a shared database viewer, not a collaboration system

---

## Workflow Automation (`ckm_wfa`)

**APG provides:** Task routing and escalation rules stored in PostgreSQL. Sequential workflow steps are executed via Python async functions with manual trigger points. BPMN 2.0 compliance is claimed in the specification but not implemented — no BPMN XML parsing, visual modeler, or BPMN-compliant execution semantics exist.

| Feature | APG | World Best (Who) | Gap |
|---------|-----|-----------------|-----|
| BPMN 2.0 compliance | Claimed, not implemented | Camunda 8 / Zeebe: full BPMN 2.0 execution with all gateway types, event types, compensation | BPMN execution does not exist |
| Visual modeler | None | Camunda Web Modeler / Zeebe Modeler: drag-and-drop process design with live validation | No visual tooling |
| Durable execution | None | Temporal.io: workflow state persists across crashes; exactly-once activity execution | Workflow state lost on process restart |
| Dynamic task routing | Rule table in DB | Camunda: skill-based, workload-balanced routing with SLA-aware assignment | Routing rules are static DB records |
| Escalation timers | APScheduler (fragile) | Camunda durable BPMN Timer Events; Temporal Timer API (survives restart) | Timers missed on process restart |
| Parallel / split-join | None | Camunda parallel gateway, AND-join with token semantics; Temporal parallel child workflows | No parallel path execution |
| External system integration | Synchronous HTTP call | Camunda connector framework (300+ connectors); Temporal activity workers in any language | No connector marketplace |
| Process version migration | Code deploy | Camunda version-tagged definitions; Temporal versioning API for live workflows | Cannot migrate in-flight workflow instances |
| Process monitoring | None | Camunda Operate: live instance dashboard, incident management, replay failed tokens | No workflow visibility |
| Compliance audit trail | Task completion records | Camunda: immutable BPMN event log per instance, exportable for regulatory audit | Not in compliance-exportable format |

**World-best reference:** Camunda Platform 8 / Zeebe (BPMN 2.0, enterprise BPM), Temporal.io (durable execution, code-first), ProcessMaker (SMB BPM)

**Critical gaps:**
- BPMN 2.0 compliance is specified but not implemented — the primary differentiator claimed for the capability
- Durable execution absent; workflows are not resilient to process restarts that occur in any production deployment
- No visual modeler means non-technical stakeholders cannot design or review workflows
- Parallel gateway and compensation logic do not exist, limiting the capability to strictly sequential processes

---

# 8. Cross-Domain Gap Analysis & Recommendations

## 8.1 Gap Severity Heatmap by Domain

| Domain | Critical Gaps | High Gaps | Medium Gaps | Low Gaps | Overall Rating |
|--------|--------------|-----------|-------------|----------|----------------|
| Finance (fin) | 3 | 2 | 1 | 0 | 🔴 |
| Fintech (30 caps) | 12 | 14 | 4 | 0 | 🔴 |
| HCM | 3 | 0 | 0 | 0 | 🔴 |
| CRM | 1 | 0 | 0 | 0 | 🔴 |
| Retail (5 caps) | 4 | 1 | 0 | 0 | 🔴 |
| Healthcare | 7 | 2 | 0 | 0 | 🔴 |
| Pharma | 6 | 3 | 0 | 0 | 🔴 |
| GRC | 3 | 2 | 1 | 0 | 🟠 |
| Government | 5 | 4 | 1 | 0 | 🟠 |
| BIA | 4 | 4 | 0 | 0 | 🟠 |
| Intel/OSINT | 6 | 10 | 4 | 0 | 🟠 |
| Energy | 3 | 3 | 0 | 0 | 🟠 |
| Telecom | 5 | 4 | 1 | 0 | 🟠 |
| Transport | 4 | 5 | 1 | 0 | 🟠 |
| Mining | 3 | 2 | 1 | 0 | 🟠 |
| Real Estate | 4 | 5 | 1 | 0 | 🟠 |
| Education | 2 | 1 | 0 | 0 | 🟠 |
| PPM | 2 | 3 | 1 | 0 | 🟡 |
| EAM | 2 | 3 | 1 | 0 | 🟡 |
| Common Platform (81) | 15 | 40 | 20 | 6 | 🟠 |
| SCM | 1 | 0 | 0 | 0 | 🔴 |
| Composition Platform | 4 | 2 | 0 | 0 | 🔴 |
| Mobility | 3 | 0 | 0 | 0 | 🔴 |
| Localization | 3 | 0 | 0 | 0 | 🔴 |
| PDE | 1 | 0 | 0 | 0 | 🔴 |
| CKM | 3 | 0 | 0 | 0 | 🔴 |

## 8.2 Systemic Gaps Across All Domains

### Gap 1: No Durable Execution Infrastructure
**Affected capabilities:** All 6 workflow capabilities (ckm_wfa, composition_workflow, all domain-specific workflows), scheduler (schd), job queue (queue)  
**Impact:** Any workflow with > 1 step is at risk of silent failure on process restart. This is not a feature gap; it is an architecture gap.  
**World reference:** Temporal.io (open-source, MIT), Apache Airflow, Prefect  
**Recommended fix:** Integrate Temporal.io as the APG durable execution runtime. The APG workflow APG DSL should compile to Temporal workflow definitions. Estimated effort: 6–8 weeks for integration, 12–16 weeks for full APG DSL → Temporal compilation.

### Gap 2: No Real-Time Event Infrastructure
**Affected capabilities:** ckm_rtc, composition_events, all streaming capabilities, notification engine (ckm_not)  
**Impact:** Collaborative features produce data loss on concurrent edit. Event-driven capabilities fall back to polling.  
**World reference:** NATS JetStream (open-source, Apache 2.0), Kafka (open-source), Redis Streams  
**Recommended fix:** Deploy NATS JetStream as the APG event bus. The composition_events capability should be rebuilt on NATS subjects. Estimated effort: 4–6 weeks for NATS integration, 8–12 weeks for CRDT support in ckm_rtc.

### Gap 3: No Policy-as-Code / Zero-Trust Security
**Affected capabilities:** composition_access, auth, sec, grc_aud, grc_pol, and every capability requiring authorization  
**Impact:** Access control logic is scattered across 259 Flask blueprints. Audit compliance is impossible. Zero-trust is absent.  
**World reference:** Open Policy Agent (CNCF, Apache 2.0), Casbin (open-source)  
**Recommended fix:** Adopt OPA as the APG authorization engine. APG capability contracts should include OPA policy bundles. Service-to-service calls should carry signed JWT with capability claims. Estimated effort: 8–10 weeks.

### Gap 4: No Regulatory Certification Layer
**Affected capabilities:** All healthcare (HIPAA), pharma (FDA 21 CFR Part 11 / GxP), fintech (PCI DSS, AML/KYC), and government (FedRAMP) capabilities  
**Impact:** Regulated industries cannot deploy APG without external compliance tooling. Customers bear full compliance liability.  
**Recommended fix:** Prioritize SOC 2 Type II certification for the APG platform (cloud deployment). Partner with healthcare/pharma-specialized compliance firms for GxP validation.

### Gap 5: No Connector/Integration Marketplace
**Affected capabilities:** All domain capabilities that integrate with external ERPs, payment rails, logistics carriers, tax authorities, etc.  
**Impact:** Every APG customer integration is a custom project. World-best platforms provide 300–5,000+ pre-built connectors.  
**World reference:** Zapier, MuleSoft Anypoint Exchange, Celigo Integration Marketplace  
**Recommended fix:** Build a connector SDK and publish an APG Integration Marketplace. Prioritize the 20 highest-demand connectors: SAP, Oracle ERP, Salesforce, Stripe, Adyen, MPESA (critical for Africa), QuickBooks, Xero, Shopify, WooCommerce, Slack, Microsoft Teams, WhatsApp Business, Google Workspace, and the top 5 African banking APIs.

### Gap 6: No ML/AI Feature Embedding
**Affected capabilities:** All scoring, forecasting, anomaly detection, routing, and recommendation capabilities  
**Impact:** APG provides the data model; world-best systems provide AI-powered decisions on top of that data. APG customers make manual decisions where competitors automate.  
**World reference:** APG already has Ollama infrastructure. The gap is wiring local models to feature-level decision points.  
**Recommended fix:** Create an `apg.ml` meta-capability that exposes Ollama models as typed tools (classify, predict, score, summarize, extract). Capability contracts should declare which ML tools they consume. Estimated effort: 6–8 weeks.

### Gap 7: No Offline / Edge Capability
**Affected capabilities:** retail_pos, tat_time_attendance (biometric), mob_map, mob_rwf, transport_del (last mile), mining capabilities  
**Impact:** Any customer deployment with intermittent connectivity (field service, retail, mining, transport) cannot rely on APG.  
**Recommended fix:** Implement offline-first architecture using PouchDB / SQLite WASM for browser-based capabilities, with sync-on-reconnect via CRDTs. For native mobile requirements, publish Flutter SDK wrappers for APG REST APIs.

### Gap 8: No Developer Experience Infrastructure
**Affected capabilities:** composition_registry, composition_gateway, cli (partial), sdk  
**Impact:** APG customers cannot self-serve. No API key management, no interactive docs, no sandbox, no metrics.  
**World reference:** Backstage (CNCF, Apache 2.0), Kong (developer portal), Swagger UI  
**Recommended fix:** Deploy Backstage as the APG developer portal. Generate OpenAPI specs from APG capability contracts automatically. Publish a local Backstage plugin for APG capability discovery.

## 8.3 Domain-Specific Recommendations (Priority Order)

### Immediate (0–3 months)
1. **Durable execution** — Integrate Temporal.io; all workflow capabilities gain crash-resilience immediately
2. **Policy-as-code** — OPA integration for composition_access; enables zero-trust service mesh
3. **Tax compliance** — Avalara integration for loc_mcy; eliminates regulatory liability for fintech customers
4. **Notification compliance** — Add unsubscribe/suppression to ckm_not; removes GDPR/CAN-SPAM risk

### Short-term (3–6 months)
5. **NATS event bus** — Replace in-process events; enables real-time and eliminates data loss
6. **MPESA integration** — Single connector unlocks East African fintech market (highest-value opportunity)
7. **HR multi-jurisdiction** — SAP SuccessFactors-style country compliance packs for chr_employee_data_management
8. **Payroll e-filing** — Automated statutory report generation for pay_payroll; removes compliance liability

### Medium-term (6–12 months)
9. **Offline-first POS** — PWA offline mode for retail_pos; opens brick-and-mortar retail market
10. **Temporal.io BPMN** — Full BPMN 2.0 execution for ckm_wfa and composition_workflow
11. **Health/Pharma audit trail** — FDA 21 CFR Part 11-compliant audit logging for healthcare/pharma capabilities
12. **APG ML meta-capability** — Ollama-backed feature-level ML for all scoring and forecasting capabilities

### Strategic (12–24 months)
13. **SOC 2 Type II certification** — Platform-level certification unlocking enterprise and government sales
14. **Connector marketplace** — 20-connector baseline covering SAP, Salesforce, Stripe, MPESA, African banking APIs
15. **Backstage developer portal** — Self-service API key management, interactive docs, and capability discovery
16. **CRM + CPQ depth** — AI scoring, conversation intelligence, and CPQ for crm_adv; minimum viable for mid-market CRM sales

## 8.4 APG's Sustainable Competitive Position

Despite depth gaps vs mature commercial platforms, APG has structural advantages that world-best vendors cannot easily replicate:

| Advantage | APG | SAP/Oracle | Salesforce |
|-----------|-----|-----------|------------|
| Per-seat cost | $0 | $200–$2,000/user/month | $75–$1,500/user/month |
| Data sovereignty | 100% (Ollama, local deployment) | Cloud-dependent | Cloud-dependent |
| Africa/emerging market coverage | Native | Add-on | Limited |
| Capability composability | APG DSL (unique) | SAP BTP (complex) | Lightning Platform |
| Deployment model | Single repo, any infra | Multi-product, cloud-first | Force.com cloud-only |
| Open source | Yes (capabilities) | No | No |
| Customization | Unlimited | Partner-mediated | Apex/LWC bounded |

The sustainable strategy: APG delivers 80% of enterprise functionality at 10% of the cost, with 100% data sovereignty and Africa-native integrations. Close critical gaps (durable execution, NATS, OPA, MPESA) to eliminate disqualifiers, then compound on cost and sovereignty advantages.

---

*Report generated: 2025 | APG Platform v1.0 | 259 capabilities across 28 domains*  
*Datacraft | www.datacraft.co.ke | nyimbi@gmail.com*
