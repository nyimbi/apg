# Supply Planning (scm_spl) — World-Class Improvement Roadmap

© 2025 Datacraft | Author: Nyimbi Odero

---

## 1. Probabilistic Demand Forecasting with Prediction Intervals

Replace point forecasts with full predictive distributions. Implement Monte Carlo simulation over demand scenarios so every forecast carries P10/P50/P90 quantiles. This lets downstream safety stock and MRP calculations propagate uncertainty correctly instead of hiding it inside a single confidence_pct scalar.

**Impact**: Eliminates systematic under-stocking at high service levels caused by ignoring forecast distribution tails.

---

## 2. Dynamic Safety Stock with Demand Sensing

Current Z-score formula treats lead time and demand variability as static inputs. Replace with a rolling recalculation that ingests near-real-time POS or warehouse movement data to update `demand_std_dev` and `lead_time_days` continuously. Trigger automatic safety stock revision when the 7-day rolling CV deviates by more than a configurable threshold from the baseline.

**Impact**: Reduces excess inventory 15–25% while maintaining service levels during demand spikes.

---

## 3. Multi-Echelon Inventory Optimisation (MEIO)

Current safety stock is calculated per SKU per warehouse in isolation. Implement a multi-echelon model that optimises safety stock across distribution centres and stores jointly, allocating buffer optimally across the supply chain network rather than at each node independently.

**Impact**: Network-wide inventory reduction of 20–40% with equivalent service level.

---

## 4. Supplier Lead-Time Variability Tracking

Add a `SupplierPerformance` entity that tracks actual vs promised lead times per supplier/SKU, computes running lead-time mean and std-dev, and feeds these into the safety stock formula automatically. Raise supply exceptions when a supplier's lead-time sigma exceeds contract SLA.

**Impact**: Accurate safety stock based on actual supplier behaviour, not assumed constants.

---

## 5. Economic Order Quantity (EOQ) and Total Cost Optimisation

Current order quantities are rule-based (fixed or min-max fill). Add EOQ computation incorporating ordering cost, holding cost rate, and demand rate. Extend to include quantity-discount break-point analysis (Wagner-Whitin for dynamic lot sizing) and produce a total cost comparison across lot-sizing policies.

**Impact**: Reduces total inventory cost (ordering + holding) by optimising order cadence.

---

## 6. Constrained MRP with Capacity Feedback Loop

Current MRP generates planned orders without respecting capacity limits. Integrate the capacity plan into the MRP explosion loop: when planned demand in a period exceeds available capacity, automatically time-shift orders to adjacent periods (forward scheduling, then backward from due date) and flag residual violations as hard exceptions.

**Impact**: Eliminates infeasible plans that blow through capacity, reducing expediting cost.

---

## 7. Vendor-Managed Inventory (VMI) and Collaborative Replenishment

Add a VMI mode where suppliers receive read-access to stock positions and planned orders, can propose replenishment schedules, and the system reconciles supplier proposals against internal rules. Include a Collaborative Planning Forecasting and Replenishment (CPFR) workflow with supplier forecast submission and conflict resolution.

**Impact**: Reduces ordering overhead and improves supplier fill rates.

---

## 8. ABC-XYZ Segmentation Engine

Classify the SKU universe by value (ABC) and demand volatility (XYZ) to drive differentiated planning policies. High-value stable items (AX) get tight service-level targets and frequent review; low-value erratic items (CZ) get wider safety bands and periodic review. Automatically assign replenishment rule templates by segment.

**Impact**: Focuses planning effort on high-leverage items, reduces blanket safety stock waste.

---

## 9. Perishability and Shelf-Life Constraints

Add `shelf_life_days` and `manufactured_date` tracking to demand forecasts and planned orders. Integrate FEFO (First Expired First Out) logic into supply/demand balancing and raise pre-expiry alerts when projected closing stock will expire before consumption. Support write-down cost modelling.

**Impact**: Critical for food, pharma, and chemical sectors; reduces write-off losses.

---

## 10. Scenario Planner and What-If Simulation

Allow planners to fork the current supply plan into named scenarios (e.g. "demand +20%", "supplier delay 2 weeks"), run full MRP and safety stock recalculation for each scenario, and produce a side-by-side comparison of KPIs (total inventory cost, service level, capacity utilisation). Scenarios are isolated and non-destructive to the baseline plan.

**Impact**: De-risks planning decisions; enables rapid response to demand or supply shocks.

---

## 11. Rolling Wave Planning with Frozen Zone Enforcement

Implement planning horizon segmentation: a frozen zone (0–2 weeks, orders locked), a firm zone (2–6 weeks, changes require approval), and a flexible zone (6+ weeks, free planning). Enforce zone rules in the MRP run and expose approval workflow for firm-zone changes.

**Impact**: Reduces production and procurement disruption from late plan changes.

---

## 12. Forecast Bias Detection and Auto-Correction

Track cumulative forecast error (CFE) and tracking signal (TS = CFE / MAD) per SKU. When |TS| > configurable threshold (typically 4–6), flag a forecast bias exception and apply an auto-correction factor (Trigg's tracking signal) to the next forecast. Surface bias metrics in the planning dashboard.

**Impact**: Eliminates systematic over/under-forecasting that compounds into chronic inventory imbalances.

---

## 13. Supply Chain Risk Scoring

Add a risk scoring subsystem that evaluates each planned order against supplier concentration, geopolitical risk flags, single-source dependencies, and lead-time variability. Produce a composite supply risk score per SKU and recommend diversification or buffer-stock adjustments for high-risk items.

**Impact**: Proactive risk management; reduces probability of stockouts from supply disruptions.

---

## 14. Planned Order Firm-and-Release Workflow

Current planned orders stay perpetually in "planned" status. Add a firm-and-release lifecycle: planned → firmed (manual or rule-based confirmation) → released (sent to procurement or production) → confirmed → received. Each transition fires an audit event and can trigger downstream capabilities (scm_po, scm_wms).

**Impact**: Closes the loop between planning and execution; enables procurement automation.

---

## 15. Inventory Turnover and Days-on-Hand Analytics

Extend the planning dashboard with inventory turnover ratio (COGS / average inventory), days on hand (DOH = inventory / daily demand), and fill-rate tracking per SKU and warehouse. Add trend sparklines and alerting when DOH breaches configured upper/lower bounds. Export to BI tools via a structured analytics endpoint.

**Impact**: Gives planners and finance teams a shared KPI language; drives continuous inventory reduction.

---

*Improvements are sequenced by implementation complexity: 1–5 are algorithmic enhancements to existing methods, 6–10 require new entity types, 11–15 require workflow and integration extensions.*
