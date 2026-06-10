# Succession Planning — User Guide

## Overview

The Succession Planning (SCP) capability enables organisations to identify, assess, and develop talent for critical roles. It covers talent pools, readiness assessments, the nine-box grid, succession scenarios, and critical role management.

## Use Cases

- Build talent pools with minimum readiness thresholds
- Assess employees' readiness (developing / ready_in_1_year / ready_now) for specific roles
- Place employees on the nine-box grid (performance x potential) for a review cycle
- Create succession scenarios with ranked successors for each role
- Identify and track critical roles with succession coverage reporting

## Talent Pools

Create pools via `POST /api/hcm/scp/talent-pools`. Set `min_readiness_level` to gate entry. Add employees via `POST /api/hcm/scp/talent-pools/{id}/members` — the API enforces the minimum readiness gate.

## Readiness Assessments

Provide `performance_rating` (1–5) and `potential_rating` (1–5) along with `readiness_level`. The API automatically computes the nine-box quadrant for the assessment record.

## Nine-Box Grid

Place employees on the grid via `POST /api/hcm/scp/nine-box` using axes of 1.0–3.0 (low/medium/high). Retrieve a grouped grid view for a review cycle via `GET /api/hcm/scp/nine-box/grid?review_cycle=2026-H1`.

Nine-box quadrant labels: `star`, `high_performer`, `solid_contributor`, `high_potential`, `core_employee`, `inconsistent_player`, `enigma`, `average_performer`, `underperformer`.

## Succession Scenarios

Each scenario lists successors with `employee_id`, `readiness`, and `rank`. Scenarios flow `draft → active`. Activate via `PUT /api/hcm/scp/scenarios/{id}/activate`.

## Critical Roles

Flag roles as critical via `POST /api/hcm/scp/critical-roles`. Use `GET /api/hcm/scp/coverage-report` to see which critical roles have no successors.

## API Quick Reference

```
GET  /api/hcm/scp/health
POST /api/hcm/scp/talent-pools
POST /api/hcm/scp/readiness-assessments
POST /api/hcm/scp/nine-box
GET  /api/hcm/scp/nine-box/grid?review_cycle=2026-H1
POST /api/hcm/scp/scenarios
POST /api/hcm/scp/critical-roles
GET  /api/hcm/scp/coverage-report
GET  /api/hcm/scp/dashboard
```
