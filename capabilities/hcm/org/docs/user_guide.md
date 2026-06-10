# Organizational Management — User Guide

## Overview

The Organizational Management (ORG) capability provides tools for designing, maintaining, and analysing your organisational structure. It manages the formal hierarchy of business units, positions, and reporting relationships.

## Use Cases

- Build and maintain an org chart with departments, divisions, and teams
- Define positions with job grades and critical role flags
- Assign employees to positions and track vacancies
- Define direct, dotted-line, and functional reporting relationships
- Plan annual headcount by org unit
- Manage organisational restructuring programmes with approval workflows

## Org Chart

The org chart is a tree structure. Create a root unit first (type `company` or `division`), then create child units with `parent_unit_id`. Retrieve the full chart via `GET /api/hcm/org/chart`.

## Positions

Positions belong to an org unit. A position has a `status` of `open`, `filled`, `frozen`, or `abolished`. Critical positions (`is_critical: true`) are highlighted in succession planning.

## Reporting Lines

Three line types are supported: `direct` (solid line), `dotted` (informal), and `functional` (cross-functional). Use `GET /api/hcm/org/reporting-lines?manager_employee_id=<id>` to list direct reports.

## Headcount Planning

Create headcount plans per org unit per year via `POST /api/hcm/org/headcount-plans`. The API computes the variance between current and planned headcount automatically.

## Restructuring

Restructurings flow through: `draft → proposed → approved → in_progress → completed`. Only draft restructurings can be deleted.

## API Quick Reference

```
GET  /api/hcm/org/health
POST /api/hcm/org/units
GET  /api/hcm/org/chart
POST /api/hcm/org/positions
PUT  /api/hcm/org/positions/{id}/assign
POST /api/hcm/org/reporting-lines
POST /api/hcm/org/restructurings
GET  /api/hcm/org/analytics
```
