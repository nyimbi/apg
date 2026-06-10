# Professional Development — User Guide

## Overview

The Professional Development (PRO) capability supports employee growth through structured development plans, skill gap analysis, mentoring relationships, certification tracking, and career pathing.

## Use Cases

- Create annual individual development plans with objectives and focus areas
- Build a skills catalogue and assess employee proficiency against targets
- Identify skill gaps and generate gap reports per employee
- Pair mentors and mentees in structured programmes, log sessions
- Track professional certifications with expiry alerts
- Define career paths with milestones from current to target role

## Development Plans

Plans flow: `draft → active → completed`. Activate a plan via `PUT /api/hcm/pro/development-plans/{id}/activate` with a `reviewed_by` field. Track progress (0–100%) via the `/progress` endpoint.

## Skill Gap Analysis

1. Add skills to the catalogue via `POST /api/hcm/pro/skills`.
2. Assess an employee's current vs. target level via `POST /api/hcm/pro/skill-assessments`.
3. Retrieve a structured gap report via `GET /api/hcm/pro/skill-gap-report/{employee_id}`.

Proficiency levels (ascending): `beginner → intermediate → advanced → expert`.

## Certifications

Add certifications via `POST /api/hcm/pro/certifications`. The API auto-computes `days_to_expiry`. Filter certifications expiring within N days using `?expiring_within_days=90`.

## Career Paths

Define milestones as a list of objects with `title` and optional `completed` flags. Mark milestones complete via `PUT /api/hcm/pro/career-paths/{id}` with updated milestones array. When all milestones are completed the path status moves to `achieved`.

## API Quick Reference

```
GET  /api/hcm/pro/health
POST /api/hcm/pro/development-plans
POST /api/hcm/pro/skill-assessments
GET  /api/hcm/pro/skill-gap-report/{employee_id}
POST /api/hcm/pro/mentoring-programmes
POST /api/hcm/pro/certifications
POST /api/hcm/pro/career-paths
GET  /api/hcm/pro/report/{employee_id}
GET  /api/hcm/pro/dashboard
```
