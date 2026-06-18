# Advanced Planning and Scheduling Specification

## Overview

`mfg_aps` is an APG platform capability that provides Advanced Planning and Scheduling functionality.

## Requirements

- Tenant-scoped operations with full data isolation
- Audit trail via `audl` capability
- Role-based access control via `auth` capability
- Event sourcing for all state mutations

## Functional Requirements

- Expose REST endpoints for CRUD operations
- Emit structured events on all state changes
- Support pagination, filtering, and sorting

## Non-Functional Requirements

- Response time < 200 ms at p95
- 99.9 % availability SLA
