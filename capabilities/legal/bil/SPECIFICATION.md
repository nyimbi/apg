# leg_bil Specification

## Overview

`leg_bil` is an APG platform capability that provides leg_bil functionality.

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
