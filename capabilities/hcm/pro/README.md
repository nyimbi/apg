# Professional Development (hcm_pro)

Capability for employee professional development: development plans, skill gap analysis, mentoring programmes, certification tracking, career pathing, 360 feedback, learning activities, training providers, automated nudges, and the Professional Development Index (PDI).

## Core Concepts

| Entity | Purpose |
|---|---|
| Development Plan | Annual growth plan per employee with objectives, focus areas, and progress |
| Plan Template | Reusable plan scaffolding keyed by role or department |
| Skill | Catalogue entry for a named competency in one of 6 categories |
| Skill Assessment | Point-in-time gap measurement (current level vs. target level) |
| Skill Endorsement | Peer endorsement of an employee's proficiency at a given level |
| 360 Feedback Request | Multi-rater feedback collection (self, peer, manager, skip-level) |
| Mentoring Programme | Structured pairing between mentor and mentee with session logging |
| Certification | Professional credential with expiry and renewal tracking |
| Career Path | Role-to-role trajectory with milestones and timeline |
| Learning Activity | Course, workshop, conference, book, e-learning, or coaching event |
| Training Provider | External vendor catalogue for spend aggregation |
| Learning Budget | Annual budget envelope per employee with utilisation tracking |
| PDI Snapshot | Professional Development Index composite score (0–100) |

## API Endpoints

### Infrastructure
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/health | Health check |
| GET | /api/hcm/pro/describe | Capability contract |
| GET | /api/hcm/pro/audit-events | Audit trail |

### Development Plans
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/development-plans | List plans |
| POST | /api/hcm/pro/development-plans | Create plan |
| GET | /api/hcm/pro/development-plans/{id} | Get plan |
| PUT | /api/hcm/pro/development-plans/{id} | Update plan |
| PUT | /api/hcm/pro/development-plans/{id}/activate | Activate plan |
| PUT | /api/hcm/pro/development-plans/{id}/progress | Update progress |
| DELETE | /api/hcm/pro/development-plans/{id} | Delete draft plan |
| POST | /api/hcm/pro/development-plans/{id}/clone | Clone to new year |

### Plan Templates
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/plan-templates | List templates |
| POST | /api/hcm/pro/plan-templates | Create template |
| POST | /api/hcm/pro/plan-templates/{id}/apply | Apply template → new plan |

### Skills & Assessments
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/skills | List skills catalogue |
| POST | /api/hcm/pro/skills | Add skill |
| GET | /api/hcm/pro/skills/{id} | Get skill |
| PUT | /api/hcm/pro/skills/{id} | Update skill |
| DELETE | /api/hcm/pro/skills/{id} | Remove skill |
| GET | /api/hcm/pro/skill-assessments | List assessments |
| POST | /api/hcm/pro/skill-assessments | Assess skill |
| GET | /api/hcm/pro/skill-gap-report/{employee_id} | Individual gap report |
| POST | /api/hcm/pro/team-skill-gap-report | Team-wide gap report |

### Skill Endorsements
| Method | Path | Description |
|--------|------|-------------|
| POST | /api/hcm/pro/skill-endorsements | Endorse a skill |
| GET | /api/hcm/pro/skill-endorsements/{employee_id}/{skill_id} | Endorsement summary |

### 360 Feedback
| Method | Path | Description |
|--------|------|-------------|
| POST | /api/hcm/pro/feedback-requests | Create 360 feedback request |
| POST | /api/hcm/pro/feedback-requests/{id}/respond | Submit rater response |
| GET | /api/hcm/pro/feedback-requests/{id}/aggregate | Aggregated results |

### Mentoring
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/mentoring-programmes | List programmes |
| POST | /api/hcm/pro/mentoring-programmes | Create programme |
| GET | /api/hcm/pro/mentoring-programmes/{id} | Get programme |
| PUT | /api/hcm/pro/mentoring-programmes/{id} | Update programme |
| DELETE | /api/hcm/pro/mentoring-programmes/{id} | Delete programme |
| POST | /api/hcm/pro/mentoring-programmes/{id}/sessions | Log session |

### Certifications
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/certifications | List certifications |
| POST | /api/hcm/pro/certifications | Add certification |
| GET | /api/hcm/pro/certifications/{id} | Get certification |
| PUT | /api/hcm/pro/certifications/{id} | Update certification |
| DELETE | /api/hcm/pro/certifications/{id} | Delete certification |

### Career Paths
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/career-paths | List career paths |
| POST | /api/hcm/pro/career-paths | Create career path |
| GET | /api/hcm/pro/career-paths/{id} | Get career path |
| PUT | /api/hcm/pro/career-paths/{id} | Update career path |
| POST | /api/hcm/pro/career-paths/{id}/milestones/{idx}/complete | Complete milestone |
| DELETE | /api/hcm/pro/career-paths/{id} | Delete career path |

### Learning Activities & Budget
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/learning-activities | List activities |
| POST | /api/hcm/pro/learning-activities | Add activity |
| POST | /api/hcm/pro/learning-activities/{id}/complete | Complete activity |
| GET | /api/hcm/pro/training-providers | List providers |
| POST | /api/hcm/pro/training-providers | Add provider |
| GET | /api/hcm/pro/training-providers/{id}/stats | Provider spend stats |
| POST | /api/hcm/pro/learning-budgets | Set budget |
| GET | /api/hcm/pro/learning-budgets/{employee_id}/{year} | Budget utilisation |

### Analytics
| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/report/{employee_id} | Full dev report |
| GET | /api/hcm/pro/dashboard | Tenant dashboard |
| GET | /api/hcm/pro/nudges | Automated nudges list |
| GET | /api/hcm/pro/pdi/{employee_id} | Compute PDI |
| GET | /api/hcm/pro/pdi/{employee_id}/trend | PDI trend over time |

## Proficiency Levels

`beginner → intermediate → advanced → expert`

## Skill Categories

`technical | leadership | communication | analytical | domain | soft`

## Activity Types

`course | workshop | conference | book | elearning | coaching | webinar`

## Feedback Rater Types

`self | peer | manager | skip_level | report`
