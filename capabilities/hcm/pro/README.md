# Professional Development (hcm_pro)

Capability for employee professional development: development plans, skill gap analysis, mentoring programmes, certification tracking, and career pathing.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/hcm/pro/health | Health check |
| GET | /api/hcm/pro/describe | Capability contract |
| GET | /api/hcm/pro/development-plans | List development plans |
| POST | /api/hcm/pro/development-plans | Create development plan |
| PUT | /api/hcm/pro/development-plans/{id} | Update plan |
| PUT | /api/hcm/pro/development-plans/{id}/activate | Activate plan |
| PUT | /api/hcm/pro/development-plans/{id}/progress | Update progress |
| DELETE | /api/hcm/pro/development-plans/{id} | Delete plan |
| GET | /api/hcm/pro/skills | List skills catalogue |
| POST | /api/hcm/pro/skills | Add skill |
| DELETE | /api/hcm/pro/skills/{id} | Remove skill |
| GET | /api/hcm/pro/skill-assessments | List assessments |
| POST | /api/hcm/pro/skill-assessments | Assess skill |
| GET | /api/hcm/pro/skill-gap-report/{employee_id} | Skill gap report |
| GET | /api/hcm/pro/mentoring-programmes | List mentoring |
| POST | /api/hcm/pro/mentoring-programmes | Create programme |
| POST | /api/hcm/pro/mentoring-programmes/{id}/sessions | Log session |
| GET | /api/hcm/pro/certifications | List certifications |
| POST | /api/hcm/pro/certifications | Add certification |
| PUT | /api/hcm/pro/certifications/{id} | Update certification |
| DELETE | /api/hcm/pro/certifications/{id} | Delete certification |
| GET | /api/hcm/pro/career-paths | List career paths |
| POST | /api/hcm/pro/career-paths | Create career path |
| PUT | /api/hcm/pro/career-paths/{id} | Update career path |
| DELETE | /api/hcm/pro/career-paths/{id} | Delete career path |
| GET | /api/hcm/pro/report/{employee_id} | Full dev report |
| GET | /api/hcm/pro/dashboard | Dashboard |
| GET | /api/hcm/pro/audit-events | Audit trail |
