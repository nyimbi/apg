# leg_mat — Matter Management

Legal matter lifecycle, task management, team assignment, deadline tracking, court dockets.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/legal/mat/health | Health check |
| GET | /api/legal/mat/describe | Capability descriptor |
| GET | /api/legal/mat/matters | List matters |
| GET | /api/legal/mat/matters/{id} | Get matter |
| POST | /api/legal/mat/matters | Create matter |
| PUT | /api/legal/mat/matters/{id} | Update matter |
| DELETE | /api/legal/mat/matters/{id} | Archive matter |
| POST | /api/legal/mat/matters/{id}/close | Close matter |
| GET | /api/legal/mat/tasks | List tasks |
| GET | /api/legal/mat/tasks/{id} | Get task |
| POST | /api/legal/mat/tasks | Create task |
| PUT | /api/legal/mat/tasks/{id} | Update task |
| DELETE | /api/legal/mat/tasks/{id} | Cancel task |
| GET | /api/legal/mat/deadlines | List deadlines |
| GET | /api/legal/mat/deadlines/{id} | Get deadline |
| POST | /api/legal/mat/deadlines | Create deadline |
| PUT | /api/legal/mat/deadlines/{id} | Update deadline |
| DELETE | /api/legal/mat/deadlines/{id} | Remove deadline |
| GET | /api/legal/mat/docket | List docket entries |
| POST | /api/legal/mat/docket | Create docket entry |
| GET | /api/legal/mat/dashboard | Matter dashboard |
| GET | /api/legal/mat/audit | Audit events |

## Service Class

`MatterManagementService` — 42+ async methods covering matter CRUD, team assignment, tasks, deadlines, dockets, notes, time budgets, and analytics.
