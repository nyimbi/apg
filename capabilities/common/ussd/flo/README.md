# ussd_flo — USSD Flow Designer

Visual USSD menu flow builder with conditional routing, multi-language support, and A/B test flows.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/ussd/flo/health` | Service health |
| GET | `/api/ussd/flo/flows` | List flows |
| POST | `/api/ussd/flo/flows` | Create flow |
| GET | `/api/ussd/flo/flows/<id>` | Get flow |
| PUT | `/api/ussd/flo/flows/<id>` | Update flow |
| DELETE | `/api/ussd/flo/flows/<id>` | Delete flow |
| POST | `/api/ussd/flo/flows/<id>/activate` | Activate flow |
| POST | `/api/ussd/flo/flows/<id>/archive` | Archive flow |
| GET | `/api/ussd/flo/flows/<id>/validate` | Validate flow graph |
| POST | `/api/ussd/flo/flows/<id>/export` | Export flow |
| POST | `/api/ussd/flo/flows/import` | Import flow |
| GET | `/api/ussd/flo/flows/<id>/nodes` | List nodes |
| POST | `/api/ussd/flo/flows/<id>/nodes` | Add node |
| GET | `/api/ussd/flo/flows/<id>/nodes/<nid>` | Get node |
| PUT | `/api/ussd/flo/flows/<id>/nodes/<nid>` | Update node |
| DELETE | `/api/ussd/flo/flows/<id>/nodes/<nid>` | Delete node |
| GET | `/api/ussd/flo/flows/<id>/edges` | List edges |
| POST | `/api/ussd/flo/flows/<id>/edges` | Add edge |
| DELETE | `/api/ussd/flo/flows/<id>/edges/<eid>` | Delete edge |
| POST | `/api/ussd/flo/flows/<id>/route` | Resolve next node |
| GET | `/api/ussd/flo/flows/<id>/translations` | List translations |
| POST | `/api/ussd/flo/flows/<id>/translations` | Add translation |
| GET | `/api/ussd/flo/flows/<id>/translations/<lang>` | Get translation |
| DELETE | `/api/ussd/flo/flows/<id>/translations/<lang>` | Delete translation |
| GET | `/api/ussd/flo/flows/<id>/versions` | List versions |
| POST | `/api/ussd/flo/flows/<id>/versions` | Snapshot flow |
| POST | `/api/ussd/flo/flows/<id>/versions/<vid>/restore` | Restore version |
| GET | `/api/ussd/flo/abtests` | List A/B tests |
| POST | `/api/ussd/flo/abtests` | Create A/B test |
| GET | `/api/ussd/flo/abtests/<id>` | Get A/B test |
| PUT | `/api/ussd/flo/abtests/<id>` | Update A/B test |
| DELETE | `/api/ussd/flo/abtests/<id>` | Delete A/B test |
| GET | `/api/ussd/flo/abtests/<id>/results` | A/B test results |
| GET | `/api/ussd/flo/dashboard` | Summary dashboard |
| GET | `/api/ussd/flo/audit` | Audit event log |
