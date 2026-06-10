# GraphQL Federation Gateway (gql_gw)

Federated GraphQL gateway with auto-schema from semantic_model.json, DataLoader batching, persisted queries, and introspection.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/gql/health | Service health |
| POST | /api/gql/graphql | Execute GraphQL query |
| GET | /api/gql/graphql | Introspection schema |
| GET | /api/gql/subgraphs | List subgraphs |
| POST | /api/gql/subgraphs | Register subgraph |
| GET | /api/gql/subgraphs/{name} | Get subgraph |
| PUT | /api/gql/subgraphs/{name} | Update subgraph |
| DELETE | /api/gql/subgraphs/{name} | Remove subgraph |
| GET | /api/gql/subgraphs/{name}/health | Probe subgraph |
| GET | /api/gql/subgraphs/health/all | Probe all subgraphs |
| GET | /api/gql/schema | Composed federated schema |
| POST | /api/gql/schema/auto | Auto-generate SDL from semantic model |
| POST | /api/gql/schema/flush | Flush schema cache |
| POST | /api/gql/schema/{name}/diff | Detect breaking changes |
| GET | /api/gql/persisted | List persisted queries |
| POST | /api/gql/persisted | Register persisted query |
| POST | /api/gql/persisted/{id}/execute | Execute persisted query |
| DELETE | /api/gql/persisted/{id} | Delete persisted query |
| POST | /api/gql/dataloader/batch | DataLoader batch load |
| GET | /api/gql/analytics | Query analytics |
| GET | /api/gql/statistics | Gateway statistics |
| GET | /api/gql/querylog | Query execution log |
| GET | /api/gql/audit | Audit trail |
