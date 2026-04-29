# API Overview

The Epalea hosted API is available at `https://epalea.ai/api/v1`.

> **Note:** The API server code is in the private web-app repository.
> This documentation covers the public endpoints.

## Authentication

```bash
curl https://epalea.ai/api/v1/models \
  -H "X-API-Key: ek_live_xxxxxxxxxxxxxxxxxxxx"
```

## Rate Limits

| Tier | Req/min | Req/day |
|---|---|---|
| Free (beta) | 20 | 500 |
| Standard *(soon)* | 200 | 10,000 |

## Endpoint Status

| Endpoint | Status |
|---|---|
| `GET /api/v1/models` | ✅ Live |
| `GET /api/v1/models/{id}` | ✅ Live |
| `POST /api/v1/infer` | ✅ Live (SPN mode) |
| `POST /api/v1/infer/batch` | ✅ Live (SPN mode) |
| `POST /api/v1/evaluate` | ✅ Live (SPN mode) |
| `POST /api/v1/models/train` | 🔜 Coming Soon |
| `POST /api/v1/models/index` | 🔜 Coming Soon |
| `POST /api/v1/models/train-aggregator` | 🔜 Coming Soon |
| `POST /api/v1/models/train-full` | 🔜 Coming Soon |
