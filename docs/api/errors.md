# Error Reference

| HTTP | Code | Cause | Fix |
|---|---|---|---|
| `400` | `invalid_request` | Malformed request body | Check schema |
| `401` | `invalid_api_key` | Missing or invalid key | Check `X-API-Key` header |
| `404` | `model_not_found` | Unknown model_id | Use `GET /api/v1/models` |
| `422` | `validation_error` | Field validation failed | See `errors[]` array |
| `429` | `rate_limit_exceeded` | Too many requests | Wait for `X-RateLimit-Reset` |
| `503` | `service_unavailable` | Temporary outage | Retry after 5s |

## Error response shape

```json
{
  "error": {
    "code": "validation_error",
    "message": "Field 'entity_id' is required.",
    "errors": [
      {"field": "entity_id", "message": "required field missing"}
    ]
  }
}
```
