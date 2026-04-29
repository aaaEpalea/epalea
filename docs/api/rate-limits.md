# Rate Limits

| Tier | Req/min | Req/day |
|---|---|---|
| Free (beta) | 20 | 500 |
| Standard *(soon)* | 200 | 10,000 |
| Enterprise *(soon)* | Custom | Custom |

Rate limit headers are returned on every response:

```
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 17
X-RateLimit-Reset: 1735689600
```

When the rate limit is exceeded, the API returns `429 rate_limit_exceeded`.
Wait until `X-RateLimit-Reset` before retrying.
