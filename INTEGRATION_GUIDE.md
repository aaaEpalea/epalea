# Epalea v1.0.0 — Complete Integration Guide

---

## Part 1: What Was Fixed and Why

### The evaluate bug (now fixed everywhere)

The `evaluate` command and API route were collecting per-entity predictions but discarding uncertainty information — it only showed aggregate metrics. Now:

- **CLI terminal output** has two new columns: `Ep.Unc` (mean epistemic) and `Al.Unc` (mean aleatoric) per mode
- **`evaluation.json`** now has a `per_entity` array with each entity's full prediction and per-mode uncertainty
- **API `POST /evaluate`** `per_entity` rows now include `uncertainty.spn` and `uncertainty.aggregator`
- **Notebook 03** metrics table cell now shows both uncertainty columns

### The infer bug (also fixed)

In `mode="both"`, uncertainty previously ran a single query defaulting to aggregator weights, discarding SPN uncertainty entirely. Now both are computed independently and returned as `uncertainty.spn` and `uncertainty.aggregator`. This affects `_inference.py`, `_model.py`, `cli.py`, and all docs that referenced the old flat `uncertainty.epistemic` shape.

### Summary of changed files

| File | Change |
|---|---|
| `epalea/_inference.py` | Per-mode uncertainty in all three modes |
| `epalea/_model.py` | Same fix in `infer()` + `_flatten_result()` |
| `epalea/cli.py` | `_run_single_infer` + `evaluate` command |
| `notebooks/03_inference.ipynb` | Cells 5 and 11 |
| `docs/concepts/uncertainty.md` | Updated to per-mode format |
| `docs/getting-started/quickstart.md` | Updated examples |
| `docs/api/endpoints/evaluate.md` | Updated response schema |
| `docs/cli/evaluate.md` | Updated output table + JSON schema |
| `docs/intro/index.md` | NEW |
| `docs/research/index.md` | NEW |
| `docs/contributing/overview.md` | Updated |

---

## Part 2: Files to Copy — epalea Package

Drop these files into your `epalea/` repo, replacing the originals:

```
epalea/
├── epalea/
│   ├── _inference.py          ← REPLACE
│   ├── _model.py              ← REPLACE
│   └── cli.py                 ← REPLACE
├── notebooks/
│   └── 03_inference.ipynb     ← REPLACE
└── docs/
    ├── intro/
    │   └── index.md           ← NEW
    ├── research/
    │   └── index.md           ← NEW
    ├── contributing/
    │   └── overview.md        ← REPLACE
    ├── concepts/
    │   └── uncertainty.md     ← REPLACE
    ├── getting-started/
    │   └── quickstart.md      ← REPLACE
    ├── api/endpoints/
    │   └── evaluate.md        ← REPLACE
    └── cli/
        └── evaluate.md        ← REPLACE
```

---

## Part 3: Files to Copy — Web App Backend

```
epalea-web/backend/
├── api/
│   ├── main.py                    ← REPLACE (see Part 4 for instructions)
│   ├── middleware/
│   │   ├── __init__.py            ← NEW (empty file)
│   │   ├── auth.py                ← NEW
│   │   └── rate_limit.py          ← NEW
│   └── routers/
│       └── v1/
│           ├── __init__.py        ← NEW (empty file)
│           ├── models.py          ← NEW
│           ├── infer.py           ← NEW
│           ├── batch_infer.py     ← NEW
│           ├── evaluate.py        ← NEW
│           ├── data.py            ← NEW
│           ├── train.py           ← NEW (stub)
│           └── account_keys.py   ← NEW (stub)
└── pretrained/
    └── compliance-v1/
        ├── schema.json
        ├── best_model.pt          ← copy from checkpoints/
        ├── aggregator.pt          ← copy from checkpoints/ (if available)
        └── evidence_index/        ← copy from existing index (if available)
```

To populate `pretrained/`:

```bash
cd epalea-web/backend
mkdir -p pretrained/compliance-v1

cp checkpoints/compliance/best_model.pt  pretrained/compliance-v1/
cp checkpoints/compliance/aggregator.pt  pretrained/compliance-v1/   # if available

cat > pretrained/compliance-v1/schema.json << 'EOF'
{
  "model_id": "compliance-v1",
  "version": "1.0.0",
  "domain": "compliance",
  "predicate": "compliance_level",
  "domain_values": ["low", "medium", "high"],
  "description": "EU AI Act compliance classification model.",
  "released": "2025-01-01",
  "embedding_dim": 384,
  "latent_dim": 64
}
EOF
```

---

## Part 4: Updating `main.py`

Your existing `main.py` mounts demos, Gradio, health checks, and other routes. **Do not replace it wholesale** — instead, add the v1 router to it. Open your existing `main.py` and add these lines:

```python
# At the top, add this import:
from api.routers.v1 import router as v1_router

# Then, after you create your FastAPI `app`, add:
app.include_router(v1_router, prefix="/api/v1")
```

That's it. Your existing routes stay exactly as they are.

### Environment variables

```bash
# .env (backend)
EPALEA_API_KEYS=ek_live_xxx,ek_live_yyy    # comma-separated valid keys
EPALEA_ENV=development                      # use "production" for live
```

In development mode (`EPALEA_ENV=development`):
- Swagger UI is available at `/api/docs`
- CORS allows all origins
- The key `ek_dev_test_key` is automatically valid (no need to set `EPALEA_API_KEYS`)

---

## Part 5: Frontend — Next.js Setup

### API proxy

Open `epalea-web/frontend/next.config.ts` and add:

```typescript
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: 'http://localhost:8000/api/v1/:path*',
      },
    ]
  },
}

export default nextConfig
```

This means the frontend calls `/api/v1/infer` and Next.js silently forwards it to your FastAPI backend — no CORS issues.

---

## Part 6: Setting Up the Docs in the Web App

Your docs currently live as plain markdown files in `epalea/docs/`. Your web app serves them at `epalea.ai/docs`. Here is the simplest concrete way to connect them — **no MDX, no Docusaurus, no framework magic required**.

### The approach: one shared `docs/` folder + a simple Next.js page

You need two things:

**1. A symlink (or copy) of the docs folder into the frontend**

```bash
# From the frontend directory
cd epalea-web/frontend

# Option A: Symlink (recommended — edits to epalea/docs/ automatically appear in the web app)
ln -s ../../../epalea/docs public/docs-content

# Option B: Copy (if you can't use symlinks)
cp -r ../../../epalea/docs public/docs-content
```

**2. A Next.js API route that reads and returns the markdown**

Create this file at `epalea-web/frontend/src/app/api/docs/[...slug]/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: { slug: string[] } }
) {
  // slug is e.g. ["intro", "index"] for /docs/intro/index
  const filePath = path.join(process.cwd(), 'public', 'docs-content', ...params.slug) + '.md'

  if (!fs.existsSync(filePath)) {
    return NextResponse.json({ error: 'Not found' }, { status: 404 })
  }

  const content = fs.readFileSync(filePath, 'utf-8')
  return NextResponse.json({ content })
}
```

**3. A single docs page component that fetches and renders markdown**

Install the markdown renderer once:

```bash
cd epalea-web/frontend
npm install react-markdown remark-gfm
```

Create `epalea-web/frontend/src/components/DocsPage.tsx`:

```tsx
'use client'
import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function DocsPage({ slug }: { slug: string }) {
  const [content, setContent] = useState('')

  useEffect(() => {
    fetch(`/api/docs/${slug}`)
      .then(r => r.json())
      .then(data => setContent(data.content || ''))
  }, [slug])

  return (
    <div className="prose prose-invert max-w-4xl mx-auto px-6 py-12">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {content}
      </ReactMarkdown>
    </div>
  )
}
```

**4. Create each docs page as a one-liner**

`epalea-web/frontend/src/app/docs/intro/page.tsx`:
```tsx
import DocsPage from '@/components/DocsPage'
export default function IntroPage() {
  return <DocsPage slug="intro/index" />
}
```

`epalea-web/frontend/src/app/docs/research/page.tsx`:
```tsx
import DocsPage from '@/components/DocsPage'
export default function ResearchPage() {
  return <DocsPage slug="research/index" />
}
```

`epalea-web/frontend/src/app/docs/contributing/page.tsx`:
```tsx
import DocsPage from '@/components/DocsPage'
export default function ContributingPage() {
  return <DocsPage slug="contributing/overview" />
}
```

All existing docs pages follow the same pattern. For example your existing quickstart page becomes:
```tsx
import DocsPage from '@/components/DocsPage'
export default function QuickstartPage() {
  return <DocsPage slug="getting-started/quickstart" />
}
```

**That's it.** Every time you update a `.md` file in `epalea/docs/`, the web app automatically serves the updated content — no rebuilds needed if you used the symlink approach.

### Adding syntax highlighting (optional but recommended)

```bash
npm install rehype-highlight highlight.js
```

Update `DocsPage.tsx`:

```tsx
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'

// In the return:
<ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
  {content}
</ReactMarkdown>
```

---

## Part 7: Running Everything

### Prerequisites

```bash
# Backend
cd epalea-web/backend
pip install -r requirements.txt

# Frontend
cd epalea-web/frontend
npm install
```

### A: CLI

```bash
cd epalea
pip install -e .

epalea --version     # 1.0.0
epalea models list

# Single inference — check per-mode uncertainty in output
epalea infer \
  --checkpoint pretrained/compliance-v1/best_model.pt \
  --aggregator-checkpoint pretrained/compliance-v1/aggregator.pt \
  --schema pretrained/compliance-v1/schema.json \
  --index-dir pretrained/compliance-v1/evidence_index \
  --entity-id C0001 \
  --mode both

# You should see:
# "uncertainty": {
#   "spn":        { "epistemic": 0.12, "aleatoric": 0.08, ... },
#   "aggregator": { "epistemic": 0.09, "aleatoric": 0.11, ... }
# }

# Evaluate — check per-mode columns in terminal and per_entity in evaluation.json
epalea evaluate \
  --checkpoint pretrained/compliance-v1/best_model.pt \
  --index-dir pretrained/compliance-v1/evidence_index \
  --test-companies pretrained/compliance-v1/sample_data/test_companies.json \
  --mode both

# Terminal should show:
# System             Acc   Macro F1  ...  Ep.Unc   Al.Unc
# LPF-SPN          0.947  ...             0.120    0.080
# LPF-Agg          0.951  ...             0.090    0.110
```

### B: Notebooks

```bash
cd epalea
pip install -e ".[notebooks]"
jupyter lab

# Run notebooks/03_inference.ipynb top-to-bottom.
# In the inference cell, you should see:
#   uncertainty.spn: epistemic=X  aleatoric=Y  total=Z
#   uncertainty.aggregator: epistemic=X  aleatoric=Y  total=Z
#
# In the metrics table cell, you should see Ep.Unc and Al.Unc columns.
```

### C: Web App

```bash
# Terminal 1
cd epalea-web/backend
EPALEA_ENV=development uvicorn api.main:app --reload --port 8000

# Terminal 2
cd epalea-web/frontend
npm run dev
```

Open:
- `http://localhost:3000` — homepage
- `http://localhost:3000/docs/intro` — new intro page
- `http://localhost:3000/docs/research` — new research page
- `http://localhost:3000/docs/contributing` — updated contributing guide
- `http://localhost:3000/docs/api/live-testing` — live API explorer
- `http://localhost:8000/api/redoc` — API reference

### D: API

```bash
KEY="ek_dev_test_key"
BASE="http://localhost:8000/api/v1"

# Health
curl $BASE/health

# List models
curl -H "X-API-Key: $KEY" $BASE/models

# Infer — verify BOTH uncertainty keys present
curl -X POST $BASE/infer \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{
    "model_id": "compliance-v1",
    "entity_id": "C0001",
    "evidence": [{"evidence_id":"E001","text_content":"Passed all audits.","credibility":0.95,"evidence_type":"audit_report"}],
    "options": {"mode": "both", "output_format": "nested"}
  }'
# ✓ Check: response.uncertainty.spn is not null
# ✓ Check: response.uncertainty.aggregator is not null

# Evaluate — verify per_entity rows have uncertainty
curl -X POST $BASE/evaluate \
  -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{
    "model_id": "compliance-v1",
    "test_data": [
      {"entity_id":"C001","true_label":"high","evidence":[{"evidence_id":"E1","text_content":"Excellent compliance.","credibility":0.9,"evidence_type":"audit_report"}]},
      {"entity_id":"C002","true_label":"low","evidence":[{"evidence_id":"E2","text_content":"Multiple violations.","credibility":0.9,"evidence_type":"regulatory_filing"}]}
    ],
    "options": {"mode": "both"}
  }'
# ✓ Check: response.per_entity[0].uncertainty.spn is not null
# ✓ Check: response.per_entity[0].uncertainty.aggregator is not null

# Auth checks
curl $BASE/models                              # should return 401
curl -H "X-API-Key: bad_key" $BASE/models     # should return 401
```

---

## Part 8: Pre-Release Checklist

### epalea package
- [ ] `epalea infer --mode both` output has `uncertainty.spn` AND `uncertainty.aggregator`
- [ ] `epalea evaluate --mode both` terminal shows `Ep.Unc` and `Al.Unc` columns
- [ ] `evaluation.json` has `per_entity` array with `uncertainty` per entity
- [ ] Notebook 03 cell 5 prints both uncertainty decompositions
- [ ] Notebook 03 metrics table shows uncertainty columns

### Web app backend
- [ ] `EPALEA_ENV=production`, `EPALEA_API_KEYS` set to real keys
- [ ] `pretrained/` populated for all domains
- [ ] `/health` returns `{"status": "ok"}`
- [ ] `/api/v1/infer` returns per-mode uncertainty
- [ ] `/api/v1/evaluate` per_entity rows have uncertainty

### Web app frontend
- [ ] `/docs/intro` renders
- [ ] `/docs/research` renders all 10 papers
- [ ] `/docs/contributing` renders updated guide
- [ ] `/docs/api/live-testing` live explorer works
- [ ] Code blocks have syntax highlighting

---

## Part 9: Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `uncertainty.spn` is null in both-mode | Still running old `_inference.py` | Replace with patched file |
| `uncertainty.aggregator` is null in both-mode | No `aggregator.pt` checkpoint | Train aggregator first, or use `mode: "spn"` |
| `per_entity` missing from evaluate response | Old API `evaluate.py` | Replace with new file |
| Docs page shows blank | `public/docs-content` symlink broken | Check symlink target with `ls -la public/` |
| Markdown not rendering | `react-markdown` not installed | `npm install react-markdown remark-gfm` |
| Code blocks not highlighted | `rehype-highlight` not installed | `npm install rehype-highlight highlight.js` |
| `401 invalid_api_key` | `EPALEA_API_KEYS` not set | Set env var or use `EPALEA_ENV=development` |
| `503 service_unavailable` | Weights missing in `pretrained/` | Copy `best_model.pt` to `pretrained/{model_id}/` |
| `404 model_not_found` | `schema.json` missing | Create `pretrained/{model_id}/schema.json` |
| CORS error in browser | Origin not in allow list | Add your domain to `allowed_origins` in `main.py` |
