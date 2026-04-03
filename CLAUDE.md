# Paramount Venue Classifier

FastAPI service that classifies Australian hospitality venues into one of six segments using live web research + Claude AI.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API keys to .env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...

# 3. Run the server
uvicorn main:app --reload

# 4. Server is live at http://localhost:8000
```

---

## Endpoints

### `GET /health`
Liveness probe. Returns model name and status.

```json
{"status": "ok", "model": "claude-3-5-sonnet-20241022"}
```

### `POST /classify`
Classify a venue into one of 6 segments.

**Request:**
```json
{
  "business_name": "The Everleigh",
  "suburb": "Fitzroy"
}
```

**Response:**
```json
{
  "segment": "Bar",
  "confidence": 95,
  "reasoning": [
    "The Everleigh is a celebrated cocktail bar with a drinks-first identity, known nationally for its curated cocktail menu.",
    "Food offering is limited to bar snacks — primary revenue driver is alcohol service.",
    "Multiple reviews and its own website describe it explicitly as a cocktail bar, not a restaurant."
  ]
}
```

---

## Segments

| Segment | Definition |
|---|---|
| **Bar** | Primary revenue/identity is drinks (cocktails, wine, beer). Food is secondary or snack-only. |
| **Restaurant** | Primary experience is food dining. Guests come to eat. Drinks support the meal. |
| **Pub** | Traditional Australian pub/hotel. Beer on tap, TAB/gaming, casual bistro. Community feel. |
| **Club** | RSL/sports/nightclub/large entertainment complex. Membership structures or gaming rooms. |
| **Accommodation** | Hotel/motel where overnight rooms are the primary commercial service. |
| **Caterer** | Business whose primary activity is catering for events at external locations. |

---

## 6-Step Workflow

1. **Input** — Business name + suburb received via POST /classify
2. **Query Generation** — Claude generates 3 targeted Tavily search queries
3. **Live Research** — 3 concurrent Tavily searches gather venue context (menu, reviews, Google profile)
4. **Classification** — Claude analyzes research → segment + confidence (0-100) + bullet reasoning
5. **Judge** — Claude reviews its own classification; if confidence < 75 or weak reasoning → retry once
6. **Output** — Clean JSON response

---

## Key Edge Cases

- **Bar vs Restaurant**: Ask "why do most customers come here?" — drinks first = Bar, food first = Restaurant
- **Pub vs Club**: Pub = single community venue. Club = RSL/members/large complex.
- **Pub vs Accommodation**: Historic pub hotel with famous bar → Pub (bar identity dominates)
- **Multi-purpose venues** (e.g. Crown Casino): Use dominant commercial identity → Club (gaming/entertainment)
- **No online presence**: Fallback gracefully with low-confidence result, no crash

---

## Stress Test Expected Results

| Venue | Suburb | Expected | Reason |
|---|---|---|---|
| The Everleigh | Fitzroy | Bar | Celebrated cocktail bar, drinks-first |
| The Continental | Sorrento | Pub | Historic pub/hotel, bar identity dominates |
| Garden State Hotel | Melbourne CBD | Pub | Multi-level pub with beer garden |
| Crown Casino | Southbank | Club | Gaming/entertainment complex |
| MoVida | Melbourne CBD | Restaurant | Spanish tapas, food-first dining |

---

## Environment Variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key from console.anthropic.com |
| `TAVILY_API_KEY` | Tavily search API key from app.tavily.com |

Both keys are required. The server will refuse to start if either is missing.

---

## Error Codes

| Code | Meaning |
|---|---|
| 401 | Invalid API key (Anthropic or Tavily) |
| 422 | Invalid request body (missing fields) |
| 429 | Rate limit hit — retry shortly |
| 503 | No search results retrieved for this venue |
| 502 | Claude returned unexpected output |
| 504 | Search timed out — retry |
| 500 | Unexpected internal error |
