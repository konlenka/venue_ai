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
{"status": "ok", "model": "claude-sonnet-4-6"}
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
  "confidence": 97,
  "reasoning": [
    "Google Maps category label: 'Cocktail bar' — unambiguous primary classification.",
    "Website menu is drinks-only with a curated seasonal cocktail list; food limited to bar snacks.",
    "Every review focuses on cocktail craft and spirits selection, not dining."
  ],
  "venue_profile": {
    "address": "150 Gertrude St, Fitzroy VIC 3065",
    "phone": "+61 3 9416 2229",
    "website": "theeverleigh.com",
    "hours": "Tue–Sun 5pm–1am",
    "google_maps_url": "https://www.google.com/maps/search/?api=1&query=The+Everleigh+Fitzroy+Australia"
  },
  "sales_playbook": [
    "Lead with Paramount's premium spirits range — The Everleigh's identity is built on craft cocktails and allocated bottles.",
    "Pitch exclusive or allocated spirit partnerships; their seasonal cocktail menu rotates and needs consistent high-end supply.",
    "Highlight Paramount's competitive pricing on high-velocity spirits (gin, whisky, vodka) to protect their margin on signature serves."
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
2. **Query Generation** — Claude generates 4 targeted Tavily search queries (menu, Google profile, review sites, atmosphere)
3. **Live Research** — 4 concurrent Tavily searches with raw page content enabled (actual menu text, Google Maps category labels, venue descriptions — not just snippets)
4. **Classification** — Claude reads all research evidence and must find 2+ verifiable facts before classifying above 75%; explicit evidence priority: Google Maps category → website → reviews
5. **Judge** — Claude reviews its own classification; if confidence < 75 or weak reasoning → retry once with fresh queries
6. **Output** — Clean JSON response with venue profile and Paramount Liquor sales playbook

---

## Key Edge Cases

- **Bar vs Restaurant**: Ask "why do most customers come here?" — drinks first = Bar, food first = Restaurant
- **Pub vs Club**: Pub = single community venue. Club = RSL/members/large complex.
- **Pub vs Accommodation**: Historic pub hotel with famous bar → Pub (bar identity dominates)
- **Multi-purpose venues** (e.g. Crown Casino): Use dominant commercial identity → Club (gaming/entertainment)
- **No online presence**: Fallback gracefully with low-confidence result, no crash

---

## Stress Test Expected Results

Results based on evidence-first classification using raw page content, Google Maps category labels, and actual menu data.

| Venue | Suburb | Segment | Confidence | Key Evidence |
|---|---|---|---|---|
| The Everleigh | Fitzroy | **Bar** | ~97% | Google Maps: "Cocktail bar". Website: drinks-only menu, bar snacks. Every review is drinks-focused. |
| The Continental | Sorrento | **Pub** | ~85% | Heritage-listed pub hotel. Famous front bar. TAB and bistro. Rooms secondary to bar identity. |
| Garden State Hotel | Melbourne CBD | **Bar** | ~82% | Google Maps: "Bar" (Drink and Dine group). No TAB/pokies. Multi-level cocktail bars and beer garden. Upscale drinks-first venue, not a traditional pub. |
| Crown Casino | Southbank | **Club** | ~90% | Google Maps: "Casino". Massive gaming floor + Crown Towers hotel + multiple restaurants and bars. Entertainment complex identity dominates. |
| MoVida | Melbourne CBD | **Restaurant** | ~97% | Google Maps: "Spanish restaurant". Full tapas menu, food-first. Nationally acclaimed for cuisine. Drinks support the dining experience. |

> **Note on Garden State Hotel:** This flips from the intuitive "Pub" to **Bar** under strict evidence rules.
> The name contains "Hotel" but the venue is operated by Drink and Dine (a bar group), has no TAB or gaming machines,
> and is explicitly listed as a "Bar" on Google Maps. Evidence overrides intuition.

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
