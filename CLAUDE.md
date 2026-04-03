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

| Segment | Primary Focus | Food vs Drink | Atmosphere | Access |
|---|---|---|---|---|
| **Bar** | Drinks/alcohol — cocktails, wine, spirits, craft beer. Food is snacks only. | Heavy drink, minimal food | Relaxed/conversational, drink-social | Open walk-in |
| **Restaurant** | Food dining — full kitchen, multi-course or substantial meals. Drinks secondary. | Heavy food, drinks support meal | Dining-focused, table service | Open walk-in |
| **Pub** | Balanced drinks + substantial food. Beer on tap, community feel, often TAB/pokies. | Both — hearty meals + beer | Relaxed local, community social | Open walk-in |
| **Club** | Entertainment/dancing (nightclub) OR membership-based hospitality with gaming (RSL/Leagues). | Drinks secondary to entertainment | High-energy (nightclub) or member leisure | Cover charge or membership |
| **Accommodation** | Overnight rooms are the primary commercial product. Food/bar serves guests incidentally. | Incidental — serves guests | Rest/travel stay | Room booking |
| **Caterer** | Event-based food service at client/external locations. No fixed walk-in venue. | Food-focused, event/bulk | Event-specific, mobile | Event booking |

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

Results based on evidence-first classification using raw page content, Google Maps category labels, ANZSIC codes, state liquor licences, and actual menu data.

| Venue | Suburb | Segment | Confidence | ANZSIC | Liquor Licence | Key Evidence |
|---|---|---|---|---|---|---|
| The Everleigh | Fitzroy | **Bar** | ~97% | 4520 — Pubs, Taverns and Bars | VIC — Bar Licence | VIC Bar Licence is definitive: on-premise only, no gaming permitted. Google Maps: "Cocktail bar". Drinks-only menu, bar snacks. ANZSIC 4520 + Bar Licence = Bar (not Pub). No dance floor (not nightclub). |
| The Continental | Sorrento | **Pub** | ~90% | 4520 — Pubs, Taverns and Bars | VIC — General Licence | VIC General Licence = standard pub licence with gaming/TAB authority. ANZSIC 4520 + General Licence + TAB + bistro meals = Pub. Heritage-listed; rooms present but pub identity dominates. |
| Garden State Hotel | Melbourne CBD | **Bar** | ~90% | 4520 — Pubs, Taverns and Bars | VIC — On-Premises Licence (or Bar Licence) | Drink and Dine bar group; no pokies/TAB. Google Maps: "Bar". If On-Premises or Bar Licence confirmed = Bar. ANZSIC 4520 + absence of General Licence + no gaming = Bar (not Pub). |
| Crown Casino | Southbank | **Club** | ~95% | 9201 — Casino Operations | VIC — Casino Licence (special legislation) | ANZSIC 9201 (Casino Operations, Division R) + Casino Licence = unambiguous Club (entertainment complex). Massive gaming floor + Crown Towers hotel + multiple F&B venues. Not classifiable as Pub or Bar. |
| MoVida | Melbourne CBD | **Restaurant** | ~98% | 4511 — Cafes and Restaurants | VIC — Restaurant and Café Licence | VIC Restaurant and Café Licence is definitive: primary activity must be substantial meals; cannot operate as a bar at night. ANZSIC 4511 + R&C Licence + Google Maps "Spanish restaurant" = unambiguous Restaurant. |

> **Note on ANZSIC 4520 (Pubs, Taverns and Bars):** This code covers both Bar and Pub — ANZSIC does not distinguish
> between them. The liquor licence resolves the ambiguity: VIC **Bar Licence** = Bar; VIC **General Licence** = Pub.
> The Continental and Garden State Hotel both return 4520 but carry different licences and operational signals.

> **Note on Crown Casino:** ANZSIC **9201 Casino Operations** (Division R) and a special Casino Licence confirm
> the entertainment complex classification. Confidence reaches ~95% — no other segment is consistent with casino
> operations at this scale.

> **Note on Garden State Hotel:** Flips from intuitive "Pub" to **Bar** via licence + operational evidence. A General
> Licence would have allowed gaming machines; the absence of pokies/TAB and presence of an On-Premises or Bar Licence
> confirms it operates as a bar, not a pub. Drink and Dine bar group identity reinforces this.

> **Note on confidence increases:** Liquor licence data adds a government-sourced signal that removes ambiguity.
> The Continental rises to ~90% (General Licence confirms Pub), Garden State to ~90% (no General Licence + no gaming
> confirms Bar), Crown to ~95% (Casino Licence + 9201 = definitive), MoVida to ~98% (R&C Licence + 4511 = definitive).

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
