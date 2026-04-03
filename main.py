"""
Paramount Venue Classifier
FastAPI service that classifies Australian hospitality venues into one of six segments
using live Tavily web research + Claude 3.5 Sonnet AI reasoning.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
CONFIDENCE_THRESHOLD = 75
VALID_SEGMENTS = {"Bar", "Restaurant", "Pub", "Club", "Accommodation", "Caterer"}
MAX_RESEARCH_CHARS = 8_000
MAX_SEARCH_RESULTS = 10

# In-memory cache: (name_lower, suburb_lower) → ClassifyResponse
# Persists for the lifetime of the Railway deployment.
_cache: Dict[Tuple[str, str], ClassifyResponse] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

QUERY_GENERATION_SYSTEM = """You are a research assistant specialising in Australian hospitality venues.

Your task: generate exactly 3 targeted web search queries that will surface the most useful
information for classifying a venue into one of these segments:
  Bar | Restaurant | Pub | Club | Accommodation | Caterer

The 3 queries must collectively cover:
1. The venue's primary offering — menu, food vs drinks focus, cocktails, gaming, accommodation
2. Google Maps / TripAdvisor / aggregator listing that shows an explicit venue category label
3. Press reviews, editorial descriptions, or customer reviews that reveal atmosphere and purpose

Return ONLY a valid JSON array of exactly 3 strings. No explanation. No markdown. Example:
["The Everleigh cocktail bar Fitzroy menu drinks", "The Everleigh Fitzroy Google Maps venue type", "The Everleigh Fitzroy review atmosphere experience"]
"""

QUERY_GENERATION_SYSTEM_RETRY = """You are a research assistant specialising in Australian hospitality venues.

Your task: generate exactly 3 targeted web search queries to definitively confirm the TYPE of venue.
Focus specifically on:
1. Finding explicit category labels: "bar", "restaurant", "pub", "RSL club", "hotel", "caterer"
2. Confirming the primary revenue driver: food service vs alcohol vs gaming vs accommodation
3. Finding any industry directory listings (Australian Hotels Association, Restaurant & Catering Australia, etc.)

Return ONLY a valid JSON array of exactly 3 strings. No explanation. No markdown. Example:
["The Everleigh Fitzroy type cocktail bar OR restaurant OR pub", "The Everleigh Fitzroy primary business category hospitality", "The Everleigh Fitzroy industry listing classification"]
"""

CLASSIFICATION_SYSTEM = """You are an expert classifier of Australian hospitality venues for a B2B sales team.
Your job: assign exactly ONE segment from this list to a venue based on research evidence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEGMENT DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BAR
  Primary revenue and public identity is alcohol — cocktails, wine, craft beer, spirits.
  Customers come primarily to drink. Food exists but is secondary: bar snacks, share plates,
  a small bar menu. The venue bills itself as a "bar", "cocktail bar", "wine bar", or "bottle bar".
  Examples: cocktail bars, wine bars, brewery taprooms, gin bars.
  → The Everleigh (Fitzroy): renowned cocktail bar. Drinks-first. Bar snacks only. = Bar ✓

RESTAURANT
  Primary experience is a food dining service. Customers come primarily to eat a meal.
  Has a full kitchen with a proper multi-course or substantial food menu. Alcohol (wine, beer,
  cocktails) supports the dining experience but is not the primary draw.
  Examples: tapas restaurants, fine dining, casual dining, bistros, cafes with full meals.
  → MoVida (Melbourne CBD): Spanish tapas, serious food menu, food-first identity. = Restaurant ✓

PUB
  Traditional Australian pub or licensed hotel. Core identity: beer on tap, casual community feel.
  Often includes TAB/wagering terminal, gaming machines (pokies), a casual bistro or dining section,
  and may have accommodation rooms above. Single-entity venue with a local or heritage feel.
  Trades under a "Hotel" or "Pub" name but is NOT primarily an accommodation business.
  Examples: corner pubs, heritage hotel pubs, beer garden pubs, gastropubs.
  → Garden State Hotel (Melbourne CBD): multi-level pub, beer garden, food, pokies. = Pub ✓
  → The Continental (Sorrento): historic pub hotel with famous bar and bistro. = Pub ✓

CLUB
  RSL clubs, sports clubs, social/community clubs, nightclubs, large gaming complexes, or
  massive multi-use entertainment venues. Distinguishing features: formal membership structure,
  large dedicated gaming rooms, multiple bars and restaurants under one roof, or explicitly
  named as a "Club" (RSL, Leagues, Bowling, Sports, Nightclub).
  Also includes large entertainment complexes (casino + hotel + restaurants + bars in one complex)
  where no single element dominates — the entertainment/gaming complex identity wins.
  Examples: RSL clubs, Leagues clubs, nightclubs, casino complexes.
  → Crown Casino (Southbank): massive gaming/entertainment complex, casino + hotels + bars
    + restaurants + entertainment. Entertainment complex identity dominates. = Club ✓

ACCOMMODATION
  Hotels, motels, serviced apartments, B&Bs, resorts where the PRIMARY commercial purpose is
  selling overnight rooms. Food and beverage (bar, restaurant) is incidental — it serves guests
  but is not the main reason the business exists.
  Distinguish from Pub: if the venue's bar/pub identity is MORE famous than its rooms → Pub.
  If the venue is primarily marketed as a place to stay → Accommodation.
  Examples: business hotels, boutique hotels, motels, serviced apartments, holiday resorts.

CATERER
  A business whose PRIMARY activity is catering for events at external or client-specified
  locations. They may have a commercial kitchen but do not operate a permanent public-facing
  venue where customers walk in. Often listed as "catering company" or "event caterer".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BAR vs RESTAURANT — ask yourself one question:
  "Why do MOST customers come here?"
  → To drink, with food as an option = BAR
  → To eat a meal, with drinks as an accompaniment = RESTAURANT
  A wine bar with a cheese board = Bar.
  A restaurant with a full cocktail list = Restaurant.
  If cocktails/drinks are the star of any review → likely Bar.
  If the kitchen and food menu are the star of any review → likely Restaurant.

PUB vs CLUB:
  → Single community/local venue, one "pub" entity = Pub
  → Formal membership, large gaming room, named RSL/Leagues/Sports Club = Club
  → Massive multi-venue complex (casino scale) = Club

PUB vs ACCOMMODATION:
  → The pub/bar identity is the dominant public reputation → Pub
  → The hotel's primary marketing is "book a room" → Accommodation
  Historic hotels that are famous for their public bar: Pub.

MULTI-PURPOSE VENUES — choose the dominant commercial identity:
  → Crown Casino: gaming + entertainment complex. The casino/club identity wins. = Club
  → Brewery with taproom + small food menu: drinks-first. = Bar
  → Resort hotel with a day spa and restaurant: if rooms are the primary product → Accommodation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE GUIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
90-100: Strong online presence, multiple unambiguous signals, perfect segment fit
75-89:  Good evidence, minor ambiguity clearly resolved
50-74:  Ambiguous or limited information, reasonable but uncertain call
0-49:   Very sparse data, best guess only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SALES PLAYBOOK RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
You are generating talking points for a Paramount Liquor sales rep. Paramount Liquor is one of
Australia's largest liquor wholesalers — they supply wine, spirits, beer, and RTDs to licensed
venues at competitive trade pricing with reliable delivery.

Generate exactly 3 concise, actionable bullet points the rep can use in their first conversation
with this venue. Each bullet should be a specific angle for selling Paramount's liquor wholesale
products to THIS venue based on its type and context from the research.

Segment angles for Paramount Liquor reps:
- Bar: Lead with premium spirits and cocktail-base spirits supply — reference any specific spirits
  or cocktail styles visible in research. Pitch Paramount's competitive pricing on high-velocity
  spirits (gin, whisky, vodka). Mention exclusive label or allocated bottle access.
- Restaurant: Pitch wine-by-the-glass programs and curated bottle lists. Emphasise Paramount's
  sommelier support and food-pairing range. Reference any cuisine style to suggest matching varietals.
- Pub: Lead with bulk beer and tap product supply — competitive kegs, craft beer range. Reference
  gaming/TAB revenue as context for why reliable stock = reliable income. Pitch spirits for the bar.
- Club: Volume pricing and account management for high-throughput venues. Pitch event stock packages
  and RTD/beer bulk deals. Reference any gaming or function revenue visible in research.
- Accommodation: Minibar restocking programs and in-room wine selections. Pitch function/event
  liquor packages for weddings and corporate events. Emphasise reliable delivery SLAs.
- Caterer: Flexible event-by-event ordering and short-lead-time delivery. Pitch mixed case options
  and wine/beer packages tailored to event types (weddings, corporate, private dining).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY this JSON. No explanation before or after. No markdown fences.
{
  "segment": "<exactly one of: Bar, Restaurant, Pub, Club, Accommodation, Caterer>",
  "confidence": <integer 0-100>,
  "reasoning": [
    "<bullet 1: most important evidence — cite specific facts about this venue>",
    "<bullet 2: supporting evidence>",
    "<bullet 3: address any competing signals and explain why they were discounted>"
  ],
  "venue_profile": {
    "address": "<full street address if found in research, else null>",
    "phone": "<phone number if found, else null>",
    "website": "<website URL if found, else null>",
    "hours": "<opening hours if found, else null>"
  },
  "sales_playbook": [
    "<tailored next-action bullet 1 — specific to this venue's context>",
    "<tailored next-action bullet 2>",
    "<tailored next-action bullet 3>"
  ]
}

Reasoning bullets MUST be:
- Specific to THIS venue (name actual facts: drink names, food type, specific reviewer quotes, gaming machines, room count)
- Concise (one sentence each)
- Written for a sales rep reading on their phone in the field
- 3 to 5 bullets maximum
- NOT generic (do NOT write "Based on available information, this appears to be a bar")
"""

JUDGE_SYSTEM = """You are a quality-control reviewer for a venue classification system serving a B2B sales team.

You will receive:
1. Research text gathered about a venue
2. A classification result (segment, confidence, reasoning)

Your job: decide whether this classification is reliable enough to return to a sales team.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PASS criteria — ALL must be true:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Confidence score is 75 or above
2. At least 2 reasoning bullets contain a specific, verifiable claim about this venue
   (e.g. "serves tapas", "has 200 gaming machines", "explicit cocktail bar on Google Maps")
3. The segment choice is clearly supported by the evidence in the research text
4. No obvious contradictions between the reasoning and the segment

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FAIL criteria — ANY of these triggers a FAIL:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Confidence below 75
2. Reasoning bullets are generic (could apply to ANY venue of that type)
3. The segment seems inconsistent with what the research actually says
4. Fewer than 2 bullets contain a specific, verifiable claim

Return ONLY this JSON. No explanation before or after. No markdown fences.
{
  "verdict": "PASS",
  "reason": "<one sentence explaining why this classification is or is not reliable>"
}

OR

{
  "verdict": "FAIL",
  "reason": "<one sentence explaining the specific weakness>"
}
"""

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ClassifyRequest(BaseModel):
    business_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The trading name of the venue",
        examples=["The Everleigh"],
    )
    suburb: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Suburb or locality in Australia",
        examples=["Fitzroy"],
    )


class VenueProfile(BaseModel):
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    hours: Optional[str] = None
    google_maps_url: Optional[str] = None


class ClassifyResponse(BaseModel):
    segment: str = Field(..., description="One of: Bar, Restaurant, Pub, Club, Accommodation, Caterer")
    confidence: int = Field(..., ge=0, le=100, description="Classification confidence 0-100")
    reasoning: List[str] = Field(..., min_length=1, description="Bullet-point reasoning")
    venue_profile: Optional[VenueProfile] = Field(None, description="Contact info and location details")
    sales_playbook: List[str] = Field(default_factory=list, description="Tailored next-action prompts for sales reps")


class HealthResponse(BaseModel):
    status: str
    model: str


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    if not tavily_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

    app.state.anthropic = anthropic.AsyncAnthropic(api_key=anthropic_key)
    app.state.tavily = AsyncTavilyClient(api_key=tavily_key)
    logger.info("Clients initialised — server ready.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Paramount Venue Classifier",
    description="Classifies Australian hospitality venues into Bar, Restaurant, Pub, Club, Accommodation, or Caterer.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins so Lovable (and any browser-based frontend) can call this API.
# Tighten origins to your Lovable app URL once you have it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    """Extract the first complete JSON object from a string."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"No valid JSON found in Claude response: {text[:300]}") from exc


async def generate_search_queries(
    client: anthropic.AsyncAnthropic,
    business_name: str,
    suburb: str,
    retry: bool = False,
) -> List[str]:
    """Ask Claude to generate 3 targeted Tavily search queries for this venue."""
    system_prompt = QUERY_GENERATION_SYSTEM_RETRY if retry else QUERY_GENERATION_SYSTEM
    user_message = f"Venue: {business_name}, {suburb}, Australia\n\nGenerate 3 search queries now."

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=256,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text.strip()

        # Find JSON array in response
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            queries = json.loads(raw[start : end + 1])
            if isinstance(queries, list) and len(queries) >= 1:
                return [str(q) for q in queries[:3]]
    except Exception as exc:
        logger.warning(f"Query generation failed, using fallback: {exc}")

    # Hardcoded fallback — always works
    suffix = "venue type category" if retry else "menu food drinks"
    return [
        f"{business_name} {suburb} {suffix}",
        f'"{business_name}" {suburb} Google Maps TripAdvisor bar restaurant pub',
        f"{business_name} {suburb} review atmosphere experience",
    ]


async def run_tavily_searches(tavily: AsyncTavilyClient, queries: List[str]) -> str:
    """Run all queries concurrently, merge and deduplicate results, return formatted string."""

    async def search_one(query: str):
        return await tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=False,
            include_raw_content=False,
        )

    results = await asyncio.gather(*[search_one(q) for q in queries], return_exceptions=True)

    # Collect all result items, skip failed searches
    all_items = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"Search {i+1} failed: {type(result).__name__}: {result}")
            continue
        items = result.get("results", []) if isinstance(result, dict) else []
        all_items.extend(items)

    if not all_items:
        raise HTTPException(
            status_code=503,
            detail="No venue research data could be retrieved. Check your Tavily API key or try again.",
        )

    # Deduplicate by URL, sort by score descending
    seen_urls: dict = {}
    for item in all_items:
        url = item.get("url", "")
        if url and url not in seen_urls:
            seen_urls[url] = item

    deduped = sorted(seen_urls.values(), key=lambda x: x.get("score", 0), reverse=True)
    top = deduped[:MAX_SEARCH_RESULTS]

    # Format into a text block for Claude
    lines = []
    for i, item in enumerate(top, 1):
        lines.append(f"[Result {i}]")
        lines.append(f"Title: {item.get('title', 'N/A')}")
        lines.append(f"URL: {item.get('url', 'N/A')}")
        lines.append(f"Content: {item.get('content', 'N/A')}")
        lines.append("---")

    research_text = "\n".join(lines)

    # Cap total size to keep Claude prompt manageable
    if len(research_text) > MAX_RESEARCH_CHARS:
        # Trim at a natural boundary
        cutoff = research_text.rfind("---", 0, MAX_RESEARCH_CHARS)
        research_text = research_text[:cutoff] if cutoff != -1 else research_text[:MAX_RESEARCH_CHARS]

    logger.info(f"Research: {len(top)} results, {len(research_text)} chars")
    return research_text


async def classify_venue(
    client: anthropic.AsyncAnthropic,
    business_name: str,
    suburb: str,
    research_text: str,
) -> dict:
    """Feed research into Claude and get segment + confidence + reasoning."""
    user_message = (
        f"Venue: {business_name}, {suburb}, Australia\n\n"
        f"Research findings:\n{research_text}\n\n"
        f"Classify this venue now. Return only the JSON."
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.1,
        system=CLASSIFICATION_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text
    classification = extract_json(raw)

    # Validate segment
    segment = classification.get("segment", "")
    if segment not in VALID_SEGMENTS:
        raise ValueError(
            f"Claude returned an unknown segment: '{segment}'. "
            f"Valid segments: {', '.join(sorted(VALID_SEGMENTS))}"
        )

    # Validate confidence
    confidence = classification.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Claude returned a non-numeric confidence value: {confidence!r}")
    classification["confidence"] = int(confidence)

    # Validate reasoning
    reasoning = classification.get("reasoning", [])
    if not isinstance(reasoning, list) or len(reasoning) == 0:
        raise ValueError("Claude returned empty or invalid reasoning list.")

    # Parse venue_profile (graceful fallback to None if missing or malformed)
    raw_profile = classification.get("venue_profile")
    classification["venue_profile"] = raw_profile if isinstance(raw_profile, dict) else None

    # Parse sales_playbook (graceful fallback to empty list)
    playbook = classification.get("sales_playbook", [])
    classification["sales_playbook"] = playbook if isinstance(playbook, list) else []

    return classification


async def judge_classification(
    client: anthropic.AsyncAnthropic,
    research_text: str,
    classification: dict,
) -> bool:
    """
    Claude reviews its own classification.
    Returns True (PASS) or False (FAIL).
    Fail-open: parse errors return True to avoid spurious retries.
    """
    user_message = (
        f"Research used:\n{research_text}\n\n"
        f"Classification produced:\n{json.dumps(classification, indent=2)}\n\n"
        f"Is this classification reliable? Return only the JSON verdict."
    )

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=128,
            temperature=0.0,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
        raw = response.content[0].text
        result = extract_json(raw)
        verdict = result.get("verdict", "PASS").upper()
        reason = result.get("reason", "")
        passed = verdict == "PASS"
        logger.info(f"Judge verdict: {verdict} — {reason}")
        return passed
    except Exception as exc:
        logger.warning(f"Judge parse failed (fail-open, returning PASS): {exc}")
        return True  # Fail-open: return original classification rather than crashing


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Liveness probe."""
    return HealthResponse(status="ok", model=MODEL)


@app.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify(request: ClassifyRequest, req: Request):
    """
    Classify an Australian hospitality venue into one of:
    Bar, Restaurant, Pub, Club, Accommodation, Caterer.

    Uses live web research + Claude AI reasoning with a self-scoring judge step.
    """
    anthropic_client: anthropic.AsyncAnthropic = req.app.state.anthropic
    tavily_client: AsyncTavilyClient = req.app.state.tavily

    business_name = request.business_name.strip()
    suburb = request.suburb.strip()
    cache_key = (business_name.lower(), suburb.lower())

    logger.info(f"Classifying: '{business_name}' in '{suburb}'")

    # ── Cache check ───────────────────────────────────────────────────────────
    if cache_key in _cache:
        logger.info(f"Cache HIT — returning stored result for '{business_name}' in '{suburb}'")
        return _cache[cache_key]

    try:
        # ── Step 2: Generate search queries ──────────────────────────────────
        queries = await generate_search_queries(anthropic_client, business_name, suburb)
        logger.info(f"Queries: {queries}")

        # ── Step 3: Live research ─────────────────────────────────────────────
        research_text = await run_tavily_searches(tavily_client, queries)

        # ── Step 4: Classify ─────────────────────────────────────────────────
        classification = await classify_venue(anthropic_client, business_name, suburb, research_text)
        logger.info(
            f"First pass: {classification['segment']} @ {classification['confidence']}% confidence"
        )

        # ── Step 5: Judge ─────────────────────────────────────────────────────
        passed = await judge_classification(anthropic_client, research_text, classification)

        if not passed:
            logger.info("Judge FAILED — retrying with fresh search queries")
            retry_queries = await generate_search_queries(
                anthropic_client, business_name, suburb, retry=True
            )
            research_text = await run_tavily_searches(tavily_client, retry_queries)
            classification = await classify_venue(
                anthropic_client, business_name, suburb, research_text
            )
            logger.info(
                f"Retry pass: {classification['segment']} @ {classification['confidence']}% confidence"
            )

        # ── Step 6: Build response — Google Maps URL generated server-side ────
        maps_query = quote_plus(f"{business_name} {suburb} Australia")
        google_maps_url = f"https://www.google.com/maps/search/?api=1&query={maps_query}"

        raw_profile = classification.get("venue_profile") or {}
        venue_profile = VenueProfile(
            address=raw_profile.get("address"),
            phone=raw_profile.get("phone"),
            website=raw_profile.get("website"),
            hours=raw_profile.get("hours"),
            google_maps_url=google_maps_url,  # Always reliable — built from venue name
        )

        response = ClassifyResponse(
            segment=classification["segment"],
            confidence=classification["confidence"],
            reasoning=classification["reasoning"],
            venue_profile=venue_profile,
            sales_playbook=classification.get("sales_playbook", []),
        )

        # ── Cache the result ──────────────────────────────────────────────────
        _cache[cache_key] = response
        logger.info(f"Cached result for '{business_name}' in '{suburb}' (cache size: {len(_cache)})")

        return response

    except HTTPException:
        raise  # Re-raise FastAPI exceptions unchanged

    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="Anthropic API key is invalid. Check ANTHROPIC_API_KEY in .env")

    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Anthropic rate limit reached — wait a moment and retry.")

    except anthropic.APIStatusError as exc:
        logger.error(f"Anthropic API error: {exc.status_code} {exc.message}")
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {exc.message}")

    except ValueError as exc:
        logger.error(f"Classification parsing error: {exc}")
        raise HTTPException(status_code=502, detail=str(exc))

    except Exception as exc:
        logger.exception(f"Unexpected error classifying '{business_name}' in '{suburb}'")
        raise HTTPException(status_code=500, detail=f"Internal error: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
