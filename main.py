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
from typing import List, Optional
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
MAX_RESEARCH_CHARS = 15_000
MAX_SEARCH_RESULTS = 12
MAX_RAW_CONTENT_PER_RESULT = 800  # chars of raw page content to include per result

# In-memory cache: (name_lower, suburb_lower) → ClassifyResponse
# Persists for the lifetime of the Railway deployment.
_cache: dict = {}

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

Your task: generate exactly 5 targeted web search queries that will surface the most useful
information for classifying a venue into one of these segments:
  Bar | Restaurant | Pub | Club | Accommodation | Caterer

The 5 queries must collectively cover:
1. The venue's primary offering — menu, food vs drinks focus, cocktails, gaming, accommodation
2. Google Maps / TripAdvisor / aggregator listing that shows an explicit venue category label
3. Press reviews, editorial descriptions, or customer reviews that reveal atmosphere and purpose
4. Australian Business Register (ABR) or ASIC listing to surface the official ANZSIC industry code
   (ANZSIC: 4511=Cafes & Restaurants, 4513=Catering, 4520=Pubs/Taverns/Bars, 4530=Clubs, 4400=Accommodation, 9201=Casino)
5. State liquor licence type — this is a powerful classification signal:
   VIC: General Licence=Pub | Bar Licence=Bar | Restaurant & Café Licence=Restaurant | Club Licence=Club | Late Night=Bar/Club | Limited=Caterer
   NSW: Hotel Licence=Pub | Small Bar Licence=Bar | On-Premises=Restaurant/Accommodation | Club Licence=Club | Limited=Caterer
   QLD: Commercial Hotel=Pub | Restaurant Licence=Restaurant | Nightclub Licence=Club | Community Club=Club | Limited=Caterer

Return ONLY a valid JSON array of exactly 5 strings. No explanation. No markdown. Example:
["The Everleigh cocktail bar Fitzroy menu drinks", "The Everleigh Fitzroy Google Maps venue type", "The Everleigh Fitzroy review atmosphere experience", "The Everleigh Fitzroy ABN ABR ANZSIC industry classification", "The Everleigh Fitzroy liquor licence type Victorian Commission for Gambling and Liquor Regulation"]
"""

QUERY_GENERATION_SYSTEM_RETRY = """You are a research assistant specialising in Australian hospitality venues.

Your task: generate exactly 5 targeted web search queries to definitively confirm the TYPE of venue.
Focus specifically on:
1. Finding explicit category labels: "bar", "restaurant", "pub", "RSL club", "hotel", "caterer"
2. Confirming the primary revenue driver: food service vs alcohol vs gaming vs accommodation
3. Finding any industry directory listings (Australian Hotels Association, Restaurant & Catering Australia, etc.)
4. Australian Business Register / ASIC / ANZSIC industry code lookup for official classification
   (ANZSIC: 4511=Cafes & Restaurants, 4513=Catering, 4520=Pubs/Taverns/Bars, 4530=Clubs, 4400=Accommodation, 9201=Casino)
5. State liquor licence type — powerful classification signal:
   VIC: General Licence=Pub | Bar Licence=Bar | Restaurant & Café Licence=Restaurant | Club Licence=Club
   NSW: Hotel Licence=Pub | Small Bar Licence=Bar | On-Premises=Restaurant | Club Licence=Club
   QLD: Commercial Hotel=Pub | Restaurant Licence=Restaurant | Nightclub Licence=Club | Community Club=Club

Return ONLY a valid JSON array of exactly 5 strings. No explanation. No markdown. Example:
["The Everleigh Fitzroy type cocktail bar OR restaurant OR pub", "The Everleigh Fitzroy primary business category hospitality", "The Everleigh Fitzroy industry listing classification", "The Everleigh Fitzroy ABN ANZSIC 4520 OR 4511 OR 4530 industry code", "The Everleigh Fitzroy liquor licence VCGLR Victorian bar licence"]
"""

CLASSIFICATION_SYSTEM = """You are a senior sales representative for Paramount Liquor, one of Australia's largest
liquor wholesalers. Paramount supplies wine, spirits, beer, and RTDs to licensed hospitality venues
at competitive trade pricing with reliable delivery.

You are researching a prospective venue customer. Your job: assign exactly ONE segment from this list
based on research evidence, then generate a targeted sales approach for your first call with this venue.

Why accurate classification matters: the segment determines which Paramount products to pitch, what
volume to expect, and how to frame the value proposition. A mis-classified venue means the wrong
pitch — e.g. pitching minibar restocking to a cocktail bar, or craft spirits to a TAB pub.

Segment list: Bar | Restaurant | Pub | Club | Accommodation | Caterer

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT EVIDENCE RULES — READ BEFORE CLASSIFYING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER classify based on the venue name alone. Names are misleading.
   ("The Hotel" could be a pub, not a hotel. "Restaurant X" could be a bar.)
2. You MUST find at least 2 of the following before classifying above 75% confidence:
   - A menu (food or drinks) that reveals the primary offering
   - An explicit category label from Google Maps, TripAdvisor, or Zomato
   - A description from the venue's own website or a credible review
   - Opening hours pattern that reveals the business type (e.g. late-night = bar/nightclub)
3. Prioritise evidence in this order (most reliable first):
   a. Google Maps / TripAdvisor category label (e.g. "Cocktail bar", "Restaurant", "Pub")
   b. The venue's own website menu or "About" page content
   c. Food/drinks items visible in reviews or booking platforms
   d. Reviewer descriptions of what customers do there (eat vs drink vs stay vs dance)
4. If you cannot find clear evidence, set confidence below 60. Do NOT guess.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANZSIC 2006 CODES — OFFICIAL AUSTRALIAN CLASSIFICATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
These are the Australian and New Zealand Standard Industrial Classification (ANZSIC 2006)
codes used by the ABS, ABR, and ASIC to officially classify hospitality businesses.
If you find an ANZSIC code in the research (e.g. from ABR lookup, ASIC filing, or business
directory), treat it as strong evidence — weight it alongside Google Maps category labels.

  4400  Accommodation
        → Hotels, motels, serviced apartments, B&Bs providing short-term lodging.
          If a venue's ABR/ASIC registration shows 4400, classify as Accommodation.

  4511  Cafes and Restaurants
        → Units providing food service for on-premise consumption with waiter service.
          Maps to our RESTAURANT segment.

  4512  Takeaway Food Services
        → Counter service, fast food, delivery-first. NOT a primary segment here but
          helps exclude restaurant/bar misclassifications if found.

  4513  Catering Services
        → Event/off-site food and beverage preparation. Maps to our CATERER segment.

  4520  Pubs, Taverns and Bars
        → Alcoholic beverages served on-premise; may include food, gaming, entertainment.
          NOTE: This code covers BOTH our Bar AND Pub segments — ANZSIC does not
          distinguish between them. Use our Bar vs Pub decision rules to differentiate
          within 4520: TAB/pokies/bistro = Pub; curated drinks, no gaming = Bar.

  4530  Clubs (Hospitality)
        → Associations providing hospitality services to members: RSL, Leagues, sports
          clubs, nightclubs. Maps to our CLUB segment.

  9201  Casino Operations  [Division R — Arts and Recreation Services]
        → Businesses whose primary activity is operating a casino or gaming facility.
          Large entertainment complexes like Crown Casino register under 9201, NOT 4530.
          If found, treat as very strong evidence for CLUB (entertainment complex).

HOW TO USE ANZSIC IN EVIDENCE:
- If the research shows an ANZSIC code (e.g. "industry: 4530"), treat it as supporting
  evidence equivalent to a TripAdvisor category label.
- If ANZSIC code conflicts with other evidence (e.g. ABR shows 4520 but venue is clearly
  accommodation-focused), trust the stronger operational evidence and note the discrepancy.
- 4520 alone does NOT distinguish Bar from Pub — apply the Bar vs Pub decision rules.
- Include the identified ANZSIC code and description in your output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUSTRALIAN LIQUOR LICENSING — STATE-BY-STATE SIGNALS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Australian liquor licences are issued by state/territory regulators and are among the most
reliable classification signals available — the licence type is tied directly to the venue's
legal operating purpose. Weight a confirmed liquor licence type equally to a Google Maps
category label.

VICTORIA (regulated by VCGLR — Victorian Commission for Gambling and Liquor Regulation):
  General Licence
    → The standard pub/hotel licence. Permits on-premise AND takeaway alcohol. Gaming
      machines (pokies) and TAB wagering are commonly attached. = PUB
  Bar Licence (Small Bar)
    → On-premise alcohol only. No gaming machines permitted. Typically capped capacity.
      Explicitly a "bar" under Victorian law. = BAR
  Restaurant and Café Licence
    → Primary activity MUST be substantial meals. Alcohol must accompany food service.
      Cannot operate as a bar at night without a separate licence. = RESTAURANT
  Club Licence
    → Must be an incorporated association. RSL clubs, Leagues clubs, sports clubs. = CLUB
  On-Premises Licence
    → Alcohol is secondary to the main business. Subtypes include: accommodation, live
      music venue, entertainment venue. = ACCOMMODATION (if accommodation subtype) or BAR
      (if live music/entertainment subtype, especially with Late Night endorsement)
  Late Night (General) Licence / Late Night (On-Premises) Licence
    → Authorises trading after 1 am. Late Night General = large late-night pub/bar. Late
      Night On-Premises = bar or nightclub. = BAR (if conversational) or CLUB (if nightclub)
  Limited Licence
    → Event-specific or temporary authorisation. No fixed venue. = CATERER

NEW SOUTH WALES (regulated by Liquor & Gaming NSW):
  Hotel Licence
    → The broadest pub licence. On-premise and takeaway. Gaming machines permitted. = PUB
  Small Bar Licence
    → Maximum 120 patrons. On-premise only. No gaming machines permitted. = BAR
  On-Premises Licence
    → Alcohol supports a primary business (food, accommodation, entertainment). Subtypes:
      restaurant = RESTAURANT; accommodation = ACCOMMODATION; entertainment = BAR or CLUB
  Club Licence
    → Registered clubs only (RSL, Leagues, sports, community clubs). Members + guests. = CLUB
  Limited Licence
    → Temporary/event-based authority. = CATERER

QUEENSLAND (regulated by OLGR — Office of Liquor and Gaming Regulation):
  Commercial Hotel Licence
    → Hotels and taverns. On-premise and takeaway. Gaming permitted. = PUB
  Restaurant Licence
    → Meals must be the principal activity. Alcohol accompanies food. = RESTAURANT
  Nightclub Licence
    → Entertainment (live, on-site) must be the primary function. Late-night hours. = CLUB
  Community Club Licence
    → Non-proprietary clubs (RSL, sports, community). Members-based. = CLUB
  Commercial Special Facility Licence
    → Venues with special features: tourist accommodation, resorts, etc. = ACCOMMODATION
  Limited Licence
    → Events and temporary functions. = CATERER

HOW TO USE LIQUOR LICENCE IN EVIDENCE:
- A confirmed liquor licence type is one of the strongest classification signals available
  — it reflects how the venue is legally registered with a state regulator.
- VIC Bar Licence is definitive: if found, classify as BAR regardless of the venue's name.
- VIC General Licence narrows the field to PUB (or possibly Accommodation for a hotel that
  primarily rents rooms — apply the Pub vs Accommodation rule to resolve).
- NSW Small Bar Licence is definitive: classify as BAR.
- NSW/VIC/QLD Club Licence is definitive: classify as CLUB.
- QLD Nightclub Licence is definitive: classify as CLUB (nightclub sub-type).
- Restaurant/Café licences are definitive for RESTAURANT.
- Limited Licence strongly supports CATERER.
- Include the identified licence type and state in your output.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEGMENT DEFINITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BAR
  An establishment primarily licensed to serve alcoholic beverages (beer, wine, spirits,
  cocktails) for consumption on the premises, centred around a counter where drinks are
  prepared and served. Food, if offered at all, is limited to light snacks, appetisers, or
  small bar bites — NOT a full dining menu. The focus is on drinks, socialising, and a casual
  or intimate atmosphere. Bars are typically smaller and more drink-oriented than food-focused
  venues. The venue bills itself as a "bar", "cocktail bar", "wine bar", or "bottle bar".
  Primary focus: Drinks/alcohol (minimal food).
  Hours: Typically afternoon through late evening; may trade late.
  Customer experience: Social leisure outing centred on drinking.
  Examples: cocktail bars, wine bars, brewery taprooms, gin bars, whisky bars.
  → The Everleigh (Fitzroy): renowned cocktail bar. Drinks-first. Bar snacks only. = Bar ✓
  Paramount wholesale profile: Premium spirits (gin, whisky, vodka, rum, mezcal), allocated/
  exclusive bottles, craft beer, wine by the bottle, cocktail bitters and liqueurs. Moderate
  volume but high unit value. Seasonal menu rotations drive regular re-orders. Reverse signal:
  if a venue has a large tap beer setup or TAB, it is NOT a Bar — it's a Pub.

RESTAURANT
  A public establishment whose main business is preparing and serving meals/food for
  consumption on the premises. Alcohol (wine, beer, cocktails) may be served but is secondary
  — it supports the dining experience. Has a full kitchen with a proper multi-course or
  substantial food menu. Table seating and (in full-service cases) waitstaff are typical.
  Ranges from casual to fine dining. NOT primarily focused on alcohol or entertainment.
  Primary focus: Food/meals (drinks secondary).
  Hours: Lunch and/or dinner service; kitchen-driven hours.
  Customer experience: Meal-centric outing.
  Examples: tapas restaurants, fine dining, casual dining, bistros, cafes with full meals.
  → MoVida (Melbourne CBD): Spanish tapas, serious food menu, food-first identity. = Restaurant ✓
  Paramount wholesale profile: Wine dominates — by-the-glass pours and curated bottle list are
  the core sell. Beer (bottled/canned, not kegs), some spirits for cocktails/digestifs. Sommelier
  support and food-pairing suggestions are key sales hooks. Reverse signal: if a venue has no
  substantial food menu or bills itself primarily as a drinks destination, it is NOT a Restaurant.

PUB (Public House)
  A licensed establishment (with British/Irish or Australian roots) that serves alcoholic
  beverages — especially beer on tap — alongside substantial food options (pub classics like
  fish and chips, pies, burgers, or a bistro menu). Pubs emphasise a relaxed, social,
  community-oriented atmosphere for both drinking AND eating. In Australia, "pub" or "hotel"
  often overlaps and may include TAB/wagering terminals, gaming machines (pokies), live music,
  and accommodation rooms above. Single-entity venue with a local or heritage feel.
  Trades under a "Hotel" or "Pub" name but is NOT primarily an accommodation business —
  the bar/pub identity dominates over any rooms.
  Primary focus: Balanced drinks + substantial food in a casual social setting.
  Hours: Daytime through evening; bistro lunch and dinner.
  Customer experience: Relaxed local gathering; eat and drink.
  Examples: corner pubs, heritage hotel pubs, beer garden pubs, gastropubs, TAB pubs.
  → The Continental (Sorrento): historic pub hotel with famous bar and bistro. = Pub ✓
  Paramount wholesale profile: Bulk beer is the anchor — kegs (tap lines), slabs, mid-strength
  and full-strength. Spirits for the back bar (basic range + a few premium pours). Wine for the
  bistro. RTDs for the fridge. High volume, steady weekly orders, price-sensitive. TAB/gaming
  revenue means reliable cash flow = reliable wholesale account. Reverse signal: if there are no
  tap lines, no TAB, and the focus is curated craft spirits, it is NOT a Pub — it's a Bar.

CLUB
  Two distinct sub-types — both classified as Club:

  NIGHTCLUB: A late-night venue focused on music, dancing, and entertainment. Has a large
  dance floor, DJs or live performances, and loud upbeat/dance music. Alcohol is served but
  the primary draw is the energetic atmosphere, not food or quiet socialising. Entry may
  involve cover charges, dress codes, or age restrictions. Operates later than bars/pubs.
  Targets a younger, party-oriented crowd.

  HOSPITALITY/MEMBERSHIP CLUB: RSL clubs, Leagues clubs, sports clubs, bowling clubs, or
  other community clubs that provide hospitality services primarily to members. They often
  have formal membership rules, large dedicated gaming rooms, multiple bars and restaurants
  under one roof, function rooms, and entertainment facilities. Named as "Club" (RSL,
  Leagues, Bowling, Sports).

  Also includes large entertainment complexes (casino + hotel + restaurants + bars in one
  complex) where gaming/entertainment identity dominates — the complex identity wins over
  any individual food or drink component within it.
  Primary focus: Entertainment/dancing/music (nightclub) OR membership-based hospitality
  with gaming/functions (club). Drinks served but secondary to the main draw.
  Hours: Late night (nightclub); daytime through late (membership club).
  Customer experience: High-energy party outing OR member leisure/dining/gaming.
  Examples: RSL clubs, Leagues clubs, nightclubs, casino complexes.
  → Crown Casino (Southbank): massive gaming/entertainment complex. = Club ✓
  Paramount wholesale profile: Varies by sub-type. Membership clubs (RSL/Leagues): very high
  volume beer (kegs + cans), RTDs, basic spirits, large function/event packages — price and
  reliability are everything. Nightclubs: spirits (vodka, rum, gin for mixing), RTDs, bottled
  beer — high margin per serve. Casino complexes: multi-account potential across multiple bars
  and restaurants within the complex. Reverse signal: if the venue has no membership structure,
  no gaming room, and no dance floor, it is NOT a Club.

ACCOMMODATION
  Hotels, motels, serviced apartments, B&Bs, resorts, or inns where the PRIMARY commercial
  purpose is selling overnight rooms (temporary lodging for guests). Food and beverage (bar,
  restaurant) is incidental — it serves guests but is not the main reason the business exists.
  The venue is primarily marketed as a place to stay; "book a room" is the core CTA.
  May include on-site restaurants or bars, but these serve guests rather than being the
  dominant public identity.
  Distinguish from Pub: if the venue's bar/pub identity is MORE famous than its rooms → Pub.
  If the venue is primarily marketed as a place to stay → Accommodation.
  Primary focus: Overnight lodging/sleeping facilities.
  Hours: 24-hour reception; food/drink serves guests on hotel schedule.
  Customer experience: Rest/travel stay.
  Examples: business hotels, boutique hotels, motels, serviced apartments, holiday resorts.
  Paramount wholesale profile: Minibar restocking (spirits, wine, beer in single-serve formats),
  in-room wine selections, on-site restaurant/bar stock, function and event packages (weddings,
  conferences, gala dinners). Delivery reliability and varied SKU range are the key pitch.
  Reverse signal: if the venue's primary public identity is its bar or dining rather than rooms,
  it is NOT Accommodation — it's a Pub or Restaurant.

CATERER
  A business whose PRIMARY activity is preparing and serving food/beverages for events,
  functions, or off-site consumption (e.g. weddings, conferences, parties). Unlike on-premise
  venues, catering is delivered to a client's location or provided at external venues.
  They may have a commercial kitchen but do NOT operate a permanent public-facing walk-in
  venue. Often listed as "catering company", "event caterer", or "function catering".
  Ranges from full-service (including setup, staffing, equipment) to buffet/plated options.
  Primary focus: Event-based or bulk food service; no fixed venue for walk-in customers.
  Customer experience: Support for private or corporate events.
  Examples: wedding caterers, corporate catering companies, mobile food service businesses.
  Paramount wholesale profile: Irregular, event-driven ordering — mixed cases of wine, beer,
  and spirits tailored per event type. Weddings: sparkling + white + red wine + beer. Corporate:
  beer + wine + spirits. Short lead times are critical. No fixed weekly volume but strong margin
  on event packages. Reverse signal: if the business operates a fixed public-facing venue where
  customers walk in, it is NOT a Caterer.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY DIFFERENTIATORS AT A GLANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIMARY FOCUS:
  Bar          → Drinks/alcohol (minimal food)
  Restaurant   → Food/meals (drinks secondary)
  Pub          → Balanced drinks + substantial food in a casual social setting
  Club         → Entertainment/dancing/music OR membership-based gaming/hospitality
  Accommodation→ Overnight lodging
  Caterer      → Off-site or event-based food service (no fixed walk-in venue)

FOOD vs DRINK EMPHASIS:
  Heavy food focus    → Restaurant, and to a lesser extent Pub and Caterer
  Heavy drink focus   → Bar and Pub
  Minimal food focus  → Pure Bar or Nightclub
  No fixed venue      → Accommodation (food incidental) and Caterer (mobile/event)

ATMOSPHERE AND HOURS:
  Relaxed/conversational       → Bar or Pub
  Energetic/dancing, late-night→ Club (nightclub)
  Dining-focused               → Restaurant
  Overnight stay               → Accommodation
  Event-specific               → Caterer

ACCESS AND COMMUNITY FEEL:
  Open to all, community local → Pub
  Members-only or RSL/Leagues  → Club (membership)
  Cover charge, late-night     → Club (nightclub)
  Walk-in dining               → Restaurant or Bar
  Room booking required        → Accommodation
  Event booking required       → Caterer

PARAMOUNT WHOLESALE BUYING SIGNALS (use these as additional classification evidence):
  Curated spirits list, allocated bottles, cocktail menu → Bar
  Extensive wine list, food-pairing language, sommelier  → Restaurant
  Tap beer / kegs, TAB signage, weekly bulk orders       → Pub
  High-volume beer + RTDs, gaming, function packages     → Club
  Minibar, in-room wine, event/function packages         → Accommodation
  Event-by-event mixed cases, no fixed venue ordering    → Caterer
  If a venue's drinks list reveals its identity, use it as a classification signal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL DECISION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BAR vs RESTAURANT — ask yourself ONE question:
  "Why do MOST customers come here?"
  → To drink, with food as an option = BAR
  → To eat a meal, with drinks as an accompaniment = RESTAURANT
  Signal: A wine bar with a cheese board = Bar.
  Signal: A restaurant with a full cocktail list = Restaurant.
  Signal: If cocktails/drinks are the star of reviews → likely Bar.
  Signal: If the kitchen and food menu are the star of reviews → likely Restaurant.
  Signal: If the food is described as "light bites", "bar snacks", or "share plates" → Bar.
  Signal: If there is a multi-course menu, mains, or set menu → Restaurant.

BAR vs PUB — ask yourself:
  "Is this a community local with beer on tap and a bistro, or a specialist drinks venue?"
  → Specialist drinks (cocktails, wine, craft beer as the art form) + minimal food = Bar
  → Beer on tap + hearty pub meals + TAB/pokies + community vibe = Pub
  Signal: No TAB, no pokies, curated spirits list → lean Bar.
  Signal: TAB terminal, gaming machines, counter meals → Pub.

PUB vs CLUB:
  → Single community/local venue, one "pub" entity, open to all = Pub
  → Formal membership structure, named RSL/Leagues/Sports/Bowling Club = Club
  → Massive multi-venue complex (casino scale) = Club
  → Late-night dancing, DJ, dance floor, cover charge = Club (nightclub)

PUB vs ACCOMMODATION:
  → The pub/bar identity is the dominant public reputation → Pub
  → The hotel's primary marketing is "book a room" → Accommodation
  → Historic hotels famous for their public bar: Pub.
  → Business or boutique hotel whose bar serves hotel guests: Accommodation.

NIGHTCLUB vs BAR:
  → Dance floor, DJ/live music, late-night (past midnight), cover charge = Club (nightclub)
  → No dance floor, conversational atmosphere, closes before midnight = Bar

MULTI-PURPOSE VENUES — choose the dominant commercial identity:
  → Crown Casino: gaming + entertainment complex. Casino/club identity wins. = Club
  → Brewery with taproom + small food menu: drinks-first. = Bar
  → Resort hotel with day spa and restaurant: rooms are primary product → Accommodation
  → RSL with restaurant and bar: membership + gaming dominates → Club

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
You are a Paramount Liquor sales rep preparing for your FIRST call with this venue.
Write exactly 3 concise, actionable talking points you will use in that call — written in
first-person sales language ("Lead with...", "Pitch...", "Open by asking...").
Each point must reference something specific from the research about THIS venue.
Do NOT write generic points that could apply to any venue of this type.

What to lead with per segment:
- Bar: Open on premium spirits and allocated bottles — reference any cocktail menu, spirits
  brands, or signature serves visible in research. Pitch Paramount's competitive pricing on
  high-velocity spirits (gin, whisky, vodka) and exclusive label access. Ask about their
  seasonal cocktail rotation as the hook for a recurring supply relationship.
- Restaurant: Lead with a wine-by-the-glass program — reference any cuisine type or existing
  wine list visible in research. Pitch Paramount's sommelier support and food-pairing range.
  Offer to build a house pour shortlist matched to their menu style.
- Pub: Open on tap beer and keg supply — reference any brands, tap lines, or beer garden
  visible in research. Frame Paramount's reliable delivery as protecting their TAB/gaming
  revenue (no stock = no trade). Upsell spirits for the back bar.
- Club: Lead with volume pricing and a dedicated account manager. Reference gaming, function
  rooms, or membership numbers if visible. Pitch bulk beer/RTD deals and event stock packages.
  For nightclubs, focus on spirits + RTD mix for high-margin serves.
- Accommodation: Open on minibar restocking and in-room wine selections. Reference any
  function or event space visible in research to pitch event liquor packages. Emphasise
  Paramount's delivery reliability and varied SKU range for multi-format orders.
- Caterer: Lead with flexibility — event-by-event ordering, short lead times, mixed case
  options. Reference any event types (weddings, corporate, private) visible in research.
  Pitch pre-built event packages to simplify their ordering process.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY this JSON. No explanation before or after. No markdown fences.
{
  "segment": "<exactly one of: Bar, Restaurant, Pub, Club, Accommodation, Caterer>",
  "confidence": <integer 0-100>,
  "anzsic_code": "<ANZSIC 2006 code if found in research, e.g. '4520', else null>",
  "anzsic_description": "<Official ANZSIC class description if code found, e.g. 'Pubs, Taverns and Bars', else null>",
  "liquor_licence": "<State liquor licence type if found in research, e.g. 'VIC — Bar Licence', 'NSW — Hotel Licence', else null>",
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
    anzsic_code: Optional[str] = Field(None, description="ANZSIC 2006 code if found in research (e.g. '4520')")
    anzsic_description: Optional[str] = Field(None, description="Official ANZSIC class description (e.g. 'Pubs, Taverns and Bars')")
    liquor_licence: Optional[str] = Field(None, description="State liquor licence type if found (e.g. 'VIC — Bar Licence')")
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
                return [str(q) for q in queries[:5]]
    except Exception as exc:
        logger.warning(f"Query generation failed, using fallback: {exc}")

    # Hardcoded fallback — always works
    suffix = "venue type category" if retry else "menu food drinks"
    return [
        f"{business_name} {suburb} {suffix}",
        f'"{business_name}" {suburb} Google Maps TripAdvisor bar restaurant pub',
        f"{business_name} {suburb} review atmosphere experience",
        f"{business_name} {suburb} ABN ABR ANZSIC industry code classification",
        f"{business_name} {suburb} liquor licence type VCGLR OR OLGR bar hotel club restaurant",
    ]


async def run_tavily_searches(tavily: AsyncTavilyClient, queries: List[str]) -> str:
    """Run all queries concurrently, merge and deduplicate results, return formatted string."""

    async def search_one(query: str):
        return await tavily.search(
            query=query,
            search_depth="advanced",
            max_results=7,
            include_answer=False,
            include_raw_content=True,  # Get actual page text: menus, profile descriptions, reviews
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

    # Format into a text block for Claude — include truncated raw page content where available
    lines = []
    for i, item in enumerate(top, 1):
        lines.append(f"[Result {i}]")
        lines.append(f"Title: {item.get('title', 'N/A')}")
        lines.append(f"URL: {item.get('url', 'N/A')}")
        lines.append(f"Summary: {item.get('content', 'N/A')}")
        raw = item.get("raw_content") or ""
        if raw:
            truncated = raw[:MAX_RAW_CONTENT_PER_RESULT].strip()
            lines.append(f"Page content: {truncated}")
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

    # Parse ANZSIC fields (graceful fallback to None if not found in research)
    anzsic_code = classification.get("anzsic_code")
    classification["anzsic_code"] = str(anzsic_code).strip() if anzsic_code else None
    anzsic_desc = classification.get("anzsic_description")
    classification["anzsic_description"] = str(anzsic_desc).strip() if anzsic_desc else None

    # Parse liquor licence field (graceful fallback to None if not found in research)
    liquor_licence = classification.get("liquor_licence")
    classification["liquor_licence"] = str(liquor_licence).strip() if liquor_licence else None

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
            anzsic_code=classification.get("anzsic_code"),
            anzsic_description=classification.get("anzsic_description"),
            liquor_licence=classification.get("liquor_licence"),
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
