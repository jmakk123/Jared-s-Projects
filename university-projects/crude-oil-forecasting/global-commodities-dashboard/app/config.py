from datetime import timedelta
from pathlib import Path


COMMODITIES = [
    {
        "id": "wti",
        "name": "Cushing, OK WTI Spot Price FOB",
        "unit": "$/barrel",
        "accent": "#f5a623",
        "short_name": "WTI",
        "category": "Crude Oil",
        "definition": "West Texas Intermediate is a crude stream produced in Texas and southern Oklahoma and traded at Cushing, Oklahoma as a domestic pricing marker.",
        "location_note": "Cushing, Oklahoma marker crude",
        "sheet": "Data 1",
        "header_match": "cushing ok wti spot price fob dollars per barrel",
    },
    {
        "id": "brent",
        "name": "Europe Brent Spot Price FOB",
        "unit": "$/barrel",
        "accent": "#28d7ff",
        "short_name": "Brent",
        "category": "Crude Oil",
        "definition": "Brent is a blended North Sea crude stream used globally as a benchmark for pricing many other crude oils.",
        "location_note": "North Sea benchmark crude",
        "sheet": "Data 1",
        "header_match": "europe brent spot price fob dollars per barrel",
    },
    {
        "id": "conventional_gasoline",
        "name": "Conventional Gasoline",
        "unit": "$/gallon",
        "accent": "#57e389",
        "short_name": "Conventional Gasoline",
        "category": "Gasoline",
        "definition": "Finished motor gasoline outside the oxygenated and reformulated categories, excluding RBOB and other blending stock.",
        "location_note": "New York Harbor regular gasoline spot",
        "sheet": "Data 2",
        "header_match": "new york harbor conventional gasoline regular spot price fob dollars per gallon",
        "source_series": "New York Harbor Conventional Gasoline Regular Spot Price FOB",
    },
    {
        "id": "rbob_gasoline",
        "name": "RBOB Regular Gasoline",
        "unit": "$/gallon",
        "accent": "#8eea57",
        "short_name": "RBOB Gasoline",
        "category": "Gasoline Blendstock",
        "definition": "Reformulated Gasoline Blendstock for Oxygenate Blending, intended to be mixed with oxygenates to produce finished reformulated gasoline.",
        "location_note": "Los Angeles RBOB spot pricing",
        "sheet": "Data 3",
        "header_match": "los angeles reformulated rbob regular gasoline spot price dollars per gallon",
        "source_series": "Los Angeles Reformulated RBOB Regular Gasoline Spot Price",
    },
    {
        "id": "heating_oil",
        "name": "No. 2 Heating Oil",
        "unit": "$/gallon",
        "accent": "#ffd166",
        "short_name": "Heating Oil",
        "category": "Distillate",
        "definition": "A No. 2 distillate fuel used for heating applications and closely tied to middle-distillate refinery economics.",
        "location_note": "New York Harbor heating oil spot",
        "sheet": "Data 4",
        "header_match": "new york harbor no 2 heating oil spot price fob dollars per gallon",
        "source_series": "New York Harbor No. 2 Heating Oil Spot Price FOB",
    },
    {
        "id": "ulsd_diesel",
        "name": "Ultra-Low-Sulfur No. 2 Diesel Fuel",
        "unit": "$/gallon",
        "accent": "#ff8c42",
        "short_name": "ULSD Diesel",
        "category": "Distillate",
        "definition": "No. 2 diesel fuel with sulfur content at or below 15 ppm, the standard on-highway ultra-low-sulfur diesel grade.",
        "location_note": "New York Harbor ULSD spot",
        "sheet": "Data 5",
        "header_match": "new york harbor ultra low sulfur no 2 diesel spot price dollars per gallon",
        "source_series": "New York Harbor Ultra-Low Sulfur No. 2 Diesel Spot Price",
    },
    {
        "id": "jet_fuel",
        "name": "Kerosene-Type Jet Fuel",
        "unit": "$/gallon",
        "accent": "#b388ff",
        "short_name": "Jet Fuel",
        "category": "Distillate",
        "definition": "A kerosene-based aviation fuel meeting ASTM and military specifications for commercial and military aircraft turbines.",
        "location_note": "U.S. Gulf Coast jet fuel spot",
        "sheet": "Data 6",
        "header_match": "u s gulf coast kerosene type jet fuel spot price fob dollars per gallon",
        "source_series": "U.S. Gulf Coast Kerosene-Type Jet Fuel Spot Price FOB",
    },
    {
        "id": "propane",
        "name": "Propane",
        "unit": "$/gallon",
        "accent": "#ff5fa2",
        "short_name": "Propane",
        "category": "NGL",
        "definition": "A normally gaseous hydrocarbon extracted from natural gas or refinery streams and quoted here at Mont Belvieu, Texas.",
        "location_note": "Mont Belvieu propane hub",
        "sheet": "Data 7",
        "header_match": "mont belvieu tx propane spot price fob dollars per gallon",
        "source_series": "Mont Belvieu, TX Propane Spot Price FOB",
    },
]

COMMODITY_BY_ID = {item["id"]: item for item in COMMODITIES}

DATA_FILENAME = "PET_PRI_SPT_S1_D.xls"
BOOTSTRAP_CACHE_FILENAME = "bootstrap_cache.json"
NEWS_HISTORY_FILENAME = "news_history.json"
INDEX_TEMPLATE_PATH = Path("static/index.html")
CACHE_VERSION = 7
DEFAULT_FORECAST_RANGE = "1Y"
EIA_SOURCE_PAGE_URL = "https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm"
EIA_SOURCE_XLS_URL = "https://www.eia.gov/dnav/pet/xls/PET_PRI_SPT_S1_D.xls"
NEWS_CACHE_TTL = timedelta(hours=6)
SOURCE_REFRESH_TTL = timedelta(hours=6)
NEWS_HISTORY_LOOKBACK_DAYS = 30
NEWS_HISTORY_TOP_K = 3
ZERO_SHOT_MODEL_ID = "valhalla/distilbart-mnli-12-1"

POSITIVE_SENTIMENT_TERMS = {
    "ceasefire", "deal", "easing", "boost", "expand", "surplus",
    "stabilize", "stability", "cooling", "resume", "recovery", "relief",
    "agreement", "lower", "drop", "fall", "soften", "discount",
    "glut", "oversupply", "build", "builds", "ease", "eases", "weakens",
    "slump", "decline", "declines", "down", "downturn",
    "bearish", "cool", "cools", "cooler", "dovish", "disinflation",
    "normalize", "normalizes", "unwind", "unwinds", "deflationary",
    "ample", "abundant", "capacity", "restart", "restarts", "reopen",
    "reopens", "resolve", "resolves", "settlement", "truce", "lowering",
    "decrease", "decreases", "declining", "retreat", "retreats", "slides",
    "slide", "slid", "plunge", "plunges", "plunged", "tumble", "tumbles",
    "tumbled", "sink", "sinks", "sank", "cheaper", "discounted",
}

NEGATIVE_SENTIMENT_TERMS = {
    "war", "conflict", "attack", "sanction", "tariff", "strike", "outage",
    "disruption", "tight", "shortage", "cut", "cuts", "embargo", "risk",
    "surge", "spike", "higher", "hike", "hikes", "halt", "shutdown", "missile",
    "volatile", "delay", "bottleneck", "soar", "soars", "soared", "soaring",
    "strengthen", "strengthens", "stronger", "jump", "jumps", "jumped",
    "rally", "rallies", "tighten", "tightens", "squeeze", "squeezes",
    "squeezed", "escalation", "escalates", "escalating",
    "bullish", "inflation", "inflationary", "scarcity", "scarce", "stress",
    "stressed", "shock", "shocks", "supply shock", "supply shocks",
    "premium", "premiums", "dislocated", "dislocation", "curb", "curbs",
    "restrict", "restricts", "restricted", "crunch", "crisis", "crises",
    "reroute", "reroutes", "rerouted", "houthi", "drone", "blast",
    "explosion", "tension", "tensions", "military", "retaliation",
    "retaliatory", "blockade", "freeze", "freezes", "freeze-up", "panic",
    "panic buying", "draw", "draws", "drawdown", "drawdowns",
}

PRICE_BULLISH_TERMS = {
    "war", "conflict", "attack", "sanction", "sanctions", "strike", "outage",
    "disruption", "tight", "shortage", "cut", "cuts", "embargo", "surge",
    "spike", "higher", "hike", "hikes", "halt", "shutdown", "missile",
    "delay", "bottleneck", "soar", "soars", "soared", "soaring", "strengthen",
    "strengthens", "stronger", "jump", "jumps", "jumped", "rally", "rallies",
    "tighten", "tightens", "squeeze", "squeezes", "squeezed", "escalation",
    "escalates", "escalating", "scarcity", "scarce", "stress", "stressed",
    "shock", "shocks", "premium", "premiums", "curb", "curbs", "restrict",
    "restricts", "restricted", "crunch", "crisis", "crises", "reroute",
    "reroutes", "rerouted", "houthi", "drone", "blast", "explosion",
    "tension", "tensions", "military", "retaliation", "retaliatory",
    "blockade", "panic", "draw", "draws", "drawdown", "drawdowns",
    "cold", "freeze", "freezes", "storm", "hurricane", "outperform",
}

PRICE_BEARISH_TERMS = {
    "ceasefire", "deal", "easing", "expand", "surplus", "stabilize",
    "stability", "cooling", "resume", "recovery", "relief", "agreement",
    "lower", "drop", "fall", "soften", "discount", "glut", "oversupply",
    "build", "builds", "ease", "eases", "weakens", "slump", "decline",
    "declines", "down", "downturn", "cool", "cools", "cooler", "normalize",
    "normalizes", "unwind", "unwinds", "ample", "abundant", "capacity",
    "restart", "restarts", "reopen", "reopens", "resolve", "resolves",
    "settlement", "truce", "lowering", "decrease", "decreases", "declining",
    "retreat", "retreats", "slides", "slide", "slid", "plunge", "plunges",
    "plunged", "tumble", "tumbles", "tumbled", "sink", "sinks", "sank",
    "cheaper", "discounted", "inventory", "inventories", "build-up",
    "increase", "increases", "raise", "raises", "raised", "boost",
    "output", "production", "supply", "surplus", "glutted", "weak",
    "recession", "slowdown", "demand destruction",
}

PRICE_BULLISH_PHRASES = {
    "prices soar": 1.8,
    "costs soar": 1.8,
    "price surge": 1.5,
    "prices jump": 1.5,
    "prices rise": 1.0,
    "output cuts": 1.6,
    "supply disruption": 1.7,
    "shipping disruption": 1.7,
    "middle east conflict": 1.6,
    "refinery outage": 1.7,
    "refinery fire": 1.7,
    "pipeline outage": 1.7,
    "inventory draw": 1.4,
    "inventory draws": 1.4,
    "export curbs": 1.5,
    "export cuts": 1.5,
    "production cuts": 1.6,
    "supply cuts": 1.6,
    "sanctions risk": 1.3,
    "shipping attacks": 1.7,
    "red sea": 1.2,
    "opec cuts": 1.7,
    "opec+ cuts": 1.7,
    "refinery closures": 1.5,
    "gas prices rise": 1.4,
    "diesel prices rise": 1.4,
}

PRICE_BEARISH_PHRASES = {
    "prices fall": -1.0,
    "prices ease": -1.0,
    "supply builds": -1.1,
    "inventory build": -1.4,
    "inventory builds": -1.4,
    "ceasefire talks": -1.1,
    "production increase": -1.2,
    "output increase": -1.2,
    "output increases": -1.2,
    "supply increase": -1.1,
    "supply increases": -1.1,
    "refinery restart": -1.2,
    "refinery restarts": -1.2,
    "exports resume": -1.1,
    "peace talks": -1.0,
    "demand weakness": -1.2,
    "demand slows": -1.2,
    "economic slowdown": -1.0,
    "oversupply fears": -1.2,
    "record supply": -1.2,
    "inventories rise": -1.1,
}

COMMODITY_NEWS_CONFIG = {
    "wti": {
        "queries": [
            '("wti" OR "west texas intermediate" OR crude oil OR oil prices) ("middle east" OR opec OR sanctions OR shipping OR tariff OR pipeline OR refinery OR output OR "supply cuts")',
            '("wti crude" OR "us crude") (prices OR rally OR slump OR forecasts OR sanctions OR outage)',
        ],
        "must_have": ["wti", "west texas intermediate", "crude", "oil", "opec", "brent"],
    },
    "brent": {
        "queries": [
            '("brent crude" OR brent OR crude oil OR oil prices) ("middle east" OR opec OR sanctions OR shipping OR north sea OR output)',
            '("brent crude") (prices OR rally OR slump OR sanctions OR outage OR cuts)',
        ],
        "must_have": ["brent", "crude", "oil", "opec", "north sea"],
    },
    "conventional_gasoline": {
        "queries": [
            '(gasoline OR petrol OR "motor fuel" OR "gas prices" OR refinery) (prices OR supply OR outage OR demand OR tariff OR conflict)',
            '("gas prices" OR gasoline OR petrol) (rises OR falls OR surge OR shortage OR refinery)',
        ],
        "must_have": ["gasoline", "petrol", "gas prices", "refinery", "motor fuel", "fuel"],
    },
    "rbob_gasoline": {
        "queries": [
            '(rbob OR gasoline futures OR reformulated gasoline OR refinery) (prices OR supply OR outage OR crack spread)',
            '(rbob OR gasoline futures) (surge OR slump OR refinery OR crack spread OR outage)',
        ],
        "must_have": ["rbob", "gasoline", "reformulated", "refinery", "petrol", "futures"],
    },
    "heating_oil": {
        "queries": [
            '("heating oil" OR distillate OR diesel OR refinery) (prices OR supply OR weather OR outage)',
            '("heating oil" OR distillates) (cold OR storm OR shortage OR inventories OR prices)',
        ],
        "must_have": ["heating oil", "distillate", "diesel", "refinery", "fuel oil"],
    },
    "ulsd_diesel": {
        "queries": [
            '(diesel OR ulsd OR distillate OR trucking OR refinery) (prices OR supply OR outage OR sanctions OR shipping)',
            '(diesel OR trucking fuel OR ulsd) (prices OR surge OR slump OR shortages OR refinery)',
        ],
        "must_have": ["diesel", "ulsd", "distillate", "truck", "refinery", "shipping", "freight"],
    },
    "jet_fuel": {
        "queries": [
            '("jet fuel" OR aviation fuel OR airline fuel OR refinery) (prices OR demand OR travel OR outage)',
            '("jet fuel" OR aviation fuel) (travel demand OR refinery OR prices OR shortage)',
        ],
        "must_have": ["jet fuel", "aviation fuel", "airline", "refinery", "kerosene", "travel"],
    },
    "propane": {
        "queries": [
            '(propane OR ngl OR "natural gas liquids" OR mont belvieu) (prices OR exports OR petrochemical OR weather OR inventory)',
            '(propane OR mont belvieu) (prices OR cold OR inventory OR exports OR petrochemical)',
        ],
        "must_have": ["propane", "ngl", "natural gas liquids", "mont belvieu", "petchem", "inventory"],
    },
}

RANGE_LABEL_TO_DAYS = {
    "1M": 30,
    "3M": 90,
    "6M": 180,
    "1Y": 365,
    "2Y": 730,
    "5Y": 365 * 5,
    "10Y": 365 * 10,
    "All": None,
}
