from datetime import date, datetime, timedelta
import xml.etree.ElementTree as ET
import re
from typing import Any, Optional
from urllib.parse import quote_plus

import numpy as np
from transformers import pipeline

from app import state
from app.config import (
    COMMODITY_BY_ID,
    COMMODITY_NEWS_CONFIG,
    NEGATIVE_SENTIMENT_TERMS,
    NEWS_CACHE_TTL,
    NEWS_HISTORY_LOOKBACK_DAYS,
    NEWS_HISTORY_TOP_K,
    POSITIVE_SENTIMENT_TERMS,
    PRICE_BEARISH_PHRASES,
    PRICE_BEARISH_TERMS,
    PRICE_BULLISH_PHRASES,
    PRICE_BULLISH_TERMS,
    ZERO_SHOT_MODEL_ID,
)
from app.utils import http_get_text, news_history_path, parse_published_datetime, read_json, write_json

UPWARD_PRICE_PATTERNS = {
    "gas prices soar": 1.0,
    "prices soar": 0.95,
    "costs soar": 0.95,
    "prices spike": 0.92,
    "prices surge": 0.9,
    "prices jump": 0.82,
    "prices rally": 0.75,
    "prices rise": 0.58,
    "pump prices rise": 0.7,
    "pump prices soar": 1.0,
    "price hike": 0.78,
    "price hikes": 0.82,
    "higher prices": 0.55,
    "record high": 0.78,
    "tight supply": 0.62,
    "supply crunch": 0.88,
    "output cuts": 0.82,
    "refinery outage": 0.86,
    "shipping disruption": 0.84,
}

DOWNWARD_PRICE_PATTERNS = {
    "prices plunge": -1.0,
    "prices tumble": -0.95,
    "prices slump": -0.9,
    "prices slide": -0.82,
    "prices drop": -0.78,
    "prices fall": -0.72,
    "prices ease": -0.58,
    "lower prices": -0.52,
    "cheaper fuel": -0.65,
    "inventory build": -0.7,
    "inventory builds": -0.7,
    "record supply": -0.78,
    "oversupply": -0.72,
    "output increase": -0.62,
    "production increase": -0.62,
    "refinery restart": -0.7,
}

UPWARD_MOVE_TERMS = {
    "up": 0.66,
    "upward": 0.66,
    "soar": 0.95,
    "soars": 0.95,
    "soared": 0.95,
    "soaring": 0.95,
    "surge": 0.9,
    "surges": 0.9,
    "surged": 0.9,
    "spike": 0.9,
    "spikes": 0.9,
    "spiked": 0.9,
    "jump": 0.78,
    "jumps": 0.78,
    "jumped": 0.78,
    "rally": 0.7,
    "rallies": 0.7,
    "rise": 0.58,
    "rises": 0.58,
    "higher": 0.48,
    "hike": 0.74,
    "hikes": 0.78,
    "strengthen": 0.62,
    "strengthens": 0.62,
    "stronger": 0.52,
}

DOWNWARD_MOVE_TERMS = {
    "down": -0.66,
    "downward": -0.66,
    "plunge": -0.95,
    "plunges": -0.95,
    "plunged": -0.95,
    "tumble": -0.9,
    "tumbles": -0.9,
    "tumbled": -0.9,
    "slump": -0.84,
    "slumps": -0.84,
    "slide": -0.78,
    "slides": -0.78,
    "drop": -0.72,
    "drops": -0.72,
    "fall": -0.66,
    "falls": -0.66,
    "ease": -0.52,
    "eases": -0.52,
    "lower": -0.46,
    "decline": -0.58,
    "declines": -0.58,
    "weaken": -0.52,
    "weakens": -0.52,
}

PRICE_CONTEXT_TERMS = {
    "price", "prices", "cost", "costs", "pump", "gasoline", "gas", "diesel",
    "fuel", "oil", "crude", "brent", "wti", "propane", "rbob", "ulsd",
}

CLAUSE_SPLIT_RE = re.compile(r"\s[-–—]\s|[,:;|]+|\.\s+")
PERCENT_MOVE_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")


def news_queries_for_commodity(commodity: dict[str, Any], days: int) -> list[str]:
    config = COMMODITY_NEWS_CONFIG.get(commodity["id"], {})
    configured = config.get("queries")
    if configured:
        return [f"({query}) when:{days}d" for query in configured]
    topic = config.get("query", commodity["short_name"])
    return [f'({topic}) (OPEC OR sanctions OR refinery OR shipping OR conflict OR tariff OR pipeline OR output OR demand) when:{days}d']


def fallback_news_queries_for_commodity(commodity: dict[str, Any], days: int) -> list[str]:
    short_name = commodity.get("short_name", "")
    category = commodity.get("category", "")
    source_series = commodity.get("source_series", short_name)
    return [
        f'("{short_name}" OR "{source_series}" OR {category} OR energy) (prices OR supply OR demand OR outage OR conflict OR sanctions OR shipping) when:{days}d',
        '(oil OR crude OR gasoline OR diesel OR jet fuel OR propane OR refinery OR OPEC OR shipping) (prices OR supply OR demand OR outage OR sanctions OR conflict) when:7d',
    ]


def commodity_context_terms(commodity: Optional[dict[str, Any]] = None) -> set[str]:
    terms = set(PRICE_CONTEXT_TERMS)
    if not commodity:
        return terms
    raw_values = [
        commodity.get("id", ""),
        commodity.get("short_name", ""),
        commodity.get("category", ""),
        commodity.get("name", ""),
        commodity.get("definition", ""),
        commodity.get("location_note", ""),
    ]
    for raw in raw_values:
        for token in re.findall(r"[A-Za-z]+", str(raw).lower()):
            if len(token) >= 3:
                terms.add(token)
    return terms


def contextual_headline_segments(title: str, commodity: Optional[dict[str, Any]] = None) -> list[str]:
    lowered = title.lower()
    clauses = [part.strip() for part in CLAUSE_SPLIT_RE.split(lowered) if part.strip()]
    if not clauses:
        return [lowered]
    context_terms = commodity_context_terms(commodity)
    contextual = []
    for clause in clauses:
        words = set(re.findall(r"[A-Za-z']+", clause))
        if words.intersection(context_terms):
            contextual.append(clause)
    return contextual or [lowered]


def magnitude_score(percent_move: float) -> float:
    bounded = min(abs(percent_move), 30.0)
    return min(0.98, 0.62 + (bounded / 30.0) * 0.33)


def explicit_clause_direction_score(clause: str) -> float:
    words = re.findall(r"[A-Za-z']+", clause)
    upward = max((UPWARD_MOVE_TERMS[word] for word in words if word in UPWARD_MOVE_TERMS), default=0.0)
    downward = min((DOWNWARD_MOVE_TERMS[word] for word in words if word in DOWNWARD_MOVE_TERMS), default=0.0)

    if re.search(r"\bup\s+\d+(?:\.\d+)?\s*%", clause) or re.search(r"\+\d+(?:\.\d+)?\s*%", clause):
        upward = max(upward, 0.72)
    if re.search(r"\bdown\s+\d+(?:\.\d+)?\s*%", clause) or re.search(r"-\d+(?:\.\d+)?\s*%", clause):
        downward = min(downward, -0.72)

    percents = [float(match.group(1)) for match in PERCENT_MOVE_RE.finditer(clause)]
    if percents:
        largest = max(percents, key=lambda value: abs(value))
        explicit_sign = 0
        if largest > 0 and (upward > 0 or re.search(r"\bup\b", clause) or "+" in clause):
            explicit_sign = 1
        elif largest < 0 or downward < 0 or re.search(r"\bdown\b", clause):
            explicit_sign = -1
        elif upward > abs(downward):
            explicit_sign = 1
        elif abs(downward) > upward:
            explicit_sign = -1
        if explicit_sign > 0:
            upward = max(upward, magnitude_score(largest))
        elif explicit_sign < 0:
            downward = min(downward, -magnitude_score(largest))

    if upward > 0 and downward >= 0:
        return upward
    if downward < 0 and upward <= 0:
        return downward
    return upward if upward >= abs(downward) else downward


def headline_tone_score(title: str, commodity: Optional[dict[str, Any]] = None) -> float:
    lowered = title.lower()
    contextual_text = " ".join(contextual_headline_segments(title, commodity))
    words = re.findall(r"[A-Za-z']+", contextual_text)
    if not words:
        return 0.0
    positive = sum(1 for word in words if word in POSITIVE_SENTIMENT_TERMS)
    negative = sum(1 for word in words if word in NEGATIVE_SENTIMENT_TERMS)
    phrase_weights = {
        "prices soar": -2.0,
        "costs soar": -2.0,
        "price surge": -1.5,
        "prices jump": -1.5,
        "prices rise": -1.0,
        "output cuts": -1.5,
        "supply disruption": -1.5,
        "middle east conflict": -1.5,
        "shipping disruption": -1.5,
        "refinery outage": -1.5,
        "prices fall": 1.0,
        "supply builds": 1.0,
        "ceasefire talks": 1.0,
        "production increase": 1.0,
    }
    phrase_adjustment = sum(weight for phrase, weight in phrase_weights.items() if phrase in contextual_text)
    raw = positive - negative
    lexical_score = max(min((raw + phrase_adjustment) / 4.0, 1.0), -1.0)
    semantic_score = price_phrase_tone_score(title, commodity)
    if abs(semantic_score) >= 0.55:
        return semantic_score
    combined = (lexical_score * 0.35) + (semantic_score * 0.65)
    return max(min(combined, 1.0), -1.0)


def price_phrase_tone_score(title: str, commodity: Optional[dict[str, Any]] = None) -> float:
    contextual_segments = contextual_headline_segments(title, commodity)
    score = 0.0
    for clause in contextual_segments:
        explicit_score = explicit_clause_direction_score(clause)
        if explicit_score > 0:
            score = max(score, explicit_score)
        elif explicit_score < 0:
            score = min(score, explicit_score)

        for phrase, weight in UPWARD_PRICE_PATTERNS.items():
            if phrase in clause:
                score = max(score, weight)
        for phrase, weight in DOWNWARD_PRICE_PATTERNS.items():
            if phrase in clause:
                score = min(score, weight)

        words = re.findall(r"[A-Za-z']+", clause)
        for word in words:
            if word in UPWARD_MOVE_TERMS:
                score = max(score, UPWARD_MOVE_TERMS[word])
            elif word in DOWNWARD_MOVE_TERMS:
                score = min(score, DOWNWARD_MOVE_TERMS[word])

    contextual_text = " ".join(contextual_segments)
    if "price" in contextual_text or "prices" in contextual_text:
        if any(token in contextual_text for token in ("record high", "all-time high", "multi-year high")):
            score = max(score, 0.82)
        if any(token in contextual_text for token in ("record low", "multi-year low")):
            score = min(score, -0.82)

    return max(min(score, 1.0), -1.0)


def price_impact_rule_score(title: str, commodity: Optional[dict[str, Any]] = None) -> float:
    contextual_segments = contextual_headline_segments(title, commodity)
    contextual_text = " ".join(contextual_segments)
    words = re.findall(r"[A-Za-z']+", contextual_text)
    if not words:
        return 0.0
    bullish = sum(1 for word in words if word in PRICE_BULLISH_TERMS)
    bearish = sum(1 for word in words if word in PRICE_BEARISH_TERMS)
    phrase_adjustment = sum(weight for phrase, weight in PRICE_BULLISH_PHRASES.items() if phrase in contextual_text)
    phrase_adjustment += sum(weight for phrase, weight in PRICE_BEARISH_PHRASES.items() if phrase in contextual_text)
    directional = max((explicit_clause_direction_score(clause) for clause in contextual_segments), key=lambda value: abs(value), default=0.0)
    explicit_score = abs(directional)
    if "price" in contextual_text or "prices" in contextual_text:
        if any(token in contextual_text for token in ("rise", "rises", "higher", "hike", "soar", "surge", "jump", "up")):
            phrase_adjustment += 0.8
        if any(token in contextual_text for token in ("fall", "falls", "lower", "ease", "slump", "drop", "down")):
            phrase_adjustment -= 0.8
    raw = bullish - bearish
    score = max(min((raw / 5.0) + (phrase_adjustment / 3.0), 1.0), -1.0)
    if explicit_score >= 0.72:
        score = directional
    return score


def zero_shot_classifier() -> Optional[Any]:
    if state.ZERO_SHOT_CLASSIFIER is not None:
        return state.ZERO_SHOT_CLASSIFIER
    if state.ZERO_SHOT_LOAD_ERROR is not None:
        return None
    try:
        state.ZERO_SHOT_CLASSIFIER = pipeline(
            "zero-shot-classification",
            model=ZERO_SHOT_MODEL_ID,
            device=-1,
        )
        return state.ZERO_SHOT_CLASSIFIER
    except Exception as exc:
        state.ZERO_SHOT_LOAD_ERROR = str(exc)
        return None


def transformer_price_direction_score(title: str, commodity: dict[str, Any]) -> dict[str, Any]:
    cached = state.HEADLINE_SIGNAL_CACHE.get((commodity["id"], title, "direction"))
    if cached is not None:
        return cached
    classifier = zero_shot_classifier()
    if classifier is None:
        payload = {
            "score": 0.0,
            "label": "unavailable",
            "confidence": 0.0,
            "model": ZERO_SHOT_MODEL_ID,
            "available": False,
            "error": state.ZERO_SHOT_LOAD_ERROR,
        }
        state.HEADLINE_SIGNAL_CACHE[(commodity["id"], title, "direction")] = payload
        return payload

    candidate_labels = [
        f"higher near-term {commodity['short_name']} spot prices",
        f"lower near-term {commodity['short_name']} spot prices",
        f"unclear impact on {commodity['short_name']} spot prices",
    ]
    try:
        result = classifier(
            title,
            candidate_labels,
            hypothesis_template="This headline implies {}.",
            multi_label=False,
        )
        scores = dict(zip(result["labels"], result["scores"]))
        score = float(scores.get(candidate_labels[0], 0.0) - scores.get(candidate_labels[1], 0.0))
        label = "bullish" if score > 0.12 else "bearish" if score < -0.12 else "neutral"
        confidence = max(scores.get(candidate_labels[0], 0.0), scores.get(candidate_labels[1], 0.0), scores.get(candidate_labels[2], 0.0))
        payload = {
            "score": round(score, 4),
            "label": label,
            "confidence": round(float(confidence), 4),
            "model": ZERO_SHOT_MODEL_ID,
            "available": True,
        }
    except Exception as exc:
        payload = {
            "score": 0.0,
            "label": "unavailable",
            "confidence": 0.0,
            "model": ZERO_SHOT_MODEL_ID,
            "available": False,
            "error": str(exc),
        }
    state.HEADLINE_SIGNAL_CACHE[(commodity["id"], title, "direction")] = payload
    return payload


def transformer_market_tone_score(title: str, commodity: dict[str, Any]) -> dict[str, Any]:
    cached = state.HEADLINE_SIGNAL_CACHE.get((commodity["id"], title, "tone"))
    if cached is not None:
        return cached
    classifier = zero_shot_classifier()
    if classifier is None:
        payload = {
            "score": 0.0,
            "label": "unavailable",
            "confidence": 0.0,
            "available": False,
        }
        state.HEADLINE_SIGNAL_CACHE[(commodity["id"], title, "tone")] = payload
        return payload

    candidate_labels = [
        f"strong upward pressure on {commodity['short_name']} prices",
        f"strong downward pressure on {commodity['short_name']} prices",
        f"neutral market backdrop for {commodity['short_name']}",
    ]
    try:
        result = classifier(
            title,
            candidate_labels,
            hypothesis_template="This headline signals {}.",
            multi_label=False,
        )
        scores = dict(zip(result["labels"], result["scores"]))
        score = float(scores.get(candidate_labels[0], 0.0) - scores.get(candidate_labels[1], 0.0))
        label = "positive" if score > 0.12 else "negative" if score < -0.12 else "neutral"
        confidence = max(scores.get(candidate_labels[0], 0.0), scores.get(candidate_labels[1], 0.0), scores.get(candidate_labels[2], 0.0))
        payload = {
            "score": round(score, 4),
            "label": label,
            "confidence": round(float(confidence), 4),
            "available": True,
        }
    except Exception as exc:
        payload = {
            "score": 0.0,
            "label": "unavailable",
            "confidence": 0.0,
            "available": False,
            "error": str(exc),
        }
    state.HEADLINE_SIGNAL_CACHE[(commodity["id"], title, "tone")] = payload
    return payload


def combined_headline_signal(title: str, commodity: dict[str, Any]) -> dict[str, Any]:
    phrase_tone_score = price_phrase_tone_score(title, commodity)
    lexical_tone_score = headline_tone_score(title, commodity)
    transformer_tone = transformer_market_tone_score(title, commodity)
    transformer_tone_score = float(transformer_tone.get("score", 0.0) or 0.0)
    transformer_tone_confidence = float(transformer_tone.get("confidence", 0.0) or 0.0)
    transformer_tone_weight = 0.5 if transformer_tone.get("available") else 0.0
    if transformer_tone_confidence < 0.45:
        transformer_tone_weight *= 0.5
    rule_tone_weight = 0.35
    lexical_tone_weight = max(0.0, 1.0 - transformer_tone_weight - rule_tone_weight)
    tone_score = (
        transformer_tone_score * transformer_tone_weight
        + phrase_tone_score * rule_tone_weight
        + lexical_tone_score * lexical_tone_weight
    )
    if abs(phrase_tone_score) >= 0.8:
        tone_score = phrase_tone_score
    tone_score = max(min(tone_score, 1.0), -1.0)
    rule_score = price_impact_rule_score(title, commodity)
    transformer = transformer_price_direction_score(title, commodity)
    transformer_score = float(transformer.get("score", 0.0) or 0.0)
    transformer_confidence = float(transformer.get("confidence", 0.0) or 0.0)
    transformer_weight = 0.55 if transformer.get("available") else 0.0
    if transformer_confidence < 0.45:
        transformer_weight *= 0.5
    rule_weight = 0.45 if transformer_weight > 0 else 0.8
    tone_weight = max(0.0, 1.0 - transformer_weight - rule_weight)
    event_score = (transformer_score * transformer_weight) + (rule_score * rule_weight) + (tone_score * tone_weight * -0.15)
    if abs(phrase_tone_score) >= 0.72:
        event_score = max(abs(event_score), abs(phrase_tone_score)) * (1 if phrase_tone_score > 0 else -1)
    if abs(rule_score) >= 0.45 and abs(transformer_score) < 0.15:
        event_score = rule_score * 0.85
    event_score = max(min(event_score, 1.0), -1.0)
    event_label = "bullish" if event_score > 0.12 else "bearish" if event_score < -0.12 else "neutral"
    tone_label = "positive" if tone_score > 0.12 else "negative" if tone_score < -0.12 else "neutral"
    return {
        "tone_score": round(float(tone_score), 4),
        "tone_label": tone_label,
        "phrase_tone_score": round(float(phrase_tone_score), 4),
        "transformer_tone_score": round(float(transformer_tone_score), 4),
        "transformer_tone_label": transformer_tone.get("label", "unavailable"),
        "rule_score": round(float(rule_score), 4),
        "transformer_score": transformer.get("score", 0.0),
        "transformer_label": transformer.get("label", "unavailable"),
        "transformer_confidence": transformer.get("confidence", 0.0),
        "transformer_available": transformer.get("available", False),
        "transformer_model": transformer.get("model", ZERO_SHOT_MODEL_ID),
        "event_score": round(float(event_score), 4),
        "event_label": event_label,
    }


def headline_sort_key(item: dict[str, Any]) -> tuple[float, datetime]:
    published = parse_published_datetime(item.get("published_at", "")) or datetime.min
    magnitude = abs(float(item.get("price_signal", 0.0) or 0.0))
    return (magnitude, published)


def daily_signal_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "headline_count": 0,
            "event_score": 0.0,
            "event_label": "neutral",
            "tone_score": 0.0,
            "tone_label": "neutral",
        }
    event_score = float(np.mean([float(item.get("price_signal", 0.0) or 0.0) for item in items]))
    tone_score = float(np.mean([float(item.get("tone_score", 0.0) or 0.0) for item in items]))
    return {
        "headline_count": len(items),
        "event_score": round(event_score, 3),
        "event_label": "bullish" if event_score > 0.15 else "bearish" if event_score < -0.15 else "neutral",
        "tone_score": round(tone_score, 3),
        "tone_label": "positive" if tone_score > 0.15 else "negative" if tone_score < -0.15 else "neutral",
    }


def summarize_news_history(items: list[dict[str, Any]], days: int = NEWS_HISTORY_LOOKBACK_DAYS, top_k: int = NEWS_HISTORY_TOP_K) -> list[dict[str, Any]]:
    cutoff = datetime.utcnow().date() - timedelta(days=max(days - 1, 0))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        published = parse_published_datetime(item.get("published_at", ""))
        day_key = published.date().isoformat() if published else item.get("day")
        if not day_key:
            continue
        if date.fromisoformat(day_key) < cutoff:
            continue
        grouped.setdefault(day_key, []).append(item)
    rows: list[dict[str, Any]] = []
    for day_key, entries in sorted(grouped.items(), key=lambda item: item[0], reverse=True):
        ranked = sorted(entries, key=headline_sort_key, reverse=True)[:top_k]
        rows.append({"date": day_key, **daily_signal_summary(ranked), "items": ranked})
    return rows


def weighted_recent_event_signal(daily_history: list[dict[str, Any]], days: int = 7) -> dict[str, Any]:
    if not daily_history:
        return {
            "event_score": 0.0,
            "event_label": "neutral",
            "tone_score": 0.0,
            "tone_label": "neutral",
            "headline_days": 0,
        }
    weighted_event = 0.0
    weighted_tone = 0.0
    total_weight = 0.0
    for idx, daily in enumerate(daily_history[:days]):
        weight = 1.0 / (1.0 + idx * 0.65)
        weighted_event += float(daily.get("event_score", 0.0) or 0.0) * weight
        weighted_tone += float(daily.get("tone_score", 0.0) or 0.0) * weight
        total_weight += weight
    event_score = weighted_event / total_weight if total_weight else 0.0
    tone_score = weighted_tone / total_weight if total_weight else 0.0
    return {
        "event_score": round(event_score, 3),
        "event_label": "bullish" if event_score > 0.15 else "bearish" if event_score < -0.15 else "neutral",
        "tone_score": round(tone_score, 3),
        "tone_label": "positive" if tone_score > 0.15 else "negative" if tone_score < -0.15 else "neutral",
        "headline_days": min(len(daily_history), days),
    }


def load_news_history() -> dict[str, list[dict[str, Any]]]:
    path = news_history_path()
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    history: dict[str, list[dict[str, Any]]] = {}
    for commodity_id, items in payload.items():
        if commodity_id in COMMODITY_BY_ID and isinstance(items, list):
            history[commodity_id] = items
    return history


def write_news_history(history: dict[str, list[dict[str, Any]]]) -> None:
    write_json(news_history_path(), history)


def merge_news_history(commodity_id: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = state.STORE.news_history.get(commodity_id, [])
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for item in existing + items:
        title = item.get("title", "").strip()
        day_key = item.get("day")
        if not day_key:
            published = parse_published_datetime(item.get("published_at", ""))
            day_key = published.date().isoformat() if published else None
        if not title or not day_key:
            continue
        merged[(day_key, title.lower())] = {**item, "day": day_key}
    cutoff = datetime.utcnow().date() - timedelta(days=NEWS_HISTORY_LOOKBACK_DAYS)
    filtered = [item for item in merged.values() if date.fromisoformat(item["day"]) >= cutoff]
    filtered.sort(key=lambda item: (parse_published_datetime(item.get("published_at", "")) or datetime.min), reverse=True)
    state.STORE.news_history[commodity_id] = filtered
    write_news_history(state.STORE.news_history)
    return filtered


def is_title_relevant(commodity_id: str, title: str, relaxed: bool = False) -> bool:
    config = COMMODITY_NEWS_CONFIG.get(commodity_id, {})
    must_have = config.get("must_have", [])
    lowered = title.lower()
    if relaxed:
        relaxed_terms = must_have + [
            COMMODITY_BY_ID[commodity_id]["short_name"].lower(),
            COMMODITY_BY_ID[commodity_id]["category"].lower(),
            "oil",
            "energy",
            "refinery",
            "shipping",
        ]
        return any(term in lowered for term in relaxed_terms if term)
    return any(term in lowered for term in must_have) if must_have else True


def collect_news_history_items(
    commodity: dict[str, Any],
    commodity_id: str,
    now: datetime,
    queries: list[str],
    seen_titles: set[str],
    relaxed: bool = False,
) -> tuple[list[dict[str, Any]], int, list[float]]:
    items: list[dict[str, Any]] = []
    risk_terms = ("sanction", "opec", "conflict", "war", "tariff", "refinery", "shipping", "pipeline", "strike")
    risk_hits = 0
    transformer_scores: list[float] = []
    for query in queries:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        xml_payload = http_get_text(url)
        root = ET.fromstring(xml_payload)
        for item in root.findall(".//item")[:16]:
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            source = item.findtext("source", default="").strip()
            normalized_title = title.lower()
            if not title or not link or normalized_title in seen_titles:
                continue
            if not is_title_relevant(commodity_id, title, relaxed=relaxed):
                continue
            signal = combined_headline_signal(title, commodity)
            published = parse_published_datetime(pub_date)
            items.append(
                {
                    "title": title,
                    "link": link,
                    "source": source or "Google News",
                    "published_at": pub_date,
                    "day": published.date().isoformat() if published else now.date().isoformat(),
                    "sentiment": round(float(signal["tone_score"]), 3),
                    "tone_score": round(float(signal["tone_score"]), 3),
                    "tone_label": signal["tone_label"],
                    "price_signal": round(float(signal["event_score"]), 3),
                    "price_signal_label": signal["event_label"],
                    "rule_score": round(float(signal["rule_score"]), 3),
                    "transformer_score": round(float(signal["transformer_score"]), 3),
                    "transformer_label": signal["transformer_label"],
                }
            )
            seen_titles.add(normalized_title)
            risk_hits += sum(1 for term in risk_terms if term in normalized_title)
            if signal["transformer_available"]:
                transformer_scores.append(float(signal["transformer_score"]))
        if len(items) >= 24:
            break
    return items, risk_hits, transformer_scores


def fetch_news_for_commodity(commodity_id: str, force: bool = False) -> dict[str, Any]:
    commodity = COMMODITY_BY_ID[commodity_id]
    cached = state.STORE.news_cache.get(commodity_id)
    now = datetime.utcnow()
    if cached and not force:
        cached_at = datetime.fromisoformat(cached["fetched_at"].replace("Z", ""))
        if now - cached_at < NEWS_CACHE_TTL:
            return cached

    history_items: list[dict[str, Any]] = []
    risk_hits = 0
    transformer_scores: list[float] = []
    seen_titles: set[str] = set()
    try:
        strict_items, strict_risk_hits, strict_transformer_scores = collect_news_history_items(
            commodity,
            commodity_id,
            now,
            news_queries_for_commodity(commodity, days=NEWS_HISTORY_LOOKBACK_DAYS),
            seen_titles,
            relaxed=False,
        )
        history_items.extend(strict_items)
        risk_hits += strict_risk_hits
        transformer_scores.extend(strict_transformer_scores)
        if len(history_items) < 8:
            fallback_items, fallback_risk_hits, fallback_transformer_scores = collect_news_history_items(
                commodity,
                commodity_id,
                now,
                fallback_news_queries_for_commodity(commodity, days=NEWS_HISTORY_LOOKBACK_DAYS),
                seen_titles,
                relaxed=True,
            )
            history_items.extend(fallback_items)
            risk_hits += fallback_risk_hits
            transformer_scores.extend(fallback_transformer_scores)
    except Exception as exc:
        return {
            "commodity_id": commodity_id,
            "fetched_at": now.isoformat() + "Z",
            "risk_level": "unknown",
            "risk_score": 0.0,
            "items": [],
            "error": str(exc),
        }

    merged_history = merge_news_history(commodity_id, history_items)
    daily_history = summarize_news_history(merged_history)
    items = [headline for day in daily_history[:3] for headline in day.get("items", [])][:6]
    recent_signal = weighted_recent_event_signal(daily_history, days=7)
    risk_score = min((risk_hits / max(len(items), 1)) + abs(float(recent_signal["event_score"])), 3.0)
    risk_level = "high" if risk_score >= 1.5 else "elevated" if risk_score >= 0.75 else "calm"
    transformer_mean = round(float(np.mean(transformer_scores)), 3) if transformer_scores else 0.0
    transformer_applied = bool(transformer_scores) or zero_shot_classifier() is not None
    payload = {
        "commodity_id": commodity_id,
        "fetched_at": now.isoformat() + "Z",
        "risk_level": risk_level,
        "risk_score": round(risk_score, 3),
        "sentiment_score": round(float(recent_signal["tone_score"]), 3),
        "sentiment_label": recent_signal["tone_label"],
        "tone_score": round(float(recent_signal["tone_score"]), 3),
        "tone_label": recent_signal["tone_label"],
        "event_score": round(float(recent_signal["event_score"]), 3),
        "event_label": recent_signal["event_label"],
        "transformer_score": transformer_mean,
        "transformer_available": transformer_applied,
        "transformer_model": ZERO_SHOT_MODEL_ID,
        "transformer_error": state.ZERO_SHOT_LOAD_ERROR,
        "items": items,
        "daily_history": daily_history,
        "history_window_days": NEWS_HISTORY_LOOKBACK_DAYS,
        "history_top_k": NEWS_HISTORY_TOP_K,
        "headline_days": recent_signal["headline_days"],
        "transformer_applied": transformer_applied and len(history_items) > 0,
    }
    state.STORE.news_cache[commodity_id] = payload
    return payload
