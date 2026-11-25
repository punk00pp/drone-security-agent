"""
analyzer.py — Frame analysis: rule-based (offline) and LLM-backed.

Two implementations share a common analyze(frame, telemetry) -> AnalyzedFrame interface.

RuleBasedAnalyzer  — regex/keyword dictionaries, no API key, used in all tests.
LLMAnalyzer        — sends a strict JSON prompt to the configured LLM provider,
                     validates with Pydantic, falls back to RuleBasedAnalyzer on error.
"""

from __future__ import annotations
import os
import json
import logging
import re
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from drone_agent.models import AnalyzedFrame, Detection, Frame, Telemetry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern tables used by RuleBasedAnalyzer
# ---------------------------------------------------------------------------

_LABEL_PATTERNS = [
    (r"\b(cat|dog|bird|fox|raccoon|deer|rabbit)\b", "animal"),
    (r"\b(person|man|woman|individual|pedestrian|intruder|guard|driver|cyclist|jogger)\b", "person"),
    (r"\b(vehicle|car|truck|van|suv|pickup|sedan|motorcycle|bicycle|bus|f.?150|f150|camry|civic)\b", "vehicle"),
    (r"\bsomething\b", "unknown"),
]

_SUBTYPE_PATTERNS = [
    (r"\b(f.?150|pickup|truck|lorry|garbage.?truck|delivery.?truck)\b", "truck"),
    (r"\b(van|delivery.?van|minivan)\b", "van"),
    (r"\b(suv|sport.?utility)\b", "suv"),
    (r"\b(sedan|car|camry|civic)\b", "car"),
    (r"\b(motorcycle|motorbike)\b", "motorcycle"),
]

_COLOR_PATTERNS = [
    (r"\bblue\b", "blue"), (r"\bred\b", "red"), (r"\bwhite\b", "white"),
    (r"\bblack\b", "black"), (r"\bsilver\b", "silver"), (r"\bgrey\b|\bgray\b", "grey"),
    (r"\bgreen\b", "green"), (r"\byellow\b", "yellow"),
]

_MAKE_PATTERNS = [
    (r"\bford\b", "ford"), (r"\btoyota\b", "toyota"), (r"\bhonda\b", "honda"),
    (r"\bchevrolet\b|\bchevy\b", "chevrolet"),
]

_MODEL_PATTERNS = [
    (r"\bf.?150\b", "f150"), (r"\bcamry\b", "camry"), (r"\bcivic\b", "civic"),
]

_ACTIVITY_PATTERNS = [
    (r"\bclimbing\b|\bclimbs\b", "climbing"),
    (r"\bjumping\b|\bjumps\b", "jumping"),
    (r"\brunning\b|\bjogging\b", "running"),
    (r"\bwalking\b|\bwalks\b", "walking"),
    (r"\bstanding\b|\bwaiting\b", "standing"),
    (r"\bparked\b|\bparking\b", "parked"),
    (r"\bentering\b|\benters\b|\bentered\b", "entering"),
    (r"\bleaving\b|\bleaves\b|\bdeparting\b", "leaving"),
]


def _first_match(text: str, patterns: list) -> str | None:
    for pat, val in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return val
    return None


# ---------------------------------------------------------------------------
# RuleBasedAnalyzer
# ---------------------------------------------------------------------------

class RuleBasedAnalyzer:
    """Keyword/regex analyzer — deterministic, offline, no API key required."""

    def analyze(self, frame: Frame, telemetry: Telemetry) -> AnalyzedFrame:
        detections = self._extract(frame.description)
        summary = self._summary(detections, telemetry)
        return AnalyzedFrame(frame=frame, telemetry=telemetry,
                             detections=detections, summary=summary)

    def _extract(self, text: str) -> list[Detection]:
        detections: list[Detection] = []
        for pat, label in _LABEL_PATTERNS:
            if not re.search(pat, text, re.IGNORECASE):
                continue
            attrs: dict = {}
            subtype = None
            if label == "vehicle":
                subtype = _first_match(text, _SUBTYPE_PATTERNS)
                for key, table in [("color", _COLOR_PATTERNS), ("make", _MAKE_PATTERNS), ("model", _MODEL_PATTERNS)]:
                    val = _first_match(text, table)
                    if val:
                        attrs[key] = val
            activity = _first_match(text, _ACTIVITY_PATTERNS)
            confidence = 1.0 if (subtype or attrs) else 0.6
            detections.append(Detection(label=label, subtype=subtype, attributes=attrs,
                                        confidence=confidence, activity=activity))
        return detections or [Detection(label="unknown", confidence=0.3)]

    def _summary(self, detections: list[Detection], telemetry: Telemetry) -> str:
        parts = []
        for d in detections:
            if d.label == "unknown":
                continue
            color = d.attributes.get("color", "")
            make = d.attributes.get("make", "").title()
            model = d.attributes.get("model", "").upper()
            sub = d.subtype or ""
            if d.label == "vehicle":
                name = " ".join(p for p in [color, make, model, sub] if p) or "Vehicle"
            else:
                name = d.label.title()
            parts.append(f"{name} {d.activity or 'detected'}")
        body = "; ".join(parts) if parts else "Activity"
        return f"{body} at {telemetry.location}, {telemetry.timestamp.strftime('%H:%M')}."


# ---------------------------------------------------------------------------
# LLM Analyzer
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
You are a computer-vision system for a property security drone.
Extract ALL entities from the frame description and return ONLY valid JSON — no markdown.

Schema:
{{
  "detections": [
    {{"label":"<person|vehicle|animal|unknown>","subtype":"<truck|van|car|suv|null>",
      "attributes":{{"color":"<or null>","make":"<or null>","model":"<or null>"}},
      "confidence":<0-1>,"activity":"<parked|entering|leaving|standing|walking|running|climbing|null>"}}
  ],
  "summary":"<one sentence: subject + activity + location + time>"
}}

Frame: {description}
Location: {location}
Time: {time}
"""


def _build_llm():
    """Build the Groq LLM client via LangChain's OpenAI-compatible interface."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv('API_KEY'),
        model="qwen/qwen3.8-27b",
        temperature=0,
        max_tokens=512,
    )


def _llm_call(description: str, location: str, time_str: str) -> str:
    from langchain_core.messages import HumanMessage
    prompt = _LLM_PROMPT.format(description=description, location=location, time=time_str)
    return _build_llm().invoke([HumanMessage(content=prompt)]).content


class LLMAnalyzer:
    """LLM-backed analyzer with Pydantic validation and RuleBasedAnalyzer fallback."""

    def __init__(self):
        self._fallback = RuleBasedAnalyzer()

    def analyze(self, frame: Frame, telemetry: Telemetry) -> AnalyzedFrame:
        try:
            return self._llm_analyze(frame, telemetry)
        except Exception as exc:
            logger.warning("LLMAnalyzer failed for frame %d: %s — using fallback", frame.frame_id, exc)
            return self._fallback.analyze(frame, telemetry)

    def _llm_analyze(self, frame: Frame, telemetry: Telemetry) -> AnalyzedFrame:
        time_str = telemetry.timestamp.strftime("%H:%M")
        raw = _llm_call(frame.description, telemetry.location, time_str)
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        data = json.loads(clean)

        detections = []
        for d in data.get("detections", []):
            attrs = {k: v for k, v in (d.get("attributes") or {}).items() if v}
            detections.append(Detection(
                label=d.get("label", "unknown"),
                subtype=d.get("subtype") or None,
                attributes=attrs,
                confidence=float(d.get("confidence", 0.8)),
                activity=d.get("activity") or None,
            ))

        summary = data.get("summary") or self._fallback._summary(detections, telemetry)
        return AnalyzedFrame(frame=frame, telemetry=telemetry,
                             detections=detections or [Detection(label="unknown", confidence=0.3)],
                             summary=summary)
