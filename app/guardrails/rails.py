
"""
NeMo Guardrails gateway for the production RAG system.

Configured rails:
    - off_topic
    - jailbreak
    - harmful_content
    - greeting
    - capabilities
    - farewell

Important:
    Guardrails must run BEFORE the RAG pipeline.
    If a rail fires, the caller must skip retrieval.
"""

from typing import Optional
import re

import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import (
    COLANG_CONTENT,
    YAML_CONTENT,
    RAIL_INDICATORS,
)


# ---------------------------------------------------------------------------
# Global NeMo instance
# ---------------------------------------------------------------------------

_rails: Optional[LLMRails] = None


# ---------------------------------------------------------------------------
# Fallback responses
# These match the responses defined in colang_rules.py
# ---------------------------------------------------------------------------

DEFAULT_RESPONSES = {
    "off_topic": (
        "I'm an Enterprise IT Assistant focused on Kubernetes, Intel hardware, "
        "and networking. I can't help with that — but ask me anything technical!"
    ),

    "jailbreak": (
        "I maintain consistent guidelines regardless of how I am prompted. "
        "I am here to help with Kubernetes, Intel, and networking. "
        "What can I help you with?"
    ),

    "harmful_content": (
        "I can't help with that. I'm here to assist with Kubernetes, Intel "
        "hardware, and enterprise networking questions."
    ),

    "greeting": (
        "Hello! I'm your Enterprise IT Assistant. I specialise in Kubernetes, "
        "Intel hardware, and enterprise networking. What can I help you with today?"
    ),

    "capabilities": (
        "I'm an Enterprise AI Assistant with deep expertise in: Kubernetes "
        "(deployment, scaling, networking, operators), Intel Hardware "
        "(CPUs, FPGAs, SRIOV, NICs), Enterprise Networking "
        "(SDN, VLANs, BGP, routing). Ask me anything in these areas!"
    ),

    "farewell": (
        "Goodbye! Feel free to return whenever you have more enterprise IT "
        "questions. Have a great day!"
    ),
}

HARMFUL_INPUT_PATTERNS = (
    r"\bhow\s+to\s+(?:kill|beat|hit|attack|harm)\s+(?:a|the)\s+(?:person|man|woman|someone|somebody)\b",
    r"\bhow\s+(?:do|can)\s+i\s+(?:hurt|beat|hit|attack|harm)\s+(?:someone|somebody|a person|a man|a woman)\b",
    r"\b(?:ways|methods|instructions)\s+to\s+(?:kill|beat|hit|attack|harm)\s+(?:someone|somebody|a person)\b",
    r"\bhow\s+to\s+make\s+(?:a\s+)?bomb\b",
    r"\bhow\s+to\s+make\s+poison\b",
    r"\bhelp\s+me\s+plan\s+an\s+attack\b",
    r"\bhow\s+to\s+make\s+(?:a\s+)?weapon\b",
    r"\bhow\s+to\s+make\s+explosives?\b",
    r"\bhow\s+to\s+create\s+(?:malware|a\s+virus)\b",
)


# ---------------------------------------------------------------------------
# Normalize text
# ---------------------------------------------------------------------------

def _normalize_text(value: object) -> str:
    """
    Normalize text before comparing NeMo output
    with RAIL_INDICATORS.
    """

    if value is None:
        return ""

    text = str(value)

    # Normalize Unicode punctuation.
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

    # Collapse multiple spaces/newlines.
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


# ---------------------------------------------------------------------------
# Extract NeMo response
# ---------------------------------------------------------------------------

def _extract_content(result: object) -> str:
    """
    Extract assistant content from NeMo's response.

    Expected response:

        {
            "role": "assistant",
            "content": "..."
        }
    """

    if result is None:
        return ""

    if isinstance(result, dict):
        content = result.get("content", "")
        return str(content).strip()

    return str(result).strip()


# ---------------------------------------------------------------------------
# Detect which rail fired
# ---------------------------------------------------------------------------

def _detect_rail(content: str) -> Optional[str]:
    """
    Detect the configured rail from NeMo's response.

    Returns:
        off_topic
        jailbreak
        harmful_content
        greeting
        capabilities
        farewell
        None
    """

    normalized_content = _normalize_text(content)

    if not normalized_content:
        return None

    # Check longer/more-specific indicators first.
    indicators = sorted(
        RAIL_INDICATORS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for indicator, rail_name in indicators:

        normalized_indicator = _normalize_text(indicator)

        if not normalized_indicator:
            continue

        if normalized_indicator in normalized_content:
            return rail_name

    return None


def _detect_harmful_input(message: str) -> bool:
    """Catch explicit harmful requests before NeMo can paraphrase its reply."""

    normalized_message = _normalize_text(message)

    return any(
        re.search(pattern, normalized_message) is not None
        for pattern in HARMFUL_INPUT_PATTERNS
    )


# ---------------------------------------------------------------------------
# Initialize NeMo Guardrails
# ---------------------------------------------------------------------------

def initialize_rails() -> None:
    """
    Initialize NeMo Guardrails.

    Uses:
        Groq
        openai/gpt-oss-20b
    """

    global _rails

    # Prevent duplicate initialization.
    if _rails is not None:
        logfire.debug(
            "🛡️ NeMo Guardrails already initialized."
        )
        return

    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )

    logfire.info(
        "🛡️ Initializing NeMo Guardrails "
        "(openai/gpt-oss-20b)..."
    )

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT,
    )

    _rails = LLMRails(
        config,
        llm=guard_llm,
    )

    logfire.info(
        "🛡️ NeMo Guardrails initialized successfully."
    )


# ---------------------------------------------------------------------------
# Main guard function
# ---------------------------------------------------------------------------

def guard(message: str) -> tuple[bool, Optional[str]]:
    """
    Run a user message through NeMo Guardrails.

    Returns:

        (True, response)
            A guardrail fired.
            The caller MUST return this response immediately.
            RAG/retrieval must NOT run.

        (False, None)
            No configured rail fired.
            The query can continue to LangGraph/RAG.
    """

    if not isinstance(message, str):
        logfire.warning(
            "🛡️ Guardrails received a non-string message."
        )
        return False, None

    message = message.strip()

    if not message:
        logfire.warning(
            "🛡️ Guardrails received an empty message."
        )
        return False, None

    if _detect_harmful_input(message):
        logfire.warning(
            f"🛑 Harmful request blocked before retrieval | query='{message[:100]}'"
        )
        return True, DEFAULT_RESPONSES["harmful_content"]

    if _rails is None:
        logfire.error(
            "❌ Guardrails are not initialized."
        )
        return False, None

    with logfire.span("🛡️ Guardrails Check"):

        logfire.debug(
            f"Checking query: {message[:200]}"
        )

        try:

            # -------------------------------------------------------------
            # Run NeMo
            # -------------------------------------------------------------

            result = _rails.generate(
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            )

            # -------------------------------------------------------------
            # Debug raw response
            # -------------------------------------------------------------

            logfire.debug(
                f"NeMo raw result: {result}"
            )

            # -------------------------------------------------------------
            # Extract content
            # -------------------------------------------------------------

            content = _extract_content(result)

            logfire.debug(
                f"NeMo content: {content}"
            )

            # -------------------------------------------------------------
            # Detect rail
            # -------------------------------------------------------------

            rail_name = _detect_rail(content)

            if rail_name:

                response = content

                # Safety fallback.
                if not response:
                    response = DEFAULT_RESPONSES.get(
                        rail_name,
                        "I can't help with that request.",
                    )

                logfire.warning(
                    "🛑 Guardrail fired | "
                    f"rail={rail_name} | "
                    f"query='{message[:100]}'"
                )

                logfire.info(
                    f"🛑 RAG retrieval SKIPPED | rail={rail_name}"
                )

                return True, response

            # -------------------------------------------------------------
            # Clean query
            # -------------------------------------------------------------

            logfire.info(
                f"✅ Guardrails passed | query='{message[:100]}'"
            )

            return False, None

        except Exception as exc:

            logfire.exception(
                f"❌ Guardrails execution failed: {exc}"
            )

            return False, None


# ---------------------------------------------------------------------------
# Optional testing helper
# ---------------------------------------------------------------------------

def get_rail_name(message: str) -> Optional[str]:
    """
    Test which rail is detected without running the complete RAG pipeline.

    Example:

        get_rail_name("how to kill a person")

    Expected:

        harmful_content
    """

    if _rails is None:
        initialize_rails()

    if not message or not message.strip():
        return None

    try:

        result = _rails.generate(
            messages=[
                {
                    "role": "user",
                    "content": message.strip(),
                }
            ]
        )

        content = _extract_content(result)

        return _detect_rail(content)

    except Exception as exc:

        logfire.exception(
            f"❌ Rail detection failed: {exc}"
        )

        return None