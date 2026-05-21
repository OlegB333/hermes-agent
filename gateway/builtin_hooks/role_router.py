"""Built-in hook: multi-role auto-routing.

Classifies each incoming message against user-defined role definitions
and routes it to a role-specific session with a per-role system prompt.

Enabled via ``roles.enabled: true`` in config.yaml.

Sticky routing: after classification, the role "sticks" for
``roles.sticky_count`` consecutive messages from the same source,
avoiding redundant LLM calls.

Badge display: when the role switches, the hook sets a badge string
on the route decision so the gateway can prepend it to the response.
"""

import asyncio
import logging
import threading
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Config cache ──────────────────────────────────────────────────────────
_cached_roles_config: Optional[Dict[str, Any]] = None
_config_loaded = False


def _load_roles_config() -> Dict[str, Any]:
    """Load and cache the roles configuration block."""
    global _cached_roles_config, _config_loaded
    if _config_loaded:
        return _cached_roles_config or {}
    try:
        from hermes_cli.config import load_config
        config = load_config()
        _cached_roles_config = config.get("roles", {})
    except Exception:
        _cached_roles_config = {}
    _config_loaded = True
    return _cached_roles_config or {}


def reload_config() -> None:
    """Force config reload (called by /role command)."""
    global _config_loaded
    _config_loaded = False


# ── Sticky routing state ──────────────────────────────────────────────────
# Maps source_key -> (role_name, remaining_sticky_count)
_sticky_lock = threading.Lock()
_sticky_state: Dict[str, Tuple[str, int]] = {}


def _source_key(context: Dict[str, Any]) -> str:
    """Build a cache key from the source for sticky routing."""
    source = context.get("source")
    if source is None:
        return ""
    parts = [
        getattr(source, "platform", ""),
        getattr(source, "chat_id", ""),
        getattr(source, "user_id", ""),
    ]
    return ":".join(str(p) for p in parts)


def _get_sticky(key: str) -> Optional[str]:
    """Return the sticky role name if still valid, else None."""
    with _sticky_lock:
        entry = _sticky_state.get(key)
        if entry is None:
            return None
        role_name, remaining = entry
        if remaining <= 0:
            del _sticky_state[key]
            return None
        _sticky_state[key] = (role_name, remaining - 1)
        return role_name


def _set_sticky(key: str, role_name: str, count: int) -> None:
    """Set sticky routing for a source."""
    with _sticky_lock:
        _sticky_state[key] = (role_name, count)


def clear_sticky(key: str = None) -> None:
    """Clear sticky state. If key is None, clear all."""
    with _sticky_lock:
        if key is None:
            _sticky_state.clear()
        else:
            _sticky_state.pop(key, None)


# ── LLM classification ───────────────────────────────────────────────────

def _classify(prompt: str, model: str) -> str:
    """Synchronous LLM call for role classification."""
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(
        task="role_classification",
        model=model,
        messages=messages,
        max_tokens=20,
        temperature=0.0,
    )
    return extract_content_or_reasoning(response).strip().lower()


def _find_role(definitions: list, name: str) -> Optional[Dict[str, Any]]:
    """Find a role definition by name (case-insensitive)."""
    name_lower = name.lower()
    for role in definitions:
        if role.get("name", "").lower() == name_lower:
            return role
    return None


def _build_route(role: Dict[str, Any], previous_role: Optional[str],
                 roles_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the route decision dict from a matched role."""
    role_name = role.get("name", "unknown")
    emoji = role.get("emoji", "🤖")
    result = {
        "role_thread_id": f"role:{role_name}",
        "system_prompt": role.get("system_prompt"),
        "emoji": emoji,
        "role_name": role_name,
    }

    # Badge logic: inject badge when role switches
    show_badge = roles_config.get("show_badge", True)
    badge_display = roles_config.get("badge_display", "on_switch")
    if show_badge:
        if badge_display == "always" or (badge_display == "on_switch" and previous_role != role_name):
            badge_format = roles_config.get("badge_format", "emoji")
            if badge_format == "emoji":
                result["badge"] = emoji
            elif badge_format == "name":
                result["badge"] = f"[{role_name}]"
            else:
                result["badge"] = f"{emoji} {role_name}"

    return result


# ── Main handler ──────────────────────────────────────────────────────────

# Track last active role per source for badge "on_switch" logic
_last_role: Dict[str, str] = {}


async def handle(event_type: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Handle ``message:pre_route`` — classify and route to a role session."""
    if event_type != "message:pre_route":
        return None

    roles_config = _load_roles_config()
    if not roles_config.get("enabled", False):
        return None

    event = context.get("event")
    if not event or not event.text:
        return None

    definitions = roles_config.get("definitions", [])
    if not definitions:
        return None

    sk = _source_key(context)
    sticky_count = roles_config.get("sticky_count", 3)
    previous_role = _last_role.get(sk)

    # Check sticky cache first — avoid LLM call
    sticky_role_name = _get_sticky(sk)
    if sticky_role_name:
        role = _find_role(definitions, sticky_role_name)
        if role:
            logger.debug("Role routing: sticky hit %r for source %s", sticky_role_name, sk)
            return _build_route(role, previous_role, roles_config)

    # Classify via LLM
    role_list = "\n".join(
        f"{i + 1}. {r.get('name', 'unknown')} — {r.get('description', '')}"
        for i, r in enumerate(definitions)
    )
    prompt = (
        f"Classify the following user message into one of these roles:\n\n"
        f"{role_list}\n\n"
        f"Return ONLY the exact role name, nothing else.\n"
        f"Message: {event.text}"
    )

    model = roles_config.get("classifier_model", "google/gemini-2.0-flash")

    try:
        selected_name = await asyncio.to_thread(_classify, prompt, model)
        role = _find_role(definitions, selected_name)
        if role:
            role_name = role.get("name", selected_name)
            logger.info("Role routing: classified as %r for source %s", role_name, sk)
            # Set sticky for future messages
            if sticky_count > 0:
                _set_sticky(sk, role_name, sticky_count)
            _last_role[sk] = role_name
            return _build_route(role, previous_role, roles_config)
    except Exception as exc:
        logger.warning("Role classification failed: %s", exc)

    return None
