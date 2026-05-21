"""Built-in hook: multi-role auto-routing.

Classifies each incoming message against user-defined role definitions
and routes it to a role-specific session with a per-role system prompt.

Enabled via ``roles.enabled: true`` in config.yaml.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Cache for roles config — avoid re-reading YAML on every message.
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

    # Build classification prompt
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
        selected_role = await asyncio.to_thread(_classify, prompt, model)

        for role in definitions:
            if role.get("name", "").lower() == selected_role:
                logger.info("Role routing: selected role %r for message", selected_role)
                return {
                    "role_thread_id": f"role:{selected_role}",
                    "system_prompt": role.get("system_prompt"),
                    "emoji": role.get("emoji", "🤖"),
                }
    except Exception as exc:
        logger.warning("Role classification failed: %s", exc)

    return None
