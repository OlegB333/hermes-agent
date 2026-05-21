---
name: role-manager
description: Manage multi-role auto-routing definitions
---

# Role Manager

This skill teaches the agent how to manage the `roles` configuration for multi-role auto-routing.

## Instructions
When the user asks to add or modify a role, update the `roles.definitions` array in their `~/.hermes/config.yaml`.
Set `roles.enabled = true` to enable routing.

Each role definition should have:
- `name`: Short, unique identifier (e.g. "coder", "writer")
- `description`: What kinds of messages should be routed to this role
- `system_prompt`: The specific instructions for this role
- `emoji`: An emoji for UI display (e.g. "👨‍💻", "📝")

Example structure:
```yaml
roles:
  enabled: true
  definitions:
    - name: coder
      description: Write and review python code
      system_prompt: You are an expert Python engineer...
      emoji: 👨‍💻
```
