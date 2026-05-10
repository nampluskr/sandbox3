# src/config.py

import os
import re
import yaml


def resolve_vars(config: dict) -> dict:
    flat = {k: v for k, v in config.items() if isinstance(v, str)}

    def interpolate(value: str, context: dict) -> str:
        # Repeat substitution until no more changes (max 10 to prevent circular refs)
        for _ in range(10):
            new_value = re.sub(
                r'\$\{(\w+)\}',
                lambda m: context.get(m.group(1), m.group(0)),
                value
            )
            if new_value == value:
                break
            value = new_value

        # Raise if any unresolved variables remain
        unresolved = re.findall(r'\$\{(\w+)\}', value)
        if unresolved:
            raise KeyError(f"Undefined variable(s): {unresolved}")
        return value

    # Fully resolve all top-level string variables first
    resolved_flat: dict[str, str] = {}
    for key in flat:
        resolved_flat[key] = interpolate(flat[key], flat)

    # Apply resolved context to the entire config (including nested dicts)
    resolved: dict = {}
    for k, v in config.items():
        if isinstance(v, str):
            resolved[k] = resolved_flat.get(k, interpolate(v, resolved_flat))
        elif isinstance(v, dict):
            resolved[k] = {
                dk: interpolate(dv, resolved_flat) if isinstance(dv, str) else dv
                for dk, dv in v.items()
            }
        else:
            resolved[k] = v

    return resolved


def load_config(config_path):
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    return resolve_vars(raw)
