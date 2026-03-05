"""Compatibility wrapper for AI backend profile DB tools.

This module is kept for external imports and tests.
Implementation lives under `nsys_ai.ai.backend.profile_db_tool`.
"""

from .ai.backend.profile_db_tool import *  # noqa: F401,F403
from .ai.backend import profile_db_tool as _impl

# Preserve test/compat access to module-level cache internals.
_schema_cache = _impl._schema_cache
