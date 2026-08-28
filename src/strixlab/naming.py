"""Shared lexical grammars for StrixLab identifiers."""

from __future__ import annotations

import re

# POSIX-style environment variable name: a leading letter or underscore
# followed by letters, digits, or underscores.
ENV_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
ENV_NAME_RE = re.compile(ENV_NAME_PATTERN)
