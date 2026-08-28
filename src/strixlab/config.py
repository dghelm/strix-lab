"""Pure YAML parsing and explicit trusted-value resolution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

from strixlab.naming import ENV_NAME_RE


class DuplicateKeyError(ConstructorError):
    """Raised when YAML contains a duplicate mapping key."""


class EnvironmentResolutionError(ValueError):
    """Raised when trusted environment interpolation cannot be resolved."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise DuplicateKeyError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def parse_manifest_text(text: str) -> dict[str, Any]:
    """Parse one manifest without performing resolution or other side effects."""

    value = yaml.load(text, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError("manifest root must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError("manifest root keys must be strings")
    return value


def read_manifest(path: Path) -> dict[str, Any]:
    """Read and parse a UTF-8 YAML manifest without resolving its values."""

    return parse_manifest_text(path.read_text(encoding="utf-8"))


def _scan_environment_tokens(value: str) -> Iterator[tuple[str, str | None, bool]]:
    """Tokenize the ``${NAME}`` interpolation grammar in one place.

    Yields ``(kind, payload, escaped)`` tokens so every caller shares one
    definition of both escaping (``$${NAME}`` is a literal ``${NAME}``) and
    naming (validity is decided here via :data:`ENV_NAME_RE`):

    - ``text``: literal run; ``payload`` is the text.
    - ``escape``: a ``$${NAME}`` standing for the literal ``${NAME}``.
    - ``reference``: a live ``${NAME}`` to resolve; ``payload`` is the name.
    - ``unterminated``: a ``${``/``$${`` with no closing brace; ``payload`` None.
    - ``invalid``: a token whose brace content is not a valid name; ``payload``
      is that raw content.

    ``escaped`` marks whether the token began with ``$${``. Malformed tokens are
    reported rather than raised so read-only callers can skip them while
    resolving callers can raise their own contextual errors.
    """

    index = 0
    length = len(value)
    text_start = 0
    while index < length:
        escaped = value.startswith("$${", index)
        if not (escaped or value.startswith("${", index)):
            index += 1
            continue
        if index > text_start:
            yield "text", value[text_start:index], False
        prefix = 3 if escaped else 2
        closing = value.find("}", index + prefix)
        if closing < 0:
            yield "unterminated", None, escaped
            return
        name = value[index + prefix : closing]
        if ENV_NAME_RE.fullmatch(name) is None:
            yield "invalid", name, escaped
        else:
            yield ("escape" if escaped else "reference"), name, escaped
        index = closing + 1
        text_start = index
    if length > text_start:
        yield "text", value[text_start:length], False


def iter_environment_references(value: str) -> Iterator[str]:
    """Yield each environment name referenced by a live ``${NAME}``.

    Escaped ``$${NAME}`` sequences and malformed tokens are not references and
    are skipped. This is a read-only scan for callers that must inspect raw,
    not-yet-resolved values; validation belongs to :func:`resolve_environment`.
    """

    for kind, payload, _escaped in _scan_environment_tokens(value):
        if kind == "reference":
            assert payload is not None
            yield payload


def _resolve_string(value: str, environ: Mapping[str, str]) -> str:
    if "$" not in value:
        return value

    output: list[str] = []
    for kind, payload, escaped in _scan_environment_tokens(value):
        if kind == "text":
            assert payload is not None
            output.append(payload)
        elif kind == "escape":
            output.append("${" + str(payload) + "}")
        elif kind == "reference":
            name = str(payload)
            if name not in environ:
                raise EnvironmentResolutionError(f"environment variable is not set: {name}")
            replacement = environ[name]
            if not isinstance(replacement, str):
                raise EnvironmentResolutionError(f"environment value is not a string: {name}")
            output.append(_literal_environment_value(name, replacement))
        elif kind == "unterminated":
            raise EnvironmentResolutionError(
                "unterminated escaped environment token"
                if escaped
                else "unterminated environment token"
            )
        else:  # invalid
            raise EnvironmentResolutionError(
                f"invalid escaped environment token: {payload!r}"
                if escaped
                else f"invalid environment token: {payload!r}"
            )
    return "".join(output)


def _literal_environment_value(name: str, value: str) -> str:
    """Reject unresolved tokens introduced by an environment replacement."""

    if "$" not in value:
        return value

    output: list[str] = []
    for kind, payload, escaped in _scan_environment_tokens(value):
        if kind == "text":
            assert payload is not None
            output.append(payload)
        elif kind == "escape":
            output.append("${" + str(payload) + "}")
        elif kind == "unterminated" and escaped:
            raise EnvironmentResolutionError(
                f"unterminated escaped token in environment value: {name}"
            )
        elif kind == "invalid" and escaped:
            raise EnvironmentResolutionError(
                f"invalid escaped token in environment value {name}: {payload!r}"
            )
        else:  # a live ${NAME}, or a malformed non-escaped ${...}
            raise EnvironmentResolutionError(
                f"environment value introduces an unresolved token: {name}"
            )
    return "".join(output)


def resolve_environment(value: Any, environ: Mapping[str, str]) -> Any:
    """Resolve environment tokens in trusted values only.

    Mapping keys are intentionally never transformed. Callers must opt in to
    this function after parsing; imported candidate data should remain raw.
    """

    if isinstance(value, str):
        return _resolve_string(value, environ)
    if isinstance(value, list):
        return [resolve_environment(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: resolve_environment(item, environ) for key, item in value.items()}
    return value
