from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from strixlab.config import (
    DuplicateKeyError,
    EnvironmentResolutionError,
    iter_environment_references,
    parse_manifest_text,
    read_manifest,
    resolve_environment,
)


def test_parse_manifest_is_pure() -> None:
    parsed = parse_manifest_text("schema_version: 1\npath: ${MODELS}/model.gguf\n")

    assert parsed == {"schema_version": 1, "path": "${MODELS}/model.gguf"}


@pytest.mark.parametrize(
    "text",
    [
        "key: first\nkey: second\n",
        "outer:\n  key: first\n  key: second\n",
    ],
)
def test_duplicate_keys_are_rejected_at_every_depth(text: str) -> None:
    with pytest.raises(DuplicateKeyError, match="duplicate key"):
        parse_manifest_text(text)


@pytest.mark.parametrize("text", ["", "- one\n- two\n", "plain scalar\n"])
def test_manifest_root_must_be_a_mapping(text: str) -> None:
    with pytest.raises(ValueError, match="root must be a mapping"):
        parse_manifest_text(text)


def test_manifest_root_keys_must_be_strings() -> None:
    with pytest.raises(ValueError, match="root keys must be strings"):
        parse_manifest_text("1: value\n")


def test_invalid_yaml_is_reported() -> None:
    with pytest.raises(yaml.YAMLError):
        parse_manifest_text("value: [unterminated\n")


def test_read_manifest_uses_utf8(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("name: Strix Hélo\n", encoding="utf-8")

    assert read_manifest(path) == {"name": "Strix Hélo"}


def test_explicit_environment_resolution_is_recursive_and_does_not_change_keys() -> None:
    value = {
        "${KEY}": "${ROOT}/one-${EMPTY}",
        "nested": ["${ROOT}", {"value": "prefix-${ROOT}-suffix"}],
        "number": 3,
    }

    resolved = resolve_environment(value, {"ROOT": "/models", "EMPTY": ""})

    assert resolved == {
        "${KEY}": "/models/one-",
        "nested": ["/models", {"value": "prefix-/models-suffix"}],
        "number": 3,
    }


def test_environment_token_can_be_escaped() -> None:
    assert resolve_environment("$${HOME}/${HOME}", {"HOME": "/tmp/home"}) == ("${HOME}//tmp/home")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("${MISSING}", "not set"),
        ("${BROKEN", "unterminated"),
        ("$${BROKEN", "unterminated escaped"),
        ("${NAME:-fallback}", "invalid environment token"),
        ("${1NAME}", "invalid environment token"),
        ("$${NAME:-fallback}", "invalid escaped environment token"),
    ],
)
def test_invalid_environment_resolution_fails(value: str, message: str) -> None:
    with pytest.raises(EnvironmentResolutionError, match=message):
        resolve_environment(value, {})


def test_environment_values_must_be_strings() -> None:
    with pytest.raises(EnvironmentResolutionError, match="not a string"):
        resolve_environment("${VALUE}", {"VALUE": 3})  # type: ignore[dict-item]


def test_environment_replacement_cannot_introduce_an_unresolved_token() -> None:
    with pytest.raises(EnvironmentResolutionError, match="introduces an unresolved token"):
        resolve_environment("${FIRST}", {"FIRST": "${SECOND}", "SECOND": "expanded"})


def test_environment_replacement_can_explicitly_escape_a_literal_token() -> None:
    assert resolve_environment("${FIRST}", {"FIRST": "$${SECOND}"}) == "${SECOND}"


def test_iter_environment_references_shares_escape_and_naming_rules() -> None:
    references = list(iter_environment_references("${ROOT}/$${LITERAL}/${SECOND}-${1BAD}-${BROKEN"))

    assert references == ["ROOT", "SECOND"]


def test_environment_replacement_may_carry_escaped_literals() -> None:
    assert resolve_environment("${A}", {"A": "pre-$${X}-post"}) == "pre-${X}-post"


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("$${BROKEN", "unterminated escaped token in environment value"),
        ("$${1BAD}", "invalid escaped token in environment value"),
    ],
)
def test_environment_replacement_escaped_token_errors(replacement: str, message: str) -> None:
    with pytest.raises(EnvironmentResolutionError, match=message):
        resolve_environment("${A}", {"A": replacement})
