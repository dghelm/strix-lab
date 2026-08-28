from __future__ import annotations

import getpass
from pathlib import Path

import pytest

from strixlab import paths


def test_explicit_home_wins_without_creating_it(tmp_path: Path) -> None:
    target = tmp_path / "explicit"

    resolved = paths.resolve_home(target, environ={paths.STRIXLAB_HOME_ENV: "/ignored"})

    assert resolved == target
    assert not target.exists()


def test_environment_home_is_used(tmp_path: Path) -> None:
    target = tmp_path / "environment"

    assert paths.resolve_home(environ={paths.STRIXLAB_HOME_ENV: str(target)}) == target


def test_platform_default_is_used_when_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = Path("/var/lib/strixlab-test")
    monkeypatch.setattr(paths, "user_data_path", lambda *_args, **_kwargs: expected)

    assert paths.resolve_home(environ={}) == expected


@pytest.mark.parametrize("value", ["", "   ", "relative/path", "bad\x00path"])
def test_unsafe_home_values_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        paths.resolve_home(value, environ={})


def test_empty_environment_value_is_not_treated_as_unset() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        paths.resolve_home(environ={paths.STRIXLAB_HOME_ENV: ""})


def test_current_and_named_user_expansion() -> None:
    current = Path.home()
    user = getpass.getuser()

    assert paths.resolve_home("~/strixlab", environ={}) == current / "strixlab"
    assert paths.resolve_home(f"~{user}/strixlab", environ={}) == current / "strixlab"


def test_unknown_named_user_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot expand"):
        paths.resolve_home("~strixlab-user-that-does-not-exist/state", environ={})
