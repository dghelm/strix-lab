from __future__ import annotations

import pytest

from strixlab.secret_policy import RedactionContext, UnsafeOutputError, secret_values


def test_known_nonsecret_session_flag_does_not_weaken_short_token_scanning() -> None:
    environ = {
        "API_TOKEN": "abc1234",
        "CLAUDE_CODE_CHILD_SESSION": "1",
        "PLAIN": "not-sensitive",
    }

    assert secret_values(environ) == ("abc1234",)

    context = RedactionContext.from_environ(environ)
    context.assert_text_safe("benign output containing 1")
    with pytest.raises(UnsafeOutputError):
        context.assert_text_safe("leaked abc1234")


@pytest.mark.parametrize("secret_name", ["API_TOKEN", "SESSION", "XDG_SESSION_CLASS_TOKEN"])
def test_session_class_exception_preserves_same_value_secrets(secret_name: str) -> None:
    context = RedactionContext.from_environ({"XDG_SESSION_CLASS": "user", secret_name: "user"})

    with pytest.raises(UnsafeOutputError):
        context.assert_text_safe("bg_user_message.xml")
