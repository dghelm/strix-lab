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
