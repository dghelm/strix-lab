from __future__ import annotations

import pytest

from strixlab.manifests import SourceLockV1
from strixlab.source_identity import (
    PatchIdentity,
    SubmoduleIdentity,
    candidate_id,
    content_tree_id,
    length_frame,
    locator_class,
    preparation_identity,
    request_digest,
)


def test_length_frame_has_a_stable_binary_vector() -> None:
    framed = length_frame("d", (("x", b"abc"),))

    assert framed.hex() == (
        "73747269786c61622d6c662d76310000000001640000000100000001780000000000000003616263"
    )


def test_source_identity_chain_has_stable_vectors() -> None:
    lock = SourceLockV1.model_validate(
        {
            "schema_version": 1,
            "id": "fixture",
            "kind": "git",
            "url": "https://example.invalid/repo.git",
            "commit": "0123456789abcdef0123456789abcdef01234567",
            "branch_hint": "main",
            "submodules": True,
            "adapter": "llama_cpp",
            "allowed_dirty_state": False,
        }
    )
    patches = (
        PatchIdentity(1, 3, "00" * 32),
        PatchIdentity(2, 5, "ff" * 32),
    )

    request = request_digest(lock, resolved_locator=lock.url, patches=patches)
    preparation, attempt = preparation_identity(
        lock.id,
        request,
        bytes.fromhex("0123456789abcdef0123456789abcdef"),
    )
    content = content_tree_id(
        "89abcdef0123456789abcdef0123456789abcdef",
        patches=patches,
        submodules=(
            SubmoduleIdentity("deps/a", "11" * 20),
            SubmoduleIdentity("vendor/b", "22" * 20),
        ),
    )

    assert request == "37ac0a8a042a73e55e14a76b49442201fee6bb359f0e7e48e5e57cf16d92b542"
    assert preparation == "prep-fixture-6ed9395f6ff1c60f6018049f"
    assert attempt == "6ed9395f6ff1c60f6018049f4ba3f2a716b3fe1f82d22c933e3cebfb4b026460"
    assert content == (
        "content-tree-sha256:1124dc607f3af7df541be33f7d8e77170766a46ec3763fb0793e45b6ff2437ce"
    )
    assert candidate_id(lock.commit, content, submodules=True) == (
        "candidate-sha256:02920bea65ea5b72b90aa2fe6e8337f537c6d0a9f7a29fe902098f19903d8931"
    )


def test_identity_input_boundaries_are_explicit() -> None:
    assert locator_class("file:///repo") == "file"
    assert locator_class("ssh://git@example.test/repo") == "ssh"
    assert locator_class("git@example.test:repo") == "scp-ssh"
    with pytest.raises(ValueError, match="exactly 16 bytes"):
        preparation_identity("fixture", "0" * 64, b"short")
