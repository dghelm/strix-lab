"""Versioned, length-framed identities for prepared source candidates."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from strixlab.manifests import SourceLockV1

_PREFIX = b"strixlab-lf-v1\0"


@dataclass(frozen=True, slots=True)
class PatchIdentity:
    order: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SubmoduleIdentity:
    path: str
    commit: str


def _u32(value: int) -> bytes:
    return struct.pack(">I", value)


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _field(name: str, value: bytes) -> bytes:
    encoded_name = name.encode("ascii")
    return _u32(len(encoded_name)) + encoded_name + _u64(len(value)) + value


def length_frame(domain: str, fields: tuple[tuple[str, bytes], ...]) -> bytes:
    encoded_domain = domain.encode("utf-8")
    return (
        _PREFIX
        + _u32(len(encoded_domain))
        + encoded_domain
        + _u32(len(fields))
        + b"".join(_field(name, value) for name, value in fields)
    )


def _optional(value: str | None) -> bytes:
    return b"\x00" if value is None else b"\x01" + value.encode("utf-8")


def _items(values: tuple[bytes, ...]) -> bytes:
    return _u32(len(values)) + b"".join(_u64(len(value)) + value for value in values)


def _sha256_frame(domain: str, fields: tuple[tuple[str, bytes], ...]) -> str:
    return hashlib.sha256(length_frame(domain, fields)).hexdigest()


def locator_class(locator: str) -> str:
    if locator.startswith("/"):
        return "local-path"
    if locator.startswith("file:"):
        return "file"
    if locator.startswith("https:"):
        return "https"
    if locator.startswith("ssh:"):
        return "ssh"
    return "scp-ssh"


def request_digest(
    lock: SourceLockV1,
    *,
    resolved_locator: str,
    patches: tuple[PatchIdentity, ...],
) -> str:
    patch_frames = tuple(
        length_frame(
            "strixlab.source.patch.v1",
            (
                ("order", _u32(patch.order)),
                ("size_bytes", _u64(patch.size_bytes)),
                ("sha256", bytes.fromhex(patch.sha256)),
            ),
        )
        for patch in patches
    )
    return _sha256_frame(
        "strixlab.source.request.v1",
        (
            ("schema_version", _u32(1)),
            ("source_id", lock.id.encode("utf-8")),
            ("kind", b"git"),
            ("locator_class", locator_class(resolved_locator).encode("ascii")),
            ("resolved_locator", resolved_locator.encode("utf-8")),
            ("commit", bytes.fromhex(lock.commit)),
            ("branch_hint", _optional(lock.branch_hint)),
            ("submodules", bytes((int(lock.submodules),))),
            ("adapter", lock.adapter.encode("utf-8")),
            ("allowed_dirty_state", b"\x00"),
            ("patches", _items(patch_frames)),
        ),
    )


def preparation_identity(source_id: str, request_sha256: str, nonce: bytes) -> tuple[str, str]:
    if len(nonce) != 16:
        raise ValueError("preparation nonce must contain exactly 16 bytes")
    attempt = _sha256_frame(
        "strixlab.source.preparation.v1",
        (
            ("schema_version", _u32(1)),
            ("source_id", source_id.encode("utf-8")),
            ("request_digest", bytes.fromhex(request_sha256)),
            ("nonce", nonce),
        ),
    )
    return f"prep-{source_id}-{attempt[:24]}", attempt


def content_tree_id(
    root_tree: str,
    *,
    patches: tuple[PatchIdentity, ...],
    submodules: tuple[SubmoduleIdentity, ...],
) -> str:
    patch_values = tuple(bytes.fromhex(patch.sha256) for patch in patches)
    submodule_values = tuple(
        length_frame(
            "strixlab.source.submodule.v1",
            (
                ("path", submodule.path.encode("utf-8")),
                ("commit", bytes.fromhex(submodule.commit)),
            ),
        )
        for submodule in sorted(submodules, key=lambda value: value.path.encode("utf-8"))
    )
    digest = _sha256_frame(
        "strixlab.source.content-tree.v1",
        (
            ("object_format", b"sha1"),
            ("root_tree", bytes.fromhex(root_tree)),
            ("patches", _items(patch_values)),
            ("submodules", _items(submodule_values)),
        ),
    )
    return f"content-tree-sha256:{digest}"


def candidate_id(root_commit: str, content_id: str, *, submodules: bool) -> str:
    content_digest = content_id.removeprefix("content-tree-sha256:")
    digest = _sha256_frame(
        "strixlab.source.candidate.v1",
        (
            ("source_protocol_version", _u32(1)),
            ("object_format", b"sha1"),
            ("root_commit", bytes.fromhex(root_commit)),
            ("content_tree_digest", bytes.fromhex(content_digest)),
            ("submodule_policy", bytes((int(submodules),))),
        ),
    )
    return f"candidate-sha256:{digest}"
