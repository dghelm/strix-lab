from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from strixlab.config import read_manifest
from strixlab.manifests import ModelManifestV1, resolve_and_validate_manifest, validate_manifest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "configs" / "models"
_PROVENANCE = json.loads(
    (_REPO_ROOT / "tests" / "fixtures" / "models" / "smoke-provenance.json").read_text()
)

SMOKE_IDS = ("qwen35-2b-smoke", "qwen35-4b-smoke")
DRAFT_IDS = ("ornith15-35b-a3b", "qwen38-27b", "qwen38-flash-next")


def _resolved(config_id: str) -> ModelManifestV1:
    raw = read_manifest(_CONFIG_DIR / f"{config_id}.yaml")
    resolved = resolve_and_validate_manifest("model", raw, {"MODELS": "/data/models"})
    assert isinstance(resolved, ModelManifestV1)
    return resolved


def test_all_checked_in_configs_validate() -> None:
    present = sorted(path.stem for path in _CONFIG_DIR.glob("*.yaml"))
    assert present == sorted((*SMOKE_IDS, *DRAFT_IDS))


@pytest.mark.parametrize("config_id", SMOKE_IDS)
def test_smoke_manifest_pins_reviewed_constants(config_id: str) -> None:
    resolved = _resolved(config_id)
    reviewed = _PROVENANCE["models"][config_id]
    assert resolved.registry_status == "registered"

    base = resolved.base_model
    assert base is not None
    assert base.repository == reviewed["base"]["repository"]
    assert base.revision == reviewed["base"]["revision"]
    assert base.license == reviewed["base"]["license"]

    file = resolved.artifact.file
    art = reviewed["artifact"]
    assert file.repository == art["repository"]
    assert file.revision == art["revision"]
    assert file.filename == art["filename"]
    assert file.size_bytes == art["size_bytes"]
    assert file.sha256 == art["sha256"]

    arch = resolved.architecture
    assert arch is not None
    assert arch.model_dump(mode="json") == reviewed["architecture"]

    quant = resolved.quantization
    assert quant.format_family == reviewed["quantization"]["format_family"]
    assert quant.measured_bits_per_weight is None
    # Unknown quant provenance means the smoke receipts can never be publishable.
    assert quant.is_fully_provenanced() is False
    assert resolved.sidecars == []


def test_smoke_manifests_keep_environment_templated() -> None:
    raw = read_manifest(_CONFIG_DIR / "qwen35-2b-smoke.yaml")
    templated = validate_manifest("model", raw)
    assert templated.artifact.file.local_path.startswith("${MODELS}/")


@pytest.mark.parametrize("config_id", DRAFT_IDS)
def test_draft_manifests_declare_no_local_identity(config_id: str) -> None:
    resolved = _resolved(config_id)
    assert resolved.registry_status == "draft"
    assert resolved.draft_reason is not None
    file = resolved.artifact.file
    assert file.local_path is None
    assert file.size_bytes is None
    assert file.sha256 is None
    assert resolved.sidecars == []


@pytest.mark.parametrize("config_id", SMOKE_IDS)
def test_smoke_provenance_records_reviewed_content_digests(config_id: str) -> None:
    # Offline: assert the reviewed-content shape and constants are locked. Never fetch.
    reviewed = _PROVENANCE["models"][config_id]
    base = reviewed["base"]
    revision = base["revision"]
    content = base["reviewed_content"]
    assert set(content) == {"config_json", "model_card"}

    config = content["config_json"]
    assert re.fullmatch(r"[0-9a-f]{64}", config["sha256"])
    assert config["size_bytes"] > 0
    assert config["url"].endswith(f"/resolve/{revision}/config.json")
    # The reviewed base config's model_type is exactly the manifest's declared family.
    assert config["model_type"] == reviewed["architecture"]["family"]
    assert config["architectures"] == ["Qwen3_5ForConditionalGeneration"]

    card = content["model_card"]
    assert re.fullmatch(r"[0-9a-f]{64}", card["sha256"])
    assert card["size_bytes"] > 0
    assert card["url"].endswith(f"/resolve/{revision}/README.md")
    # The reviewed model-card license field agrees with the manifest license (case-fold).
    assert card["license_field"].lower() == base["license"].lower()

    # The digests of the two base revisions are distinct reviewed sources.
    other = "qwen35-4b-smoke" if config_id == "qwen35-2b-smoke" else "qwen35-2b-smoke"
    other_config = _PROVENANCE["models"][other]["base"]["reviewed_content"]["config_json"]
    assert config["sha256"] != other_config["sha256"]
