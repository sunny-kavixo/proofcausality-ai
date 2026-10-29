"""Tests for release packaging, version synchronization, and distribution safety."""

from __future__ import annotations

import os
from pathlib import Path
import tomllib
from fastapi.testclient import TestClient
import pytest

from backend import __version__
from backend.app import app
from scripts.package_release import (
    REPO_ROOT,
    clean_extract_and_smoke_test,
    compute_sha256,
    get_version,
    verify_archive_contents,
)

client = TestClient(app)


def test_version_synchronization_across_repository():
    """Verify that version strings match exactly across all configuration and code files."""
    canonical_version = __version__
    assert canonical_version == "0.2.6"
    assert get_version() == canonical_version

    # 1. pyproject.toml
    pyproject_path = REPO_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject_data = tomllib.load(f)
    assert pyproject_data["project"]["version"] == canonical_version
    assert pyproject_data["project"]["requires-python"] == ">=3.11"

    # 2. Cargo.toml
    cargo_path = REPO_ROOT / "native" / "proofcausality_core" / "Cargo.toml"
    with open(cargo_path, "r", encoding="utf-8") as f:
        cargo_content = f.read()
    assert f'version = "{canonical_version}"' in cargo_content

    # 3. frontend/static/index.html
    index_path = REPO_ROOT / "frontend" / "static" / "index.html"
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    assert f"Engine v{canonical_version}" in html_content

    # 4. FastAPI /api/health
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["version"] == canonical_version


def test_gitattributes_export_ignore_rules():
    """Verify that .gitattributes strictly enforces exclusion of forbidden distribution assets."""
    gitattributes_path = REPO_ROOT / ".gitattributes"
    assert gitattributes_path.exists(), ".gitattributes must exist at repository root"

    with open(gitattributes_path, "r", encoding="utf-8") as f:
        rules = f.read()

    # Third-party restricted photographic assets must be excluded from public archive
    assert "data/coco_val2017_subset/images export-ignore" in rules
    # Virtualenv, build caches, and compiled artifacts must be excluded
    assert ".venv export-ignore" in rules
    assert "native/proofcausality_core/target export-ignore" in rules
    assert "__pycache__ export-ignore" in rules
    assert ".gitattributes export-ignore" in rules


def test_release_packaging_archive_safety(tmp_path: Path):
    """Test generating release archives and verify that required files exist and forbidden files are excluded."""
    # Build test tarball into tmp_path
    version = "0.2.6"
    prefix = f"proofcausality-ai-{version}/"
    tar_path = tmp_path / f"test-release-{version}.tar.gz"

    import subprocess
    cmd = [
        "git",
        "archive",
        "--worktree-attributes",
        "--format=tar.gz",
        f"--prefix={prefix}",
        "-o",
        str(tar_path),
        "HEAD",
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    assert tar_path.exists()
    assert tar_path.stat().st_size > 50000  # Non-empty source archive

    # Verify contents
    result = verify_archive_contents(tar_path, prefix)
    assert not result["missing_required"], f"Missing required files in archive: {result['missing_required']}"
    assert not result["present_forbidden"], f"Forbidden files found in archive: {result['present_forbidden']}"

    # Verify clean extraction smoke test
    extract_result = clean_extract_and_smoke_test(tar_path, prefix)
    assert extract_result["status"] == "PASS", f"Extraction test failed: {extract_result}"
    assert extract_result["extracted_version"] == version
