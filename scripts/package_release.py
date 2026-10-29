#!/usr/bin/env python3
"""ProofCausality AI — Release Packaging & Distribution Safety Verification Script.

Builds minimal, clean, and compliant source distribution archives (tar.gz and zip)
from git HEAD, strictly enforcing exclusion of third-party copyrighted/restricted
photographs, local binaries, virtual environments, caches, and secrets via .gitattributes.
Performs clean extraction verification and SHA-256 digest computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_version() -> str:
    """Read canonical version from backend/__init__.py."""
    init_path = REPO_ROOT / "backend" / "__init__.py"
    with open(init_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Cannot find __version__ in backend/__init__.py")


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_archive_contents(archive_path: Path, prefix: str) -> dict[str, list[str]]:
    """Inspect archive to confirm required source files exist and forbidden files are absent."""
    required_rel_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "backend/app.py",
        "backend/api/routes.py",
        "backend/__init__.py",
        "native/proofcausality_core/Cargo.toml",
        "native/proofcausality_core/src/lib.rs",
        "frontend/static/index.html",
        "data/coco_val2017_subset/annotations_coco_pilot20.json",
        "evaluation/experiments/genuine_model_predictions/dataset_manifest.json",
        "docs/research/REAL_DATASET_MANIFEST.json",
        "scripts/download_coco_subset.py",
        "docs/research/DATASET_LICENSING_AUDIT.md",
        "tests/test_api.py",
    ]

    forbidden_patterns = [
        ".venv/",
        ".git/",
        "data/coco_val2017_subset/images/",
        "native/proofcausality_core/target/",
        ".DS_Store",
        ".env",
        ".pyc",
        "__pycache__",
        ".so",
        ".dylib",
        ".dll",
    ]

    all_members: list[str] = []
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            all_members = zf.namelist()
    elif archive_path.name.endswith(".tar.gz") or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tf:
            all_members = [m.name for m in tf.getmembers()]
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    # Strip prefix
    stripped_members = set()
    for m in all_members:
        clean = m
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
        clean = clean.lstrip("/")
        if clean:
            stripped_members.add(clean)

    missing_required: list[str] = []
    for req in required_rel_files:
        if req not in stripped_members:
            missing_required.append(req)

    present_forbidden: list[str] = []
    for member in stripped_members:
        if member == ".env" or member.endswith("/.env") or (
            member.startswith(".env.") and not member.endswith(".example")
        ):
            present_forbidden.append(member)
            continue
        for pat in forbidden_patterns:
            if pat == ".env":
                continue
            if pat in member:
                present_forbidden.append(member)
                break

    return {
        "missing_required": missing_required,
        "present_forbidden": present_forbidden,
        "total_files": [str(len(stripped_members))],
    }


def clean_extract_and_smoke_test(archive_path: Path, prefix: str) -> dict[str, str]:
    """Extract archive into a clean isolated directory and test basic sanity."""
    with tempfile.TemporaryDirectory(prefix="proofcausality_release_test_") as tmpdir:
        extract_dir = Path(tmpdir)
        print(f"Extracting {archive_path.name} to clean directory: {extract_dir}")
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_dir)
        else:
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_dir)

        target_root = extract_dir / prefix.rstrip("/")
        if not target_root.exists():
            target_root = extract_dir

        # 1. Check pyproject.toml exists and matches version
        pyproject_path = target_root / "pyproject.toml"
        if not pyproject_path.exists():
            return {"status": "FAILED", "reason": "pyproject.toml not found in extract"}

        # 2. Check Rust Cargo manifest exists
        cargo_path = target_root / "native" / "proofcausality_core" / "Cargo.toml"
        if not cargo_path.exists():
            return {"status": "FAILED", "reason": "Cargo.toml not found in extract"}

        # 3. Check that python -c "import backend" works from target_root
        proc = subprocess.run(
            [sys.executable, "-c", "import backend; print(backend.__version__)"],
            cwd=str(target_root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return {"status": "FAILED", "reason": f"Import test failed: {proc.stderr}"}

        extracted_version = proc.stdout.strip()
        expected_version = get_version()
        if extracted_version != expected_version:
            return {
                "status": "FAILED",
                "reason": f"Version mismatch: {extracted_version} != {expected_version}",
            }

        # 4. Check Cargo compilation check
        cargo_proc = subprocess.run(
            ["cargo", "check", "--manifest-path", str(cargo_path)],
            capture_output=True,
            text=True,
        )
        if cargo_proc.returncode != 0:
            return {"status": "FAILED", "reason": f"Cargo check failed: {cargo_proc.stderr}"}

        return {"status": "PASS", "extracted_version": extracted_version}


def package_release(version: str | None = None, output_dir: Path | None = None) -> dict[str, any]:
    """Build and verify release packages."""
    if version is None:
        version = get_version()
    if output_dir is None:
        output_dir = REPO_ROOT / "dist"

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"proofcausality-ai-{version}/"

    tar_name = f"proofcausality-ai-{version}.tar.gz"
    zip_name = f"proofcausality-ai-{version}.zip"

    tar_path = output_dir / tar_name
    zip_path = output_dir / zip_name

    print(f"Building release archives for version {version}...")

    # 1. Build tar.gz using git archive with worktree attributes
    cmd_tar = [
        "git",
        "archive",
        "--worktree-attributes",
        "--format=tar.gz",
        f"--prefix={prefix}",
        "-o",
        str(tar_path),
        "HEAD",
    ]
    subprocess.run(cmd_tar, cwd=str(REPO_ROOT), check=True)

    # 2. Build zip using git archive with worktree attributes
    cmd_zip = [
        "git",
        "archive",
        "--worktree-attributes",
        "--format=zip",
        f"--prefix={prefix}",
        "-o",
        str(zip_path),
        "HEAD",
    ]
    subprocess.run(cmd_zip, cwd=str(REPO_ROOT), check=True)

    # 3. Compute digests
    tar_sha = compute_sha256(tar_path)
    zip_sha = compute_sha256(zip_path)

    tar_size = tar_path.stat().st_size
    zip_size = zip_path.stat().st_size

    # 4. Content verification
    tar_check = verify_archive_contents(tar_path, prefix)
    zip_check = verify_archive_contents(zip_path, prefix)

    if tar_check["missing_required"] or tar_check["present_forbidden"]:
        raise RuntimeError(f"tar.gz content verification failed: {tar_check}")
    if zip_check["missing_required"] or zip_check["present_forbidden"]:
        raise RuntimeError(f"zip content verification failed: {zip_check}")

    # 5. Clean extract & smoke test
    extract_test = clean_extract_and_smoke_test(tar_path, prefix)
    if extract_test["status"] != "PASS":
        raise RuntimeError(f"Clean extraction test failed: {extract_test}")

    manifest = {
        "version": version,
        "prefix": prefix,
        "archives": {
            "tar_gz": {
                "filename": tar_name,
                "path": str(tar_path),
                "sha256": tar_sha,
                "size_bytes": tar_size,
            },
            "zip": {
                "filename": zip_name,
                "path": str(zip_path),
                "sha256": zip_sha,
                "size_bytes": zip_size,
            },
        },
        "content_verification": {
            "tar_gz": tar_check,
            "zip": zip_check,
        },
        "clean_extract_test": extract_test,
    }

    manifest_path = output_dir / f"release_manifest_v{version}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nRelease packaging complete!")
    print(f"  Version:   {version}")
    print(f"  tar.gz:    {tar_name} ({tar_size:,} bytes)")
    print(f"  SHA-256:   {tar_sha}")
    print(f"  zip:       {zip_name} ({zip_size:,} bytes)")
    print(f"  SHA-256:   {zip_sha}")
    print(f"  Manifest:  {manifest_path.name}")
    print(f"  Clean Test: {extract_test['status']}")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package ProofCausality AI release archive.")
    parser.add_argument("--version", type=str, default=None, help="Release version (e.g. 0.2.6)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for archives")
    args = parser.parse_args()

    package_release(version=args.version, output_dir=args.output_dir)
