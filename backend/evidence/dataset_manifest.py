"""Validated source-manifest records for public photographic datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, model_validator


class DatasetSourceRecord(BaseModel):
    dataset_name: str
    dataset_release: str
    official_image_id: int
    filename: str
    canonical_source_url: str
    license_status: Literal["KNOWN", "UNKNOWN"] = "UNKNOWN"
    license_name: str | None = None
    license_url: str | None = None
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    annotation_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_image_id: int
    acquisition_procedure: str
    redistribution_status: Literal["PERMITTED", "RESTRICTED", "UNKNOWN"] = "UNKNOWN"
    preprocessing_history: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_linkage_and_license(self) -> "DatasetSourceRecord":
        if self.annotation_image_id != self.official_image_id:
            raise ValueError("annotation/image ID mismatch")
        if not self.license_name or not self.license_url:
            self.license_status = "UNKNOWN"
            self.redistribution_status = "UNKNOWN"
        return self

    def verify_files(self, image_path: str | Path, annotation_path: str | Path) -> None:
        image_bytes = Path(image_path).read_bytes()
        annotation_bytes = Path(annotation_path).read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != self.original_sha256:
            raise ValueError("source image hash mismatch")
        if hashlib.sha256(annotation_bytes).hexdigest() != self.annotation_file_sha256:
            raise ValueError("annotation file hash mismatch")
        with Image.open(image_path) as image:
            if image.size != (self.image_width, self.image_height):
                raise ValueError("source image dimension mismatch")
        annotation = json.loads(annotation_bytes)
        image_ids = {int(item["id"]) for item in annotation.get("images", [])}
        if self.annotation_image_id not in image_ids:
            raise ValueError("annotation/image ID linkage missing")

