from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class NoiseConfig(BaseModel):
    enabled: bool = False
    blur_sigma: float = 0.0
    rotation_deg: float = 0.0


class LocaleConfig(BaseModel):
    currency: str = "AED"
    date_fmt: str = "dd MMM yy"
    decimal_sep: str = ","


class SyntheticConfig(BaseModel):
    seed: int = 42
    statements: int = 10
    layouts: List[str] = Field(default_factory=lambda: ["default"])
    locale: LocaleConfig = Field(default_factory=LocaleConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    output_dir: Path = Path("data/synthetic")
    renderer: str = "auto"  # auto, reportlab, or html
    schema: str = "five_column"  # five_column | four_no_balance | random

    @field_validator("renderer")
    @classmethod
    def _validate_renderer(cls, v: str) -> str:
        if v not in {"auto", "reportlab", "html"}:
            raise ValueError("renderer must be 'auto', 'reportlab', or 'html'")
        return v

    @field_validator("schema")
    @classmethod
    def _validate_schema(cls, v: str) -> str:
        if v not in {"five_column", "four_no_balance", "random"}:
            raise ValueError("schema must be 'five_column', 'four_no_balance', or 'random'")
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SyntheticConfig":
        import yaml

        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def dump_yaml(self, path: str | Path) -> None:
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(), f) 