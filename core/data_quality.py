from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from core.state import PipelineState


QualityValidator = Callable[["PipelineState"], Sequence[str] | None]


@dataclass(frozen=True)
class QualityCheck:
    name: str
    validate: QualityValidator


class QualityCheckResult(BaseModel):
    layer: str
    check_name: str
    passed: bool
    violations: list[str] = Field(default_factory=list)


class DataQualityError(ValueError):
    def __init__(self, layer: str, check_name: str, violation_count: int) -> None:
        super().__init__(
            f"{layer} data quality check '{check_name}' failed "
            f"with {violation_count} violation(s)."
        )


def evaluate_quality_check(
    layer: str,
    state: PipelineState,
    check: QualityCheck,
) -> QualityCheckResult:
    violations = list(check.validate(state) or ())
    return QualityCheckResult(
        layer=layer,
        check_name=check.name,
        passed=not violations,
        violations=violations,
    )