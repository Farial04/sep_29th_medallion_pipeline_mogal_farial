from collections.abc import Callable, Mapping, Sequence
from uuid import uuid4

from core.audit import AuditLogger
from core.data_quality import (
    DataQualityError,
    QualityCheck,
    evaluate_quality_check,
)
from core.state import PipelineState


StageHandler = Callable[[PipelineState], None]


def run_pipeline(
    state: PipelineState,
    *,
    bronze: StageHandler,
    silver: StageHandler,
    gold: StageHandler,
    audit_logger: AuditLogger | None = None,
    quality_checks: Mapping[str, Sequence[QualityCheck]] | None = None,
) -> PipelineState:
    unknown_stages = set(quality_checks or ()) - {"bronze", "silver", "gold"}
    if unknown_stages:
        names = ", ".join(sorted(unknown_stages))
        raise ValueError(f"Quality checks configured for unknown stage(s): {names}.")

    if not state.run_id:
        state.run_id = audit_logger.run_id if audit_logger is not None else str(uuid4())
    if audit_logger is not None and audit_logger.run_id != state.run_id:
        raise ValueError("Audit logger run_id must match pipeline state run_id.")

    logger = audit_logger if audit_logger is not None else AuditLogger(state.run_id)
    state.status = "running"
    state.error = None
    state.quality_results.clear()
    logger.log("pipeline", "run_started")

    for stage_name, handler in (
        ("bronze", bronze),
        ("silver", silver),
        ("gold", gold),
    ):
        logger.log(stage_name, "stage_started")
        try:
            handler(state)
            for check in (quality_checks or {}).get(stage_name, ()):
                result = evaluate_quality_check(stage_name, state, check)
                state.quality_results.append(result)
                logger.log(
                    stage_name,
                    "quality_check_completed",
                    check_name=result.check_name,
                    passed=result.passed,
                    violation_count=len(result.violations),
                )
                if not result.passed:
                    raise DataQualityError(
                        stage_name,
                        result.check_name,
                        len(result.violations),
                    )
        except Exception as error:
            state.status = "failed"
            state.error = f"{stage_name} stage failed: {error}"
            logger.log(stage_name, "stage_failed", error=str(error))
            logger.log("pipeline", "run_failed", failed_stage=stage_name, error=str(error))
            raise
        logger.log(stage_name, "stage_completed")

    state.status = "success"
    logger.log("pipeline", "run_completed")
    return state