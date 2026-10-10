import pytest

from core.audit import AuditLogger
from core.data_quality import DataQualityError, QualityCheck
from core.pipeline import run_pipeline
from core.state import PipelineState


def _state() -> PipelineState:
    return PipelineState(
        sttm_bronze_path="bronze.json",
        sttm_silver_path="silver.json",
        sttm_gold_path="gold.json",
        bronze_output_paths=[],
        silver_output_paths=[],
        gold_output_paths=[],
    )


def test_run_pipeline_executes_stages_in_order_and_audits(tmp_path) -> None:
    state = _state()
    logger = AuditLogger(run_id="test-run")
    logger.log_file = tmp_path / "test-run.jsonl"
    stages_run = []

    result = run_pipeline(
        state,
        bronze=lambda current: stages_run.append("bronze"),
        silver=lambda current: stages_run.append("silver"),
        gold=lambda current: stages_run.append("gold"),
        audit_logger=logger,
    )

    assert result is state
    assert state.run_id == "test-run"
    assert state.status == "success"
    assert state.error is None
    assert stages_run == ["bronze", "silver", "gold"]
    assert [entry["action"] for entry in logger.get_logs()] == [
        "run_started",
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "stage_started",
        "stage_completed",
        "run_completed",
    ]


def test_run_pipeline_stops_and_records_failure(tmp_path) -> None:
    state = _state()
    logger = AuditLogger(run_id="failed-run")
    logger.log_file = tmp_path / "failed-run.jsonl"
    stages_run = []

    def fail_silver(current: PipelineState) -> None:
        stages_run.append("silver")
        raise RuntimeError("invalid silver input")

    with pytest.raises(RuntimeError, match="invalid silver input"):
        run_pipeline(
            state,
            bronze=lambda current: stages_run.append("bronze"),
            silver=fail_silver,
            gold=lambda current: stages_run.append("gold"),
            audit_logger=logger,
        )

    assert state.status == "failed"
    assert state.error == "silver stage failed: invalid silver input"
    assert stages_run == ["bronze", "silver"]
    assert [entry["action"] for entry in logger.get_logs()][-2:] == [
        "stage_failed",
        "run_failed",
    ]


def test_quality_check_pass_is_recorded_and_audited(tmp_path) -> None:
    state = _state()
    logger = AuditLogger(run_id="quality-pass")
    logger.log_file = tmp_path / "quality-pass.jsonl"

    run_pipeline(
        state,
        bronze=lambda current: None,
        silver=lambda current: None,
        gold=lambda current: None,
        audit_logger=logger,
        quality_checks={
            "bronze": [QualityCheck("bronze-output-present", lambda current: None)],
        },
    )

    assert len(state.quality_results) == 1
    assert state.quality_results[0].passed is True
    quality_events = [
        entry for entry in logger.get_logs()
        if entry["action"] == "quality_check_completed"
    ]
    assert len(quality_events) == 1
    assert quality_events[0]["check_name"] == "bronze-output-present"
    assert quality_events[0]["passed"] is True
    assert quality_events[0]["violation_count"] == 0


def test_quality_check_failure_stops_downstream_stages(tmp_path) -> None:
    state = _state()
    logger = AuditLogger(run_id="quality-fail")
    logger.log_file = tmp_path / "quality-fail.jsonl"
    stages_run = []

    with pytest.raises(DataQualityError, match="silver data quality check"):
        run_pipeline(
            state,
            bronze=lambda current: stages_run.append("bronze"),
            silver=lambda current: stages_run.append("silver"),
            gold=lambda current: stages_run.append("gold"),
            audit_logger=logger,
            quality_checks={
                "silver": [
                    QualityCheck("required-columns", lambda current: ["missing column: total"]),
                ],
            },
        )

    assert stages_run == ["bronze", "silver"]
    assert state.status == "failed"
    assert state.quality_results[0].passed is False
    assert state.quality_results[0].violations == ["missing column: total"]
    assert [entry["action"] for entry in logger.get_logs()][-3:] == [
        "quality_check_completed",
        "stage_failed",
        "run_failed",
    ]


def test_quality_check_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match=r"unknown stage\(s\): silveer"):
        run_pipeline(
            _state(),
            bronze=lambda current: None,
            silver=lambda current: None,
            gold=lambda current: None,
            quality_checks={"silveer": [QualityCheck("check", lambda current: None)]},
        )