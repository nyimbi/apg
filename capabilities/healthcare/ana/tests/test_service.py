"""Tests for ClinicalAnalyticsService."""

from __future__ import annotations

import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ana.models import (
	AnalyticsReportCreate, CareGapCreate, CohortCreate, CohortUpdate,
	MetricRecordCreate, PredictionModelCreate, QualityIndicatorCreate,
)
from ana.service import ClinicalAnalyticsService, PolicyViolationError


def run(coro):
	loop = asyncio.get_event_loop()
	return loop.run_until_complete(coro)


def make_service():
	return ClinicalAnalyticsService()


def make_cohort_payload(tenant_id: str = "test") -> CohortCreate:
	return CohortCreate(
		tenant_id=tenant_id,
		name="Diabetic Cohort",
		description="Patients with T2DM",
		segment="chronic_disease",
		criteria={"icd10_prefix": "E11"},
		icd10_codes=["E11.9"],
		created_by="analyst_1",
	)


def test_create_cohort_returns_id():
	svc = make_service()
	cohort = run(svc.create_cohort(make_cohort_payload()))
	assert cohort.id
	assert cohort.status == "draft"
	assert cohort.tenant_id == "test"


def test_get_cohort_returns_created():
	svc = make_service()
	created = run(svc.create_cohort(make_cohort_payload()))
	fetched = run(svc.get_cohort("test", created.id))
	assert fetched is not None
	assert fetched.id == created.id


def test_get_cohort_wrong_tenant_returns_none():
	svc = make_service()
	created = run(svc.create_cohort(make_cohort_payload()))
	result = run(svc.get_cohort("other_tenant", created.id))
	assert result is None


def test_list_cohorts_filtered_by_segment():
	svc = make_service()
	run(svc.create_cohort(make_cohort_payload()))
	other = CohortCreate(tenant_id="test", name="Other", description="x", segment="geriatric", created_by="u")
	run(svc.create_cohort(other))
	results = run(svc.list_cohorts("test", segment="chronic_disease"))
	assert all(c.segment == "chronic_disease" for c in results)


def test_activate_cohort():
	svc = make_service()
	cohort = run(svc.create_cohort(make_cohort_payload()))
	activated = run(svc.activate_cohort("test", cohort.id))
	assert activated.status == "active"


def test_update_cohort_name():
	svc = make_service()
	cohort = run(svc.create_cohort(make_cohort_payload()))
	updated = run(svc.update_cohort("test", cohort.id, CohortUpdate(name="Updated Name")))
	assert updated.name == "Updated Name"


def test_delete_cohort_success():
	svc = make_service()
	cohort = run(svc.create_cohort(make_cohort_payload()))
	deleted = run(svc.delete_cohort("test", cohort.id))
	assert deleted
	assert run(svc.get_cohort("test", cohort.id)) is None


def test_record_metric_success():
	svc = make_service()
	payload = MetricRecordCreate(
		tenant_id="test",
		metric_type="readmission_rate",
		value=12.5,
		unit="percent",
		period="monthly",
		period_start=datetime(2026, 1, 1),
		period_end=datetime(2026, 1, 31),
		data_source="emr",
		created_by="analyst_1",
	)
	rec = run(svc.record_metric(payload))
	assert rec.id
	assert rec.metric_type == "readmission_rate"


def test_record_metric_unsupported_type_denied():
	svc = make_service()
	payload = MetricRecordCreate(
		tenant_id="test",
		metric_type="invalid_metric_xyz",
		value=5.0,
		unit="percent",
		period="monthly",
		period_start=datetime(2026, 1, 1),
		period_end=datetime(2026, 1, 31),
		data_source="emr",
		created_by="analyst_1",
	)
	try:
		run(svc.record_metric(payload))
		assert False, "should have raised"
	except PolicyViolationError:
		pass


def test_create_prediction_model_success():
	svc = make_service()
	payload = PredictionModelCreate(
		tenant_id="test",
		name="30-day Readmission Model",
		model_type="gradient_boosting",
		target_outcome="30_day_readmission",
		feature_set=["age", "prior_admits", "icd10_count"],
		auc_score=0.82,
		training_cohort_id="cohort_001",
		approval_reference="approval_001",
		created_by="data_scientist_1",
	)
	model = run(svc.create_prediction_model(payload))
	assert model.id
	assert model.status == "active"
	assert model.auc_score == 0.82


def test_create_model_low_auc_denied():
	svc = make_service()
	payload = PredictionModelCreate(
		tenant_id="test",
		name="Bad Model",
		model_type="logistic_regression",
		target_outcome="30_day_readmission",
		auc_score=0.55,
		training_cohort_id="cohort_001",
		approval_reference="appr_001",
		created_by="ds_1",
	)
	try:
		run(svc.create_prediction_model(payload))
		assert False, "should have raised"
	except PolicyViolationError:
		pass


def test_generate_prediction_returns_score():
	svc = make_service()
	payload = PredictionModelCreate(
		tenant_id="test", name="M", model_type="random_forest",
		target_outcome="readmission", auc_score=0.80,
		training_cohort_id="c1", approval_reference="a1", created_by="u",
	)
	model = run(svc.create_prediction_model(payload))
	result = run(svc.generate_prediction("test", model.id, {"age": 72, "prior_admits": 3}))
	assert "probability_score" in result
	assert result["risk_level"] in ("low", "medium", "high")


def test_identify_care_gap():
	svc = make_service()
	payload = CareGapCreate(
		tenant_id="test", patient_id="pt_001", gap_type="screening",
		description="Overdue mammogram", severity="warning",
		evidence_reference="guideline_001", created_by="analyst_1",
	)
	gap = run(svc.identify_care_gap(payload))
	assert gap.id
	assert gap.status == "open"


def test_resolve_care_gap():
	svc = make_service()
	payload = CareGapCreate(
		tenant_id="test", patient_id="pt_001", gap_type="screening",
		description="Overdue mammogram", severity="warning",
		evidence_reference="guideline_001", created_by="analyst_1",
	)
	gap = run(svc.identify_care_gap(payload))
	resolved = run(svc.resolve_care_gap("test", gap.id))
	assert resolved.status == "resolved"
	assert resolved.resolved_at is not None


def test_generate_report():
	svc = make_service()
	payload = AnalyticsReportCreate(
		tenant_id="test", report_name="Monthly QI Report", report_type="quality",
		format="pdf", period="monthly",
		period_start=datetime(2026, 1, 1), period_end=datetime(2026, 1, 31),
		created_by="analyst_1",
	)
	report = run(svc.generate_report(payload))
	assert report.id
	assert report.status == "completed"
	assert report.download_url is not None


def test_dashboard_summary():
	svc = make_service()
	run(svc.create_cohort(make_cohort_payload()))
	summary = run(svc.dashboard_summary("test"))
	assert "cohorts" in summary
	assert summary["cohorts"]["total"] == 1


def test_record_quality_indicator():
	svc = make_service()
	payload = QualityIndicatorCreate(
		tenant_id="test", indicator_code="PSI-90", indicator_name="Patient Safety",
		value=88.5, numerator=177, denominator=200, period="quarterly",
		data_source="emr", benchmark_value=90.0, benchmark_type="national",
		created_by="analyst_1",
	)
	qi = run(svc.record_quality_indicator(payload))
	assert qi.id
	assert qi.performance_status == "below_target"
