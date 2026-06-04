"""Tests for ExpService."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from capabilities.mining.exp.models import (
	AssayMethod,
	AssayResultCreate,
	ComplianceReportCreate,
	DrillholeCollarCreate,
	GeologyIntervalCreate,
	HoleType,
	OxidationState,
	ResourceClassification,
	ResourceEstimateCreate,
	ResourceEstimateUpdate,
	ReportingStandard,
	ReviewStatus,
	SampleType,
)
from capabilities.mining.exp.service import ExpService

TENANT = "test_tenant"


def run(coro):
	return asyncio.get_event_loop().run_until_complete(coro)


def make_service():
	return ExpService(tenant_id=TENANT)


def make_collar(hole_id: str = "ABDD001", tenant_id: str = TENANT) -> DrillholeCollarCreate:
	return DrillholeCollarCreate(
		hole_id=hole_id,
		tenant_id=tenant_id,
		hole_type=HoleType.REVERSE_CIRCULATION,
		easting=500000.0,
		northing=7500000.0,
		elevation_m=450.0,
		coordinate_system="mga_zone_54",
		azimuth_deg=270.0,
		dip_deg=-60.0,
		planned_depth_m=200.0,
		drilled_by="driller_001",
		drilled_at=datetime(2026, 1, 15),
	)


def make_assay(hole_id: str = "ABDD001") -> AssayResultCreate:
	return AssayResultCreate(
		tenant_id=TENANT,
		hole_id=hole_id,
		sample_id=f"S{hole_id}001",
		from_m=0.0,
		to_m=1.0,
		sample_type=SampleType.CHIP,
		assay_method=AssayMethod.FIRE_ASSAY,
		commodity="gold",
		grade_value=2.5,
		grade_units="g/t",
		detection_limit=0.01,
		lab_name="SGS Minerals",
		lab_certificate_ref="LAB2026-001",
	)


def test_create_drillhole_collar():
	svc = make_service()
	result = run(svc.create_drillhole_collar(make_collar(), created_by="geo_user"))
	assert result.hole_id == "ABDD001"
	assert result.tenant_id == TENANT
	assert result.id is not None


def test_drillhole_id_uniqueness():
	svc = make_service()
	run(svc.create_drillhole_collar(make_collar(), created_by="geo_user"))
	with pytest.raises(ValueError, match="already exists"):
		run(svc.create_drillhole_collar(make_collar(), created_by="geo_user"))


def test_list_drillhole_collars():
	svc = make_service()
	run(svc.create_drillhole_collar(make_collar("ABDD001"), created_by="geo_user"))
	run(svc.create_drillhole_collar(make_collar("ABDD002"), created_by="geo_user"))
	results = run(svc.list_drillhole_collars())
	assert len(results) == 2


def test_update_actual_depth():
	svc = make_service()
	collar = run(svc.create_drillhole_collar(make_collar(), created_by="geo_user"))
	updated = run(svc.update_drillhole_actual_depth(collar.id, 195.0))
	assert updated.actual_depth_m == 195.0


def test_get_collar_not_found():
	svc = make_service()
	result = run(svc.get_drillhole_collar("nonexistent"))
	assert result is None


def test_import_assay_requires_collar():
	svc = make_service()
	with pytest.raises(ValueError, match="does not exist"):
		run(svc.import_assay_results([make_assay()], created_by="lab_user"))


def test_import_assay_success():
	svc = make_service()
	run(svc.create_drillhole_collar(make_collar(), created_by="geo_user"))
	results = run(svc.import_assay_results([make_assay()], created_by="lab_user"))
	assert len(results) == 1
	assert results[0].hole_id == "ABDD001"
	assert results[0].grade_value == 2.5


def test_assay_interval_overlap_rejected():
	svc = make_service()
	run(svc.create_drillhole_collar(make_collar(), created_by="geo_user"))
	assay1 = make_assay()
	run(svc.import_assay_results([assay1], created_by="lab_user"))
	# Same interval — should fail
	assay2 = AssayResultCreate(
		tenant_id=TENANT, hole_id="ABDD001", sample_id="S002",
		from_m=0.0, to_m=1.0, sample_type=SampleType.CHIP,
		assay_method=AssayMethod.FIRE_ASSAY, commodity="gold",
		grade_value=1.0, grade_units="g/t", detection_limit=0.01,
		lab_name="SGS", lab_certificate_ref="LAB2"
	)
	with pytest.raises(ValueError, match="overlaps"):
		run(svc.import_assay_results([assay2], created_by="lab_user"))


def test_log_geology_interval():
	svc = make_service()
	run(svc.create_drillhole_collar(make_collar(), created_by="geo_user"))
	geo = GeologyIntervalCreate(
		tenant_id=TENANT, hole_id="ABDD001",
		from_m=0.0, to_m=10.0,
		lithology_code="granite",
		oxidation_state=OxidationState.OXIDISED,
		geologist_id="geo_001",
		logged_at=datetime(2026, 1, 16),
	)
	result = run(svc.log_geology_interval(geo, created_by="geo_user"))
	assert result.from_m == 0.0
	assert result.lithology_code == "granite"


def test_create_and_approve_resource_estimate():
	svc = make_service()
	payload = ResourceEstimateCreate(
		tenant_id=TENANT, name="Main Zone Resource",
		commodity="gold", classification=ResourceClassification.INDICATED,
		reporting_standard=ReportingStandard.JORC_2012,
		estimation_method="ordinary_kriging",
		tonnes=1_500_000.0, grade_value=1.8, grade_units="g/t",
		effective_date=datetime(2026, 3, 1),
		competent_person_id="cp_001",
		competent_person_qualification="FAusIMM(CP)",
	)
	resource = run(svc.create_resource_estimate(payload, created_by="geo_user"))
	assert resource.review_status == ReviewStatus.PENDING

	approved = run(svc.approve_resource_estimate(resource.id, reviewer_id="reviewer_001"))
	assert approved.review_status == ReviewStatus.APPROVED


def test_publish_requires_approval():
	svc = make_service()
	payload = ResourceEstimateCreate(
		tenant_id=TENANT, name="Test Resource",
		commodity="copper", classification=ResourceClassification.INFERRED,
		reporting_standard=ReportingStandard.JORC_2012,
		estimation_method="inverse_distance",
		tonnes=500_000.0, grade_value=0.5, grade_units="%",
		effective_date=datetime(2026, 3, 1),
		competent_person_id="cp_002",
		competent_person_qualification="FAusIMM(CP)",
	)
	resource = run(svc.create_resource_estimate(payload, created_by="geo_user"))
	with pytest.raises(PermissionError):
		run(svc.publish_resource_estimate(resource.id))


def test_cross_tenant_access_denied():
	svc = make_service()
	collar = make_collar()
	collar_wrong_tenant = DrillholeCollarCreate(
		**{**collar.model_dump(), "tenant_id": "other_tenant"}
	)
	with pytest.raises(AssertionError):
		run(svc.create_drillhole_collar(collar_wrong_tenant, created_by="geo_user"))


def test_exploration_summary():
	svc = make_service()
	run(svc.create_drillhole_collar(make_collar("ABDD001"), created_by="geo_user"))
	run(svc.create_drillhole_collar(make_collar("ABDD002"), created_by="geo_user"))
	summary = run(svc.get_exploration_summary())
	assert summary["total_drillholes"] == 2
	assert summary["tenant_id"] == TENANT
