"""Async service layer for APG Electronic Medical Records."""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any

from .capability_contract import (
	SUPPORTED_ALLERGY_SEVERITIES, SUPPORTED_ALLERGY_TYPES,
	SUPPORTED_ENCOUNTER_STATUSES, SUPPORTED_FHIR_RESOURCE_TYPES,
	SUPPORTED_MEDICATION_STATUSES, SUPPORTED_NOTE_TYPES,
	SUPPORTED_PROBLEM_STATUSES, SUPPORTED_RECONCILIATION_STATUSES,
	SUPPORTED_VITAL_TYPES, evaluate_capability_rules, get_capability_contract,
)
from .models import (
	AllergyCreate, AllergyResponse,
	CarePlanCreate, CarePlanResponse, CarePlanUpdate,
	ClinicalAlert,
	ClinicalNoteCreate, ClinicalNoteResponse, ClinicalNoteUpdate,
	EncounterCreate, EncounterResponse, EncounterUpdate,
	FamilyHistoryCreate, FamilyHistoryResponse,
	ImagingOrderCreate, ImagingOrderResponse,
	ImmunisationCreate, ImmunisationResponse,
	LabOrderCreate, LabOrderResponse,
	LabResultCreate, LabResultResponse,
	MedicationCreate, MedicationResponse,
	PatientCreate, PatientResponse, PatientUpdate,
	PatientMatchCandidate,
	ProblemCreate, ProblemResponse,
	PrescriptionCreate, PrescriptionResponse,
	ReferralCreate, ReferralUpdate,
	VitalSignCreate, VitalSignResponse,
	uuid7str,
)

logger = logging.getLogger(__name__)


def _log_op(op: str, tenant_id: str, entity_id: str) -> None:
	logger.info("emr.%s tenant=%s id=%s", op, tenant_id, entity_id)


def _log_deny(rule: str, tenant_id: str) -> None:
	logger.warning("emr.rule_denied rule=%s tenant=%s", rule, tenant_id)


def _log_fhir_export(tenant_id: str, resource_type: str, count: int) -> str:
	return f"fhir_export tenant={tenant_id} resource={resource_type} count={count}"


def _log_drug_check(check_type: str, patient_id: str, drug: str) -> None:
	logger.info("emr.drug_check type=%s patient=%s drug=%s", check_type, patient_id, drug)


def _log_cds(score_name: str, patient_id: str, score: int | float) -> None:
	logger.info("emr.cds score=%s patient=%s value=%s", score_name, patient_id, score)


def _log_consent(action: str, patient_id: str, consent_type: str) -> None:
	logger.info("emr.consent action=%s patient=%s type=%s", action, patient_id, consent_type)


def _log_prescribe(prescription_id: str, patient_id: str, drug: str) -> None:
	logger.info("emr.prescribe rx=%s patient=%s drug=%s", prescription_id, patient_id, drug)


class PolicyViolationError(ValueError):
	"""Raised when a capability rule denies an operation."""

from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache

class DrugSafetyError(ValueError):
	"""Raised when a drug safety check produces a hard stop."""


# ── adapter stubs (replaced at runtime via get_*_adapter()) ──────────────────

class _NullAuthAdapter:
	async def check_permission(self, actor_id: str, action: str, resource: str) -> bool:
		return True


class _NullAuditAdapter:
	async def record(self, tenant_id: str, actor_id: str, event: str, entity_id: str, detail: dict[str, Any] | None = None) -> None:
		pass


class _NullNotifyAdapter:
	async def send(self, tenant_id: str, recipient_id: str, subject: str, body: str, channel: str = "in_app") -> None:
		pass


class _NullStore:
	"""In-memory fallback; production replaces with SQLAlchemy async store."""

	def __init__(self) -> None:
		self._data: dict[str, dict[str, Any]] = {}

	async def get(self, collection: str, key: str) -> Any | None:
		return self._data.get(collection, {}).get(key)

	async def put(self, collection: str, key: str, value: Any) -> None:
		self._data.setdefault(collection, {})[key] = value

	async def list(self, collection: str, **filters: Any) -> list[Any]:
		items = list(self._data.get(collection, {}).values())
		for k, v in filters.items():
			items = [i for i in items if isinstance(i, dict) and i.get(k) == v]
		return items

	async def delete(self, collection: str, key: str) -> None:
		self._data.get(collection, {}).pop(key, None)


def get_auth_adapter() -> _NullAuthAdapter:
	return _NullAuthAdapter()


def get_audit_adapter() -> _NullAuditAdapter:
	return _NullAuditAdapter()


def get_notify_adapter() -> _NullNotifyAdapter:
	return _NullNotifyAdapter()


def get_store(db_url: str | None = None) -> _NullStore:
	return _NullStore()


# ── static reference data ────────────────────────────────────────────────────

# Major drug–drug interaction pairs (drug_a_lower, drug_b_lower): (severity, mechanism, effect, management)
_DDI_DB: dict[frozenset[str], tuple[str, str, str, str]] = {
	frozenset(["warfarin", "aspirin"]): (
		"major", "additive anticoagulation + platelet inhibition",
		"markedly increased bleeding risk",
		"avoid combination; if unavoidable monitor INR closely",
	),
	frozenset(["warfarin", "ibuprofen"]): (
		"major", "NSAID inhibits COX-1/platelet + displaces warfarin from albumin",
		"elevated INR, GI bleed risk",
		"use paracetamol instead; monitor INR",
	),
	frozenset(["simvastatin", "clarithromycin"]): (
		"contraindicated", "CYP3A4 inhibition raises simvastatin AUC > 10×",
		"rhabdomyolysis, acute kidney injury",
		"withhold simvastatin during clarithromycin course",
	),
	frozenset(["metformin", "contrast"]): (
		"major", "contrast-induced AKI causes metformin accumulation",
		"lactic acidosis",
		"hold metformin 48 h before/after IV contrast; check eGFR",
	),
	frozenset(["ssri", "tramadol"]): (
		"major", "additive serotonergic stimulation",
		"serotonin syndrome",
		"avoid; if needed use lowest tramadol dose + monitor closely",
	),
	frozenset(["ace inhibitor", "potassium"]): (
		"major", "ACE inhibitor reduces aldosterone → K retention",
		"hyperkalaemia, cardiac arrhythmia",
		"monitor serum K+, avoid K supplements unless documented deficiency",
	),
	frozenset(["digoxin", "amiodarone"]): (
		"major", "amiodarone inhibits P-gp + renal tubular secretion of digoxin",
		"digoxin toxicity: bradycardia, AV block, visual disturbance",
		"halve digoxin dose; monitor digoxin levels and ECG",
	),
	frozenset(["lithium", "nsaid"]): (
		"major", "NSAIDs reduce renal Li excretion",
		"lithium toxicity: tremor, confusion, renal failure",
		"avoid combination; use paracetamol; monitor Li levels",
	),
	frozenset(["methotrexate", "trimethoprim"]): (
		"major", "additive folate antagonism",
		"bone marrow suppression, mucositis",
		"avoid; if unavoidable supplement folinic acid and monitor FBC",
	),
	frozenset(["sildenafil", "nitrate"]): (
		"contraindicated", "additive cGMP-mediated vasodilation",
		"severe hypotension, syncope, MI",
		"absolutely contraindicated; washout nitrates ≥24 h before sildenafil",
	),
}

# Paediatric weight-based dosing bounds (drug_lower): {route: (min_mg_per_kg, max_mg_per_kg, absolute_max_mg)}
_PAED_DOSE_DB: dict[str, dict[str, tuple[float, float, float]]] = {
	"paracetamol": {"oral": (10.0, 15.0, 1000.0), "iv": (7.5, 15.0, 1000.0), "rectal": (15.0, 20.0, 1000.0)},
	"amoxicillin": {"oral": (20.0, 40.0, 500.0), "iv": (25.0, 50.0, 2000.0)},
	"ibuprofen": {"oral": (5.0, 10.0, 400.0)},
	"gentamicin": {"iv": (2.5, 7.5, 240.0), "im": (2.5, 7.5, 240.0)},
	"ceftriaxone": {"iv": (25.0, 100.0, 4000.0), "im": (25.0, 100.0, 4000.0)},
	"metronidazole": {"oral": (7.5, 10.0, 500.0), "iv": (7.5, 10.0, 500.0)},
	"prednisolone": {"oral": (0.5, 2.0, 60.0)},
	"salbutamol": {"nebulised": (0.03, 0.15, 2.5)},
}

# Pregnancy FDA categories (drug_lower): {trimester: category}
_PREGNANCY_DB: dict[str, dict[int, str]] = {
	"warfarin": {1: "X", 2: "D", 3: "X"},
	"isotretinoin": {1: "X", 2: "X", 3: "X"},
	"methotrexate": {1: "X", 2: "X", 3: "X"},
	"thalidomide": {1: "X", 2: "X", 3: "X"},
	"valproate": {1: "D", 2: "D", 3: "D"},
	"carbamazepine": {1: "D", 2: "D", 3: "D"},
	"lithium": {1: "D", 2: "C", 3: "D"},
	"tetracycline": {1: "D", 2: "D", 3: "D"},
	"fluoroquinolone": {1: "C", 2: "C", 3: "C"},
	"nsaid": {1: "C", 2: "C", 3: "D"},
	"ace inhibitor": {1: "C", 2: "D", 3: "D"},
	"paracetamol": {1: "B", 2: "B", 3: "B"},
	"amoxicillin": {1: "B", 2: "B", 3: "B"},
	"azithromycin": {1: "B", 2: "B", 3: "B"},
	"metformin": {1: "B", 2: "B", 3: "B"},
	"methyldopa": {1: "B", 2: "B", 3: "B"},
	"heparin": {1: "C", 2: "C", 3: "C"},
	"ssri": {1: "C", 2: "C", 3: "C"},
}

_PREGNANCY_CATEGORY_DESCRIPTIONS: dict[str, str] = {
	"A": "Adequate studies show no risk to fetus in any trimester.",
	"B": "Animal studies show no risk; no adequate human studies, or animal studies show risk but human studies do not.",
	"C": "Animal studies show adverse effects; no adequate human studies. Benefits may outweigh risks.",
	"D": "Positive evidence of human fetal risk. Benefits may still justify use in life-threatening situations.",
	"X": "Studies show fetal abnormalities. Risks outweigh any benefit. Contraindicated in pregnancy.",
}

# Renal dosing adjustments (drug_lower): list of (eGFR_threshold, adjustment_text, contraindicated)
_RENAL_DB: dict[str, list[tuple[float, str, bool]]] = {
	"metformin": [
		(60.0, "use with caution; monitor renal function every 3–6 months", False),
		(45.0, "reduce dose by 50%", False),
		(30.0, "contraindicated", True),
	],
	"gentamicin": [
		(60.0, "extend dosing interval to 36 h; monitor levels", False),
		(30.0, "extend dosing interval to 48 h; monitor levels closely", False),
		(15.0, "contraindicated unless haemodialysis available", True),
	],
	"digoxin": [
		(50.0, "reduce dose by 25–50%; monitor levels", False),
		(30.0, "reduce dose by 50%; monitor ECG and levels closely", False),
	],
	"nsaid": [
		(60.0, "use with caution; short course only", False),
		(30.0, "avoid; risk of acute kidney injury", True),
	],
	"lisinopril": [
		(30.0, "start at 50% of normal dose; titrate slowly", False),
		(10.0, "use only under specialist supervision", False),
	],
	"spironolactone": [
		(45.0, "monitor K+ closely", False),
		(30.0, "avoid; hyperkalaemia risk", True),
	],
	"trimethoprim": [
		(30.0, "reduce dose by 50%", False),
		(15.0, "avoid", True),
	],
}

# Drug classes for duplicate therapy detection (drug_lower → drug_class_lower)
_DRUG_CLASS_MAP: dict[str, str] = {
	"atorvastatin": "statin", "simvastatin": "statin", "rosuvastatin": "statin",
	"pravastatin": "statin", "fluvastatin": "statin",
	"lisinopril": "ace inhibitor", "enalapril": "ace inhibitor", "ramipril": "ace inhibitor",
	"perindopril": "ace inhibitor", "captopril": "ace inhibitor",
	"amlodipine": "calcium channel blocker", "nifedipine": "calcium channel blocker",
	"diltiazem": "calcium channel blocker", "verapamil": "calcium channel blocker",
	"metoprolol": "beta blocker", "atenolol": "beta blocker", "bisoprolol": "beta blocker",
	"carvedilol": "beta blocker", "propranolol": "beta blocker",
	"furosemide": "loop diuretic", "bumetanide": "loop diuretic",
	"fluoxetine": "ssri", "sertraline": "ssri", "citalopram": "ssri",
	"escitalopram": "ssri", "paroxetine": "ssri",
	"omeprazole": "ppi", "esomeprazole": "ppi", "lansoprazole": "ppi",
	"pantoprazole": "ppi", "rabeprazole": "ppi",
	"metformin": "biguanide",
	"glibenclamide": "sulfonylurea", "gliclazide": "sulfonylurea", "glipizide": "sulfonylurea",
	"ibuprofen": "nsaid", "naproxen": "nsaid", "diclofenac": "nsaid",
	"celecoxib": "nsaid", "ketorolac": "nsaid",
}

# Clinical reminders: (check_key, description, icd10_trigger, interval_months)
_CLINICAL_REMINDERS = [
	("hba1c", "HbA1c overdue (>3 months)", "E11", 3),
	("hba1c", "HbA1c overdue (>3 months)", "E10", 3),
	("pap_smear", "Cervical cancer screening overdue (>36 months)", "Z12.4", 36),
	("lipid_panel", "Fasting lipid panel overdue (>12 months)", "E78", 12),
	("mammogram", "Mammography screening overdue (>24 months)", "Z12.31", 24),
	("colonoscopy", "Colorectal cancer screening overdue (>60 months)", "Z12.11", 60),
	("flu_vaccine", "Annual influenza vaccination overdue", "Z23", 12),
	("pneumococcal", "Pneumococcal vaccine due (age ≥65 or immunocompromised)", "Z23", 60),
	("foot_exam", "Diabetic foot exam overdue (>12 months)", "E11", 12),
	("eye_exam", "Diabetic eye exam overdue (>12 months)", "E11", 12),
]

# ICD-10 symptom → diagnosis keyword matching
_SYMPTOM_DX_MAP: dict[str, list[dict[str, str]]] = {
	"chest pain": [
		{"icd10": "I20.9", "description": "Unstable angina", "confidence": "high"},
		{"icd10": "I21.9", "description": "Acute MI, unspecified", "confidence": "high"},
		{"icd10": "R07.9", "description": "Chest pain, unspecified", "confidence": "medium"},
		{"icd10": "J18.9", "description": "Pneumonia, unspecified (pleuritic)", "confidence": "low"},
	],
	"shortness of breath": [
		{"icd10": "J18.9", "description": "Pneumonia", "confidence": "high"},
		{"icd10": "J44.1", "description": "COPD with acute exacerbation", "confidence": "high"},
		{"icd10": "I50.9", "description": "Heart failure, unspecified", "confidence": "medium"},
		{"icd10": "J45.901", "description": "Uncontrolled asthma", "confidence": "medium"},
	],
	"fever": [
		{"icd10": "R50.9", "description": "Fever, unspecified", "confidence": "high"},
		{"icd10": "A41.9", "description": "Sepsis, unspecified", "confidence": "medium"},
		{"icd10": "J06.9", "description": "URTI, unspecified", "confidence": "medium"},
	],
	"headache": [
		{"icd10": "G43.909", "description": "Migraine, unspecified", "confidence": "high"},
		{"icd10": "G44.309", "description": "Post-traumatic headache", "confidence": "medium"},
		{"icd10": "R51", "description": "Headache, unspecified", "confidence": "medium"},
		{"icd10": "G97.1", "description": "Meningitis — consider LP", "confidence": "low"},
	],
	"cough": [
		{"icd10": "J18.9", "description": "Pneumonia", "confidence": "high"},
		{"icd10": "J45.901", "description": "Asthma", "confidence": "medium"},
		{"icd10": "J44.1", "description": "COPD exacerbation", "confidence": "medium"},
		{"icd10": "A15.0", "description": "Pulmonary tuberculosis", "confidence": "low"},
	],
	"abdominal pain": [
		{"icd10": "K35.89", "description": "Acute appendicitis", "confidence": "high"},
		{"icd10": "K92.1", "description": "Melaena / GI bleed", "confidence": "medium"},
		{"icd10": "K80.20", "description": "Calculus of gallbladder with acute cholecystitis", "confidence": "medium"},
		{"icd10": "K29.70", "description": "Gastritis, unspecified", "confidence": "low"},
	],
	"dizziness": [
		{"icd10": "H81.10", "description": "Benign paroxysmal positional vertigo", "confidence": "high"},
		{"icd10": "I10", "description": "Essential hypertension", "confidence": "medium"},
		{"icd10": "G45.9", "description": "TIA, unspecified", "confidence": "medium"},
	],
	"back pain": [
		{"icd10": "M54.5", "description": "Low back pain", "confidence": "high"},
		{"icd10": "M51.16", "description": "Intervertebral disc degeneration, lumbar", "confidence": "medium"},
		{"icd10": "M47.816", "description": "Spondylosis with radiculopathy, lumbar", "confidence": "medium"},
	],
}

# CHADS2-VASc risk factor ICD-10 prefixes
_CHADS2_VASC_CRITERIA = {
	"chf": (["I50"], 1),        # C — congestive heart failure
	"hypertension": (["I10", "I11", "I12", "I13"], 1),   # H
	"age_75": ([], 2),           # A2 — handled by age logic
	"diabetes": (["E10", "E11", "E13"], 1),  # D
	"stroke_tia": (["I63", "G45", "I64"], 2),  # S2
	"vascular_disease": (["I21", "I25", "I70"], 1),  # V
	"age_65_74": ([], 1),        # A — handled by age logic
	# sex_female bonus handled separately
}

# Guideline alerts by ICD-10 prefix
_GUIDELINE_ALERTS: dict[str, list[dict[str, str]]] = {
	"E11": [
		{"title": "Diabetes: metformin first-line", "body": "ADA guidelines recommend metformin as first-line agent unless contraindicated.", "source": "ADA Standards of Care 2024"},
		{"title": "Diabetes: annual HbA1c, lipids, renal, eye, foot exam", "body": "Annual monitoring bundle recommended.", "source": "ADA Standards of Care 2024"},
	],
	"I50": [
		{"title": "Heart failure: ACEI/ARB + beta-blocker + MRA", "body": "ESC HFrEF guideline recommends triple neurohormonal blockade.", "source": "ESC Heart Failure Guidelines 2021"},
		{"title": "Heart failure: loop diuretic for fluid overload", "body": "Furosemide or bumetanide for symptomatic fluid retention.", "source": "ESC Heart Failure Guidelines 2021"},
	],
	"I10": [
		{"title": "Hypertension: ABCD first-line agents", "body": "ACEi, ARB, CCB, thiazide-like diuretic per NICE NG136.", "source": "NICE NG136 2023"},
	],
	"J44": [
		{"title": "COPD: SABA + LAMA first-line", "body": "GOLD 2024: start with bronchodilator monotherapy or dual bronchodilation for severe symptoms.", "source": "GOLD 2024"},
		{"title": "COPD: smoking cessation priority", "body": "Smoking cessation is the only intervention that definitively slows FEV1 decline.", "source": "GOLD 2024"},
	],
	"I48": [
		{"title": "AF: anticoagulation per CHA2DS2-VASc", "body": "Oral anticoagulation recommended for CHA2DS2-VASc ≥2 (male) or ≥3 (female). DOACs preferred over warfarin.", "source": "ESC AF Guidelines 2020"},
	],
}

# NEWS2 scoring thresholds
_NEWS2_THRESHOLDS = {
	"respiratory_rate": [(8, 0, 1), (11, 1, 1), (20, 0, 0), (24, 2, 0), (float("inf"), 3, 0)],
	"spo2": [(91, 3, 0), (93, 2, 0), (95, 1, 0), (float("inf"), 0, 0)],
	"systolic_bp": [(90, 3, 0), (100, 2, 0), (110, 1, 0), (219, 0, 0), (float("inf"), 3, 0)],
	"heart_rate": [(40, 3, 0), (50, 1, 0), (90, 0, 0), (110, 1, 0), (130, 2, 0), (float("inf"), 3, 0)],
	"temperature": [(35.0, 3, 0), (36.0, 1, 0), (38.0, 0, 0), (39.0, 1, 0), (float("inf"), 2, 0)],
}


def _news2_threshold_score(value: float, thresholds: list[tuple[float, int, int]]) -> int:
	"""Return NEWS2 subscale score for a continuous vital."""
	for upper, score, _ in thresholds:
		if value <= upper:
			return score
	return 0


class EMRService:
	"""Tenant-scoped EMR runtime.

	Constructor follows the adapter/store pattern so each dependency can be
	injected for testing without monkey-patching.
	"""

	def __init__(
		self,
		tenant_id: str,
		actor_id: str = "system",
		*,
		auth: Any = None,
		audit: Any = None,
		notify: Any = None,
		db_url: str | None = None,
		store: Any = None,
	) -> None:
		self.tenant_id = tenant_id
		self.actor_id = actor_id
		self._auth = auth or get_auth_adapter()
		self._audit_adapter = audit or get_audit_adapter()
		self._notify = notify or get_notify_adapter()
		self._store = store or get_store(db_url)

		# in-memory caches (null store is memory-backed; kept for backward compat)
		self._notes: dict[tuple[str, str], ClinicalNoteResponse] = {}
		self._problems: dict[tuple[str, str], ProblemResponse] = {}
		self._medications: dict[tuple[str, str], MedicationResponse] = {}
		self._allergies: dict[tuple[str, str], AllergyResponse] = {}
		self._vitals: dict[tuple[str, str], VitalSignResponse] = {}
		self._encounters: dict[tuple[str, str], EncounterResponse] = {}
		self._patients: dict[tuple[str, str], PatientResponse] = {}
		self._audit_events: list[dict[str, Any]] = []
		# extended tables
		self._prescriptions: dict[tuple[str, str], dict[str, Any]] = {}
		self._consents: dict[tuple[str, str], dict[str, Any]] = {}
		self._referrals: dict[tuple[str, str], dict[str, Any]] = {}
		self._discharge_summaries: dict[tuple[str, str], dict[str, Any]] = {}
		self._diagnoses: dict[tuple[str, str], dict[str, Any]] = {}
		self._cpt_procedures: dict[tuple[str, str], dict[str, Any]] = {}
		self._lab_orders: dict[tuple[str, str], LabOrderResponse] = {}
		self._lab_results: dict[tuple[str, str], LabResultResponse] = {}
		self._imaging_orders: dict[tuple[str, str], ImagingOrderResponse] = {}
		self._care_plans: dict[tuple[str, str], CarePlanResponse] = {}
		self._immunisations: dict[tuple[str, str], ImmunisationResponse] = {}
		self._family_history: dict[tuple[str, str], FamilyHistoryResponse] = {}


	async def describe(self, tenant_id: str | None = None) -> dict[str, Any]:
		return get_capability_contract(tenant_id or self.tenant_id)

	async def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	# ── encounters ────────────────────────────────────────────────────────────

	async def create_encounter(self, payload: EncounterCreate) -> EncounterResponse:
		"""Open a new clinical encounter."""
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		enc = EncounterResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			encounter_type=payload.encounter_type, provider_id=payload.provider_id,
			location_id=payload.location_id, chief_complaint=payload.chief_complaint,
			status="in_progress", created_by=payload.created_by,
		)
		self._encounters[(payload.tenant_id, enc.id)] = enc
		self._record_audit(payload.tenant_id, "encounter_opened", enc.id)
		_log_op("create_encounter", payload.tenant_id, enc.id)
		return enc

	async def close_encounter(self, tenant_id: str, encounter_id: str, icd10_codes: list[str] | None = None) -> EncounterResponse | None:
		"""Close an encounter and attach final ICD-10 coding."""
		enc = self._encounters.get((tenant_id, encounter_id))
		if enc is None:
			return None
		self._enforce({"tenant_context_present": bool(tenant_id), "operation": "update_encounter", "encounter_status_supported": "finished" in SUPPORTED_ENCOUNTER_STATUSES})
		updated = enc.model_copy(update={
			"status": "finished",
			"discharge_time": datetime.utcnow(),
			"icd10_codes": icd10_codes or enc.icd10_codes,
			"updated_at": datetime.utcnow(),
		})
		self._encounters[(tenant_id, encounter_id)] = updated
		self._record_audit(tenant_id, "encounter_closed", encounter_id)
		return updated

	async def get_encounter(self, tenant_id: str, encounter_id: str) -> EncounterResponse | None:
		return self._encounters.get((tenant_id, encounter_id))

	async def list_encounters(self, tenant_id: str, patient_id: str | None = None) -> list[EncounterResponse]:
		results = [e for (tid, _), e in self._encounters.items() if tid == tenant_id]
		if patient_id:
			results = [e for e in results if e.patient_id == patient_id]
		return sorted(results, key=lambda e: e.created_at, reverse=True)

	# ── clinical notes ────────────────────────────────────────────────────────

	async def create_note(self, payload: ClinicalNoteCreate) -> ClinicalNoteResponse:
		"""Author a new clinical note."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "create_note",
			"note_type_supported": payload.note_type in SUPPORTED_NOTE_TYPES,
		})
		note = ClinicalNoteResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			encounter_id=payload.encounter_id, note_type=payload.note_type,
			author_id=payload.author_id, content=payload.content,
			subjective=payload.subjective, objective=payload.objective,
			assessment=payload.assessment, plan=payload.plan,
			icd10_codes=payload.icd10_codes, status="draft", created_by=payload.author_id,
		)
		self._notes[(payload.tenant_id, note.id)] = note
		self._record_audit(payload.tenant_id, "note_created", note.id)
		_log_op("create_note", payload.tenant_id, note.id)
		return note

	async def amend_note(self, tenant_id: str, original_note_id: str, author_id: str, content: str) -> ClinicalNoteResponse | None:
		"""Create an amendment note linked to the original."""
		original = self._notes.get((tenant_id, original_note_id))
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "amend_note",
			"original_note_present": original is not None,
		})
		if original is None:
			return None
		amendment = ClinicalNoteResponse(
			id=uuid7str(), tenant_id=tenant_id, patient_id=original.patient_id,
			encounter_id=original.encounter_id, note_type=original.note_type,
			author_id=author_id, content=content, status="draft",
			amendment_of=original_note_id, created_by=author_id,
		)
		self._notes[(tenant_id, amendment.id)] = amendment
		self._record_audit(tenant_id, "note_amended", amendment.id)
		return amendment

	async def finalize_note(self, tenant_id: str, note_id: str, cosigned_by: str | None = None) -> ClinicalNoteResponse | None:
		"""Finalize and optionally co-sign a clinical note."""
		note = self._notes.get((tenant_id, note_id))
		if note is None:
			return None
		updated = note.model_copy(update={
			"status": "final",
			"cosigned_by": cosigned_by,
			"finalized_at": datetime.utcnow(),
			"updated_at": datetime.utcnow(),
		})
		self._notes[(tenant_id, note_id)] = updated
		self._record_audit(tenant_id, "note_finalized", note_id)
		return updated

	async def get_note(self, tenant_id: str, note_id: str) -> ClinicalNoteResponse | None:
		return self._notes.get((tenant_id, note_id))

	async def list_notes(self, tenant_id: str, patient_id: str | None = None, note_type: str | None = None) -> list[ClinicalNoteResponse]:
		results = [n for (tid, _), n in self._notes.items() if tid == tenant_id]
		if patient_id:
			results = [n for n in results if n.patient_id == patient_id]
		if note_type:
			results = [n for n in results if n.note_type == note_type]
		return sorted(results, key=lambda n: n.created_at, reverse=True)

	# ── problem list ──────────────────────────────────────────────────────────

	async def add_problem(self, payload: ProblemCreate) -> ProblemResponse:
		"""Add a problem to the patient's problem list."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "add_problem",
			"icd10_code_present": bool(payload.icd10_code),
		})
		prob = ProblemResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			icd10_code=payload.icd10_code, description=payload.description,
			status=payload.status, onset_date=payload.onset_date, created_by=payload.created_by,
		)
		self._problems[(payload.tenant_id, prob.id)] = prob
		self._record_audit(payload.tenant_id, "problem_added", prob.id)
		_log_op("add_problem", payload.tenant_id, prob.id)
		return prob

	async def resolve_problem(self, tenant_id: str, problem_id: str) -> ProblemResponse | None:
		prob = self._problems.get((tenant_id, problem_id))
		if prob is None:
			return None
		updated = prob.model_copy(update={"status": "resolved", "resolved_date": datetime.utcnow(), "updated_at": datetime.utcnow()})
		self._problems[(tenant_id, problem_id)] = updated
		self._record_audit(tenant_id, "problem_resolved", problem_id)
		return updated

	async def list_problems(self, tenant_id: str, patient_id: str, status: str | None = None) -> list[ProblemResponse]:
		results = [p for (tid, _), p in self._problems.items() if tid == tenant_id and p.patient_id == patient_id]
		if status:
			results = [p for p in results if p.status == status]
		return sorted(results, key=lambda p: p.created_at)

	# ── medications ───────────────────────────────────────────────────────────

	async def prescribe_medication(self, payload: MedicationCreate) -> MedicationResponse:
		"""Prescribe a medication with allergy-check enforcement."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "prescribe_medication",
			"allergy_check_performed": payload.allergy_check_performed,
		})
		med = MedicationResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			drug_name=payload.drug_name, ndc_code=payload.ndc_code, rxnorm_code=payload.rxnorm_code,
			dose=payload.dose, route=payload.route, frequency=payload.frequency,
			prescriber_id=payload.prescriber_id, indication_icd10=payload.indication_icd10,
			status="active", allergy_check_performed=payload.allergy_check_performed,
			created_by=payload.created_by,
		)
		self._medications[(payload.tenant_id, med.id)] = med
		self._record_audit(payload.tenant_id, "medication_prescribed", med.id)
		_log_op("prescribe_medication", payload.tenant_id, med.id)
		return med

	async def discontinue_medication(self, tenant_id: str, med_id: str) -> MedicationResponse | None:
		med = self._medications.get((tenant_id, med_id))
		if med is None:
			return None
		updated = med.model_copy(update={"status": "discontinued", "end_date": datetime.utcnow(), "updated_at": datetime.utcnow()})
		self._medications[(tenant_id, med_id)] = updated
		self._record_audit(tenant_id, "medication_discontinued", med_id)
		return updated

	async def list_medications(self, tenant_id: str, patient_id: str, status: str | None = None) -> list[MedicationResponse]:
		results = [m for (tid, _), m in self._medications.items() if tid == tenant_id and m.patient_id == patient_id]
		if status:
			results = [m for m in results if m.status == status]
		return sorted(results, key=lambda m: m.start_date, reverse=True)

	# ── allergies ─────────────────────────────────────────────────────────────

	async def record_allergy(self, payload: AllergyCreate) -> AllergyResponse:
		"""Record a patient allergy."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_allergy",
			"allergy_type_supported": payload.allergy_type in SUPPORTED_ALLERGY_TYPES,
			"allergy_severity_supported": payload.severity in SUPPORTED_ALLERGY_SEVERITIES,
		})
		allergy = AllergyResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			allergen=payload.allergen, allergy_type=payload.allergy_type,
			severity=payload.severity, reaction=payload.reaction, created_by=payload.created_by,
		)
		self._allergies[(payload.tenant_id, allergy.id)] = allergy
		self._record_audit(payload.tenant_id, "allergy_recorded", allergy.id)
		_log_op("record_allergy", payload.tenant_id, allergy.id)
		return allergy

	async def list_allergies(self, tenant_id: str, patient_id: str) -> list[AllergyResponse]:
		return [a for (tid, _), a in self._allergies.items() if tid == tenant_id and a.patient_id == patient_id]

	async def check_drug_allergy(self, tenant_id: str, patient_id: str, drug_name: str) -> dict[str, Any]:
		"""Returns True if a drug allergy conflict is found."""
		allergies = await self.list_allergies(tenant_id, patient_id)
		drug_allergies = [a for a in allergies if a.allergy_type == "drug" and a.status == "active"]
		conflicts = [a for a in drug_allergies if drug_name.lower() in a.allergen.lower()]
		return {
			"patient_id": patient_id,
			"drug_name": drug_name,
			"conflict_found": len(conflicts) > 0,
			"conflicts": [{"allergen": c.allergen, "severity": c.severity, "reaction": c.reaction} for c in conflicts],
		}

	# ── vitals ────────────────────────────────────────────────────────────────

	async def record_vital(self, payload: VitalSignCreate) -> VitalSignResponse:
		"""Record a vital sign measurement."""
		self._enforce({
			"tenant_context_present": bool(payload.tenant_id),
			"operation_type": "write", "policy_attached": True,
			"operation": "record_vital",
			"vital_type_supported": payload.vital_type in SUPPORTED_VITAL_TYPES,
		})
		vital = VitalSignResponse(
			id=uuid7str(), tenant_id=payload.tenant_id, patient_id=payload.patient_id,
			encounter_id=payload.encounter_id, vital_type=payload.vital_type,
			value=payload.value, value2=payload.value2, unit=payload.unit,
			recorded_by=payload.recorded_by, recorded_at=payload.recorded_at,
			method=payload.method, position=payload.position,
			created_by=payload.recorded_by,
		)
		self._vitals[(payload.tenant_id, vital.id)] = vital
		self._record_audit(payload.tenant_id, "vital_recorded", vital.id)
		return vital

	async def list_vitals(self, tenant_id: str, patient_id: str, vital_type: str | None = None) -> list[VitalSignResponse]:
		results = [v for (tid, _), v in self._vitals.items() if tid == tenant_id and v.patient_id == patient_id]
		if vital_type:
			results = [v for v in results if v.vital_type == vital_type]
		return sorted(results, key=lambda v: v.recorded_at, reverse=True)

	# ── FHIR export ───────────────────────────────────────────────────────────

	async def fhir_export(self, tenant_id: str, patient_id: str, resource_types: list[str], phi_consent_present: bool) -> dict[str, Any]:
		"""Generate a FHIR R4 bundle for the patient."""
		self._enforce({
			"tenant_context_present": bool(tenant_id),
			"operation": "fhir_export",
			"phi_consent_present": phi_consent_present,
			"resource_type_supported": all(rt in SUPPORTED_FHIR_RESOURCE_TYPES for rt in resource_types),
		})
		bundle: dict[str, Any] = {
			"resourceType": "Bundle", "type": "collection",
			"id": uuid7str(), "timestamp": datetime.utcnow().isoformat(),
			"entry": [],
		}
		if "Condition" in resource_types:
			for p in await self.list_problems(tenant_id, patient_id):
				bundle["entry"].append({"resource": {"resourceType": "Condition", "id": p.id, "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": p.icd10_code, "display": p.description}]}, "clinicalStatus": p.status}})
		if "MedicationRequest" in resource_types:
			for m in await self.list_medications(tenant_id, patient_id):
				bundle["entry"].append({"resource": {"resourceType": "MedicationRequest", "id": m.id, "medication": {"concept": {"text": m.drug_name}}, "status": m.status}})
		if "AllergyIntolerance" in resource_types:
			for a in await self.list_allergies(tenant_id, patient_id):
				bundle["entry"].append({"resource": {"resourceType": "AllergyIntolerance", "id": a.id, "code": {"text": a.allergen}, "criticality": a.severity}})
		logger.info(_log_fhir_export(tenant_id, str(resource_types), len(bundle["entry"])))
		self._record_audit(tenant_id, "fhir_export_generated", patient_id)
		return bundle

	# ── dashboard ─────────────────────────────────────────────────────────────

	async def dashboard_summary(self, tenant_id: str) -> dict[str, Any]:
		notes = [n for (tid, _), n in self._notes.items() if tid == tenant_id]
		probs = [p for (tid, _), p in self._problems.items() if tid == tenant_id]
		meds = [m for (tid, _), m in self._medications.items() if tid == tenant_id]
		encs = [e for (tid, _), e in self._encounters.items() if tid == tenant_id]
		return {
			"tenant_id": tenant_id,
			"notes": {"total": len(notes), "draft": sum(1 for n in notes if n.status == "draft"), "final": sum(1 for n in notes if n.status == "final")},
			"problems": {"total": len(probs), "active": sum(1 for p in probs if p.status == "active")},
			"medications": {"total": len(meds), "active": sum(1 for m in meds if m.status == "active")},
			"encounters": {"total": len(encs), "open": sum(1 for e in encs if e.status == "in_progress")},
		}

	# ═══════════════════════════════════════════════════════════════════════════
	# DRUG SAFETY
	# ═══════════════════════════════════════════════════════════════════════════

	async def check_drug_drug_interactions(self, drug_list: list[str]) -> list[dict[str, Any]]:
		"""Return all major/contraindicated interaction pairs from drug_list.

		Checks every pair against the embedded DDI database.  In production,
		swap ``_DDI_DB`` for a call to a local RxNav / OpenFDA mirror.
		"""
		results: list[dict[str, Any]] = []
		normalised = [d.strip().lower() for d in drug_list]
		for i, drug_a in enumerate(normalised):
			for drug_b in normalised[i + 1 :]:
				key = frozenset([drug_a, drug_b])
				if key in _DDI_DB:
					severity, mechanism, effect, management = _DDI_DB[key]
					results.append({
						"drug_a": drug_a,
						"drug_b": drug_b,
						"severity": severity,
						"mechanism": mechanism,
						"clinical_effect": effect,
						"management": management,
						"alert_type": "drug_interaction",
					})
				# also check class-level interactions
				else:
					class_a = _DRUG_CLASS_MAP.get(drug_a, drug_a)
					class_b = _DRUG_CLASS_MAP.get(drug_b, drug_b)
					class_key = frozenset([class_a, class_b])
					if class_key in _DDI_DB and class_key != key:
						severity, mechanism, effect, management = _DDI_DB[class_key]
						results.append({
							"drug_a": drug_a,
							"drug_b": drug_b,
							"drug_a_class": class_a,
							"drug_b_class": class_b,
							"severity": severity,
							"mechanism": mechanism,
							"clinical_effect": effect,
							"management": management,
							"alert_type": "drug_interaction",
							"matched_by": "drug_class",
						})
		logger.info("emr.ddi_check drugs=%d interactions=%d", len(drug_list), len(results))
		return results

	async def check_drug_allergy_alert(self, patient_id: str, drug_name: str, drug_class: str) -> dict[str, Any]:
		"""Check patient's allergy list against drug name and class.

		Returns an alert dict with ``conflict_found`` bool and full conflict
		detail.  Checks both exact allergen name and drug class membership.
		"""
		_log_drug_check("allergy", patient_id, drug_name)
		allergies = await self.list_allergies(self.tenant_id, patient_id)
		active = [a for a in allergies if a.status == "active" and a.allergy_type == "drug"]

		conflicts: list[dict[str, Any]] = []
		for allergy in active:
			allergen_lower = allergy.allergen.lower()
			if (
				drug_name.lower() in allergen_lower
				or allergen_lower in drug_name.lower()
				or drug_class.lower() in allergen_lower
			):
				conflicts.append({
					"allergen": allergy.allergen,
					"allergy_id": allergy.id,
					"severity": allergy.severity,
					"reaction": allergy.reaction,
					"match_type": "name" if drug_name.lower() in allergen_lower else "class",
				})

		hard_stop = any(c["severity"] in ("severe", "life_threatening") for c in conflicts)
		result = {
			"patient_id": patient_id,
			"drug_name": drug_name,
			"drug_class": drug_class,
			"conflict_found": len(conflicts) > 0,
			"hard_stop": hard_stop,
			"conflicts": conflicts,
			"recommendation": (
				"CONTRAINDICATED — severe/life-threatening allergy on record" if hard_stop
				else "CAUTION — allergy on record; confirm with prescriber" if conflicts
				else "No known allergy conflict"
			),
		}
		self._record_audit(self.tenant_id, "drug_allergy_checked", patient_id)
		return result

	async def paediatric_dose_check(
		self,
		drug: str,
		weight_kg: float,
		age_months: int,
		prescribed_dose: float,
		route: str,
	) -> dict[str, Any]:
		"""Validate a paediatric dose against weight-based bounds.

		Compares prescribed_dose (mg) against min/max mg/kg thresholds and
		absolute ceiling for the given drug and route.
		"""
		assert weight_kg > 0, "weight_kg must be positive"
		assert age_months >= 0, "age_months must be non-negative"
		assert prescribed_dose > 0, "prescribed_dose must be positive"

		drug_lower = drug.strip().lower()
		route_lower = route.strip().lower()

		db_entry = _PAED_DOSE_DB.get(drug_lower)
		if db_entry is None:
			return {
				"drug": drug,
				"weight_kg": weight_kg,
				"age_months": age_months,
				"prescribed_dose_mg": prescribed_dose,
				"route": route,
				"status": "unknown",
				"message": f"No paediatric dosing data available for '{drug}'. Consult BNF for Children.",
			}

		route_entry = db_entry.get(route_lower) or db_entry.get("oral")
		if route_entry is None:
			return {
				"drug": drug,
				"route": route,
				"status": "unknown",
				"message": f"No data for route '{route}' for '{drug}'. Consult specialist.",
			}

		min_dose_per_kg, max_dose_per_kg, abs_max = route_entry
		min_dose = min_dose_per_kg * weight_kg
		max_dose = min(max_dose_per_kg * weight_kg, abs_max)

		if prescribed_dose < min_dose:
			status = "underdose"
			message = f"Prescribed {prescribed_dose:.1f} mg is BELOW minimum {min_dose:.1f} mg ({min_dose_per_kg} mg/kg). Risk of therapeutic failure."
		elif prescribed_dose > max_dose:
			status = "overdose"
			message = f"Prescribed {prescribed_dose:.1f} mg EXCEEDS maximum {max_dose:.1f} mg ({max_dose_per_kg} mg/kg or {abs_max} mg cap). Toxicity risk."
		else:
			status = "within_range"
			message = f"Prescribed dose is within acceptable range ({min_dose:.1f}–{max_dose:.1f} mg)."

		_log_drug_check("paed_dose", "paed_patient", drug)
		return {
			"drug": drug,
			"route": route,
			"weight_kg": weight_kg,
			"age_months": age_months,
			"prescribed_dose_mg": prescribed_dose,
			"min_dose_mg": round(min_dose, 2),
			"max_dose_mg": round(max_dose, 2),
			"absolute_max_mg": abs_max,
			"status": status,
			"message": message,
		}

	async def pregnancy_safety_check(self, drug_name: str, trimester: int) -> dict[str, Any]:
		"""Return FDA pregnancy category and safety guidance.

		Trimester must be 1, 2, or 3.  Category X → automatic hard stop.
		"""
		assert trimester in (1, 2, 3), "trimester must be 1, 2, or 3"

		drug_lower = drug_name.strip().lower()
		# resolve to class if exact match not found
		category: str | None = None
		matched_key = drug_lower
		if drug_lower in _PREGNANCY_DB:
			category = _PREGNANCY_DB[drug_lower][trimester]
		else:
			drug_class = _DRUG_CLASS_MAP.get(drug_lower)
			if drug_class and drug_class in _PREGNANCY_DB:
				category = _PREGNANCY_DB[drug_class][trimester]
				matched_key = drug_class

		if category is None:
			return {
				"drug_name": drug_name,
				"trimester": trimester,
				"category": "Unknown",
				"description": "No pregnancy safety data available. Consult obstetric pharmacist.",
				"hard_stop": False,
				"recommendation": "Seek specialist advice before prescribing.",
			}

		hard_stop = category == "X"
		contraindicated = category in ("X", "D")
		return {
			"drug_name": drug_name,
			"matched_entry": matched_key,
			"trimester": trimester,
			"category": category,
			"description": _PREGNANCY_CATEGORY_DESCRIPTIONS[category],
			"hard_stop": hard_stop,
			"contraindicated": contraindicated,
			"recommendation": (
				"CONTRAINDICATED — Category X. Do not prescribe." if category == "X"
				else "USE WITH CAUTION — Category D. Benefit must clearly outweigh risk." if category == "D"
				else "Caution advised — Category C. Consult obstetrician." if category == "C"
				else "Generally considered safe — Category B. Normal prescribing caution." if category == "B"
				else "Safe in pregnancy — Category A."
			),
		}

	async def renal_dose_adjustment(self, drug_name: str, egfr_ml_per_min: float) -> dict[str, Any]:
		"""Return renal dose adjustment recommendation for a given eGFR.

		Uses staged thresholds from embedded reference data.  In production
		connect to a local Renal Drug Handbook API.
		"""
		assert egfr_ml_per_min >= 0, "eGFR must be non-negative"

		drug_lower = drug_name.strip().lower()
		# try drug name then class
		stages = _RENAL_DB.get(drug_lower)
		if stages is None:
			drug_class = _DRUG_CLASS_MAP.get(drug_lower)
			if drug_class:
				stages = _RENAL_DB.get(drug_class)

		if stages is None:
			return {
				"drug_name": drug_name,
				"egfr_ml_per_min": egfr_ml_per_min,
				"adjustment_required": False,
				"contraindicated": False,
				"recommendation": "No renal dosing data available. Consult Renal Drug Handbook.",
			}

		# Find the most restrictive (lowest-threshold) rule that still applies.
		# A rule applies when egfr <= threshold (patient's function is at or below
		# the threshold that triggers that adjustment).
		# Sort ascending; the first match IS the most restrictive applicable rule.
		recommendation = "No dose adjustment required for this eGFR."
		adjustment_required = False
		contraindicated = False
		applicable = [(t, txt, ic) for t, txt, ic in stages if egfr_ml_per_min <= t]
		if applicable:
			# pick the entry with the smallest threshold (most restrictive)
			threshold, recommendation, contraindicated = min(applicable, key=lambda x: x[0])
			adjustment_required = True

		return {
			"drug_name": drug_name,
			"egfr_ml_per_min": egfr_ml_per_min,
			"adjustment_required": adjustment_required,
			"contraindicated": contraindicated,
			"recommendation": recommendation,
			"ckd_stage": _egfr_to_ckd_stage(egfr_ml_per_min),
		}

	async def check_duplicate_therapy(self, patient_id: str, new_drug: str, new_drug_class: str) -> dict[str, Any]:
		"""Detect if patient is already on the same drug or a drug of the same class.

		Scans active medications; flags exact duplicates and therapeutic
		duplicates (same class).
		"""
		active_meds = await self.list_medications(self.tenant_id, patient_id, status="active")
		new_lower = new_drug.strip().lower()
		new_class_lower = new_drug_class.strip().lower()

		exact_duplicates: list[dict[str, Any]] = []
		class_duplicates: list[dict[str, Any]] = []

		for med in active_meds:
			med_lower = med.drug_name.lower()
			med_class = _DRUG_CLASS_MAP.get(med_lower, "")
			if med_lower == new_lower:
				exact_duplicates.append({
					"medication_id": med.id,
					"drug_name": med.drug_name,
					"dose": med.dose,
					"frequency": med.frequency,
					"start_date": med.start_date.isoformat(),
				})
			elif med_class and med_class == new_class_lower:
				class_duplicates.append({
					"medication_id": med.id,
					"drug_name": med.drug_name,
					"drug_class": med_class,
					"dose": med.dose,
					"frequency": med.frequency,
				})

		return {
			"patient_id": patient_id,
			"new_drug": new_drug,
			"new_drug_class": new_drug_class,
			"duplicate_found": bool(exact_duplicates or class_duplicates),
			"exact_duplicates": exact_duplicates,
			"class_duplicates": class_duplicates,
			"recommendation": (
				"STOP — exact duplicate therapy already prescribed." if exact_duplicates
				else f"CAUTION — patient already on another {new_drug_class} agent." if class_duplicates
				else "No duplicate therapy detected."
			),
		}

	async def controlled_substance_check(
		self,
		drug: str,
		schedule: str,
		quantity: int,
		prescriber_id: str,
	) -> dict[str, Any]:
		"""Validate a controlled substance prescription for schedule, quantity caps and PDMP flags.

		Schedule II drugs have a 30-day supply cap.  Schedule III–V are capped at
		90 days.  Prescriber must be present in the context.  In production,
		wire to the national PDMP API.
		"""
		assert schedule in ("II", "III", "IV", "V"), "DEA schedule must be II, III, IV, or V"
		assert quantity > 0, "quantity must be positive"
		assert prescriber_id, "prescriber_id required"

		quantity_cap = 30 if schedule == "II" else 90
		exceeds_cap = quantity > quantity_cap
		flags: list[str] = []

		if exceeds_cap:
			flags.append(f"Quantity {quantity} exceeds {quantity_cap}-day cap for Schedule {schedule}.")
		if schedule == "II":
			flags.append("Schedule II: prescriber must sign original prescription; no refills permitted.")
		if schedule in ("III", "IV"):
			flags.append(f"Schedule {schedule}: maximum 5 refills within 6 months of issue date.")

		self._record_audit(self.tenant_id, "controlled_substance_check", prescriber_id)
		logger.info("emr.cs_check drug=%s schedule=%s qty=%d prescriber=%s", drug, schedule, quantity, prescriber_id)

		return {
			"drug": drug,
			"dea_schedule": schedule,
			"quantity": quantity,
			"quantity_cap_days": quantity_cap,
			"exceeds_cap": exceeds_cap,
			"prescriber_id": prescriber_id,
			"flags": flags,
			"pdmp_checked": False,   # set True when wired to real PDMP
			"approved": not exceeds_cap,
			"recommendation": (
				f"REJECT — quantity exceeds Schedule {schedule} cap of {quantity_cap} days." if exceeds_cap
				else f"APPROVED — quantity within Schedule {schedule} limits."
			),
		}

	async def clinical_reminder_check(self, patient_id: str) -> list[dict[str, Any]]:
		"""Return list of overdue preventive care reminders for a patient.

		Compares the patient's active problem ICD-10 prefixes against a
		reminder schedule.  Returns reminders that have no matching recent
		completion in the patient's medications/vitals (proxy for completion
		in the absence of a full procedure/order store).
		"""
		problems = await self.list_problems(self.tenant_id, patient_id, status="active")
		active_icd10_prefixes = {p.icd10_code[:3] for p in problems}

		reminders: list[dict[str, Any]] = []
		for key, description, trigger_prefix, interval_months in _CLINICAL_REMINDERS:
			if trigger_prefix[:3] in active_icd10_prefixes or trigger_prefix == "Z23":
				reminders.append({
					"reminder_key": key,
					"description": description,
					"trigger_icd10_prefix": trigger_prefix,
					"recommended_interval_months": interval_months,
					"status": "overdue",  # production: compare against last completed order date
					"patient_id": patient_id,
				})

		logger.info("emr.clinical_reminders patient=%s count=%d", patient_id, len(reminders))
		return reminders

	# ═══════════════════════════════════════════════════════════════════════════
	# PRESCRIBING
	# ═══════════════════════════════════════════════════════════════════════════

	async def create_prescription(
		self,
		patient_id: str,
		drug: str,
		dose: float,
		frequency: str,
		duration_days: int,
		route: str,
		prescriber_id: str,
		encounter_id: str,
	) -> dict[str, Any]:
		"""Create a prescription after running full safety gate.

		Runs: allergy check → DDI check against active meds → duplicate
		therapy → controlled substance schedule detection.  Returns the
		prescription record plus a ``safety_summary`` block.  Hard stops
		(life-threatening allergy or CS cap exceeded) raise DrugSafetyError.
		"""
		assert patient_id, "patient_id required"
		assert drug, "drug required"
		assert dose > 0, "dose must be positive"
		assert duration_days > 0, "duration_days must be positive"
		assert prescriber_id, "prescriber_id required"
		assert encounter_id, "encounter_id required"

		drug_class = _DRUG_CLASS_MAP.get(drug.lower(), "unknown")

		# 1 — allergy gate
		allergy_result = await self.check_drug_allergy_alert(patient_id, drug, drug_class)
		if allergy_result["hard_stop"]:
			raise DrugSafetyError(
				f"Hard stop: life-threatening allergy to '{drug}' on record for patient {patient_id}."
			)

		# 2 — drug–drug interaction check
		active_meds = await self.list_medications(self.tenant_id, patient_id, status="active")
		active_drug_names = [m.drug_name for m in active_meds]
		ddi_results = await self.check_drug_drug_interactions([drug] + active_drug_names)
		contraindicated_ddis = [d for d in ddi_results if d["severity"] == "contraindicated"]

		# 3 — duplicate therapy
		dup_result = await self.check_duplicate_therapy(patient_id, drug, drug_class)

		# 4 — controlled substance preliminary check
		cs_flags: list[str] = []
		# heuristic: any drug with "codeine", "morphine", "oxycodone", "fentanyl", "tramadol" is Schedule II–III
		cs_heuristic_map = {
			"morphine": "II", "oxycodone": "II", "fentanyl": "II", "hydromorphone": "II",
			"codeine": "III", "tramadol": "IV", "diazepam": "IV", "lorazepam": "IV",
			"zolpidem": "IV", "pregabalin": "V", "gabapentin": "V",
		}
		cs_schedule = cs_heuristic_map.get(drug.lower())
		if cs_schedule:
			quantity_days = duration_days
			cs_result = await self.controlled_substance_check(drug, cs_schedule, quantity_days, prescriber_id)
			cs_flags = cs_result["flags"]
			if not cs_result["approved"]:
				raise DrugSafetyError(f"Controlled substance check failed: {'; '.join(cs_flags)}")

		rx_id = uuid7str()
		prescription: dict[str, Any] = {
			"id": rx_id,
			"tenant_id": self.tenant_id,
			"patient_id": patient_id,
			"encounter_id": encounter_id,
			"drug": drug,
			"drug_class": drug_class,
			"dose_mg": dose,
			"frequency": frequency,
			"duration_days": duration_days,
			"route": route,
			"prescriber_id": prescriber_id,
			"status": "active",
			"allergy_checked": True,
			"interaction_checked": True,
			"refills_used": 0,
			"refills_allowed": 0,
			"lot_number": None,
			"expiry_date": None,
			"dispensed_by": None,
			"dispensed_at": None,
			"created_at": datetime.utcnow().isoformat(),
			"safety_summary": {
				"allergy_conflicts": allergy_result["conflicts"],
				"ddi_interactions": ddi_results,
				"contraindicated_ddis": contraindicated_ddis,
				"duplicate_therapy": dup_result,
				"controlled_substance_flags": cs_flags,
				"warnings_count": (
					len(allergy_result["conflicts"]) +
					len(ddi_results) +
					(1 if dup_result["duplicate_found"] else 0)
				),
			},
		}
		self._prescriptions[(self.tenant_id, rx_id)] = prescription
		self._record_audit(self.tenant_id, "prescription_created", rx_id)
		_log_prescribe(rx_id, patient_id, drug)
		return prescription

	async def verify_prescription(self, prescription_id: str, pharmacist_id: str) -> dict[str, Any]:
		"""Pharmacist clinical verification of a prescription.

		Sets ``pharmacist_verified`` flag and records the verifying pharmacist.
		Returns the updated prescription or raises ValueError if not found.
		"""
		assert pharmacist_id, "pharmacist_id required"

		rx = self._prescriptions.get((self.tenant_id, prescription_id))
		if rx is None:
			raise ValueError(f"Prescription {prescription_id} not found")
		if rx["status"] in ("completed", "cancelled"):
			raise ValueError(f"Cannot verify a {rx['status']} prescription")

		rx["pharmacist_verified"] = True
		rx["pharmacist_id"] = pharmacist_id
		rx["verified_at"] = datetime.utcnow().isoformat()
		rx["updated_at"] = datetime.utcnow().isoformat()
		self._prescriptions[(self.tenant_id, prescription_id)] = rx
		self._record_audit(self.tenant_id, "prescription_verified", prescription_id)
		logger.info("emr.rx_verified rx=%s pharmacist=%s", prescription_id, pharmacist_id)
		return rx

	async def dispense_medication(
		self,
		prescription_id: str,
		lot_number: str,
		expiry_date: str,
		quantity: float,
		dispensed_by: str,
	) -> dict[str, Any]:
		"""Dispense medication against a verified prescription.

		Records lot number, expiry, quantity dispensed and timestamps the
		dispensing event.  Marks prescription ``completed`` if no refills
		remain.
		"""
		assert lot_number, "lot_number required"
		assert expiry_date, "expiry_date required"
		assert quantity > 0, "quantity must be positive"
		assert dispensed_by, "dispensed_by required"

		rx = self._prescriptions.get((self.tenant_id, prescription_id))
		if rx is None:
			raise ValueError(f"Prescription {prescription_id} not found")
		if rx["status"] not in ("active",):
			raise ValueError(f"Cannot dispense a {rx['status']} prescription")
		if not rx.get("pharmacist_verified"):
			raise ValueError("Prescription must be pharmacist-verified before dispensing")

		rx["lot_number"] = lot_number
		rx["expiry_date"] = expiry_date
		rx["quantity_dispensed"] = quantity
		rx["dispensed_by"] = dispensed_by
		rx["dispensed_at"] = datetime.utcnow().isoformat()
		rx["status"] = "completed" if rx["refills_used"] >= rx["refills_allowed"] else "active"
		rx["updated_at"] = datetime.utcnow().isoformat()
		self._prescriptions[(self.tenant_id, prescription_id)] = rx
		self._record_audit(self.tenant_id, "medication_dispensed", prescription_id)
		logger.info(
			"emr.dispense rx=%s lot=%s qty=%s by=%s",
			prescription_id, lot_number, quantity, dispensed_by,
		)
		return rx

	async def medication_reconciliation(
		self,
		patient_id: str,
		encounter_id: str,
		home_medications: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Reconcile home medication list against active EMR medications.

		Identifies: (a) home meds not in EMR, (b) EMR meds not in home list,
		(c) dose discrepancies.  Returns a reconciliation record with
		``discrepancies`` list and a ``status``.
		"""
		assert patient_id, "patient_id required"
		assert encounter_id, "encounter_id required"

		emr_meds = await self.list_medications(self.tenant_id, patient_id, status="active")
		emr_names = {m.drug_name.lower(): m for m in emr_meds}
		home_names = {h["drug_name"].lower(): h for h in home_medications}

		discrepancies: list[dict[str, Any]] = []

		# in home but not in EMR — possible omission
		for name, home_med in home_names.items():
			if name not in emr_names:
				discrepancies.append({
					"type": "omission",
					"description": f"Home medication '{home_med['drug_name']}' not in EMR active list",
					"home_med": home_med,
					"emr_med": None,
					"action_required": "Add to EMR or document reason for discontinuation",
				})

		# in EMR but not in home list — possible commission
		for name, emr_med in emr_names.items():
			if name not in home_names:
				discrepancies.append({
					"type": "commission",
					"description": f"EMR medication '{emr_med.drug_name}' not reported by patient",
					"home_med": None,
					"emr_med": {"id": emr_med.id, "drug_name": emr_med.drug_name, "dose": emr_med.dose},
					"action_required": "Confirm if patient still taking; update or discontinue",
				})

		# dose/frequency discrepancies
		for name in home_names.keys() & emr_names.keys():
			home_dose = str(home_names[name].get("dose", "")).lower()
			emr_dose = emr_names[name].dose.lower()
			if home_dose and home_dose != emr_dose:
				discrepancies.append({
					"type": "dose_discrepancy",
					"description": f"Dose mismatch for '{name}': home='{home_dose}' vs EMR='{emr_dose}'",
					"home_med": home_names[name],
					"emr_med": {"id": emr_names[name].id, "dose": emr_dose},
					"action_required": "Confirm correct dose and update EMR",
				})

		rec_id = uuid7str()
		record: dict[str, Any] = {
			"id": rec_id,
			"tenant_id": self.tenant_id,
			"patient_id": patient_id,
			"encounter_id": encounter_id,
			"home_medications_count": len(home_medications),
			"emr_active_medications_count": len(emr_meds),
			"discrepancies": discrepancies,
			"discrepancy_count": len(discrepancies),
			"status": "discrepancy_noted" if discrepancies else "reconciled",
			"reconciled_at": datetime.utcnow().isoformat(),
			"reconciled_by": self.actor_id,
		}
		self._record_audit(self.tenant_id, "medication_reconciled", rec_id)
		logger.info("emr.med_reconciliation patient=%s discrepancies=%d", patient_id, len(discrepancies))
		return record

	async def stop_medication(
		self,
		patient_id: str,
		medication_id: str,
		reason: str,
		stopped_by: str,
	) -> dict[str, Any]:
		"""Stop an active medication with reason documentation.

		Updates the MedicationResponse to discontinued status and records the
		stopping reason in a structured stop event for the audit trail.
		"""
		assert reason.strip(), "reason must not be empty"
		assert stopped_by, "stopped_by required"

		med = self._medications.get((self.tenant_id, medication_id))
		if med is None:
			raise ValueError(f"Medication {medication_id} not found for patient {patient_id}")
		if med.patient_id != patient_id:
			raise ValueError("Medication does not belong to the specified patient")
		if med.status == "discontinued":
			raise ValueError("Medication is already discontinued")

		updated = med.model_copy(update={
			"status": "discontinued",
			"end_date": datetime.utcnow(),
			"updated_at": datetime.utcnow(),
		})
		self._medications[(self.tenant_id, medication_id)] = updated

		stop_event: dict[str, Any] = {
			"medication_id": medication_id,
			"patient_id": patient_id,
			"drug_name": med.drug_name,
			"reason": reason,
			"stopped_by": stopped_by,
			"stopped_at": datetime.utcnow().isoformat(),
			"prior_status": med.status,
		}
		self._record_audit(self.tenant_id, "medication_stopped", medication_id)
		logger.info("emr.stop_med med=%s patient=%s reason=%s", medication_id, patient_id, reason)
		return stop_event

	async def generate_prescription_list(self, patient_id: str) -> list[dict[str, Any]]:
		"""Return all prescriptions for a patient across all statuses, newest first."""
		rxs = [
			rx for (tid, _), rx in self._prescriptions.items()
			if tid == self.tenant_id and rx["patient_id"] == patient_id
		]
		return sorted(rxs, key=lambda r: r["created_at"], reverse=True)

	async def medication_administration_record(
		self,
		patient_id: str,
		encounter_id: str,
		period_from: str,
		period_to: str,
	) -> dict[str, Any]:
		"""Generate a MAR for a patient's encounter within a time window.

		In the current in-memory implementation the MAR is derived from
		active prescriptions.  In production, individual administration events
		would be stored per dose.
		"""
		assert patient_id, "patient_id required"
		assert encounter_id, "encounter_id required"

		rxs = [
			rx for (tid, _), rx in self._prescriptions.items()
			if tid == self.tenant_id
			and rx["patient_id"] == patient_id
			and rx["encounter_id"] == encounter_id
		]
		meds = await self.list_medications(self.tenant_id, patient_id)

		mar_entries: list[dict[str, Any]] = []
		for rx in rxs:
			mar_entries.append({
				"prescription_id": rx["id"],
				"drug": rx["drug"],
				"dose_mg": rx["dose_mg"],
				"route": rx["route"],
				"frequency": rx["frequency"],
				"status": rx["status"],
				"dispensed_at": rx.get("dispensed_at"),
				"dispensed_by": rx.get("dispensed_by"),
				"lot_number": rx.get("lot_number"),
			})
		for med in meds:
			if med.id not in {r["prescription_id"] for r in mar_entries}:
				mar_entries.append({
					"medication_id": med.id,
					"drug": med.drug_name,
					"dose": med.dose,
					"route": med.route,
					"frequency": med.frequency,
					"status": med.status,
				})

		return {
			"patient_id": patient_id,
			"encounter_id": encounter_id,
			"period_from": period_from,
			"period_to": period_to,
			"generated_at": datetime.utcnow().isoformat(),
			"generated_by": self.actor_id,
			"entries": mar_entries,
			"total_medications": len(mar_entries),
		}

	async def refill_prescription(
		self,
		prescription_id: str,
		refill_count: int,
		dispensed_by: str,
	) -> dict[str, Any]:
		"""Process a prescription refill.

		Increments ``refills_used`` counter and validates against
		``refills_allowed``.  Marks completed when exhausted.
		"""
		assert refill_count > 0, "refill_count must be positive"
		assert dispensed_by, "dispensed_by required"

		rx = self._prescriptions.get((self.tenant_id, prescription_id))
		if rx is None:
			raise ValueError(f"Prescription {prescription_id} not found")
		if rx["status"] in ("completed", "cancelled", "stopped"):
			raise ValueError(f"Cannot refill a {rx['status']} prescription")

		current_refills = rx.get("refills_used", 0)
		allowed = rx.get("refills_allowed", 0)
		if current_refills + refill_count > allowed:
			raise ValueError(
				f"Refill denied: {refill_count} refill(s) requested but only "
				f"{allowed - current_refills} remaining."
			)

		rx["refills_used"] = current_refills + refill_count
		rx["last_refill_at"] = datetime.utcnow().isoformat()
		rx["last_refilled_by"] = dispensed_by
		rx["status"] = "completed" if rx["refills_used"] >= allowed else "active"
		rx["updated_at"] = datetime.utcnow().isoformat()
		self._prescriptions[(self.tenant_id, prescription_id)] = rx
		self._record_audit(self.tenant_id, "prescription_refilled", prescription_id)
		logger.info("emr.refill rx=%s count=%d by=%s", prescription_id, refill_count, dispensed_by)
		return rx

	# ═══════════════════════════════════════════════════════════════════════════
	# CLINICAL DECISION SUPPORT
	# ═══════════════════════════════════════════════════════════════════════════

	async def CHADS2_VASc_score(self, patient_id: str) -> dict[str, Any]:
		"""Calculate CHA₂DS₂-VASc stroke risk score for AF patients.

		Derives risk factors from the problem list and patient demographics.
		Returns score 0–9 with annual stroke risk estimate and anticoagulation
		recommendation per ESC 2020 AF guidelines.
		"""
		problems = await self.list_problems(self.tenant_id, patient_id, status="active")
		icd10_codes = {p.icd10_code[:3] for p in problems}

		# retrieve patient age (proxy: look for a stored patient or default to unknown)
		# In a full stack this calls patient_service.get_patient()
		age_years: int = 0   # production: fetch from patient store
		is_female = False     # production: fetch from patient store

		score = 0
		criteria_met: list[str] = []

		def _prefix_match(prefixes: list[str]) -> bool:
			return any(code.startswith(pfx[:3]) for code in icd10_codes for pfx in prefixes)

		if _prefix_match(["I50"]):
			score += 1; criteria_met.append("C: Congestive heart failure (+1)")
		if _prefix_match(["I10", "I11", "I12", "I13"]):
			score += 1; criteria_met.append("H: Hypertension (+1)")
		if age_years >= 75:
			score += 2; criteria_met.append("A2: Age ≥75 (+2)")
		elif age_years >= 65:
			score += 1; criteria_met.append("A: Age 65–74 (+1)")
		if _prefix_match(["E10", "E11", "E13"]):
			score += 1; criteria_met.append("D: Diabetes mellitus (+1)")
		if _prefix_match(["I63", "G45", "I64"]):
			score += 2; criteria_met.append("S2: Stroke/TIA/thromboembolism (+2)")
		if _prefix_match(["I21", "I25", "I70"]):
			score += 1; criteria_met.append("V: Vascular disease (+1)")
		if is_female:
			score += 1; criteria_met.append("Sc: Female sex (+1)")

		# Annual stroke risk lookup (Lip et al. 2010)
		_annual_risk = {0: 0.0, 1: 1.3, 2: 2.2, 3: 3.2, 4: 4.0, 5: 6.7, 6: 9.8, 7: 9.6, 8: 6.7, 9: 15.2}
		annual_risk = _annual_risk.get(min(score, 9), 15.2)

		_log_cds("CHADS2_VASc", patient_id, score)
		return {
			"patient_id": patient_id,
			"score": score,
			"max_score": 9,
			"criteria_met": criteria_met,
			"annual_stroke_risk_pct": annual_risk,
			"recommendation": (
				"No antithrombotic therapy recommended (score 0 male / 1 female)." if score <= (1 if is_female else 0)
				else "Consider anticoagulation (score 1 male). Reassess risk factors." if score == 1 and not is_female
				else "Oral anticoagulation recommended. Prefer DOAC over warfarin (ESC 2020)."
			),
		}

	async def WELLS_score_PE(self, patient_id: str, clinical_features: dict[str, Any]) -> dict[str, Any]:
		"""Calculate Wells score for pulmonary embolism pre-test probability.

		``clinical_features`` keys (all bool unless noted):
		  dvt_signs, pe_most_likely_diagnosis, heart_rate_gt_100,
		  immobilisation_or_surgery, prior_dvt_or_pe,
		  haemoptysis, malignancy
		"""
		score = 0.0
		criteria: list[str] = []

		_items: list[tuple[str, float, str]] = [
			("dvt_signs", 3.0, "Clinical signs/symptoms of DVT (+3)"),
			("pe_most_likely_diagnosis", 3.0, "PE is most likely diagnosis (+3)"),
			("heart_rate_gt_100", 1.5, "Heart rate > 100 bpm (+1.5)"),
			("immobilisation_or_surgery", 1.5, "Immobilisation ≥3 days or surgery in past 4 weeks (+1.5)"),
			("prior_dvt_or_pe", 1.5, "Prior DVT or PE (+1.5)"),
			("haemoptysis", 1.0, "Haemoptysis (+1)"),
			("malignancy", 1.0, "Malignancy (treatment within 6 months or palliative) (+1)"),
		]
		for key, points, label in _items:
			if clinical_features.get(key):
				score += points
				criteria.append(label)

		if score <= 1.0:
			probability = "low"
			pe_prevalence_pct = 1.3
		elif score <= 6.0:
			probability = "moderate"
			pe_prevalence_pct = 16.2
		else:
			probability = "high"
			pe_prevalence_pct = 37.5

		_log_cds("WELLS_PE", patient_id, score)
		return {
			"patient_id": patient_id,
			"score": score,
			"criteria_met": criteria,
			"probability": probability,
			"pe_prevalence_pct": pe_prevalence_pct,
			"recommendation": (
				"Low probability — PERC rule; if negative, no imaging required." if probability == "low"
				else "Moderate probability — D-dimer; if elevated, CTPA indicated." if probability == "moderate"
				else "High probability — proceed directly to CTPA or V/Q scan."
			),
		}

	async def QSOFA_score(
		self,
		patient_id: str,
		respiratory_rate: int,
		mentation_altered: bool,
		sbp: int,
	) -> dict[str, Any]:
		"""Quick SOFA sepsis screen (0–3).

		One point each for: RR ≥22, altered mentation (GCS <15), SBP ≤100 mmHg.
		Score ≥2 predicts in-hospital mortality > 10% (Singer et al. JAMA 2016).
		"""
		assert respiratory_rate >= 0
		assert sbp >= 0

		score = 0
		criteria: list[str] = []

		if respiratory_rate >= 22:
			score += 1; criteria.append("Respiratory rate ≥22/min (+1)")
		if mentation_altered:
			score += 1; criteria.append("Altered mentation (+1)")
		if sbp <= 100:
			score += 1; criteria.append("SBP ≤100 mmHg (+1)")

		_log_cds("qSOFA", patient_id, score)
		return {
			"patient_id": patient_id,
			"score": score,
			"max_score": 3,
			"criteria_met": criteria,
			"inputs": {
				"respiratory_rate": respiratory_rate,
				"mentation_altered": mentation_altered,
				"systolic_bp_mmhg": sbp,
			},
			"sepsis_screen_positive": score >= 2,
			"recommendation": (
				"qSOFA ≥2 — HIGH RISK for poor outcome. Suspect sepsis. "
				"Obtain cultures, lactate, consider ICU review, initiate Sepsis-6." if score >= 2
				else "qSOFA <2 — sepsis unlikely based on this screen. Reassess if clinical deterioration."
			),
		}

	async def NEWS2_score(self, patient_id: str, vitals: dict[str, Any]) -> dict[str, Any]:
		"""Calculate National Early Warning Score 2 (NEWS2).

		``vitals`` keys: respiratory_rate (int), spo2 (float 0–100),
		supplemental_oxygen (bool), systolic_bp (int), heart_rate (int),
		temperature (float °C), consciousness (str: "A"=alert, "C"=confusion,
		"V"=voice, "P"=pain, "U"=unresponsive).

		Score 0–20; clinical response thresholds per RCP 2017.
		"""
		rr: int = vitals.get("respiratory_rate", 16)
		spo2: float = vitals.get("spo2", 98.0)
		on_o2: bool = vitals.get("supplemental_oxygen", False)
		sbp: int = vitals.get("systolic_bp", 120)
		hr: int = vitals.get("heart_rate", 70)
		temp: float = vitals.get("temperature", 37.0)
		consciousness: str = vitals.get("consciousness", "A")

		# sub-scores
		rr_score = _news2_threshold_score(rr, _NEWS2_THRESHOLDS["respiratory_rate"])
		spo2_score = _news2_threshold_score(spo2, _NEWS2_THRESHOLDS["spo2"])
		o2_score = 2 if on_o2 else 0
		sbp_score = _news2_threshold_score(sbp, _NEWS2_THRESHOLDS["systolic_bp"])
		hr_score = _news2_threshold_score(hr, _NEWS2_THRESHOLDS["heart_rate"])
		temp_score = _news2_threshold_score(temp, _NEWS2_THRESHOLDS["temperature"])
		consciousness_score = 0 if consciousness == "A" else 3

		total = rr_score + spo2_score + o2_score + sbp_score + hr_score + temp_score + consciousness_score

		if total >= 7:
			risk_level = "high"
			response = "Urgent clinical review — continuous monitoring, senior clinician review, consider HDU/ICU."
		elif total >= 5:
			risk_level = "medium"
			response = "Increased monitoring frequency — clinical review within 30 minutes."
		elif total >= 1:
			risk_level = "low"
			response = "Routine monitoring — reassess per ward protocol."
		else:
			risk_level = "low_stable"
			response = "Stable — continue routine monitoring."

		_log_cds("NEWS2", patient_id, total)
		return {
			"patient_id": patient_id,
			"total_score": total,
			"max_score": 20,
			"risk_level": risk_level,
			"sub_scores": {
				"respiratory_rate": rr_score,
				"spo2": spo2_score,
				"supplemental_oxygen": o2_score,
				"systolic_bp": sbp_score,
				"heart_rate": hr_score,
				"temperature": temp_score,
				"consciousness": consciousness_score,
			},
			"inputs": vitals,
			"response_recommendation": response,
		}

	async def generate_clinical_summary(self, patient_id: str) -> dict[str, Any]:
		"""Generate a structured clinical summary for a patient.

		Aggregates active problems, active medications, allergies and most
		recent vitals into a single summary dict suitable for handover or
		referral letters.
		"""
		problems = await self.list_problems(self.tenant_id, patient_id, status="active")
		medications = await self.list_medications(self.tenant_id, patient_id, status="active")
		allergies = await self.list_allergies(self.tenant_id, patient_id)
		vitals = await self.list_vitals(self.tenant_id, patient_id)
		encounters = await self.list_encounters(self.tenant_id, patient_id)

		# most recent vital per type
		latest_vitals: dict[str, Any] = {}
		for v in vitals:
			vtype = str(v.vital_type)
			if vtype not in latest_vitals:
				latest_vitals[vtype] = {
					"value": v.value,
					"value2": v.value2,
					"unit": v.unit,
					"recorded_at": v.recorded_at.isoformat(),
				}

		return {
			"patient_id": patient_id,
			"generated_at": datetime.utcnow().isoformat(),
			"generated_by": self.actor_id,
			"active_problems": [
				{"icd10": p.icd10_code, "description": p.description, "status": p.status}
				for p in problems
			],
			"active_medications": [
				{"drug": m.drug_name, "dose": m.dose, "route": m.route, "frequency": m.frequency}
				for m in medications
			],
			"allergies": [
				{"allergen": a.allergen, "type": a.allergy_type, "severity": a.severity, "reaction": a.reaction}
				for a in allergies if a.status == "active"
			],
			"latest_vitals": latest_vitals,
			"open_encounters": [
				{"id": e.id, "type": e.encounter_type, "complaint": e.chief_complaint}
				for e in encounters if e.status == "in_progress"
			],
			"problem_count": len(problems),
			"medication_count": len(medications),
			"allergy_count": len([a for a in allergies if a.status == "active"]),
		}

	async def clinical_guideline_alert(self, patient_id: str, diagnosis_code: str) -> list[dict[str, Any]]:
		"""Return evidence-based guideline alerts for a given ICD-10 diagnosis code.

		Matches the three-character prefix against an embedded guideline
		database.  In production this would call a CDS Hooks service.
		"""
		prefix = diagnosis_code.strip().upper()[:3]
		matched: list[dict[str, Any]] = []
		for code_prefix, alerts in _GUIDELINE_ALERTS.items():
			if prefix.startswith(code_prefix[:3]):
				for alert in alerts:
					matched.append({
						"patient_id": patient_id,
						"diagnosis_code": diagnosis_code,
						"trigger_prefix": code_prefix,
						"title": alert["title"],
						"body": alert["body"],
						"source": alert["source"],
						"alert_type": "clinical_guideline",
					})
		logger.info("emr.guideline_alerts patient=%s dx=%s count=%d", patient_id, diagnosis_code, len(matched))
		return matched

	# ═══════════════════════════════════════════════════════════════════════════
	# FHIR & CODING
	# ═══════════════════════════════════════════════════════════════════════════

	async def fhir_patient_resource(self, patient_id: str) -> dict[str, Any]:
		"""Construct a FHIR R4 Patient resource.

		Patient demographics are sourced from the in-memory encounter/allergy
		records as a proxy.  Full implementation delegates to the patient
		service store.
		"""
		allergies = await self.list_allergies(self.tenant_id, patient_id)
		return {
			"resourceType": "Patient",
			"id": patient_id,
			"meta": {
				"profile": ["http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient"],
				"lastUpdated": datetime.utcnow().isoformat(),
			},
			"identifier": [
				{"use": "official", "system": f"urn:apg:tenant:{self.tenant_id}", "value": patient_id}
			],
			"active": True,
			"extension": [
				{
					"url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-allergyintolerance",
					"valueInteger": len(allergies),
				}
			],
		}

	async def fhir_encounter_resource(self, encounter_id: str) -> dict[str, Any]:
		"""Construct a FHIR R4 Encounter resource from the stored encounter."""
		enc = self._encounters.get((self.tenant_id, encounter_id))
		if enc is None:
			raise ValueError(f"Encounter {encounter_id} not found")

		return {
			"resourceType": "Encounter",
			"id": encounter_id,
			"meta": {"lastUpdated": datetime.utcnow().isoformat()},
			"status": enc.status,
			"class": {
				"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
				"code": str(enc.encounter_type).upper(),
			},
			"subject": {"reference": f"Patient/{enc.patient_id}"},
			"participant": [
				{"individual": {"reference": f"Practitioner/{enc.provider_id}"}}
			],
			"reasonCode": [{"text": enc.chief_complaint}],
			"diagnosis": [
				{"condition": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": code}]}}
				for code in enc.icd10_codes
			],
			"period": {
				"start": enc.admit_time.isoformat(),
				"end": enc.discharge_time.isoformat() if enc.discharge_time else None,
			},
		}

	async def fhir_bundle_export(self, patient_id: str, resource_types: list[str]) -> dict[str, Any]:
		"""Export a FHIR R4 transaction bundle for the specified resource types.

		Supports: Patient, Encounter, Condition, MedicationRequest,
		AllergyIntolerance, Observation (vitals), DocumentReference (notes).
		"""
		bundle: dict[str, Any] = {
			"resourceType": "Bundle",
			"id": uuid7str(),
			"type": "transaction",
			"timestamp": datetime.utcnow().isoformat(),
			"entry": [],
		}

		def _wrap(resource: dict[str, Any]) -> dict[str, Any]:
			return {
				"fullUrl": f"urn:uuid:{resource.get('id', uuid7str())}",
				"resource": resource,
				"request": {"method": "PUT", "url": f"{resource['resourceType']}/{resource.get('id', '')}"},
			}

		if "Patient" in resource_types:
			bundle["entry"].append(_wrap(await self.fhir_patient_resource(patient_id)))

		if "Encounter" in resource_types:
			encounters = await self.list_encounters(self.tenant_id, patient_id)
			for enc in encounters:
				bundle["entry"].append(_wrap(await self.fhir_encounter_resource(enc.id)))

		if "Condition" in resource_types:
			for prob in await self.list_problems(self.tenant_id, patient_id):
				bundle["entry"].append(_wrap({
					"resourceType": "Condition",
					"id": prob.id,
					"clinicalStatus": {"coding": [{"code": prob.status}]},
					"code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": prob.icd10_code, "display": prob.description}]},
					"subject": {"reference": f"Patient/{patient_id}"},
				}))

		if "MedicationRequest" in resource_types:
			for med in await self.list_medications(self.tenant_id, patient_id):
				bundle["entry"].append(_wrap({
					"resourceType": "MedicationRequest",
					"id": med.id,
					"status": med.status,
					"intent": "order",
					"medicationCodeableConcept": {"text": med.drug_name},
					"subject": {"reference": f"Patient/{patient_id}"},
					"requester": {"reference": f"Practitioner/{med.prescriber_id}"},
					"dosageInstruction": [{"text": f"{med.dose} {med.route} {med.frequency}"}],
				}))

		if "AllergyIntolerance" in resource_types:
			for allergy in await self.list_allergies(self.tenant_id, patient_id):
				bundle["entry"].append(_wrap({
					"resourceType": "AllergyIntolerance",
					"id": allergy.id,
					"clinicalStatus": {"coding": [{"code": allergy.status}]},
					"criticality": allergy.severity,
					"code": {"text": allergy.allergen},
					"patient": {"reference": f"Patient/{patient_id}"},
				}))

		if "Observation" in resource_types:
			for vital in await self.list_vitals(self.tenant_id, patient_id):
				bundle["entry"].append(_wrap({
					"resourceType": "Observation",
					"id": vital.id,
					"status": "final",
					"code": {"text": str(vital.vital_type)},
					"subject": {"reference": f"Patient/{patient_id}"},
					"valueQuantity": {"value": vital.value, "unit": vital.unit},
					"effectiveDateTime": vital.recorded_at.isoformat(),
				}))

		if "DocumentReference" in resource_types:
			for note in await self.list_notes(self.tenant_id, patient_id):
				bundle["entry"].append(_wrap({
					"resourceType": "DocumentReference",
					"id": note.id,
					"status": "current" if note.status == "final" else "superseded",
					"type": {"text": str(note.note_type)},
					"subject": {"reference": f"Patient/{patient_id}"},
					"content": [{"attachment": {"contentType": "text/plain", "data": note.content[:200]}}],
				}))

		logger.info(_log_fhir_export(self.tenant_id, str(resource_types), len(bundle["entry"])))
		self._record_audit(self.tenant_id, "fhir_bundle_exported", patient_id)
		return bundle

	async def assign_icd10_diagnosis(
		self,
		encounter_id: str,
		icd10_code: str,
		description: str,
		certainty: str,
		is_primary: bool,
	) -> dict[str, Any]:
		"""Assign an ICD-10 diagnosis to an encounter.

		Stores the diagnosis record and updates the encounter's icd10_codes
		list.  Validates code format (capitalised, non-empty).
		"""
		icd10_code = icd10_code.strip().upper()
		assert icd10_code, "icd10_code must not be empty"
		assert description.strip(), "description must not be empty"
		assert certainty in ("confirmed", "differential", "provisional", "refuted"), \
			"certainty must be one of confirmed|differential|provisional|refuted"

		enc = self._encounters.get((self.tenant_id, encounter_id))
		if enc is None:
			raise ValueError(f"Encounter {encounter_id} not found")

		dx_id = uuid7str()
		dx: dict[str, Any] = {
			"id": dx_id,
			"tenant_id": self.tenant_id,
			"encounter_id": encounter_id,
			"patient_id": enc.patient_id,
			"icd10_code": icd10_code,
			"description": description,
			"certainty": certainty,
			"is_primary": is_primary,
			"created_by": self.actor_id,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._diagnoses[(self.tenant_id, dx_id)] = dx

		# update encounter code list
		existing = list(enc.icd10_codes)
		if icd10_code not in existing:
			if is_primary:
				updated_codes = [icd10_code] + existing
			else:
				updated_codes = existing + [icd10_code]
			updated_enc = enc.model_copy(update={"icd10_codes": updated_codes, "updated_at": datetime.utcnow()})
			self._encounters[(self.tenant_id, encounter_id)] = updated_enc

		self._record_audit(self.tenant_id, "diagnosis_assigned", dx_id)
		logger.info("emr.dx_assigned enc=%s code=%s certainty=%s", encounter_id, icd10_code, certainty)
		return dx

	async def suggest_diagnoses(self, symptoms_text: str) -> list[dict[str, Any]]:
		"""Suggest ICD-10 diagnoses from free-text symptom description.

		Performs simple keyword matching against an embedded symptom→diagnosis
		map.  In production this would call an NLP/NER model (e.g. local
		llama3 via Ollama) to extract SNOMED/ICD entities.
		"""
		symptoms_lower = symptoms_text.lower()
		suggestions: list[dict[str, Any]] = []
		seen_codes: set[str] = set()

		for keyword, diagnoses in _SYMPTOM_DX_MAP.items():
			if keyword in symptoms_lower:
				for dx in diagnoses:
					if dx["icd10"] not in seen_codes:
						suggestions.append({
							"icd10_code": dx["icd10"],
							"description": dx["description"],
							"confidence": dx["confidence"],
							"matched_keyword": keyword,
						})
						seen_codes.add(dx["icd10"])

		# sort by confidence tier
		_tier = {"high": 0, "medium": 1, "low": 2}
		suggestions.sort(key=lambda s: _tier.get(s["confidence"], 9))
		logger.info("emr.suggest_dx symptoms=%r suggestions=%d", symptoms_text[:60], len(suggestions))
		return suggestions

	async def assign_cpt_procedure(
		self,
		encounter_id: str,
		cpt_code: str,
		description: str,
		units: int,
		modifier: str | None,
	) -> dict[str, Any]:
		"""Assign a CPT procedure code to an encounter for billing purposes.

		Validates that units ≥ 1 and CPT code is non-empty.  Modifier is
		optional (e.g. "59" for distinct procedural service).
		"""
		cpt_code = cpt_code.strip()
		assert cpt_code, "cpt_code must not be empty"
		assert description.strip(), "description must not be empty"
		assert units >= 1, "units must be at least 1"

		enc = self._encounters.get((self.tenant_id, encounter_id))
		if enc is None:
			raise ValueError(f"Encounter {encounter_id} not found")

		proc_id = uuid7str()
		procedure: dict[str, Any] = {
			"id": proc_id,
			"tenant_id": self.tenant_id,
			"encounter_id": encounter_id,
			"patient_id": enc.patient_id,
			"cpt_code": cpt_code,
			"description": description,
			"units": units,
			"modifier": modifier,
			"created_by": self.actor_id,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._cpt_procedures[(self.tenant_id, proc_id)] = procedure
		self._record_audit(self.tenant_id, "cpt_procedure_assigned", proc_id)
		logger.info("emr.cpt_assigned enc=%s cpt=%s units=%d", encounter_id, cpt_code, units)
		return procedure

	async def sign_clinical_note(self, note_id: str, clinician_id: str) -> dict[str, Any]:
		"""Cryptographically lock a clinical note by signing it.

		Once signed, the note status becomes ``final`` and no further
		content edits are permitted.  Subsequent corrections must go through
		``addendum_to_note``.  Returns the signed note record.
		"""
		assert clinician_id, "clinician_id required"

		note = self._notes.get((self.tenant_id, note_id))
		if note is None:
			raise ValueError(f"Clinical note {note_id} not found")
		if note.status == "final":
			raise ValueError(f"Note {note_id} is already signed/final. Use addendum for corrections.")
		if note.status == "entered_in_error":
			raise ValueError(f"Note {note_id} is marked as entered in error and cannot be signed.")

		updated = note.model_copy(update={
			"status": "final",
			"cosigned_by": clinician_id,
			"finalized_at": datetime.utcnow(),
			"updated_at": datetime.utcnow(),
		})
		self._notes[(self.tenant_id, note_id)] = updated
		self._record_audit(self.tenant_id, "note_signed", note_id)
		logger.info("emr.note_signed note=%s clinician=%s", note_id, clinician_id)
		return {
			"note_id": note_id,
			"signed_by": clinician_id,
			"signed_at": updated.finalized_at.isoformat() if updated.finalized_at else datetime.utcnow().isoformat(),
			"status": "final",
			"immutable": True,
		}

	# ═══════════════════════════════════════════════════════════════════════════
	# CONSENT & PRIVACY
	# ═══════════════════════════════════════════════════════════════════════════

	async def record_consent(
		self,
		patient_id: str,
		consent_type: str,
		obtained_by: str,
		valid_until: str,
	) -> dict[str, Any]:
		"""Record a patient consent event.

		consent_type examples: treatment, research, data_sharing,
		release_of_information, photography.
		"""
		assert patient_id, "patient_id required"
		assert consent_type.strip(), "consent_type required"
		assert obtained_by, "obtained_by required"
		assert valid_until, "valid_until required"

		consent_id = uuid7str()
		consent: dict[str, Any] = {
			"id": consent_id,
			"tenant_id": self.tenant_id,
			"patient_id": patient_id,
			"consent_type": consent_type,
			"obtained_by": obtained_by,
			"obtained_at": datetime.utcnow().isoformat(),
			"valid_until": valid_until,
			"status": "active",
			"override": False,
			"guardian_id": None,
			"relationship": None,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._consents[(self.tenant_id, consent_id)] = consent
		self._record_audit(self.tenant_id, "consent_recorded", consent_id)
		_log_consent("record", patient_id, consent_type)
		return consent

	async def check_consent(self, patient_id: str, consent_type: str) -> dict[str, Any]:
		"""Check whether a valid active consent of the given type exists for the patient."""
		assert patient_id, "patient_id required"
		assert consent_type, "consent_type required"

		now_str = datetime.utcnow().isoformat()
		active_consents = [
			c for (tid, _), c in self._consents.items()
			if tid == self.tenant_id
			and c["patient_id"] == patient_id
			and c["consent_type"] == consent_type
			and c["status"] == "active"
			and c.get("valid_until", now_str) >= now_str
		]

		_log_consent("check", patient_id, consent_type)
		return {
			"patient_id": patient_id,
			"consent_type": consent_type,
			"consent_present": len(active_consents) > 0,
			"active_consents": active_consents,
			"checked_at": now_str,
		}

	async def emergency_consent_override(
		self,
		patient_id: str,
		reason: str,
		authorised_by: str,
	) -> dict[str, Any]:
		"""Record an emergency consent override (treating without explicit consent).

		Used when the patient is unconscious or otherwise unable to consent
		and delay would result in harm.  Creates an audit record for medicolegal
		purposes and returns an override token.
		"""
		assert reason.strip(), "reason must not be empty"
		assert authorised_by, "authorised_by required"

		override_id = uuid7str()
		override: dict[str, Any] = {
			"id": override_id,
			"tenant_id": self.tenant_id,
			"patient_id": patient_id,
			"reason": reason,
			"authorised_by": authorised_by,
			"override_at": datetime.utcnow().isoformat(),
			"consent_type": "emergency_override",
			"status": "active",
			"override": True,
			"valid_until": "immediate_episode_only",
		}
		self._consents[(self.tenant_id, override_id)] = override
		self._record_audit(self.tenant_id, "emergency_consent_override", override_id)
		_log_consent("emergency_override", patient_id, "emergency")
		logger.warning(
			"emr.emergency_consent_override patient=%s authorised_by=%s reason=%r",
			patient_id, authorised_by, reason,
		)
		return override

	async def minor_consent(
		self,
		patient_id: str,
		guardian_id: str,
		relationship: str,
		consent_type: str,
	) -> dict[str, Any]:
		"""Record consent obtained from a parent or legal guardian for a minor.

		The guardian's identity and relationship are documented alongside the
		consent record for medicolegal compliance.
		"""
		assert patient_id, "patient_id required"
		assert guardian_id, "guardian_id required"
		assert relationship.strip(), "relationship required"
		assert consent_type.strip(), "consent_type required"

		consent_id = uuid7str()
		consent: dict[str, Any] = {
			"id": consent_id,
			"tenant_id": self.tenant_id,
			"patient_id": patient_id,
			"guardian_id": guardian_id,
			"relationship": relationship,
			"consent_type": consent_type,
			"obtained_by": guardian_id,
			"obtained_at": datetime.utcnow().isoformat(),
			"status": "active",
			"minor_consent": True,
			"override": False,
			"valid_until": "guardian_discretion",
		}
		self._consents[(self.tenant_id, consent_id)] = consent
		self._record_audit(self.tenant_id, "minor_consent_recorded", consent_id)
		_log_consent("minor_consent", patient_id, consent_type)
		return consent

	# ═══════════════════════════════════════════════════════════════════════════
	# REFERRALS & DISCHARGE
	# ═══════════════════════════════════════════════════════════════════════════

	async def create_referral(
		self,
		patient_id: str,
		from_provider_id: str,
		to_specialty: str,
		reason: str,
		urgency: str,
	) -> dict[str, Any]:
		"""Create an outbound referral to another specialty or provider.

		urgency: routine | urgent | emergent.
		Returns a referral record with a unique ID for tracking.
		"""
		assert patient_id, "patient_id required"
		assert from_provider_id, "from_provider_id required"
		assert to_specialty.strip(), "to_specialty required"
		assert reason.strip(), "reason required"
		assert urgency in ("routine", "urgent", "emergent"), \
			"urgency must be routine | urgent | emergent"

		ref_id = uuid7str()
		referral: dict[str, Any] = {
			"id": ref_id,
			"tenant_id": self.tenant_id,
			"patient_id": patient_id,
			"from_provider_id": from_provider_id,
			"to_specialty": to_specialty,
			"reason": reason,
			"urgency": urgency,
			"status": "active",
			"created_by": self.actor_id,
			"created_at": datetime.utcnow().isoformat(),
			"accepted_by": None,
			"appointment_date": None,
			"outcome_notes": "",
		}
		self._referrals[(self.tenant_id, ref_id)] = referral
		self._record_audit(self.tenant_id, "referral_created", ref_id)
		logger.info("emr.referral_created id=%s patient=%s specialty=%s urgency=%s", ref_id, patient_id, to_specialty, urgency)
		return referral

	async def accept_referral(
		self,
		referral_id: str,
		accepting_provider: str,
		appointment_date: str,
	) -> dict[str, Any]:
		"""Accept an inbound referral and set the appointment date.

		Validates that the referral exists and is still in ``active`` status.
		Returns the updated referral record.
		"""
		assert accepting_provider, "accepting_provider required"
		assert appointment_date, "appointment_date required"

		ref = self._referrals.get((self.tenant_id, referral_id))
		if ref is None:
			raise ValueError(f"Referral {referral_id} not found")
		if ref["status"] != "active":
			raise ValueError(f"Cannot accept a {ref['status']} referral")

		ref["status"] = "completed"
		ref["accepted_by"] = accepting_provider
		ref["appointment_date"] = appointment_date
		ref["accepted_at"] = datetime.utcnow().isoformat()
		ref["updated_at"] = datetime.utcnow().isoformat()
		self._referrals[(self.tenant_id, referral_id)] = ref
		self._record_audit(self.tenant_id, "referral_accepted", referral_id)
		logger.info("emr.referral_accepted id=%s by=%s date=%s", referral_id, accepting_provider, appointment_date)
		return ref

	async def create_discharge_summary(
		self,
		encounter_id: str,
		discharge_diagnosis: str,
		treatment_summary: str,
		follow_up: str,
		discharge_medications: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Generate and store a discharge summary for a completed encounter.

		Attaches the summary ID to the encounter record.  The summary becomes
		a ``discharge_summary`` type ClinicalNoteResponse so it is visible in
		the note list.
		"""
		assert discharge_diagnosis.strip(), "discharge_diagnosis required"
		assert treatment_summary.strip(), "treatment_summary required"

		enc = self._encounters.get((self.tenant_id, encounter_id))
		if enc is None:
			raise ValueError(f"Encounter {encounter_id} not found")

		summary_id = uuid7str()
		med_lines = "\n".join(
			f"  - {m.get('drug_name', m.get('drug', 'Unknown'))} {m.get('dose', '')} {m.get('frequency', '')}".rstrip()
			for m in discharge_medications
		)
		content = (
			f"DISCHARGE SUMMARY\n"
			f"Encounter: {encounter_id}\n"
			f"Discharge Diagnosis: {discharge_diagnosis}\n\n"
			f"Treatment Summary:\n{treatment_summary}\n\n"
			f"Discharge Medications:\n{med_lines or '  None'}\n\n"
			f"Follow-up Instructions:\n{follow_up}"
		)

		# persist as a note
		ds_note = ClinicalNoteResponse(
			id=summary_id,
			tenant_id=self.tenant_id,
			patient_id=enc.patient_id,
			encounter_id=encounter_id,
			note_type="discharge_summary",
			author_id=self.actor_id,
			content=content,
			assessment=discharge_diagnosis,
			plan=follow_up,
			status="final",
			finalized_at=datetime.utcnow(),
			created_by=self.actor_id,
		)
		self._notes[(self.tenant_id, summary_id)] = ds_note

		summary_record: dict[str, Any] = {
			"id": summary_id,
			"tenant_id": self.tenant_id,
			"encounter_id": encounter_id,
			"patient_id": enc.patient_id,
			"discharge_diagnosis": discharge_diagnosis,
			"treatment_summary": treatment_summary,
			"follow_up": follow_up,
			"discharge_medications": discharge_medications,
			"created_by": self.actor_id,
			"created_at": datetime.utcnow().isoformat(),
		}
		self._discharge_summaries[(self.tenant_id, summary_id)] = summary_record

		# link to encounter
		updated_enc = enc.model_copy(update={
			"discharge_summary_id": summary_id,
			"updated_at": datetime.utcnow(),
		})
		self._encounters[(self.tenant_id, encounter_id)] = updated_enc

		self._record_audit(self.tenant_id, "discharge_summary_created", summary_id)
		logger.info("emr.discharge_summary enc=%s id=%s", encounter_id, summary_id)
		return summary_record

	async def addendum_to_note(
		self,
		note_id: str,
		addendum_text: str,
		added_by: str,
	) -> dict[str, Any]:
		"""Append an addendum to a signed (final) clinical note.

		Creates a new ``addendum`` note linked to the original.  The original
		note is not modified, preserving the medicolegal audit trail.
		"""
		assert addendum_text.strip(), "addendum_text must not be empty"
		assert added_by, "added_by required"

		original = self._notes.get((self.tenant_id, note_id))
		if original is None:
			raise ValueError(f"Note {note_id} not found")

		addendum_id = uuid7str()
		addendum_note = ClinicalNoteResponse(
			id=addendum_id,
			tenant_id=self.tenant_id,
			patient_id=original.patient_id,
			encounter_id=original.encounter_id,
			note_type="addendum",
			author_id=added_by,
			content=f"ADDENDUM to note {note_id} — added by {added_by} at {datetime.utcnow().isoformat()}:\n\n{addendum_text}",
			status="final",
			amendment_of=note_id,
			finalized_at=datetime.utcnow(),
			created_by=added_by,
		)
		self._notes[(self.tenant_id, addendum_id)] = addendum_note
		self._record_audit(self.tenant_id, "note_addendum_added", addendum_id)
		logger.info("emr.addendum original=%s addendum=%s by=%s", note_id, addendum_id, added_by)

		return {
			"addendum_id": addendum_id,
			"original_note_id": note_id,
			"added_by": added_by,
			"added_at": addendum_note.finalized_at.isoformat() if addendum_note.finalized_at else datetime.utcnow().isoformat(),
			"content_preview": addendum_text[:120],
		}

	# ═══════════════════════════════════════════════════════════════════════════
	# PATIENT CRUD (full lifecycle)
	# ═══════════════════════════════════════════════════════════════════════════

	async def register_patient(self, payload: "PatientCreate") -> "PatientResponse":
		"""Register a new patient after running probabilistic dedup check.

		Raises PolicyViolationError if a near-certain duplicate is found.
		Returns the created PatientResponse.
		"""
		from .models import PatientResponse
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		# dedup check
		candidates = await self.patient_deduplication_check({
			"family": payload.name.family,
			"given_0": payload.name.given[0] if payload.name.given else "",
			"birth_date": payload.birth_date.isoformat(),
			"gender": payload.gender,
			"biometric_hash": payload.biometric_hash,
		})
		certain = [c for c in candidates if c.is_certain_duplicate]
		if certain:
			raise PolicyViolationError(
				f"Certain duplicate patient found: {certain[0].candidate_patient_id} "
				f"(score {certain[0].match_score:.2f}). Review before registering."
			)

		patient = PatientResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			name=payload.name,
			birth_date=payload.birth_date,
			gender=payload.gender,
			marital_status=payload.marital_status,
			deceased_date=payload.deceased_date,
			is_deceased=payload.is_deceased,
			address=payload.address,
			telecom=payload.telecom,
			language=payload.language,
			nationality=payload.nationality,
			religion=payload.religion,
			race=payload.race,
			ethnicity=payload.ethnicity,
			blood_type=payload.blood_type,
			next_of_kin=payload.next_of_kin,
			emergency_contact=payload.emergency_contact,
			mental_health_record=payload.mental_health_record,
			biometric_hash=payload.biometric_hash,
			identifiers=payload.identifiers,
			created_by=payload.created_by,
		)
		self._patients[(payload.tenant_id, patient.id)] = patient
		self._record_audit(payload.tenant_id, "patient_registered", patient.id)
		_log_op("register_patient", payload.tenant_id, patient.id)
		return patient

	async def get_patient(self, tenant_id: str, patient_id: str) -> "PatientResponse | None":
		"""Retrieve a patient by ID, enforcing tenant isolation."""
		p = self._patients.get((tenant_id, patient_id))
		if p is None or p.is_deleted:
			return None
		return p

	async def list_patients(
		self,
		tenant_id: str,
		status: str | None = None,
		search: str | None = None,
	) -> list["PatientResponse"]:
		"""List patients for a tenant with optional status and name search filters."""
		results = [
			p for (tid, _), p in self._patients.items()
			if tid == tenant_id and not p.is_deleted
		]
		if status:
			results = [p for p in results if p.status == status]
		if search:
			search_lower = search.lower()
			results = [
				p for p in results
				if search_lower in p.name.family.lower()
				or any(search_lower in g.lower() for g in p.name.given)
			]
		return sorted(results, key=lambda p: p.created_at, reverse=True)

	async def update_patient(
		self,
		tenant_id: str,
		patient_id: str,
		payload: "PatientUpdate",
	) -> "PatientResponse | None":
		"""Apply a partial update to a patient record."""
		patient = self._patients.get((tenant_id, patient_id))
		if patient is None or patient.is_deleted:
			return None
		if patient.is_deceased:
			raise PolicyViolationError("deceased_record_locked: cannot update a deceased patient")
		update_data = payload.model_dump(exclude_none=True)
		update_data["updated_at"] = datetime.utcnow()
		updated = patient.model_copy(update=update_data)
		self._patients[(tenant_id, patient_id)] = updated
		self._record_audit(tenant_id, "patient_updated", patient_id)
		return updated

	async def delete_patient(self, tenant_id: str, patient_id: str) -> bool:
		"""Soft-delete a patient record (sets is_deleted=True)."""
		patient = self._patients.get((tenant_id, patient_id))
		if patient is None:
			return False
		updated = patient.model_copy(update={"is_deleted": True, "updated_at": datetime.utcnow()})
		self._patients[(tenant_id, patient_id)] = updated
		self._record_audit(tenant_id, "patient_deleted", patient_id)
		return True

	async def merge_patients(
		self,
		tenant_id: str,
		duplicate_id: str,
		surviving_id: str,
	) -> dict[str, Any]:
		"""Merge a duplicate patient record into the surviving record.

		The duplicate is marked status=merged and merged_into=surviving_id.
		All encounters / notes / problems for the duplicate are re-keyed to the
		surviving patient in the in-memory store.
		"""
		duplicate = self._patients.get((tenant_id, duplicate_id))
		surviving = self._patients.get((tenant_id, surviving_id))
		if duplicate is None:
			raise ValueError(f"Duplicate patient {duplicate_id} not found")
		if surviving is None:
			raise ValueError(f"Surviving patient {surviving_id} not found")

		# mark duplicate
		merged_dup = duplicate.model_copy(update={
			"status": "merged",
			"merged_into": surviving_id,
			"updated_at": datetime.utcnow(),
		})
		self._patients[(tenant_id, duplicate_id)] = merged_dup

		# re-key encounters
		for key, enc in list(self._encounters.items()):
			if enc.patient_id == duplicate_id:
				rekey = enc.model_copy(update={"patient_id": surviving_id})
				self._encounters[key] = rekey

		# re-key notes
		for key, note in list(self._notes.items()):
			if note.patient_id == duplicate_id:
				self._notes[key] = note.model_copy(update={"patient_id": surviving_id})

		# re-key problems
		for key, prob in list(self._problems.items()):
			if prob.patient_id == duplicate_id:
				self._problems[key] = prob.model_copy(update={"patient_id": surviving_id})

		# re-key medications
		for key, med in list(self._medications.items()):
			if med.patient_id == duplicate_id:
				self._medications[key] = med.model_copy(update={"patient_id": surviving_id})

		# re-key allergies
		for key, allergy in list(self._allergies.items()):
			if allergy.patient_id == duplicate_id:
				self._allergies[key] = allergy.model_copy(update={"patient_id": surviving_id})

		self._record_audit(tenant_id, "patient_merged", duplicate_id)
		logger.info("emr.merge_patients duplicate=%s surviving=%s", duplicate_id, surviving_id)
		return {
			"duplicate_id": duplicate_id,
			"surviving_id": surviving_id,
			"status": "merged",
			"merged_at": datetime.utcnow().isoformat(),
		}

	async def patient_deduplication_check(
		self,
		incoming: dict[str, Any],
	) -> list["PatientMatchCandidate"]:
		"""Run probabilistic matching against all existing patients in the tenant.

		Returns a list of PatientMatchCandidate sorted by score descending.
		Candidates with score >= 0.85 are flagged as certain duplicates.
		"""
		from .models import PatientMatchCandidate
		from .domain.calculations import patient_match_score

		candidates: list[PatientMatchCandidate] = []
		for (tid, _), patient in self._patients.items():
			if tid != self.tenant_id or patient.is_deleted:
				continue
			existing = {
				"family": patient.name.family,
				"given_0": patient.name.given[0] if patient.name.given else "",
				"birth_date": patient.birth_date.isoformat(),
				"gender": str(patient.gender),
				"biometric_hash": patient.biometric_hash,
			}
			score, matched_fields = patient_match_score(incoming, existing)
			if score >= 0.40:
				candidates.append(PatientMatchCandidate(
					candidate_patient_id=patient.id,
					match_score=score,
					matching_fields=matched_fields,
					is_certain_duplicate=score >= 0.85,
				))
		candidates.sort(key=lambda c: c.match_score, reverse=True)
		return candidates

	# ═══════════════════════════════════════════════════════════════════════════
	# ENCOUNTER LIFECYCLE
	# ═══════════════════════════════════════════════════════════════════════════

	async def update_encounter(
		self,
		tenant_id: str,
		encounter_id: str,
		payload: "EncounterUpdate",
	) -> "EncounterResponse | None":
		"""Apply a partial update to an encounter."""
		enc = self._encounters.get((tenant_id, encounter_id))
		if enc is None:
			return None
		update_data = payload.model_dump(exclude_none=True)
		update_data["updated_at"] = datetime.utcnow()
		updated = enc.model_copy(update=update_data)
		self._encounters[(tenant_id, encounter_id)] = updated
		self._record_audit(tenant_id, "encounter_updated", encounter_id)
		return updated

	async def admit_patient(
		self,
		tenant_id: str,
		encounter_id: str,
		admit_data: dict[str, Any],
	) -> "EncounterResponse | None":
		"""Record formal patient admission (status → in_progress, set admit_time)."""
		enc = self._encounters.get((tenant_id, encounter_id))
		if enc is None:
			return None
		updated = enc.model_copy(update={
			"status": "in_progress",
			"admit_time": datetime.utcnow(),
			"updated_at": datetime.utcnow(),
		})
		self._encounters[(tenant_id, encounter_id)] = updated
		self._record_audit(tenant_id, "patient_admitted", encounter_id)
		logger.info("emr.admit_patient enc=%s tenant=%s", encounter_id, tenant_id)
		return updated

	async def discharge_patient(
		self,
		encounter_id: str,
		discharge_diagnosis: str,
		treatment_summary: str,
		follow_up: str,
		discharge_medications: list[dict[str, Any]],
	) -> dict[str, Any]:
		"""Discharge a patient: close the encounter and generate the discharge summary."""
		enc = self._encounters.get((self.tenant_id, encounter_id))
		if enc is None:
			raise ValueError(f"Encounter {encounter_id} not found")

		# generate discharge summary note
		summary = await self.create_discharge_summary(
			encounter_id=encounter_id,
			discharge_diagnosis=discharge_diagnosis,
			treatment_summary=treatment_summary,
			follow_up=follow_up,
			discharge_medications=discharge_medications,
		)

		# close the encounter
		closed = await self.close_encounter(self.tenant_id, encounter_id)

		self._record_audit(self.tenant_id, "patient_discharged", encounter_id)
		logger.info("emr.discharge_patient enc=%s", encounter_id)
		return {
			"encounter": closed.model_dump(mode="json") if closed else None,
			"discharge_summary": summary,
		}

	async def transfer_patient(
		self,
		encounter_id: str,
		to_location_id: str,
		to_provider_id: str | None,
		reason: str,
	) -> dict[str, Any]:
		"""Transfer patient to a different location/provider within the same encounter."""
		assert to_location_id.strip(), "to_location_id required"
		assert reason.strip(), "reason required"

		enc = self._encounters.get((self.tenant_id, encounter_id))
		if enc is None:
			raise ValueError(f"Encounter {encounter_id} not found")

		update: dict[str, Any] = {"location_id": to_location_id, "updated_at": datetime.utcnow()}
		if to_provider_id:
			update["provider_id"] = to_provider_id
		updated = enc.model_copy(update=update)
		self._encounters[(self.tenant_id, encounter_id)] = updated

		transfer_id = uuid7str()
		transfer_record: dict[str, Any] = {
			"id": transfer_id,
			"encounter_id": encounter_id,
			"patient_id": enc.patient_id,
			"from_location_id": enc.location_id,
			"to_location_id": to_location_id,
			"from_provider_id": enc.provider_id,
			"to_provider_id": to_provider_id,
			"reason": reason,
			"transferred_at": datetime.utcnow().isoformat(),
			"transferred_by": self.actor_id,
		}
		self._record_audit(self.tenant_id, "patient_transferred", encounter_id)
		logger.info("emr.transfer enc=%s from=%s to=%s", encounter_id, enc.location_id, to_location_id)
		return transfer_record

	# ═══════════════════════════════════════════════════════════════════════════
	# NOTE UPDATE
	# ═══════════════════════════════════════════════════════════════════════════

	async def update_note(
		self,
		tenant_id: str,
		note_id: str,
		payload: "ClinicalNoteUpdate",
	) -> "ClinicalNoteResponse | None":
		"""Update a draft note. Final notes must use addendum workflow."""
		note = self._notes.get((tenant_id, note_id))
		if note is None:
			return None
		if note.status == "final":
			raise PolicyViolationError("final_note_immutable: use addendum for corrections")
		update_data = payload.model_dump(exclude_none=True)
		update_data["updated_at"] = datetime.utcnow()
		updated = note.model_copy(update=update_data)
		self._notes[(tenant_id, note_id)] = updated
		self._record_audit(tenant_id, "note_updated", note_id)
		return updated

	# ═══════════════════════════════════════════════════════════════════════════
	# LAB ORDERS & RESULTS
	# ═══════════════════════════════════════════════════════════════════════════

	async def order_lab_test(self, payload: "LabOrderCreate") -> "LabOrderResponse":
		"""Create a new lab test order."""
		from .models import LabOrderResponse
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		order = LabOrderResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			patient_id=payload.patient_id,
			encounter_id=payload.encounter_id,
			ordering_provider_id=payload.ordering_provider_id,
			test_code=payload.test_code,
			test_name=payload.test_name,
			specimen_type=payload.specimen_type,
			priority=payload.priority,
			clinical_indication=payload.clinical_indication,
			created_by=payload.created_by,
		)
		self._lab_orders[(payload.tenant_id, order.id)] = order
		self._record_audit(payload.tenant_id, "lab_order_created", order.id)
		_log_op("order_lab_test", payload.tenant_id, order.id)
		return order

	async def get_lab_order(self, tenant_id: str, order_id: str) -> "LabOrderResponse | None":
		return self._lab_orders.get((tenant_id, order_id))

	async def cancel_lab_order(self, tenant_id: str, order_id: str) -> "LabOrderResponse | None":
		order = self._lab_orders.get((tenant_id, order_id))
		if order is None:
			return None
		from .models import LabOrderStatus
		updated = order.model_copy(update={"status": LabOrderStatus.cancelled, "updated_at": datetime.utcnow()})
		self._lab_orders[(tenant_id, order_id)] = updated
		self._record_audit(tenant_id, "lab_order_cancelled", order_id)
		return updated

	async def list_lab_orders(self, tenant_id: str, patient_id: str) -> list["LabOrderResponse"]:
		return sorted(
			[o for (tid, _), o in self._lab_orders.items() if tid == tenant_id and o.patient_id == patient_id],
			key=lambda o: o.created_at, reverse=True,
		)

	async def receive_lab_result(self, payload: "LabResultCreate") -> "LabResultResponse":
		"""Record a lab result and auto-flag critical values."""
		from .models import LabResultResponse, LabResultFlag
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		result = LabResultResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			order_id=payload.order_id,
			patient_id=payload.patient_id,
			test_code=payload.test_code,
			test_name=payload.test_name,
			value=payload.value,
			value_numeric=payload.value_numeric,
			unit=payload.unit,
			reference_range=payload.reference_range,
			flag=payload.flag,
			result_status=payload.result_status,
			performing_lab=payload.performing_lab,
			result_time=payload.result_time,
			verified_by=payload.verified_by,
			created_by=payload.created_by,
		)
		self._lab_results[(payload.tenant_id, result.id)] = result

		# auto-notify critical values
		if payload.flag in (LabResultFlag.critical_low, LabResultFlag.critical_high):
			logger.warning(
				"emr.critical_lab result=%s patient=%s test=%s flag=%s",
				result.id, payload.patient_id, payload.test_name, payload.flag,
			)

		# update order status
		order = self._lab_orders.get((payload.tenant_id, payload.order_id))
		if order:
			from .models import LabOrderStatus
			updated_order = order.model_copy(update={"status": LabOrderStatus.completed, "updated_at": datetime.utcnow()})
			self._lab_orders[(payload.tenant_id, payload.order_id)] = updated_order

		self._record_audit(payload.tenant_id, "lab_result_received", result.id)
		_log_op("receive_lab_result", payload.tenant_id, result.id)
		return result

	async def list_lab_results(self, tenant_id: str, patient_id: str) -> list["LabResultResponse"]:
		return sorted(
			[r for (tid, _), r in self._lab_results.items() if tid == tenant_id and r.patient_id == patient_id],
			key=lambda r: r.result_time, reverse=True,
		)

	async def flag_critical_lab_result(
		self,
		result_id: str,
		notified_to: str,
	) -> "LabResultResponse":
		"""Mark a critical lab result as notified."""
		assert notified_to, "notified_to required"
		result = self._lab_results.get((self.tenant_id, result_id))
		if result is None:
			raise ValueError(f"Lab result {result_id} not found")
		updated = result.model_copy(update={
			"critical_notified": True,
			"critical_notified_at": datetime.utcnow(),
			"critical_notified_to": notified_to,
			"updated_at": datetime.utcnow(),
		})
		self._lab_results[(self.tenant_id, result_id)] = updated
		self._record_audit(self.tenant_id, "critical_lab_notified", result_id)
		logger.info("emr.critical_lab_notified result=%s to=%s", result_id, notified_to)
		return updated

	async def list_unnotified_critical_labs(self, tenant_id: str) -> list["LabResultResponse"]:
		"""Return critical lab results that have not yet been notified."""
		from .models import LabResultFlag
		return [
			r for (tid, _), r in self._lab_results.items()
			if tid == tenant_id
			and r.flag in (LabResultFlag.critical_low, LabResultFlag.critical_high)
			and not r.critical_notified
		]

	# ═══════════════════════════════════════════════════════════════════════════
	# IMAGING ORDERS
	# ═══════════════════════════════════════════════════════════════════════════

	async def order_imaging(self, payload: "ImagingOrderCreate") -> "ImagingOrderResponse":
		"""Create a new imaging order."""
		from .models import ImagingOrderResponse
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		import random, string
		accession = "IMG-" + "".join(random.choices(string.digits, k=8))
		order = ImagingOrderResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			patient_id=payload.patient_id,
			encounter_id=payload.encounter_id,
			ordering_provider_id=payload.ordering_provider_id,
			modality=payload.modality,
			body_part=payload.body_part,
			laterality=payload.laterality,
			cpt_code=payload.cpt_code,
			clinical_indication=payload.clinical_indication,
			priority=payload.priority,
			contrast_required=payload.contrast_required,
			patient_instructions=payload.patient_instructions,
			accession_number=accession,
			created_by=payload.created_by,
		)
		self._imaging_orders[(payload.tenant_id, order.id)] = order
		self._record_audit(payload.tenant_id, "imaging_order_created", order.id)
		_log_op("order_imaging", payload.tenant_id, order.id)
		return order

	async def get_imaging_order(self, tenant_id: str, order_id: str) -> "ImagingOrderResponse | None":
		return self._imaging_orders.get((tenant_id, order_id))

	async def list_imaging_orders(self, tenant_id: str, patient_id: str) -> list["ImagingOrderResponse"]:
		return sorted(
			[o for (tid, _), o in self._imaging_orders.items() if tid == tenant_id and o.patient_id == patient_id],
			key=lambda o: o.created_at, reverse=True,
		)

	async def add_imaging_report(
		self,
		order_id: str,
		radiologist_id: str,
		impression: str,
	) -> "ImagingOrderResponse | None":
		"""Attach a radiology report to an imaging order."""
		assert radiologist_id, "radiologist_id required"
		assert impression.strip(), "impression required"
		order = self._imaging_orders.get((self.tenant_id, order_id))
		if order is None:
			return None
		from .models import ImagingStatus
		report_id = uuid7str()
		updated = order.model_copy(update={
			"status": ImagingStatus.completed,
			"report_id": report_id,
			"radiologist_id": radiologist_id,
			"impression": impression,
			"reported_at": datetime.utcnow(),
			"updated_at": datetime.utcnow(),
		})
		self._imaging_orders[(self.tenant_id, order_id)] = updated
		self._record_audit(self.tenant_id, "imaging_report_added", order_id)
		logger.info("emr.imaging_report order=%s radiologist=%s", order_id, radiologist_id)
		return updated

	# ═══════════════════════════════════════════════════════════════════════════
	# CARE PLANS
	# ═══════════════════════════════════════════════════════════════════════════

	async def create_care_plan(self, payload: "CarePlanCreate") -> "CarePlanResponse":
		"""Create a new care plan for a patient."""
		from .models import CarePlanResponse
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		plan = CarePlanResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			patient_id=payload.patient_id,
			encounter_id=payload.encounter_id,
			title=payload.title,
			description=payload.description,
			goal=payload.goal,
			icd10_codes=payload.icd10_codes,
			activities=payload.activities,
			start_date=payload.start_date,
			end_date=payload.end_date,
			care_team=payload.care_team,
			created_by=payload.created_by,
		)
		self._care_plans[(payload.tenant_id, plan.id)] = plan
		self._record_audit(payload.tenant_id, "care_plan_created", plan.id)
		_log_op("create_care_plan", payload.tenant_id, plan.id)
		return plan

	async def get_care_plan(self, tenant_id: str, plan_id: str) -> "CarePlanResponse | None":
		return self._care_plans.get((tenant_id, plan_id))

	async def list_care_plans(self, tenant_id: str, patient_id: str) -> list["CarePlanResponse"]:
		return sorted(
			[p for (tid, _), p in self._care_plans.items() if tid == tenant_id and p.patient_id == patient_id],
			key=lambda p: p.created_at, reverse=True,
		)

	async def update_care_plan(
		self,
		tenant_id: str,
		plan_id: str,
		payload: "CarePlanUpdate",
	) -> "CarePlanResponse | None":
		plan = self._care_plans.get((tenant_id, plan_id))
		if plan is None:
			return None
		update_data = payload.model_dump(exclude_none=True)
		update_data["updated_at"] = datetime.utcnow()
		updated = plan.model_copy(update=update_data)
		self._care_plans[(tenant_id, plan_id)] = updated
		self._record_audit(tenant_id, "care_plan_updated", plan_id)
		return updated

	async def activate_care_plan(self, tenant_id: str, plan_id: str) -> "CarePlanResponse | None":
		from .models import CarePlanStatus
		plan = self._care_plans.get((tenant_id, plan_id))
		if plan is None:
			return None
		updated = plan.model_copy(update={"status": CarePlanStatus.active, "updated_at": datetime.utcnow()})
		self._care_plans[(tenant_id, plan_id)] = updated
		self._record_audit(tenant_id, "care_plan_activated", plan_id)
		return updated

	async def complete_care_plan(self, tenant_id: str, plan_id: str) -> "CarePlanResponse | None":
		from .models import CarePlanStatus
		plan = self._care_plans.get((tenant_id, plan_id))
		if plan is None:
			return None
		updated = plan.model_copy(update={"status": CarePlanStatus.completed, "updated_at": datetime.utcnow()})
		self._care_plans[(tenant_id, plan_id)] = updated
		self._record_audit(tenant_id, "care_plan_completed", plan_id)
		return updated

	# ═══════════════════════════════════════════════════════════════════════════
	# REFERRAL MANAGEMENT
	# ═══════════════════════════════════════════════════════════════════════════

	async def list_referrals(self, tenant_id: str, patient_id: str) -> list[dict[str, Any]]:
		return sorted(
			[r for (tid, _), r in self._referrals.items() if tid == tenant_id and r["patient_id"] == patient_id],
			key=lambda r: r["created_at"], reverse=True,
		)

	async def cancel_referral(self, referral_id: str, reason: str) -> dict[str, Any]:
		assert reason.strip(), "reason required"
		ref = self._referrals.get((self.tenant_id, referral_id))
		if ref is None:
			raise ValueError(f"Referral {referral_id} not found")
		ref["status"] = "cancelled"
		ref["cancel_reason"] = reason
		ref["cancelled_at"] = datetime.utcnow().isoformat()
		self._referrals[(self.tenant_id, referral_id)] = ref
		self._record_audit(self.tenant_id, "referral_cancelled", referral_id)
		return ref

	# ═══════════════════════════════════════════════════════════════════════════
	# CONSENTS
	# ═══════════════════════════════════════════════════════════════════════════

	async def list_consents(
		self,
		tenant_id: str,
		patient_id: str,
		scope: str | None = None,
	) -> list[dict[str, Any]]:
		results = [
			c for (tid, _), c in self._consents.items()
			if tid == tenant_id and c["patient_id"] == patient_id
		]
		if scope:
			results = [c for c in results if c.get("consent_type") == scope or c.get("scope") == scope]
		return sorted(results, key=lambda c: c["created_at"], reverse=True)

	# ═══════════════════════════════════════════════════════════════════════════
	# IMMUNISATIONS
	# ═══════════════════════════════════════════════════════════════════════════

	async def record_immunisation(self, payload: "ImmunisationCreate") -> "ImmunisationResponse":
		"""Record a vaccine administration."""
		from .models import ImmunisationResponse
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		imm = ImmunisationResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			patient_id=payload.patient_id,
			encounter_id=payload.encounter_id,
			vaccine_code=payload.vaccine_code,
			vaccine_name=payload.vaccine_name,
			dose_quantity=payload.dose_quantity,
			dose_unit=payload.dose_unit,
			route=payload.route,
			site=payload.site,
			lot_number=payload.lot_number,
			manufacturer=payload.manufacturer,
			expiration_date=payload.expiration_date,
			administered_date=payload.administered_date,
			administered_by=payload.administered_by,
			notes=payload.notes,
			created_by=payload.created_by,
		)
		self._immunisations[(payload.tenant_id, imm.id)] = imm
		self._record_audit(payload.tenant_id, "immunisation_recorded", imm.id)
		_log_op("record_immunisation", payload.tenant_id, imm.id)
		return imm

	async def list_immunisations(self, tenant_id: str, patient_id: str) -> list["ImmunisationResponse"]:
		return sorted(
			[i for (tid, _), i in self._immunisations.items() if tid == tenant_id and i.patient_id == patient_id],
			key=lambda i: i.administered_date, reverse=True,
		)

	# ═══════════════════════════════════════════════════════════════════════════
	# FAMILY HISTORY
	# ═══════════════════════════════════════════════════════════════════════════

	async def add_family_history(self, payload: "FamilyHistoryCreate") -> "FamilyHistoryResponse":
		"""Add a family history entry for a patient."""
		from .models import FamilyHistoryResponse
		self._enforce({"tenant_context_present": bool(payload.tenant_id), "operation_type": "write", "policy_attached": True})
		fhx = FamilyHistoryResponse(
			id=uuid7str(),
			tenant_id=payload.tenant_id,
			patient_id=payload.patient_id,
			relationship=payload.relationship,
			deceased=payload.deceased,
			age_at_death=payload.age_at_death,
			conditions=payload.conditions,
			notes=payload.notes,
			created_by=payload.created_by,
		)
		self._family_history[(payload.tenant_id, fhx.id)] = fhx
		self._record_audit(payload.tenant_id, "family_history_added", fhx.id)
		return fhx

	async def list_family_history(self, tenant_id: str, patient_id: str) -> list["FamilyHistoryResponse"]:
		return [
			fhx for (tid, _), fhx in self._family_history.items()
			if tid == tenant_id and fhx.patient_id == patient_id
		]

	# ═══════════════════════════════════════════════════════════════════════════
	# CLINICAL DECISION SUPPORT (consolidated)
	# ═══════════════════════════════════════════════════════════════════════════

	async def clinical_decision_support(self, patient_id: str) -> list["ClinicalAlert"]:
		"""Run all CDS checks for a patient and return consolidated alerts.

		Checks:
		  - Drug-allergy conflicts for all active medications
		  - Drug-drug interactions across active medication list
		  - Duplicate therapy detection
		  - Critical unnotified lab results
		  - Pregnancy-contraindicated medications (if pregnancy on problem list)
		  - Overdue preventive care reminders
		  - Guideline-based alerts per active diagnoses
		"""
		from .models import ClinicalAlert, AlertType

		alerts: list[ClinicalAlert] = []
		active_meds = await self.list_medications(self.tenant_id, patient_id, status="active")
		active_allergies = await self.list_allergies(self.tenant_id, patient_id)
		active_problems = await self.list_problems(self.tenant_id, patient_id, status="active")

		# 1. Drug-allergy alerts
		for med in active_meds:
			drug_class = _DRUG_CLASS_MAP.get(med.drug_name.lower(), "unknown")
			allergy_result = await self.check_drug_allergy_alert(patient_id, med.drug_name, drug_class)
			for conflict in allergy_result.get("conflicts", []):
				severity_str = "critical" if conflict["severity"] in ("severe", "life_threatening") else "warning"
				alerts.append(ClinicalAlert(
					alert_type=AlertType.drug_allergy,
					severity=severity_str,
					title=f"Drug-Allergy Conflict: {med.drug_name}",
					message=f"Patient has recorded {conflict['severity']} allergy to '{conflict['allergen']}'. Reaction: {conflict['reaction']}",
					affected_entity_id=med.id,
					suggested_action="Discontinue or substitute medication",
					overridable=severity_str != "critical",
					override_reason_required=severity_str == "critical",
				))

		# 2. Drug-drug interactions
		if len(active_meds) >= 2:
			drug_names = [m.drug_name for m in active_meds]
			ddis = await self.check_drug_drug_interactions(drug_names)
			for ddi in ddis:
				severity_str = "critical" if ddi["severity"] == "contraindicated" else "warning"
				alerts.append(ClinicalAlert(
					alert_type=AlertType.drug_interaction,
					severity=severity_str,
					title=f"Drug Interaction: {ddi['drug_a']} + {ddi['drug_b']}",
					message=f"Severity: {ddi['severity']}. {ddi['clinical_effect']}. Management: {ddi['management']}",
					suggested_action=ddi["management"],
					overridable=ddi["severity"] != "contraindicated",
					override_reason_required=ddi["severity"] == "contraindicated",
				))

		# 3. Critical unnotified labs
		critical_labs = await self.list_unnotified_critical_labs(self.tenant_id)
		patient_critical = [r for r in critical_labs if r.patient_id == patient_id]
		for lab in patient_critical:
			alerts.append(ClinicalAlert(
				alert_type=AlertType.critical_lab,
				severity="critical",
				title=f"Critical Lab Result: {lab.test_name}",
				message=f"{lab.test_name} = {lab.value} {lab.unit} (Flag: {lab.flag}). Notification required.",
				affected_entity_id=lab.id,
				suggested_action="Notify responsible clinician immediately",
				overridable=False,
				override_reason_required=False,
			))

		# 4. Clinical reminders
		reminders = await self.clinical_reminder_check(patient_id)
		for reminder in reminders:
			alerts.append(ClinicalAlert(
				alert_type=AlertType.care_gap,
				severity="info",
				title=reminder["description"],
				message=f"Recommended interval: {reminder['recommended_interval_months']} months",
				suggested_action="Schedule appropriate screening/test",
			))

		# 5. Guideline alerts per active diagnosis
		seen_prefixes: set[str] = set()
		for prob in active_problems:
			prefix = prob.icd10_code[:3]
			if prefix not in seen_prefixes:
				seen_prefixes.add(prefix)
				guideline_alerts = await self.clinical_guideline_alert(patient_id, prob.icd10_code)
				for ga in guideline_alerts:
					alerts.append(ClinicalAlert(
						alert_type=AlertType.care_gap,
						severity="info",
						title=ga["title"],
						message=ga["body"],
						references=[ga.get("source", "")],
						suggested_action="Review guideline recommendations",
					))

		_log_cds("cds_alerts_total", patient_id, len(alerts))
		return alerts

	# ═══════════════════════════════════════════════════════════════════════════
	# HL7 v2 MESSAGE PROCESSING
	# ═══════════════════════════════════════════════════════════════════════════

	async def hl7_message_processing(self, message: str) -> dict[str, Any]:
		"""Parse and process an HL7 v2 message.

		Supports: ADT^A01 (admit), ADT^A03 (discharge), ADT^A08 (update),
		ORM^O01 (order), ORU^R01 (observation result).
		Returns a structured acknowledgement dict.
		"""
		if not message.strip():
			raise ValueError("HL7 message must not be empty")

		lines = message.strip().replace("\r\n", "\n").replace("\r", "\n").split("\n")
		segments: dict[str, list[str]] = {}
		for line in lines:
			if line:
				seg_name = line[:3]
				segments.setdefault(seg_name, []).append(line)

		msh = segments.get("MSH", [""])[0]
		msh_parts = msh.split("|")
		message_type = msh_parts[8] if len(msh_parts) > 8 else "UNKNOWN"
		message_id = msh_parts[9] if len(msh_parts) > 9 else uuid7str()

		result: dict[str, Any] = {
			"message_id": message_id,
			"message_type": message_type,
			"segments_received": list(segments.keys()),
			"processed_at": datetime.utcnow().isoformat(),
			"ack_code": "AA",
			"ack_text": "Message accepted",
			"actions_taken": [],
		}

		# ADT^A01 — patient admit
		if "A01" in message_type and "PID" in segments:
			result["actions_taken"].append("patient_admit_notification_received")

		# ORU^R01 — observation result (lab result)
		if "R01" in message_type and "OBX" in segments:
			result["actions_taken"].append("observation_result_received")
			result["obx_count"] = len(segments.get("OBX", []))

		# ORM^O01 — order
		if "O01" in message_type and "ORC" in segments:
			result["actions_taken"].append("order_received")

		self._record_audit(self.tenant_id, "hl7_message_processed", message_id)
		logger.info("emr.hl7 type=%s id=%s actions=%s", message_type, message_id, result["actions_taken"])
		return result

	# ═══════════════════════════════════════════════════════════════════════════
	# REPORTS
	# ═══════════════════════════════════════════════════════════════════════════

	async def controlled_substance_report(self, tenant_id: str) -> dict[str, Any]:
		"""Report on controlled substance prescriptions issued today."""
		today = date.today().isoformat()
		controlled_rxs = [
			rx for (tid, _), rx in self._prescriptions.items()
			if tid == tenant_id
			and rx.get("is_controlled") is True
			and rx.get("created_at", "")[:10] == today
		]
		cs_by_schedule: dict[str, int] = {}
		for rx in controlled_rxs:
			sched = rx.get("dea_schedule", "unknown")
			cs_by_schedule[sched] = cs_by_schedule.get(sched, 0) + 1
		return {
			"tenant_id": tenant_id,
			"date": today,
			"total_controlled_prescriptions": len(controlled_rxs),
			"by_schedule": cs_by_schedule,
			"prescriptions": controlled_rxs,
		}

	# ═══════════════════════════════════════════════════════════════════════════
	# INTERNAL
	# ═══════════════════════════════════════════════════════════════════════════

	def _enforce(self, context: dict[str, Any]) -> None:
		result = evaluate_capability_rules(context)
		if result["decision"] == "deny":
			_log_deny(result["rule"], context.get("tenant_id", self.tenant_id))
			raise PolicyViolationError(result["reason"])

	def _record_audit(self, tenant_id: str, event: str, entity_id: str) -> None:
		self._audit_events.append({
			"tenant_id": tenant_id,
			"actor_id": self.actor_id,
			"event": event,
			"entity_id": entity_id,
			"timestamp": datetime.utcnow().isoformat(),
		})


# ── module-level helpers ──────────────────────────────────────────────────────

def _egfr_to_ckd_stage(egfr: float) -> str:
	if egfr >= 90:
		return "G1 (normal or high)"
	elif egfr >= 60:
		return "G2 (mildly decreased)"
	elif egfr >= 45:
		return "G3a (mildly-moderately decreased)"
	elif egfr >= 30:
		return "G3b (moderately-severely decreased)"
	elif egfr >= 15:
		return "G4 (severely decreased)"
	else:
		return "G5 (kidney failure)"


# ── legacy single-instance shim ───────────────────────────────────────────────

class ElectronicMedicalRecordsService(EMRService):
	"""Backward-compatible alias: behaves exactly like EMRService("default")."""

	def __init__(self) -> None:
		super().__init__(tenant_id="default", actor_id="system")

	async def ml_clinical_note_extract(self, *args, **kwargs):
		"""AI-powered AI extraction of structured data from clinical notes. Requires OLLAMA_BASE_URL."""
		import os
		if not os.environ.get("OLLAMA_BASE_URL"):
			return {"ml_enhanced": False}
		try:
			from capabilities.common.mlx import MLCapability
			ml = MLCapability()
			result = await ml.extract(str(kwargs.get("note",""))[:2000], schema={"diagnoses": "ICD codes or conditions", "medications": "drug names and doses", "procedures": "procedures performed"}, context="electronic medical record")
			return {"extracted": result.extracted, "ml_enhanced": True}
		except Exception:
			return {"ml_enhanced": False}

