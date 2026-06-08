"""Tests for FHIR R4 adapter — APG Healthcare EMR interoperability."""
from capabilities.healthcare.emr.fhir import FHIRAdapter

adapter = FHIRAdapter(base_url="http://emr.test/fhir/r4")

SAMPLE_PATIENT = {
    "id": "pat-001",
    "first_name": "Jane",
    "last_name": "Wanjiru",
    "full_name": "Jane Wanjiru",
    "date_of_birth": "1985-03-15",
    "gender": "female",
    "status": "active",
    "phone_number": "254712345678",
    "email": "jane@example.com",
    "national_id": "12345678",
    "address": {
        "street_address": "123 Ngong Road",
        "city": "Nairobi",
        "county": "Nairobi County",
        "country": "KE",
    },
    "updated_at": "2025-06-01T10:00:00Z",
}

SAMPLE_ENCOUNTER = {
    "id": "enc-001",
    "patient_id": "pat-001",
    "encounter_type": "outpatient",
    "status": "completed",
    "encounter_date": "2025-06-01",
    "start_time": "2025-06-01T09:00:00Z",
    "end_time": "2025-06-01T09:45:00Z",
    "provider_id": "dr-001",
}


# ── Patient → FHIR ──────────────────────────────────────────────────────────

def test_patient_to_fhir_resource_type():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    assert fhir["resourceType"] == "Patient"


def test_patient_to_fhir_id():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    assert fhir["id"] == "pat-001"


def test_patient_to_fhir_name():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    names = fhir["name"]
    assert len(names) >= 1
    assert names[0]["family"] == "Wanjiru"
    assert "Jane" in names[0]["given"]


def test_patient_to_fhir_birthdate():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    assert fhir["birthDate"] == "1985-03-15"


def test_patient_to_fhir_gender():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    assert fhir["gender"] == "female"


def test_patient_to_fhir_telecom_phone():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    phones = [t for t in fhir["telecom"] if t["system"] == "phone"]
    assert phones[0]["value"] == "254712345678"


def test_patient_to_fhir_telecom_email():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    emails = [t for t in fhir["telecom"] if t["system"] == "email"]
    assert emails[0]["value"] == "jane@example.com"


def test_patient_to_fhir_address():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    addr = fhir["address"][0]
    assert addr["city"] == "Nairobi"
    assert addr["country"] == "KE"


def test_patient_to_fhir_national_id_identifier():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    nid_idents = [i for i in fhir["identifier"] if "nid" in i.get("system", "")]
    assert nid_idents[0]["value"] == "12345678"


def test_patient_to_fhir_active():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    assert fhir["active"] is True


# ── FHIR → Patient ──────────────────────────────────────────────────────────

def test_patient_from_fhir_name():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    apg = adapter.patient_from_fhir(fhir)
    assert apg["first_name"] == "Jane"
    assert apg["last_name"] == "Wanjiru"


def test_patient_from_fhir_birthdate():
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    apg = adapter.patient_from_fhir(fhir)
    assert apg["date_of_birth"] == "1985-03-15"


def test_patient_roundtrip():
    """Converting APG→FHIR→APG preserves key fields."""
    fhir = adapter.patient_to_fhir(SAMPLE_PATIENT)
    apg = adapter.patient_from_fhir(fhir)
    assert apg["first_name"] == SAMPLE_PATIENT["first_name"]
    assert apg["gender"] == SAMPLE_PATIENT["gender"]
    assert apg["email"] == SAMPLE_PATIENT["email"]


# ── Encounter → FHIR ────────────────────────────────────────────────────────

def test_encounter_to_fhir_resource_type():
    fhir = adapter.encounter_to_fhir(SAMPLE_ENCOUNTER)
    assert fhir["resourceType"] == "Encounter"


def test_encounter_to_fhir_status_mapping():
    fhir = adapter.encounter_to_fhir(SAMPLE_ENCOUNTER)
    assert fhir["status"] == "finished"  # "completed" → "finished"


def test_encounter_to_fhir_subject_reference():
    fhir = adapter.encounter_to_fhir(SAMPLE_ENCOUNTER)
    assert fhir["subject"]["reference"] == "Patient/pat-001"


def test_encounter_to_fhir_period():
    fhir = adapter.encounter_to_fhir(SAMPLE_ENCOUNTER)
    assert "period" in fhir
    assert "2025-06-01" in fhir["period"]["start"]


def test_encounter_to_fhir_class_outpatient():
    fhir = adapter.encounter_to_fhir(SAMPLE_ENCOUNTER)
    assert fhir["class"]["code"] == "AMB"  # outpatient → ambulatory


def test_encounter_to_fhir_participant():
    fhir = adapter.encounter_to_fhir(SAMPLE_ENCOUNTER)
    assert fhir["participant"][0]["individual"]["reference"] == "Practitioner/dr-001"


# ── Observation → FHIR ──────────────────────────────────────────────────────

def test_observation_to_fhir_blood_pressure():
    obs = {"id": "obs-001", "type": "blood_pressure_systolic", "value": 120, "unit": "mmHg",
           "recorded_at": "2025-06-01T09:15:00Z"}
    fhir = adapter.observation_to_fhir(obs, patient_id="pat-001", encounter_id="enc-001")
    assert fhir["resourceType"] == "Observation"
    assert fhir["code"]["coding"][0]["system"] == "http://loinc.org"
    assert fhir["code"]["coding"][0]["code"] == "55284-4"
    assert fhir["valueQuantity"]["value"] == 120.0
    assert fhir["valueQuantity"]["unit"] == "mmHg"
    assert fhir["encounter"]["reference"] == "Encounter/enc-001"


def test_observation_to_fhir_with_loinc_code():
    obs = {"id": "obs-002", "type": "heart_rate", "value": 72, "unit": "bpm"}
    fhir = adapter.observation_to_fhir(obs, patient_id="pat-001")
    assert fhir["code"]["coding"][0]["code"] == "8867-4"


# ── Capability Statement ────────────────────────────────────────────────────

def test_capability_statement():
    stmt = adapter.capability_statement()
    assert stmt["resourceType"] == "CapabilityStatement"
    assert stmt["fhirVersion"] == "4.0.1"
    resource_types = [r["type"] for r in stmt["rest"][0]["resource"]]
    assert "Patient" in resource_types
    assert "Encounter" in resource_types
    assert "Observation" in resource_types
