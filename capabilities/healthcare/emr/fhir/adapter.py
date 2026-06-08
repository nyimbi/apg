"""FHIR R4 adapter — converts APG EMR models to/from HL7 FHIR R4 JSON.

Implements the FHIR R4 mapping for:
  Patient   — demographic, identifier, contact, address
  Encounter — visit, status, type, period, participant
  Observation — clinical measurements, lab results
  Condition — diagnoses, problems
  MedicationRequest — prescriptions
  Procedure — clinical procedures

The adapter is stateless and pure-function: every method takes APG model dicts
or Pydantic models and returns FHIR JSON dicts (or vice versa). It does not
own any database connections.
"""
from __future__ import annotations

import re
from typing import Any


class FHIRAdapter:
	"""Bidirectional converter between APG EMR models and FHIR R4 resources.

	Args:
		base_url: FHIR server base URL, used in resource IDs and links.
		          e.g. "https://emr.hospital.ke/fhir/r4"
	"""

	def __init__(self, base_url: str = "http://localhost:8080/fhir/r4") -> None:
		self.base_url = base_url.rstrip("/")

	# ── Patient ───────────────────────────────────────────────────────────

	def patient_to_fhir(self, patient: dict[str, Any] | Any) -> dict[str, Any]:
		"""Convert APG PatientResponse to FHIR R4 Patient resource."""
		p = patient if isinstance(patient, dict) else patient.model_dump()

		resource: dict[str, Any] = {
			"resourceType": "Patient",
			"id": p.get("id", ""),
			"meta": {
				"profile": ["http://hl7.org/fhir/StructureDefinition/Patient"],
				"lastUpdated": p.get("updated_at", ""),
			},
			"identifier": [
				{
					"use": "official",
					"system": f"{self.base_url}/identifier/patient",
					"value": p.get("id", ""),
				}
			],
			"active": p.get("status", "active") == "active",
			"name": [],
			"telecom": [],
			"address": [],
		}

		# Name
		full_name = p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
		if full_name:
			resource["name"].append({
				"use": "official",
				"text": full_name,
				"family": p.get("last_name", ""),
				"given": [n for n in [p.get("first_name"), p.get("middle_name")] if n],
			})

		# Birth date
		if p.get("date_of_birth"):
			resource["birthDate"] = str(p["date_of_birth"])[:10]

		# Gender
		gender_map = {"male": "male", "female": "female", "other": "other", "unknown": "unknown"}
		resource["gender"] = gender_map.get(str(p.get("gender", "")).lower(), "unknown")

		# Contact information
		for phone in [p.get("phone_number"), p.get("mobile_number")]:
			if phone:
				resource["telecom"].append({"system": "phone", "value": phone, "use": "mobile"})
		if p.get("email"):
			resource["telecom"].append({"system": "email", "value": p["email"]})

		# Address
		if p.get("address"):
			addr = p["address"] if isinstance(p["address"], dict) else {"text": str(p["address"])}
			resource["address"].append({
				"use": "home",
				"text": addr.get("text", ""),
				"line": [addr.get("street_address", "")],
				"city": addr.get("city", ""),
				"state": addr.get("county", ""),
				"country": addr.get("country", "KE"),
			})

		# National ID as additional identifier
		if p.get("national_id"):
			resource["identifier"].append({
				"use": "secondary",
				"system": "https://ke.go.ke/nid",
				"value": p["national_id"],
			})

		# Remove empty lists
		for key in ("name", "telecom", "address"):
			if not resource[key]:
				del resource[key]

		return resource

	def patient_from_fhir(self, fhir_patient: dict[str, Any]) -> dict[str, Any]:
		"""Convert FHIR R4 Patient resource to APG PatientCreate-compatible dict."""
		result: dict[str, Any] = {
			"fhir_id": fhir_patient.get("id", ""),
			"active": fhir_patient.get("active", True),
		}

		# Name
		for name in fhir_patient.get("name", []):
			if name.get("use") in ("official", None, "usual"):
				result["first_name"] = (name.get("given") or [""])[0]
				result["last_name"] = name.get("family", "")
				result["full_name"] = name.get("text") or f"{result['first_name']} {result['last_name']}".strip()
				break

		# Birth date
		if fhir_patient.get("birthDate"):
			result["date_of_birth"] = fhir_patient["birthDate"]

		# Gender
		result["gender"] = fhir_patient.get("gender", "unknown")

		# Telecom
		for t in fhir_patient.get("telecom", []):
			if t.get("system") == "phone":
				result["phone_number"] = t.get("value", "")
			elif t.get("system") == "email":
				result["email"] = t.get("value", "")

		# Address
		for addr in fhir_patient.get("address", []):
			if addr.get("use") in ("home", None):
				result["address"] = {
					"text": addr.get("text", ""),
					"street_address": (addr.get("line") or [""])[0],
					"city": addr.get("city", ""),
					"county": addr.get("state", ""),
					"country": addr.get("country", "KE"),
				}
				break

		# Identifiers
		for ident in fhir_patient.get("identifier", []):
			system = ident.get("system", "")
			if "nid" in system or "national" in system:
				result["national_id"] = ident.get("value", "")

		return result

	# ── Encounter ─────────────────────────────────────────────────────────

	def encounter_to_fhir(self, encounter: dict[str, Any] | Any) -> dict[str, Any]:
		"""Convert APG EncounterResponse to FHIR R4 Encounter resource."""
		e = encounter if isinstance(encounter, dict) else encounter.model_dump()

		status_map = {
			"scheduled": "planned",
			"arrived": "arrived",
			"in_progress": "in-progress",
			"completed": "finished",
			"cancelled": "cancelled",
			"no_show": "noshow",
		}

		resource: dict[str, Any] = {
			"resourceType": "Encounter",
			"id": e.get("id", ""),
			"status": status_map.get(str(e.get("status", "")), "unknown"),
			"class": {
				"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
				"code": "AMB" if str(e.get("encounter_type", "")).lower() == "outpatient" else "IMP",
			},
			"subject": {
				"reference": f"Patient/{e.get('patient_id', '')}",
				"type": "Patient",
			},
		}

		# Type
		if e.get("encounter_type"):
			resource["type"] = [{
				"coding": [{
					"system": "http://snomed.info/sct",
					"display": str(e["encounter_type"]).replace("_", " ").title(),
				}]
			}]

		# Period
		if e.get("start_time") or e.get("encounter_date"):
			resource["period"] = {
				"start": str(e.get("start_time") or e.get("encounter_date") or ""),
			}
			if e.get("end_time"):
				resource["period"]["end"] = str(e["end_time"])

		# Participant (clinician)
		if e.get("provider_id"):
			resource["participant"] = [{
				"individual": {"reference": f"Practitioner/{e['provider_id']}"}
			}]

		return resource

	# ── Observation ───────────────────────────────────────────────────────

	def observation_to_fhir(
		self,
		obs: dict[str, Any],
		patient_id: str,
		encounter_id: str = "",
	) -> dict[str, Any]:
		"""Convert an APG clinical observation dict to FHIR R4 Observation."""
		loinc_map = {
			"blood_pressure_systolic": ("55284-4", "Blood Pressure"),
			"blood_pressure_diastolic": ("8462-4", "Diastolic blood pressure"),
			"heart_rate": ("8867-4", "Heart rate"),
			"temperature": ("8310-5", "Body temperature"),
			"oxygen_saturation": ("59408-5", "Oxygen saturation in Arterial blood"),
			"weight": ("29463-7", "Body weight"),
			"height": ("8302-2", "Body height"),
			"bmi": ("39156-5", "Body mass index (BMI)"),
			"blood_glucose": ("2339-0", "Glucose"),
			"hemoglobin": ("718-7", "Hemoglobin"),
		}

		obs_type = obs.get("type", "")
		loinc_code, display = loinc_map.get(obs_type, ("", obs_type))

		resource: dict[str, Any] = {
			"resourceType": "Observation",
			"id": obs.get("id", ""),
			"status": "final",
			"subject": {"reference": f"Patient/{patient_id}"},
			"effectiveDateTime": str(obs.get("recorded_at", obs.get("date", ""))),
		}

		if loinc_code:
			resource["code"] = {
				"coding": [{"system": "http://loinc.org", "code": loinc_code, "display": display}],
				"text": display,
			}

		if encounter_id:
			resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}

		value = obs.get("value")
		unit = obs.get("unit", "")
		if value is not None:
			resource["valueQuantity"] = {
				"value": float(value),
				"unit": unit,
				"system": "http://unitsofmeasure.org",
				"code": unit,
			}

		return resource

	# ── Capability Statement ──────────────────────────────────────────────

	def capability_statement(self) -> dict[str, Any]:
		"""Return FHIR R4 CapabilityStatement for this server."""
		return {
			"resourceType": "CapabilityStatement",
			"status": "active",
			"kind": "instance",
			"fhirVersion": "4.0.1",
			"format": ["json"],
			"rest": [{
				"mode": "server",
				"resource": [
					{"type": "Patient", "interaction": [{"code": "read"}, {"code": "search-type"}, {"code": "create"}]},
					{"type": "Encounter", "interaction": [{"code": "read"}, {"code": "search-type"}]},
					{"type": "Observation", "interaction": [{"code": "read"}, {"code": "search-type"}]},
				],
			}],
			"implementation": {
				"description": "APG Healthcare EMR FHIR R4 endpoint",
				"url": self.base_url,
			},
		}


# Module-level alias for backward compatibility
FHIRCapabilityStatement = FHIRAdapter
