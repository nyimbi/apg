"""APG Healthcare EMR — FHIR R4 Adapter.

Converts APG PatientResponse, EncounterResponse, and clinical records to/from
HL7 FHIR R4 JSON resources, enabling interoperability with:
  - Epic Systems (FHIR R4 bulk export)
  - Oracle Health / Cerner (SMART on FHIR)
  - OpenMRS, OpenEMR, and other FHIR-capable systems
  - Apple Health Records, Google Health Connect
  - National health data exchanges (Kenya MoH IHE, etc.)

Reference: https://hl7.org/fhir/R4/

Usage::

    from capabilities.healthcare.emr.fhir import FHIRAdapter
    adapter = FHIRAdapter(base_url="https://emr.hospital.ke/fhir/r4")

    # Convert APG patient to FHIR R4 Patient resource
    fhir_patient = adapter.patient_to_fhir(apg_patient_response)

    # Convert incoming FHIR R4 to APG PatientCreate
    apg_create = adapter.patient_from_fhir(fhir_json)
"""
from .adapter import FHIRAdapter, FHIRCapabilityStatement

__all__ = ["FHIRAdapter", "FHIRCapabilityStatement"]
