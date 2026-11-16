# Synthetic patients

**Every record in this folder is synthetic.** The people, names, addresses and
clinical histories are invented for testing trialmatch-agent. None of them describes
a real person. Each Bundle and resource carries a `meta.tag` with code `synthetic`
(plus the HL7 `HTEST` test-data tag), and names follow the Synthea convention of
numeric suffixes (for example `Marieke482 Dalen731`) so they cannot be mistaken for
real people.

## Format

One FHIR R4 `Bundle` of type `collection` per file, in the style of Synthea output:
a `Patient` (with `address[0].country`), plus `Condition`, `MedicationRequest`,
`Observation` and `Procedure` resources. Resource ids are stable and unique within a
bundle (`<patient>-<kind>-<n>`, for example `p002-obs-4`), so evidence references
such as `Observation/p002-obs-4` stay valid across regenerations.

Codes: SNOMED CT for conditions and procedures, LOINC for observations, RxNorm
ingredient codes for medications, UCUM for units. Where we were not confident of a
code we left it out and kept only `code.text` (for example "FEV1 % predicted").

Ages are computed by `trialmatch.patients.FhirPatientStore` against a fixed as-of
date of **2026-11-01**, so results are reproducible over time.

## Patients

| id | sex, age | country | primary diagnosis | planted issue (for eval) |
|---|---|---|---|---|
| p001 | F, 56 | Netherlands | Type 2 diabetes mellitus (metformin; HbA1c 8.1%, eGFR 72) | none, clean candidate |
| p002 | M, 68 | United States | Type 2 diabetes mellitus, Chronic kidney disease stage 4 (insulin) | latest eGFR 24, below common eGFR < 30 exclusions |
| p003 | F, 47 | United States | HER2-positive invasive ductal carcinoma of left breast | prior chemotherapy: doxorubicin and cyclophosphamide (completed) plus a chemotherapy Procedure in 2024 |
| p004 | F, 34 | Netherlands | Hormone receptor positive malignant neoplasm of right breast | currently pregnant: Observation LOINC 82810-3 = Pregnant (SNOMED 77386006), 2026-10-14 |
| p005 | M, 72 | Netherlands | Chronic obstructive pulmonary disease (tiotropium; current daily smoker; FEV1 48% predicted) | none |
| p006 | M, 81 | United States | Heart failure with reduced ejection fraction (LVEF 30%) | age 81, above common 80-year maximums |
| p007 | F, 17 | Netherlands | Moderate persistent asthma (inhaled budesonide) | minor: born 2008-11-20, turns 18 only after the as-of date |
| p008 | F, 52 | Germany | Rheumatoid arthritis (methotrexate) | none |

## Using real Synthea data

`trialmatch.patients.load_synthea_directory(path)` reads a Synthea `output/fhir`
folder (transaction bundles with `urn:uuid` fullUrls) with the same parser. Hospital
and practitioner bundles, which contain no Patient, are skipped.
