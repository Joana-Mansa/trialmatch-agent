# Design notes

Short write-up of the choices behind trialmatch-agent, written for the person who would deploy it at a hospital or research site.

## 1. Code checks what code can check

The model reads free-text criteria and connects them to the record. That is the part that needs a language model. Comparing an 81-year-old against an "80 Years" maximum does not, so `hard_checks.py` does it, and the result is handed to the model labelled as authoritative. After the run, `grounding.py` flags any trial marked eligible that failed a hard check. The same idea applies to citations: the model must cite FHIR resource ids, and code verifies each one exists in that patient's record.

The rule: anything deterministic is computed, anything interpretive is cited, and contradictions between the two are surfaced in the result instead of being smoothed over.

## 2. A hand-written agent loop instead of the SDK tool runner

The Anthropic SDK has a tool runner that drives the loop. This project writes the loop itself (`agent.py`) because it needs:

- a full trace per run (every tool call, its duration, token usage, request ids) for debugging and the eval;
- a hard turn cap and explicit handling of each stop reason (refusal, max tokens, pause, ending without a report);
- tests that replay scripted model responses built from the SDK's real response types, with no API key.

The loop keeps history append-only: each assistant turn goes back with its full content, and all tool results for a turn go back in one message. The system prompt and tool list are constants so prompt caching hits on every turn after the first.

## 3. Structured output through a tool

The final answer is a `submit_match_report` call with a strict schema, validated again with Pydantic. A report that fails validation goes back to the model as a tool error with the validation messages, so it can correct itself. Forcing a tool call is not available on this model, so the prompt asks for it and the loop nudges once if the model stops without it.

## 4. Ports, so the integration is swappable

The agent only knows two protocols in `ports.py`:

- `PatientStore`: summary, lab history, and "does this resource id exist". The local implementation reads FHIR bundles. A site deployment would implement the same four methods against its FHIR server (`GET /Patient/{id}`, `/Condition?patient=`, `/Observation?patient=&code=`), with its own auth.
- `TrialsSource`: search and eligibility. The live implementation is ClinicalTrials.gov; a site could add its internal trial registry behind the same interface.

Nothing in the agent, tools, eval or API changes when either is replaced.

## 5. Record and replay for external APIs

ClinicalTrials.gov changes daily and has rate limits. The client can record responses to JSON cassettes and replay them, so CI runs the whole suite offline and deterministically, and an eval run can be pinned to a snapshot.

## 6. Failure handling

- ClinicalTrials.gov: retries with backoff on 429, 5xx and network errors, honours `Retry-After`, then raises a typed error that the tool returns to the model as an error result rather than crashing the run.
- Claude API: the SDK retries transient errors; a refusal is detected by stop reason, and server-side fallback to another model is enabled by default and recorded in the trace.
- Service: runs execute in a bounded worker pool; a crashing run becomes an `error` job and the worker keeps going; logs carry run id, status and usage, never patient data beyond the id.

## 7. Evaluation without an LLM judge

Every metric in `eval.py` is a rule: hard-constraint violations, citation validity, criterion coverage, and recall on planted disqualifiers (pregnancy, low eGFR, prior chemotherapy, age edges). Rules are weaker than expert labels but cheap, reproducible and hard to argue with. `evals/gold.jsonl` holds space for human-labelled verdicts when a clinician can provide them.

## 8. What a real deployment would add

- Postgres or Redis for jobs instead of in-memory state; authentication on the API.
- A data processing agreement and the provider's zero-data-retention or regional options before any real patient data is sent to a model API.
- De-identification before the model sees the record, and audit logging of who requested which match.
- Clinician-labelled evaluation data and a regression gate in CI on those labels.
