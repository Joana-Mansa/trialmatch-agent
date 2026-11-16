# Evaluation harness

`python -m trialmatch.eval` runs the matching agent once per synthetic patient and
scores the runs in code. There is no LLM judge: every number below is computed from
the `MatchResult`, the patient store and the same `TrialsSource` the agent used.

```
.venv/Scripts/python -m trialmatch.eval --patients all --http live --yes
.venv/Scripts/python -m trialmatch.eval --patients p002,p004 --effort high --concurrency 2
```

Options: `--patients p001,p004|all`, `--http live|record|replay` (default:
`TRIALMATCH_HTTP` or live), `--model`, `--effort`, `--max-turns`, `--concurrency N`
(default 1), `--out` (default `evals/results`), `--expectations`, `--gold`,
`--no-fallback`, `--yes`.

**This costs money.** Every run makes paid Anthropic API calls, whatever the `--http`
mode. Before starting, the harness prints the number of patients and the model and
asks `Continue? [y/N]`; `--yes` skips the prompt. `ANTHROPIC_API_KEY` must be set.

## Outputs

Each run writes `evals/results/<UTC timestamp>/`:

- `report.md`: readable tables (also printed to stdout)
- `results.json`: all metrics, per patient and overall
- `runs/<patient>.json`: the full `MatchResult` (report, trace, grounding issues) so a
  reviewer can audit any number back to the run

## Metrics

| Metric | Definition | Target |
|---|---|---|
| Run health | counts of completed, refused, incomplete, error | all completed |
| Hard-constraint violations | a reported match whose code hard check (age, sex) failed but whose verdict is not `likely_ineligible`. Union of the run's `hard_check_contradiction` grounding issues and a recomputation of `hard_checks.check` on re-fetched eligibility | 0 |
| Citation validity | share of evidence ref occurrences that exist in the patient record (`store.resource_exists`); the rest are listed as fabricated | 100% |
| Criterion coverage | per match: number of assessments, share `unknown`, share with at least one evidence ref | higher coverage, lower unknown |
| Planted-disqualifier recall | see below | 100% |
| Cost and latency | tokens from `trace.usage`, dollars from `PRICING` in `eval.py` (Opus 5.5, taken 2026-11: input $4, output $20, cache read $0.20, cache write $5 per MTok), wall seconds, turns, tool calls by name, fallback used | |
| Gold agreement | verdict agreement with human labels in `gold.jsonl`, for trials the agent assessed | |

### Planted-disqualifier recall

Driven by `expectations.json`. The planted patients are p002 (eGFR about 24), p003
(prior chemotherapy), p004 (pregnant), p006 (age 81) and p007 (age 17); p001, p005 and
p008 are clean and have no rule.

- `criterion` rules (p002, p003, p004): a reported trial counts toward the denominator
  only if its eligibility text, re-fetched through the same trials source, matches the
  rule's `criteria_regex`. It is a hit if some assessment is an exclusion with status
  `met`, or an inclusion with status `not_met`, citing one of `correct_refs`, and the
  verdict is not `likely_eligible`.
- `hard_check` rules (p006, p007): a reported trial counts if the code hard check fails
  for the patient; it is a hit if the verdict is `likely_ineligible`.

When no reported trial is relevant, recall is reported as `n/a`, not 0 or 100%.

## Gold labels

`gold.jsonl` is to be filled by a human reviewer: one JSON object per line,
`{"patient_id", "nct_id", "expected_verdict", "notes"}`. Lines starting with `#` and
rows with `"placeholder": true` are skipped, so the shipped file has 0 active rows.
Label trials the agent actually reports (see `runs/<patient>.json`), otherwise they
cannot be compared.

## Limitations

- **Recall only sees the trials the agent chose to report.** A trial the agent
  silently dropped is neither a hit nor a miss, and an agent that reports fewer
  relevant trials can score higher. Read recall together with the denominator.
- **Keyword relevance is approximate.** The regexes can match text that is not an
  exclusion (for example "kidney" in an inclusion about diabetic kidney disease, or
  "pregnancy test required"), and miss phrasings they do not cover. This inflates or
  deflates the denominator; the per-trial table shows the matched text for review.
- **A hit checks the citation, not the reasoning.** Any exclusion met / inclusion
  not_met that cites a correct ref counts, even if it is attached to a different
  criterion than the relevant one.
- **Citation validity checks existence, not relevance.** A real resource id cited for
  the wrong criterion still counts as valid.
- **Hard checks cover age and sex only.** They do not catch other structured
  disqualifiers.
- **Live data drifts.** In live mode the re-fetch happens after the run, so the
  criteria can change in between (rare), and results are not reproducible across days
  as recruiting trials change. Use `--http record` then `replay` to pin a run.
- **Costs are estimates.** Prices are hard-coded as of 2026-11; runs that used the
  server-side fallback may have had some turns billed at another model's rate, and
  models without a price entry are reported as unpriced.
- **Small sample.** Eight synthetic patients, one run each; the agent is not
  deterministic, so repeat runs before drawing conclusions.
