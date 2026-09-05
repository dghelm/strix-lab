# Bounded profile-guided campaigns

> **Status: local controller.** Campaigns sit above the existing evidence,
> suite, and comparison-judge contracts. They do not download models, provision
> ROCm, push upstream, or fabricate measurements. `create` only freezes a plan.
> `resume` is hardware.

StrixLab's goal for this work is **bounded profile-guided optimization
campaigns** on top of the evidence scaffold that already exists. A campaign is a
local, finite investigation of a reviewed source-patch list against one frozen
evaluator. It is not a hosted challenge, leaderboard, official score, or
automatic upstream patch bot.

The ranked problem portfolio is
[Profile-guided llama.cpp research problems](research-problems.md): bounded
hypotheses and the smallest honest post-merge pilot. It separates v1 patch
campaigns from configuration tuning and blocked workloads. It is docs-only, not
an experiment catalog.

## Procedure

Keep exploration and confirmation separate. Preserve failed, negative,
inconclusive, and interrupted findings.

1. **Hypothesis** — one bounded question and a reviewed patch list.
2. **Fixed evaluator** — freeze suite, machine, source, build, model, and judge
   policy before any candidate patch. Drift on resume fails closed.
3. **Patch** — each candidate independently patches the same unmodified frozen
   source commit. No extra knobs or commands. Candidate ids cannot be
   `baseline`.
4. **Screen** — exploration only. Not a retain decision.
5. **Fresh confirmation** — new baseline and candidate runs; never reuse
   screening or calibration evidence.
6. **Retain or reject** — keep the finding. The controller does not start the
   next campaign. An external agent reads `campaign report` and may propose a
   **new** reviewed plan. Old campaigns are never rewritten.

Profiled runs remain diagnostics. Confirmation uses clean equivalent arms.

## Evaluator and acceptance

v1 uses the native smoke suite and the existing offline comparison judge.
Capsule scenarios remain deferred. Cross-arm greedy token identity restricts v1
to launch/layout-preserving candidates.

- **Calibration:** two distinct no-op baseline/baseline suite runs and one
  comparison. Fail closed unless the comparison is `inconclusive` and
  cross-arm greedy token digests and counts match.
- **Screening:** four whole-suite AB/BA runs per candidate (baseline,
  candidate, candidate, baseline) and two comparisons.
- **Confirmation:** four fresh independent AB/BA runs, only if screening is
  eligible. Confirmation reuses the screened candidate build; it does not
  silently rebuild.
- **Objective:** optional `objective_cases` (omission means all performance
  cases). Every objective case must have existing judge verdict `improvement`.
  Every remaining case is automatically protected and must satisfy
  `percent_ci_low >= -protected_regression_margin_percent`. The margin is
  finite, `0 <= m < 100`, default `0`.
- **Retain bar:** both screening comparisons and both confirmation comparisons
  must pass those objective/protected gates **and** cross-arm token parity.
  The raw judge overall verdict is preserved, including `mixed`. The campaign
  label is `objective_met_provisional`. That label is not judge improvement
  and not `best-known`. Per-case intervals are not simultaneous or
  campaign-level confidence, and the protected-case bound is not a formal
  noninferiority proof. Calibration stays two baseline/baseline runs plus one
  token-matching `inconclusive` comparison.
- **Reservation:** a whole phase is reserved before any suite work (2 slots
  for calibration, 4 for screening, 4 for confirmation). Reserved slots stay
  spent on failure, crash, or interruption; there is no automatic retry.
- **Interrupts:** an interrupted **candidate** phase is spent and terminal;
  resume continues untouched candidates and never replays the spent phase. An
  interrupted **calibration** stops the campaign.
- **Evidence:** campaign state points at existing suite and comparison runs.
  Some lower-layer failures never return an authenticatable run ID; the phase
  then records that a failure-evidence link is unavailable rather than
  claiming every failure is linked.

v1 patches may modify existing unrenamed files under `ggml/src/` with source
suffixes (`.c`, `.cc`, `.cpp`, `.h`, `.hpp`, `.cu`, `.cuh`, `.hip`). They must
not edit tests, build files, inspector files, or the evaluator. That allowlist
is a scope guard, not a sandbox against dishonest executable output.

`max_suite_runs` is a count budget (admission check), not a wall-clock
deadline. It includes calibration, screening, confirmation, failures, and
interrupted reservations. Zero is allowed and stops before source preparation.
Guaranteed complete capacity for `N` candidates is `2 + 8N` suite runs
(calibration plus screen and confirm for every candidate). Eligible-only
confirmation can cost less.

## Commands

There is no campaign JSON Schema registration; `create` parses a strict
`CampaignPlanV1`. All four commands take optional `--home PATH`. `create`,
`resume`, and `inspect` print canonical JSON state. `report` prints Markdown
from verified inspect state. Screening and confirmation are an automatic gated
sequence with no phase flags.

```bash
uv run strixlab campaign create PLAN.yaml
uv run strixlab campaign inspect ID
uv run strixlab campaign report ID
# resume evaluates remaining phases on hardware; it is not a CPU-only check.
uv run strixlab campaign resume ID
```

Copy the campaign `id` from the JSON that `create` prints. State lives at
`<home>/campaigns/<id>/state.json` with `frozen.json` and copied patches
beside it. `inspect` re-validates evidence links and does not execute or
rewrite the campaign.

The checked-in demonstration plan is
[`configs/campaigns/historical-mmvq-demo.yaml`](../configs/campaigns/historical-mmvq-demo.yaml).
It uses a historical known-negative MMVQ patch as a **harness demonstration
only**, not as a suggested optimization. `create` of that plan is CPU-only;
`resume` of it is real hardware work.

### Plan schema (v1)

Paths are relative to the plan file's directory. Unknown fields are rejected.

| Field | Contract |
|---|---|
| `schema_version` | Integer literal `1` |
| `id` | Dash identifier |
| `suite`, `machine`, `source`, `build`, `model` | Relative paths to existing manifests |
| `candidates` | Nonempty list of `{id, patches}` |
| `candidates[].id` | Dash identifier, unique in the plan, not `baseline` |
| `candidates[].patches` | Nonempty ordered list of relative patch paths |
| `max_candidates` | Integer `>=` candidate-list length and `<= 100` |
| `max_suite_runs` | Integer `>= 0` and `<= 10000` |
| `objective_cases` | Optional nonempty unique subset of the suite's performance case IDs. Omission means all |
| `protected_regression_margin_percent` | Finite number, `0 <= m < 100`, default `0` |

Create copies and hashes resolved manifests and patch bytes, records the
evaluator digest and judge policy, and rejects duplicate ordered patch
identities against the frozen base. Resume refuses evaluator drift. Exhausted
`max_suite_runs` stops before preparing source.

## Implementation plan

1. **Core controller** — freeze-only `create`, hardware `resume`, JSON
   `inspect`, Markdown `report`. Whole-phase reservation; interrupted
   candidates stay spent while untouched candidates continue; calibration
   interruption stops.
2. **Problem portfolio** — [`research-problems.md`](research-problems.md).
   Docs only; no structured catalog.
3. **Reviewer / verification** — existing `compare` unchanged; campaign-only
   token parity and frozen objective/protected gates; preserve mixed judge
   verdicts and negative findings.
4. **Later hardware experiments** — follow the post-merge pilot in the
   problem portfolio. ROCm 10, TOPK capsules, unpinned models, and
   runtime-config sweeps are not v1 prerequisites.

Community experiment PRs in [`community-workflow.md`](community-workflow.md)
remain the catalog path for human-reviewed candidates. Campaigns do not
replace that flow and do not open upstream pull requests.
