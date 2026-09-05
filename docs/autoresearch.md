# Bounded profile-guided campaigns

> **Status: local controller scaffold.** Campaigns sit above the existing
> evidence, suite, and comparison-judge contracts. They do not download models,
> provision ROCm, push upstream, or fabricate measurements. Hardware campaign
> execution is later; this document records the procedure and the v1 command
> surface.

StrixLab's goal for this work is **bounded profile-guided optimization
campaigns** on top of the evidence scaffold that already exists. A campaign is a
local, finite investigation of reviewed source patches against one frozen
evaluator. It is not a hosted challenge, leaderboard, official score, or
automatic upstream patch bot.

The ranked problem portfolio is
[Profile-guided llama.cpp research problems](research-problems.md). That
document lists bounded hypotheses and the smallest honest post-merge pilot. It
separates v1 patch campaigns from configuration tuning and blocked workloads.
It is docs-only, not an experiment catalog. Launch-reuse and profile-derived
quant dispatch lead the patch questions; configuration tuning and blocked
workloads stay out of the v1 candidate surface.

## What a campaign is

A campaign freezes one source, build, machine, model, suite, and judge policy,
then evaluates a finite list of reviewed patch candidates against that
evaluator. Screening (exploration) and confirmation are separate phases of the
same campaign. Negative, failed, and inconclusive findings are retained.

A campaign is **not**:

- a replacement for `run suite`, `run inspect`, `compare`, or bundle export
- a way to skip correctness gates or to treat a mixed/inconclusive verdict as
  a win
- a per-candidate knob sweep (v1 freezes runtime and build arguments)
- a capsule, TOPK, or ROCm 10 prerequisite
- an upstream pull-request bot

## Agent procedure

Keep this loop explicit. Do not collapse exploration into confirmation, and do
not start the next campaign until the current one has a durable retain or
reject decision.

1. **Hypothesis** — pick one bounded question from the problem portfolio, with
   a reviewed patch list small enough to falsify.
2. **Fixed evaluator** — freeze the suite, machine, source, build, model, and
   comparison policy before any candidate patch is applied. The evaluator is
   strict: resolved manifests, patch bytes, evaluator digest, and judge
   policy are hashed at create and rechecked on every admission and resume.
   Drift fails closed. The evaluator does not move mid-campaign.
3. **Patch** — each candidate is an independent patch list against the same
   unmodified frozen source commit. Candidates do not carry extra commands or
   config knobs. Candidate ids cannot be `baseline`.
4. **Screen** — exploration only. Run the candidate against the frozen
   evaluator and record the comparison. A screen is not a retain decision.
5. **Fresh independent confirmation** — if screening looks promising, run a
   **new** baseline and a **new** candidate against the same frozen
   evaluator. Do not reuse screening runs, screening comparisons, or
   calibration suites as confirmation evidence.
6. **Retain or reject** — keep the finding, including correctness failures,
   regressions, mixed results, inconclusive comparisons, and spent
   interrupted attempts. The controller does not invent the next hypothesis.
   An external agent reads `campaign report` and may propose a **new**
   campaign from that report.

Profiled runs remain diagnostics. Confirmation uses clean equivalent arms, as
the foundation design already requires.

## Frozen evaluator and acceptance

v1 campaigns consume the native smoke suite and the existing offline
comparison judge. Capsule scenarios remain deferred. The campaign does not
redesign the judge.

Acceptance is conservative and campaign-local:

- Calibration is two distinct no-op **baseline/baseline** suite runs — two
  independent executions of the unpatched frozen baseline, compared to each
  other, not to a candidate. The campaign fails closed unless they compare
  `inconclusive` and pass the campaign-only cross-arm greedy token-parity
  gate. Failed calibration does not yield a permissive campaign.
- Screening and confirmation each use whole-suite AB then BA order. Cost is
  **2** baseline/baseline calibration runs once, then **4** screening runs
  per candidate, then **4** fresh confirmation runs only for screening
  winners. Confirmation is a new independent baseline and candidate pair.
  AB/BA balances whole-suite order; it is not temporally paired sampling
  and is not a campaign-level confidence interval.
- Authenticated per-prompt greedy token digests and counts must match
  across baseline and candidate on every comparison, including calibration.
  Token mismatch is preserved as correctness-negative evidence, never as
  improvement.
- Conservative v1 retain bar: **both** screening comparisons **and both**
  confirmation comparisons must all-case improve under the existing judge,
  and the cross-arm greedy digest/count gate must pass on each of those
  comparisons. A mixed or inconclusive overall verdict is not a retain
  decision. The campaign does not rewrite the judge into `best-known`.
- The plan may still freeze optional `objective_cases` (omission means all
  performance cases) and `protected_regression_margin_percent` (default
  `0`). Until core confirms a narrower frozen objective, document and treat
  the coordinator bar as all-case improvement on every required comparison.
- Whole-phase reservation happens before any suite work: 2 slots for
  calibration, 4 for screening, 4 for confirmation. An interrupted phase is
  spent and terminal and is **not** auto-retried. Remaining unstarted
  candidates may continue only without replaying that spent phase. Failed
  and budget-exhausted reservations stay spent.

The `ggml/src` patch allowlist is a v1 scope guard: campaigns patch source
files under that tree and do not edit build, test, inspector, or evaluator
files. It is not a sandbox against dishonest executable output.

## Commands

v1 is a conservative fixed reviewed patch list. `create PLAN.yaml` freezes
that list and does not run suites. `resume ID` evaluates the finite remainder.
`inspect ID` prints canonical JSON state. `report ID` renders Markdown from
that verified state so an **external agent** can propose the next campaign;
the controller does not auto-start one. State is persisted under the resolved
StrixLab home as `campaigns/<id>/state.json` with frozen input copies beside
it. Exploration versus confirmation is phase-internal — there is no extra
flag that turns a screen into a confirmation. README command examples wait on
the checked-in CLI. Exact flags remain owned by core; the proposed surface
is:

```bash
uv run strixlab campaign create PLAN.yaml
uv run strixlab campaign inspect ID
uv run strixlab campaign report ID
# Hardware evaluation is later; resume is not a scaffold-time experiment.
uv run strixlab campaign resume ID
```

Do not invent additional flags, a "latest campaign" alias, or a hosted API.
Ask core for exact options such as `--home` before treating them as frozen.

### Plan schema (v1)

All paths are relative to the plan file's directory. The plan is strict:
unknown fields are rejected.

| Field | Contract |
|---|---|
| `schema_version` | Integer literal `1` |
| `id` | Dash identifier |
| `suite`, `machine`, `source`, `build`, `model` | Relative paths to existing manifests |
| `candidates` | Nonempty list of `{id, patches}` |
| `candidates[].id` | Dash identifier, unique in the plan, and not the reserved id `baseline` |
| `candidates[].patches` | Nonempty ordered list of relative patch paths |
| `max_candidates` | Integer `>=` the candidate list length and `<= 100`; a shorter cap is rejected |
| `max_suite_runs` | Integer `>= 0` and `<= 10000`. Count budget covering calibration, screening, confirmation, failed phases, and interrupted reservations. Zero is allowed and stops before source preparation. This is an admission check, not a wall-clock deadline |
| `objective_cases` | Optional nonempty unique subset of the suite's performance case IDs. Omission means all performance cases |
| `protected_regression_margin_percent` | Optional finite number, `0 <= m < 100`, default `0` |

Create copies and hashes the resolved manifests and patch bytes, records the
evaluator/package digest and judge policy identity, and rejects duplicate
ordered patch identities against the frozen base. Resume refuses evaluator
drift. Exhausted `max_suite_runs` stops with an explicit reason **before**
preparing source. Do not describe that count as a hard wall-clock deadline.

Suite-run cost is 2 calibration runs for the campaign, plus 4 AB/BA screening
runs per candidate, plus 4 fresh AB/BA confirmation runs per screening
winner. Failed and interrupted whole-phase reservations still consume their
reserved slots. Do not claim statistical confidence from that budget. Core
owns the schema-valid harness fixture
[`configs/campaigns/historical-mmvq-demo.yaml`](../configs/campaigns/historical-mmvq-demo.yaml)
when it lands: it is an existing historical regression used as a **harness
demo only**, not a suggested promising optimization. `create` can freeze on
CPU; `resume` on hardware is separately authorized and later.

## Implementation plan

Work is staged so the controller and the problem list can land without
pretending hardware results exist.

1. **Core controller** — freeze-only `campaign create PLAN.yaml`, bounded
   `campaign resume ID`, JSON `campaign inspect ID`, Markdown `campaign
   report ID`. Whole-phase reservation before side effects; interrupted
   phases stay spent and terminal and are not auto-retried. An external
   agent proposes the next campaign from the report. Count budgets include
   2 calibration runs, 4 screening runs per candidate, 4 confirmation runs
   per screening winner, failures, and interruptions.
2. **Problem portfolio** — [`research-problems.md`](research-problems.md)
   ranks bounded hypotheses and the post-merge pilot, separating v1 patch
   campaigns from configuration tuning and blocked workloads. Docs only; no
   structured catalog.
3. **Reviewer / verification** — independent review of campaign evidence.
   Existing `compare` stays unchanged. Conservative retain requires both
   screening comparisons and both confirmation comparisons to all-case
   improve, plus cross-arm greedy digest/count match. Failed and
   inconclusive findings remain in the campaign state.
4. **Later hardware experiments** — after the scaffold is reviewable, follow
   the post-merge pilot in the problem portfolio: calibrate, profile a
   baseline, try one-mechanism patch candidates, then confirm. ROCm 10,
   TOPK capsules, unpinned models, and runtime-config sweeps are not
   prerequisites and are not implied by v1.

Community experiment PRs in [`community-workflow.md`](community-workflow.md)
remain the catalog path for human-reviewed candidates. Campaigns do not
replace that flow and do not open upstream pull requests.
