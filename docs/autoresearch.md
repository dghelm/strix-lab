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
   interrupted attempts. Then stop or hand the next hypothesis to a **new**
   campaign.

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
- Screening and confirmation each use whole-suite AB then BA order (four
  suite runs per phase). Confirmation is a fresh independent pair of
  baseline and candidate suites. That order balances arms; it is not
  temporally paired sampling and is not a campaign-level confidence
  interval.
- Authenticated per-prompt token digests and counts must match across
  baseline and candidate on every comparison, including calibration. Token
  mismatch is preserved as correctness-negative evidence, never as
  improvement.
- The existing judge is unchanged. Its overall verdict may remain `mixed`.
  The campaign does not rewrite that verdict into improvement or
  `best-known`.
- A frozen campaign objective is declared separately. `objective_cases` is
  an optional nonempty unique subset of the suite's performance case IDs
  (omission means all of them). Every objective case must already have
  judge verdict `improvement`. Every remaining case is automatically
  protected and must satisfy `percent_ci_low >=
  -protected_regression_margin_percent`. The margin is at least 0 and
  less than 100, default 0. Both screening comparisons and both fresh
  confirmations must pass these gates.
- When those gates pass, the campaign label is `objective_met_provisional`.
  Per-case intervals are not simultaneous or campaign-level confidence, and
  the protected-case bound is not a formal noninferiority proof.
- Failed, interrupted, and budget-exhausted attempts stay spent. A resume
  that finds a `running` phase marks it `interrupted` with its reserved
  suite slots consumed and **does not replay** that work. Further attempts
  require a new campaign. Resume also does not replay a completed phase or
  confirm twice.

The `ggml/src` patch allowlist is a v1 scope guard: campaigns patch source
files under that tree and do not edit build, test, inspector, or evaluator
files. It is not a sandbox against dishonest executable output.

## Commands

`create` freezes a reviewed plan and does not run suites. `resume` evaluates
the finite remainder. `inspect` prints canonical JSON state. `report` renders
Markdown from that verified state for the next separately reviewed campaign.
State is persisted under the resolved StrixLab home as
`campaigns/<id>/state.json` with frozen input copies beside it. Exploration
versus confirmation is phase-internal — there is no extra flag that turns a
screen into a confirmation. README command examples wait on the checked-in
CLI; the controller surface is:

```bash
uv run strixlab campaign create path/to/plan.yaml
# copy the campaign identifier from the printed JSON state

uv run strixlab campaign inspect "$CAMPAIGN_ID"
uv run strixlab campaign report "$CAMPAIGN_ID"

# Hardware evaluation is later. resume continues remaining phases against the
# frozen evaluator; it is not a copy-paste experiment to run during scaffold
# work, and it never pushes upstream.
uv run strixlab campaign resume "$CAMPAIGN_ID"
```

Each command accepts optional `--home PATH`, matching the rest of the CLI.
Do not invent additional flags, a "latest campaign" alias, or a hosted API.

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

A successful candidate that reaches confirmation costs two calibration suite
runs plus four screening runs plus four confirmation runs; failed and
interrupted reservations still consume their reserved slots. Do not claim
statistical confidence from that budget. Core owns the schema-valid harness
fixture [`configs/campaigns/historical-mmvq-demo.yaml`](../configs/campaigns/historical-mmvq-demo.yaml)
when it lands: it is an existing historical regression used as a **harness
demo only**, not a suggested promising optimization. `create` can freeze on
CPU; `resume` on hardware is separately authorized and later.

## Implementation plan

Work is staged so the controller and the problem list can land without
pretending hardware results exist.

1. **Core controller** — freeze-only `campaign create`, bounded `campaign
   resume`, JSON `campaign inspect`, Markdown `campaign report`. Durable
   phase reservation before side effects; interrupted or unknown outcomes
   stay spent and are never replayed; next campaign is a new plan. Count
   budgets include calibration, confirmation, failures, and interruptions.
2. **Problem portfolio** — [`research-problems.md`](research-problems.md)
   ranks bounded hypotheses and the post-merge pilot. Docs only; no
   structured experiment catalog until something consumes one.
3. **Reviewer / verification** — independent review of campaign evidence.
   Existing `compare` stays unchanged. Campaign-only cross-arm token parity
   and the frozen objective/protected-regression gates are extra acceptance
   rules; the campaign label `objective_met_provisional` does not rewrite
   the judge. Failed and inconclusive findings remain in the campaign state.
4. **Later hardware experiments** — after the scaffold is reviewable, follow
   the post-merge pilot in the problem portfolio: calibrate, profile a
   baseline, try one-mechanism patch candidates, then confirm. ROCm 10,
   TOPK capsules, unpinned models, and runtime-config sweeps are not
   prerequisites and are not implied by v1.

Community experiment PRs in [`community-workflow.md`](community-workflow.md)
remain the catalog path for human-reviewed candidates. Campaigns do not
replace that flow and do not open upstream pull requests.
