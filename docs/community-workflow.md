# Community experiment workflow

StrixLab uses GitHub for review and coordination, not for executing untrusted
GPU code. Contributors run experiments on machines they control, StrixLab
records and compares the evidence locally, and pull requests build a small,
reviewable catalog of what was tried.

## Vocabulary

- A **scenario** is the benchmark question and its rules. In v1, the checked-in
  suite manifest in `configs/suites/` is the scenario definition; its resolved
  manifest digest fixes the exact workload, correctness gates, measurements,
  and timeouts.
- A **candidate** is the reproducible source or build change being evaluated.
  Prefer this neutral word to "solution": a candidate may improve, regress, or
  have no measurable effect.
- A **run** is one local execution of a scenario, either baseline or candidate.
- A **replication** is one contributor's attempt to reproduce the candidate
  under the scenario: a baseline run and candidate run, followed by an offline
  comparison when both pass correctness. A candidate correctness failure is a
  complete negative replication even though comparison is correctly skipped.
  A replication is evidence from one machine and time, not universal
  validation.
- An **experiment** is the community investigation of one candidate under one
  scenario. It has one catalog record and may collect multiple replications
  from different contributors.
- A **finding** is the conclusion supported by the replications so far.
  "Best-known" is always scoped to one exact scenario and is a catalog status,
  not a global score.

StrixLab reserves **verification** for checking evidence integrity. Use
**replication**, not "validation," for another contributor's rerun.

## Contribution flow

### 1. Propose a scenario in an Issue

Use the **scenario proposal** Issue form to describe the question, workloads,
correctness gates, measurements, required inputs, and hardware assumptions.
Discussion happens before code so the scenario does not move after candidates
are measured against it.

### 2. Implement the scenario in a pull request

Use the **scenario** pull-request template. The PR adds or changes the suite
manifest and any supporting checked-in configs, tests, and documentation.
Scenario PRs do not claim optimization results.

A material rule change creates a new scenario ID. This keeps old experiments
interpretable instead of silently moving their target.

### 3. Open one experiment pull request per candidate

Copy `experiments/TEMPLATE.md` to
`experiments/<scenario-id>/<experiment-id>/README.md` and use the **experiment**
pull-request template. The PR identifies the exact scenario and candidate and
may include narrow patches beneath that experiment directory or the
configuration needed to reproduce it.

An experiment PR may start as a draft before hardware is available. It becomes
mergeable after at least one complete replication is recorded. Improvement is
not required: regressions, inconclusive outcomes, and reproducible correctness
failures prevent duplicate work and belong in the catalog.

### 4. Add replications

A GPU owner reproduces the candidate locally and runs matched baseline and
candidate suites. If both pass correctness, run `strixlab compare`; if the
candidate fails correctness, stop and preserve that result. Post the small,
redacted summary from `.github/replication-comment.md` on the experiment PR. Do
not attach model weights, a raw StrixLab home, private data, credentials, or
evidence bundles.

Comments are coordination, not the source of truth. Before merge, accepted
replication summaries are copied into the experiment record. A replication
performed after merge uses a small follow-up PR that updates the same record.

## Experiment status

Use the most conservative status supported by the checked-in record:

- `proposed` — candidate is reproducibly described but has no complete
  replication yet; normally a draft PR, not a merged catalog entry.
- `observed` — at least one complete replication is recorded.
- `independently-replicated` — at least one complete replication is from a
  contributor other than the candidate author.
- `best-known` — maintainers have determined that the current evidence is the
  best known for this exact scenario. Contributors should not self-award it.

Merging means the experiment is reproducible and worth preserving. It does not
mean the candidate is universally faster, safe for production, accepted
upstream, or a "winning solution."

## What the repository stores

The repository stores scenario configs, candidate patches or config changes,
experiment records, commands, run identities, record digests, comparison
verdicts, and redacted notes. It does not store weights, raw StrixLab homes,
private datasets, credentials, or evidence bundles.

A future website may render this checked-in catalog. It does not need—and must
not silently introduce—a service that builds or executes submitted code.
