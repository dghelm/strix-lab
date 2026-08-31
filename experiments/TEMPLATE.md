<!--
Copy to experiments/<scenario-id>/<experiment-id>/README.md. Delete these comments.
One record represents one candidate under one scenario and may contain many
replications. Do not include weights, credentials, private data, raw StrixLab
homes, identifying machine values, or evidence bundles.
-->

# Experiment: <short candidate description>

- **Experiment ID:** <dash-id matching the directory name>
- **Scenario:** `configs/suites/<scenario-id>.yaml`
- **Resolved scenario SHA-256:** `<sha256 | pending>`
- **Status:** proposed | observed | independently-replicated | best-known
- **Candidate author:** <handle or name>

## Question

<What single change is being evaluated, and why might it matter?>

## Candidate

- **Source repository:** <URL>
- **Base commit:** <full commit SHA>
- **Candidate commit or patch:** <full commit SHA or repository-relative paths beneath `patches/`>
- **Build profile:** `<repository-relative config path>`

Describe any reproduction step not captured by the paths above. Keep the
candidate narrow enough that another contributor can reproduce it without
guessing.

## Replications

Add one subsection per accepted replication. A complete performance replication
has a matched baseline run, candidate run, and offline comparison. A candidate
correctness failure is also a complete negative replication; record its failed
run and use `N/A — candidate correctness failure` for the comparison fields.
Record immutable `record-sha256:` identities from inspected or verified
evidence; local run IDs alone are not portable evidence identities.

### Replication 1 — <YYYY-MM-DD, contributor>

- **Machine summary:** <non-identifying hardware / OS / ROCm summary>
- **StrixLab commit:** <full commit SHA>
- **Baseline build:** <build ID or exact config + source identity>
- **Candidate build:** <build ID or exact config + source identity>
- **Baseline run record:** `record-sha256:<digest>`
- **Candidate run record:** `record-sha256:<digest>`
- **Comparison run record:** `record-sha256:<digest>` | N/A — candidate correctness failure
- **Comparison verdict:** improvement | regression | mixed | inconclusive | N/A
- **Correctness:** passed | candidate failed
- **Notes:** <redacted conditions, anomalies, or `none`>

## Finding

<Conservative conclusion supported by the replications. State uncertainty and
disagreement. Do not generalize beyond this exact scenario.>

## Reproduction

```bash
<exact commands with repository-relative paths; redact identifying values>
```

## Privacy confirmation

- [ ] This record contains no model weights, credentials, private data, raw
      StrixLab home, evidence bundle, hostname, device serial, or identifying
      absolute path.
