<!-- Scenario implementation PR. Link the accepted proposal Issue. -->

## Scenario

- **Proposal Issue:** closes #
- **Scenario ID:**
- **Suite manifest:** `configs/suites/<scenario-id>.yaml`

## Contract

Describe the workload, correctness gates, measurements, required inputs, and
hardware assumptions. Explain why these rules are stable enough for candidates
to be compared against them.

## Verification

<!-- List exact repository checks and any real-hardware dry run. -->

## Checklist

- [ ] This PR implements a scenario, not a candidate optimization or result claim.
- [ ] The suite and supporting configs are pinned, validated, tested, and documented.
- [ ] Correctness gates run before performance measurements are interpreted.
- [ ] A material change to an existing scenario uses a new scenario ID.
- [ ] No weights, credentials, private data, raw StrixLab home, or evidence bundle are included.
