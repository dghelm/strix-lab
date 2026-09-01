<!-- One experiment PR evaluates one candidate under one exact scenario. -->

## Experiment

- **Scenario:** `configs/suites/<scenario-id>.yaml`
- **Experiment record:** `experiments/<scenario-id>/<experiment-id>/README.md`
- **Candidate:** <!-- exact commit or repository-relative patch/config -->

## Current evidence

- **Status:** proposed | observed | independently-replicated
- **Author replication:** <!-- link to record subsection or `pending` -->
- **Independent replications:** <!-- links to comments/record subsections or `none yet` -->

## Checklist

- [ ] This PR contains one candidate experiment; it does not change the scenario rules.
- [ ] The candidate and reproduction commands are exact and reviewable.
- [ ] At least one complete replication, including a reproducible correctness failure if applicable, is recorded before requesting merge.
- [ ] Accepted PR-comment replications are copied into the experiment record.
- [ ] Negative, failed, mixed, and inconclusive evidence is preserved honestly.
- [ ] I did not self-award `best-known`; maintainers assign that status within an exact scenario.
- [ ] No weights, credentials, private data, raw StrixLab home, identifying machine values, or evidence bundles are included.
