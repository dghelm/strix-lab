# Experiment catalog

This directory catalogs community experiments. Each record evaluates one
candidate under one exact scenario and may contain several independent
replications.

Create `experiments/<scenario-id>/<experiment-id>/README.md` from `TEMPLATE.md`
and open an **experiment** pull request. Candidate patches may live beside that
record beneath `patches/`. PR comments coordinate reruns, but only the checked-in
record is canonical. Preserve negative and inconclusive results.

See [`docs/community-workflow.md`](../docs/community-workflow.md) for the
definitions, lifecycle, and privacy boundary.
