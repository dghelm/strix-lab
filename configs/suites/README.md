# Scenarios

Each suite manifest in this directory is a StrixLab **scenario**: an exact
benchmark question with pinned correctness gates, performance cases, scheduling,
and timeouts. The canonical scenario identity for a run is the resolved manifest
digest captured in its evidence, not merely the filename or `id`.

Propose a new scenario with the GitHub **scenario proposal** Issue form, then
implement it using the **scenario** pull-request template. A material rule change
must use a new scenario ID so existing experiment records retain their meaning.

See [`docs/community-workflow.md`](../../docs/community-workflow.md) for the
complete workflow and terminology.
