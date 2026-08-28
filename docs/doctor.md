# Machine doctor

`strixlab doctor` is the BASE-000 read-only machine-readiness check.

```bash
uv run strixlab doctor --machine configs/machines/strix-halo-128g.yaml
```

The command resolves and validates the machine profile, freezes the process environment, captures bounded host facts, attempts the configured exclusive GPU lock without waiting, and evaluates every readiness predicate. It does not install packages, change drivers, tune clocks, select power profiles, or otherwise remediate the host.

## Exit and output contract

- Exit 0 means the report contains no blockers.
- Exit 1 means the validated machine is blocked, observation failed, or a report could not be published safely.
- Typer usage errors retain exit 2.
- Invalid profiles produce sanitized terminal diagnostics and no report.
- A lock owner atomically replaces the authoritative `doctor.json`.
- A non-owner atomically publishes a unique diagnostic sibling and never replaces the authoritative report.

The default path is `<StrixLab home>/doctor/<machine-id>/doctor.json`. `--home` changes the home used for that default, while `--output` replaces the complete destination.

## Probe ordering and identity

Only OS, CPU, memory, AC-power state, and executable presence are observed before locking. KFD, DRM, `rocminfo`, SMI, version commands, and telemetry execute while the GPU lock is held. A non-owner marks GPU-facing checks as skipped.

KFD nodes are correlated to DRM render nodes and PCI BDFs. `rocminfo` and SMI records must resolve to that same identity. Exactly one GPU must match the profile architecture and integrated/discrete classification; unrelated GPUs do not create ambiguity.

## Telemetry

One telemetry source is fixed for the complete three-sample window:

1. `amd-smi` when configured and usable;
2. read-only `rocm-smi` in `auto` mode;
3. sysfs in `auto` or `disabled` mode.

Sources are not mixed after sampling starts. All three busy readings must be valid; incomplete temperature readings warn rather than block. An `auto` fallback warns. Required `amd-smi` that cannot correlate the GPU and provide three busy readings blocks.

## Report and privacy

`doctor.json` is a strict, versioned `kind: doctor`, `schema_version: 1` document with normalized host facts, tool facts, GPU identity, compact samples, stable checks, probe errors, the resolved machine profile, and bounded environment metadata.

The full environment and raw command output are never serialized. Sensitive-name values are removed from designated free-text fields, interpolation from secret-like environment names is rejected, and final JSON bytes are scanned before publication. If secret safety cannot be established, publication fails closed.

BASE-000 ends at readiness. Immutable checksums, run bundles, build evidence, loaded-library proof, and benchmark process attribution belong to later milestones beginning with EVIDENCE-001.
