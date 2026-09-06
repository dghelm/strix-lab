# Private ROCm 10 execution plan

Prepared on 2026-09-05 before SDK execution. This plan covers implementation and
execution through HIP smoke, TOPK correctness, and capsule timing. The initial
commands were implementation specifications; executable research fixtures and
launcher now live under `research/rocm10/`. See the runtime amendment below.

## Scope amendment

Use archive SHA-256 `4feabd9f2da72352df37f6d714a54847d3fe913c0341fbe2a6542c1164024baf`
and exactly this completed private prefix:

```text
/home/dgh/.local/share/strixlab/toolchains/.stage-rocm-10.0.0-gfx1151-20260905T232532Z-99a93ed7/prefix-2
```

Completed quarantine result SHA-256:
`08f153a04c921e4d8a9429e3e92b58a43c7134e710259d6e66eb01c37c927eb4`.
Source baseline: `e927b17c832900b365372b44ce5bb7b5ab901a11`; review and pin subsequent
fixture changes before execution.

For this private unprivileged experiment, replace the installation runbook's
qualified metadata-absence and `/opt` staged/installed equality requirements with
the baseline/isolation checks below. Metadata coverage remains **unknown** and
vendor authenticity **unverified**. This is a policy amendment, not a completed
metadata audit. The previous approval covered quarantine and static inspection;
SDK/GPU execution requires acceptance of this concrete amendment. Implementation
and trusted-host launcher tests can proceed before execution.

No privileged fixture, installation, global activation, metadata removal, host
configuration change or full benchmark campaign is required. Explicit residual
limits: shared host kernel/GPU driver, no seccomp policy, and quiescent same-UID
ownership of the private prefix. Read-only mounts do not prevent host-side writes;
isolation does not establish vendor authenticity or metadata absence.

## Established evidence

The completed prefix has 27,627 descendants, 8,883,028,033 regular bytes, 305
resolved symlinks and 472 statically inspected ELFs. The mandatory candidate
compiler/HIP closure has 23 SDK and eight host ELFs, zero missing literal DT_NEEDED
candidates, and none of the 27 absolute-RUNPATH findings. This is not runtime proof.
HSA has a potential optional load of `libhsa-amd-aqlprofile64.so`, whose RUNPATH has
a foreign absolute path and empty component. Keep the SDK unchanged, `/__w` absent
and all SDK execution in empty read-only `/run/empty`; observe actual loads.

A trusted-host-only Bubblewrap probe passed with private namespaces, all capability
sets zero, NoNewPrivs=1 and ro/nosuid/nodev `/usr`; no SDK or GPU was exposed. The
final mounts still require their own preflight.

## Implementation and ownership

1. Root implements a small research launcher and smoke fixture under
   `research/rocm10/`, using existing prefix checks and GPU leases. Test launcher
   behavior with trusted host commands before SDK execution.
2. TOPK owner implements a research HIP wrapper around the existing baseline
   adapter and CPU oracle; primitives owner implements a separate rocPRIM fixture.
3. Capsule owner integrates qualified providers with existing sealed requests,
   manifests, replay, prior-response and lease/finalization checks.
4. Review each runnable increment before executing it. Do not build a generic
   framework or register a production toolchain merely to run the smoke.

## Launcher and baseline contract

Use host Python argv lists, never interpolated shell commands. Create an exclusive
private output directory mounted at `/work`; pin fixture source bytes at `/input`.
The following is the exact intended command shape; placeholders are concrete
paths recorded by the future launcher, not shell variables:

```text
/usr/bin/bwrap
  --unshare-all --unshare-user --disable-userns --assert-userns-disabled
  --die-with-parent --new-session --cap-drop ALL --clearenv
  --ro-bind /usr /usr
  --symlink usr/bin /bin --symlink usr/lib /lib --symlink usr/lib /lib64
  --ro-bind <verified-prefix-2> /sdk --ro-bind <pinned-fixtures> /input
  --bind <exclusive-output> /work --ro-bind <empty-fixture> /run/empty
  --proc /proc --dev /dev --size 2147483648 --tmpfs /tmp
  --chdir /run/empty
  --setenv PATH /usr/bin:/bin --setenv LC_ALL C --setenv TMPDIR /tmp
  /usr/bin/python -I /input/preflight_exec.py <absolute-command-and-arguments>
```

The host Python preflight starts with only the host environment shown above and
isolated Python mode. Do not expose SDK loader paths or diagnostic loader settings
before preflight: even host Python could otherwise load SDK libraries. Only after
all assertions pass, construct the SDK child environment and exec the exact command:
`PATH=/sdk/bin:/sdk/lib/llvm/bin:/usr/bin:/bin`, `HIP_PLATFORM=amd`,
`HIP_PATH=/sdk`, `ROCM_PATH=/sdk`, `HIP_CLANG_PATH=/sdk/lib/llvm/bin`,
`LD_LIBRARY_PATH=/sdk/lib:/sdk/lib/llvm/lib:/sdk/lib/rocm_sysdeps/lib`, plus the
fixed locale/temp settings. Add `LD_DEBUG=libs,files` only for diagnostic SDK
execution, never for the trusted preflight. No inherited loader/plugin variables.

The proposed preflight must assert UID/GID 1000/1000, all five capability sets zero,
NNP=1, private user/mount/PID/IPC/network namespaces, disabled nested user namespaces,
expected mounts/submounts, SDK/input/usr/empty-CWD ro/nosuid/nodev, scratch/output
nosuid/nodev, and no host home, control ROCm, network or session sockets. Verify the
bound SDK's physical identity. Keep HOME absent, stdin `/dev/null`, stdout/stderr
captured and unrelated inherited descriptors closed. Fail before SDK execution on
any mismatch. Record actual mount/namespace evidence.

Bound compiler phases to 300 seconds wall time, 240 seconds CPU, 16 GiB virtual
address space, 512 MiB per output file, 1 GiB total persistent output, 16 MiB logs,
zero core size. Bound GPU phases to 30 seconds wall time, 20 seconds CPU, 256 MiB
requested device allocations and 16 MiB logs. The launcher monitors aggregate
output and terminates the phase on a bound. Do not apply the compiler VA cap to HSA,
which may reserve large address ranges; GPU-phase process-RAM containment is not
claimed. Tear down the entire phase via private PID namespace/process group, TERM
then KILL after two seconds. Userspace timeout cannot guarantee GPU-driver recovery.
Bound changes require review, not silent retry with expanded access.

Before/after each bounded phase, use existing two-walk inventory with the reviewed
256 MiB evidence cap. Require semantic equality with the completed baseline,
physical root/entry device+inode equality, exact members/link closure and every raw
metadata observation observed-empty without errors. Pin reports and retain unknown
coverage. Keep the host prefix quiescent throughout.

## A. GPU-free build

No GPU nodes or host sysfs. Capture bounded `LD_DEBUG=libs,files` for diagnostic
commands/helpers and compare loads with pinned SDK and host allowlists:

```text
/sdk/lib/llvm/bin/clang++ --version
/sdk/lib/llvm/bin/clang++ -### -x hip --rocm-path=/sdk --hip-path=/sdk --offload-arch=gfx1151 -std=c++17 /input/hip_smoke.cpp -o /work/hip-smoke
/sdk/lib/llvm/bin/clang++ -x hip --rocm-path=/sdk --hip-path=/sdk --offload-arch=gfx1151 -std=c++17 /input/hip_smoke.cpp -o /work/hip-smoke
```

Smoke fixture specification: allocate 256 uint32 elements, launch one 256-thread
kernel computing `out[i] = 3*i + 1`, check every HIP result, synchronize, copy back,
compare all values and free resources. Emit machine-readable success only after
all checks. Record versions/device architecture and `dl_iterate_phdr` paths.
Pin source and executable digests. Host-readelf the output before GPU execution.

Host runtime allowlist starts with loader, libc/libdl/libm/libpthread/librt,
libgcc_s and libstdc++; pin resolved bytes. TOPK CPU oracle additionally needs
libcrypto and its observed z/zstd/brotli closure. Every ROCm library must resolve
inside `/sdk`. Unexpected helper/load paths stop for review; literal closure and
loader observations are separate evidence and do not prove all future dlopen paths.

If later using CMake: `CMAKE_HIP_COMPILER=/sdk/lib/llvm/bin/clang++`,
`CMAKE_HIP_COMPILER_ROCM_ROOT=/sdk`, `CMAKE_HIP_ARCHITECTURES=gfx1151`,
`GPU_TARGETS=gfx1151`. Both architecture variables matter: the HIP package otherwise
executes `amdgpu-arch`. Configuration and version probes are SDK execution.

## B. Tiny GPU smoke

Acquire the existing exclusive GPU lease before opening devices. Verify and record
character-device major/minor, PCI identity and gfx1151 mapping for `/dev/kfd` and
`/dev/dri/renderD128`; add only these individual `--dev-bind` mounts. Static host
mapping identifies renderD128 as PCI `0000:c2:00.0`; KFD node 1 reports
`drm_render_minor=128`, `gfx_target_version=110501`, vendor 4098 and device 5510.
Recheck these identities before launch. Initial read-only sysfs candidates are
`/sys/devices`, `/sys/class/kfd`, `/sys/class/drm`, `/sys/bus/pci`. The implementation
must freeze the exact discovery paths it uses, including the render-node device
link and KFD node properties, and check those paths in the sandbox. This set is
not globally symlink-closed or proven sufficient: the render device driver/module
link resolves to `/sys/module/amdgpu`, and `/sys/dev/char/226:128` is an unmounted
alias. If HIP requires either or another absent path, stop and review that explicit
additional mount; do not silently broaden exposure. Record that mounted sysfs
reveals host device topology. Device binds are deliberate exceptions to nodev,
not GPU isolation.
No broad `/dev/dri`, permission/group changes or silent alternative-device selection.

Run `/work/hip-smoke` with diagnostic loader capture, verify baseline stability,
finalize the lease on success/failure. Pass requires exact output, correct target,
reviewed actual libraries, no HIP errors and complete evidence. No performance
claims from diagnostic runs.

## C. TOPK correctness

Wrapper requirements: CPU preflight before allocation; contiguous row-major F32,
I32 indices, nonaliasing buffers, checked dimensions/scratch capacity; owned stream
and allocations; synchronization before copying back; bounds-checked reconstruction
of values from original bits; existing CPU `reference`/`validate` as oracle.

First finite distinct-value cases `(rows,columns,k)`:
`(1,1,1)`, `(1,8,3)`, `(2,8,8)`, `(1,3,2)`, `(1,1024,1)`.
The baseline supports at most 1024 columns. These cases are restricted smoke tests,
not TOPK-v1 qualification. Then test duplicates, ties at K, signed zeros, infinities,
boundary shapes, invalid dimensions/capacity and documented NaN rejection.

The bitonic strict comparisons do not establish canonical tie ordering. Preserve
counterexamples; failures block full-contract timing/registration. Repair ordering,
or use a restricted research contract only if existing scenario machinery supports
it. Never weaken the CPU oracle. Independently qualify rocPRIM signatures, output
ordering, membership, scratch sizing, and capture/replay where required. Header
presence is not conformance. Exact provider build commands and manifest identities
are outputs of this implementation step; neither exists today.

## D. Capsule timing and comparison

After declared correctness passes: build/inspect capsule, pin all identities,
prepare/verify synthetic input, describe, two serial correctness operations, then
one warmup and five benchmark samples through the existing transport under a
separately reviewed research scenario and identity. Do not reuse
`rocm10-topk-gfx1151-v1`: that fixed scenario requires five warmups, 30 samples,
the full matrix, and eager plus graph execution. Restricted baseline shapes and
reduced repetitions do not qualify for it. If no valid separate research scenario
can be registered, retain the original gates and report only direct conformance
until a provider meets them. Disable loader diagnostics for timing. Retain raw samples, statistic/method, shapes,
semantics, replay and lease/finalization evidence. Declare whether transfers/host
overhead are included; compare identical measurement boundaries and inputs.

Compare ROCm 10 providers only if both qualify for the same semantics; otherwise
publish a single-candidate result without a speedup claim. ROCm 7.2.4 execution
needs its own pinned profile and is outside this immediate experiment. Optimize
only after correctness and valid comparable measurement.

## Deliverables and gates

First implementation deliverable: reviewed launcher/smoke source plus trusted-host
checks and accepted execution amendment. First runtime deliverable: evidenced HIP
smoke. Final deliverable: oracle-qualified TOPK capsule result with reproducible
identity and raw timings. Failed gates remain recorded failures, not eligibility.
Preserve completed quarantine and existing control throughout.

## Runtime amendment: DRM character-device aliases

The first isolated GPU smoke failed in `hipGetDeviceCount` with error 100 and
`amdgpu_get_auth` diagnostics, before launching a kernel. The reviewed correction
adds only read-only `/sys/dev/char` to the existing GPU sysfs set. Preflight checks
its ro/nosuid/nodev flags, absence of unexpected submounts, and that `226:128` is
the same sysfs node as `/sys/class/drm/renderD128`; its `device/drm` path must exist.
No extra device node or permission is introduced.

The Linux implementation of `drmNodeIsDRM` checks the character-device alias path;
`drmGetNodeTypeFromFd` rejects a descriptor when that check fails. See the
[libdrm source mirror](https://android.googlesource.com/platform/external/libdrm/+/refs/heads/main/xf86drm.c#3315).
This explains why the alias mapping is required even with `/sys/devices` mounted.
The correction received independent isolation review before retrying the same
smoke binary. `-O3` is used for fixture compilation, without fast-math flags.
