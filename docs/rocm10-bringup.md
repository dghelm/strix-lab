# ROCm 10 gfx1151 bring-up

This is the provisional operator contract for comparing the installed ROCm
7.2.4 control with ROCm Core SDK 10.0.0. The lane is **planned, inactive, and
not runnable**. This document does not authorize downloading, extracting,
installing, activating, or executing ROCm 10, or running a GPU workload. The
checked-in ROCm 10 profile validates the expected shape and prefix only; it is
not proof that the prefix exists, that the artifact is authentic, or that a
binary loaded the intended libraries.

## Assessment

Observed on the bring-up host on 2026-09-01:

| Boundary | What already works | ROCm 10 status |
|---|---|---|
| Control | `/opt/rocm/.info/version` reports `7.2.4`; the existing `hip-rocm-gfx1151` profile selects it by absolute paths. | Preserved unchanged. |
| Source | `configs/sources/strix-llama.yaml` pins commit `ca94157f70a2776e8da6b6849b50b45a083d0478`. | The same preparation can feed both builds. |
| Build isolation | Profiles select absolute compilers and prefixes; profile/tool hashes and CMake selections enter the build identity; separate recipes get separate build roots. | `hip-rocm10-gfx1151` selects only `/opt/rocm-10`. |
| Runtime evidence | Build capture hashes artifacts and the CMake cache, records compile commands, `readelf -d`, and `ldd`, and attests the finalized evidence. | The same boundary applies. External ROCm dependencies are recorded by resolved path, not content hash. A separately reviewed, versioned prefix inventory verifier is still required to compare their bytes. Environment variables alone never establish isolation. |
| ROCm 10 artifact | AMD publishes a release-specific gfx1151 tarball for ROCm 10.0.0 and documents tarballs as self-contained custom-location installs. | Hard-blocked. No vendor-authenticated digest or signature for the tarball was found. A local digest would be only an observed identity. |
| Host support | The local control is usable as an existing field environment. | The host reports Omarchy/Arch with kernel `7.1.8-arch1-3`; AMD's ROCm 10 Ryzen support table names Ubuntu, not Arch. Any later result on this host is an unsupported-host field observation, not vendor-supported validation. |

The unresolved artifact-authenticity decision and missing versioned prefix
verifier are activation blockers. The missing `/opt/rocm-10` prefix and unsupported
host OS are additional blockers, not values to infer. Do not create the prefix or
execute the profile. No optimization experiment starts until the activation gate
below is separately cleared and both unmodified builds and their
dynamic-dependency evidence pass review.

## Provisional lane and retrieval observation

- Release: ROCm Core SDK `10.0.0`, released 2026-08-26.
- Official artifact URL:
  `https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-10.0.0.tar.gz`
- HTTP headers observed at `2026-09-01T19:31:17Z`:
  - `Content-Length: 1794376103`
  - `ETag: "f4f3cac4b06dfa7db2965fa7e8813603-14"`
  - `Last-Modified: Wed, 26 Aug 2026 11:46:45 GMT`
- Artifact retrieval time: not observed; the artifact was not downloaded.
- `observed_sha256`: not observed; the artifact was not downloaded.
- Vendor authenticity: not established.
- Selected explicit experiment prefix: `/opt/rocm-10`.
- Build profile: `configs/builds/hip-rocm10-gfx1151.yaml`.
- Control profile: `configs/builds/hip-rocm-gfx1151.yaml`.

Primary AMD references:

- [ROCm 10.0.0 installation](https://rocm.docs.amd.com/en/develop/install/rocm.html)
- [ROCm 10.0.0 release notes](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
- [Transition from the legacy ROCm release stream](https://rocm.docs.amd.com/en/develop/about/transition-guide-TheRock.html)

The release-specific URL identifies the selected artifact name, release, and GPU
family, but does not authenticate its bytes. The header observation above is
reproducible transfer metadata only. Do not treat the HTTP ETag, size, TLS
transport, or a locally computed digest as vendor authentication.

### Authenticity search result

As of `2026-09-01T19:31:17Z`, searches of AMD's ROCm 10 installation and release
documentation and the official `ROCm/TheRock` repository found no checksum,
detached signature, signed manifest, SBOM digest, or release asset binding this
tarball's bytes. Obvious adjacent sidecars (`.sha256`, `.sha256sum`, `.sig`,
`.asc`, `.minisig`, manifest variants, and SPDX/CycloneDX/SBOM variants) returned
HTTP 404. The `therock-10.0` GitHub release has no attached assets.

AMD's package-manager installation path does publish a GPG-keyed repository with
signed metadata, but that metadata authenticates repository packages, not this
tarball. Package installation is also outside this branch's no-install boundary,
would configure system-wide paths/alternatives, and is not a supported route on
this Omarchy/Arch host. It therefore does not clear the tarball blocker.

## Activation gate: hard blocker

The ROCm 10 profile must remain inactive until **both** gates are separately
reviewed and approved:

1. The exact tarball bytes have either a vendor-authenticated anchor, such as an
   AMD detached signature, signed manifest/repository entry, or vendor-published
   cryptographic digest, **or** an explicit human risk-acceptance decision for
   unauthenticated bytes plus a separately reviewed installation and isolation
   plan.
2. A separately reviewed, versioned inventory/verification implementation exists
   and proves exact staged-versus-installed equality under the evidence contract
   below. It must also establish an approved baseline for each installed prefix
   and fail if either prefix later differs before a build or run.

A future download record must use fields with these semantics:

```yaml
official_url: https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-gfx1151-10.0.0.tar.gz
version: 10.0.0
observed_content_length: 1794376103
observed_etag: '"f4f3cac4b06dfa7db2965fa7e8813603-14"'
artifact_retrieved_at: <UTC timestamp>
observed_sha256: <locally computed SHA-256>
vendor_authenticity: unverified
```

Never name the local value `expected_sha256`, `vendor_sha256`, or
`vendor_verified`. After human risk acceptance, pinning `observed_sha256` and
failing on any later mismatch detects subsequent drift from those observed
bytes; it does not establish that the originally observed bytes came from AMD.

## Unapproved installation plan

The following is a review contract, not a command sequence and not authorization
for an agent or operator to mutate the host. No mutation may begin until the
artifact-acceptance basis, parser/extractor/verifier implementation, and isolation
plan are separately reviewed and approved. Lane activation remains blocked until
the later post-copy equality checks actually pass. The installed `/opt/rocm`
control must remain untouched.

### Archive admission contract

The host-only `strixlab.rocm_archive` reader and its structural member manifest
implement the narrow grammar below. The synthetic extractor and unqualified prefix
inventory/comparison are implemented in the slices below. They do not provide the
qualified post-copy verifier required for SDK activation. Structural admission does
not authenticate an SDK, authorize real-SDK extraction, or clear either activation
gate. The complete preparation path must enforce all of these rules:

1. Bind review and extraction to the same exact tarball bytes: the bytes covered
   by the vendor anchor or explicit risk acceptance and by the retained
   `observed_sha256`. A single parser and policy implementation must emit a
   versioned accepted-member manifest; extraction must consume that manifest and
   re-read and revalidate every header with that same parser. A generic listing
   followed by a differently interpreting extractor is forbidden.
2. Accept exactly one gzip member with no compressed trailing data. The expanded
   tar length must be a multiple of 512 bytes. Its first two consecutive zero
   blocks terminate the sole archive; all remaining expanded bytes through EOF
   must be zero padding. Reject concatenated archives, later headers, and any
   nonzero trailing byte.
3. Limit the archive to 1,000,000 members, 8 GiB per regular file, 32 GiB total
   expanded regular-file bytes, and a 64:1 expanded-to-compressed ratio. Reject
   sparse entries or sparse markers. A change to any limit requires another
   review.
4. Decode member names and symlink targets as strict UTF-8, require Unicode NFC,
   and limit each encoded value to 4096 bytes. Reject empty names or targets,
   backslashes, and any code point whose Unicode general category begins with
   `C`, including control, formatting/bidirectional, surrogate, private-use, and
   unassigned characters. The review manifest must show both the Unicode value
   and an unambiguous escaped rendering of its UTF-8 bytes.
5. Treat names as relative POSIX paths. A directory header may lose exactly one
   trailing slash; after that, every component must be nonempty and neither `.`
   nor `..`, and joining the components must reproduce the normalized name.
   Reject absolute names, duplicate normalized names, normalization aliases, and
   any entry beneath a regular-file or symlink entry. Every non-root parent must
   have its own accepted directory header; the extractor may not synthesize an
   unlisted entry.
6. Resolve each symlink target lexically relative to the symlink's parent.
   Require the target itself to equal its POSIX lexical normalization, remain at
   or below the archive root after resolution, and satisfy the same encoding and
   display rules. The target need not precede the link in archive order. Reject
   every hard-link member, including an apparently in-root one, so no hard-link
   topology is left unrepresented.
7. Allow only directory, regular-file, and symlink members. Explicitly reject
   character or block devices, FIFOs, sockets, hard links, and every other member
   type or extension. Reject unreviewed PAX/global headers and GNU or vendor
   extensions; accepted PAX keys, if any, must be individually allowlisted by the
   later review.
8. Require regular-file and directory modes to contain only `0777` permission
   bits; reject sticky, setuid, setgid, or any other mode bit. Require archived
   symlink mode to be exactly `0777`, record it as such, and never attempt to
   chmod a symlink. Reject archive metadata for file capabilities, xattrs, or
   ACLs. Treat archived UID, GID, user name, and group name as untrusted metadata
   that must never be restored.

### ROCM10-VERIFY-A structural wire grammar, v1

This is a deliberately narrow POSIX ustar grammar, frozen against synthetic
byte-built fixtures rather than inferred from `tarfile` acceptance. The actual
SDK archive has not been retrieved or tested against it. An unsupported archive
is a negative result requiring review, never permission to fall back to another
parser. There is no extraction, network, provenance-acceptance or CLI operation.

`inspect_archive(directory_fd, name)` reads one regular file through an already
held directory descriptor; `name` must be a single nonempty component other than
`.` or `..`. The leaf is opened no-follow/nonblocking and checked against its
pre-open identity. Size, identity, mode, ownership, link count and nanosecond
mtime/ctime are compared before/after reading, including the final directory
entry. Observed change, unreadable input or premature EOF fails. This detects
observed drift; it is not a filesystem snapshot or vendor authentication.

- **Gzip:** exactly one DEFLATE gzip member. Validate gzip framing, reserved
  flags, optional-header framing/checksum when present, CRC32 and ISIZE. Optional
  extra/name/comment data are inert and not retained. Reject a second member or
  any compressed suffix, including zeros. Use at most 64 KiB compressed reads
  and 64 KiB decompressor output per call. Count **all** expanded bytes, including
  headers, member padding and terminal zeros, against `64 * compressed_size`.
  Compressed size is the opened file's snapshotted size, not a running denominator.
- **Header:** exactly 512 bytes; magic bytes 257–262 are `ustar` followed by NUL,
  version bytes 263–264 are ASCII `00`, and bytes 500–511 are zero. No GNU magic,
  base-256 numbers, PAX/global headers, sparse markers or vendor extensions.
- **Octal fields:** mode, UID, GID, size and mtime contain optional leading ASCII
  spaces, then one or more ASCII octal digits (leading zeros allowed), then a
  nonempty suffix containing only NUL and/or ASCII spaces. Reject empty/all-NUL
  fields, signs, other whitespace, non-octal digits and unterminated full-width
  numbers. Field width still bounds representable values: an 11-digit ustar size
  cannot represent the policy's inclusive 8 GiB upper bound. Device-major/minor
  fields must be all zero bytes or an accepted octal spelling of zero.
- **Checksum:** bytes 148–155 are exactly six ASCII octal digits, NUL, space.
  Compare with the unsigned-byte sum of the complete header while treating those
  eight checksum bytes as ASCII spaces. Other checksum padding/termination and
  signed checksums are rejected.
- **Text fields:** name, linkname, prefix, uname and gname either fill their field
  without NUL or terminate at the first NUL with only zero bytes afterward.
  Decode strict UTF-8/NFC; reject Unicode category `C*` and backslashes. Name is
  nonempty; join a nonempty prefix to it with one `/`, then apply the path policy
  above. UID/GID/mtime/uname/gname are validated but never applied as metadata.
- **Types:** only ASCII `0` or NUL (regular file), ASCII `5` (directory), and ASCII
  `2` (symlink). Non-symlinks have an empty linkname. Directories and symlinks have
  zero payload size. Regular-file payload bytes are arbitrary; their padding to
  the next 512-byte header must be zero. The first zero header requires a second
  zero header immediately; all subsequent expanded bytes are zero and the total
  expanded length is a multiple of 512. Even an empty archive needs both headers.
- **Topology:** explicit directory parents may occur before or after children.
  Validate the complete topology before returning a manifest. Reject duplicate
  normalized paths, missing parents and entries beneath regular files/symlinks.
  Symlink targets are validated **lexically only**: a canonical relative target
  must remain inside the archive root when joined to its parent. A target of `.`
  may name its parent, and leading `..` is allowed only when the joined path stays
  inside. Dangling targets and cycles can pass this lexical check; it does not
  establish resolved-prefix safety and must never authorize link traversal.

Fixed archive limits remain 1,000,000 members, 8 GiB per regular file, 32 GiB
aggregate regular payload and 64:1 expansion. Additionally, this implementation
admits at most **64 MiB of canonical per-entry JSON metadata**, charged before
retaining each entry; exceeding it is a resource failure, not a different parser
or relaxed policy. This bounds retained entry count/text as well as the archive
member limit; it is not a promise that Python process RSS is 64 MiB. Payloads,
terminal padding and gzip optional strings are streamed, never retained in the
manifest. No caller-supplied resource-limit overrides are accepted.

The manifest records version/parser ID, `validation: complete`, `admission: structural-only`,
`symlink_validation: lexical-only`, observed compressed SHA-256/size, expanded
size, regular-payload bytes, member count and entries sorted by UTF-8 path bytes.
Each entry records archive ordinal, byte offset of its expanded header, type,
mode, payload size, normalized path and lowercase `\xhh` escapes for every UTF-8
byte, plus a file digest or link target/escaped target as applicable. It invents
no filesystem link counts, ownership audit, tree digest or approval field.
Existing canonical JSON serialization is reused. Structural admission and
observed hashes cannot be promoted to accepted AMD provenance by a caller field.

One internal event iterator owns header/payload/padding/topology interpretation.
Every start, data chunk and end event explicitly carries `validation: provisional`;
the iterator never emits an admitted manifest, even on exhaustion. Only
`inspect_archive` returns `validation: complete` after gzip completion, tar
termination, final topology and descriptor stability checks. This is completed
structural inspection, not provenance acceptance. A future extractor must reuse
that interpretation, bind the exact reviewed archive and manifest bytes, and
revalidate them; the current slice neither provides nor authorizes an extractor.
Synthetic fixtures establish grammar behavior only. Prefix inventory, privileged
metadata auditing, real artifact authenticity, installation location approval
and runtime isolation remain separate unresolved work.

### Explicit GNU direct-hardlink profile

`strixlab.rocm_archive.inspect_gnu_archive(parent_fd, name)` explicitly selects
`rocm-archive-gnu-direct-hardlink-v1`. The default `inspect_archive` and its V1
reports/events retain the POSIX grammar above. Neither entry point auto-detects
or falls back to another dialect. The GNU subset is tested with synthetic bytes;
diagnostic vendor-archive censuses do not establish admission.

- Every header uses GNU magic `7573746172202000` at bytes 257–264 and has a
  completely zero tail at bytes 345–511. Checksum, octal, text, zero device
  fields, gzip completion, expansion and input-stability rules remain strict.
  PAX, `K`, sparse data, base-256 numbers, mixed magic and other types are rejected.
- An initial `./` directory header, mode `0755`, zero payload and empty linkname
  is mandatory. It is recorded separately and excluded from the material tree.
  It never supplies the filesystem quarantine root's mode.
- Material types are exactly `0` (regular `0644` or `0755`), `5` (directory
  `0755`), `2` (symlink `0777`), and `1` (direct hardlink `0644` or `0755`).
  Remove exactly one required leading `./` from material paths, then apply
  the existing component and explicit-parent rules. Symlink literals retain
  the existing relative, canonical, lexically-contained policy.
- `L` controls require name `././@LongLink`, mode `0644`, empty linkname and
  102–4097 payload bytes: a 101–4096-byte full name plus one final NUL, with
  no embedded NUL and zero padding. The next physical header must be regular
  `0`, whose raw 100-byte name equals the full name's first 100 bytes. Compare
  before decoding a potentially split UTF-8 character; validate the full name.
  Consecutive/dangling controls and other overrides are rejected.
- Hardlink targets require one leading `./` and a canonical archive-root-relative
  path naming an already completed regular wire member with matching mode.
  Forward references, chains, cycles, missing targets and nonregular sources
  are rejected. Wire payload size stays zero and wire SHA-256 stays null.

The separate frozen `GnuArchiveManifestV1` records the root marker, physical
header ordinals/offsets, long-name control positions, effective wire names,
wire linknames, normalized paths and original member kinds. Its separate
`hardlink_copies` projection records each destination, direct regular source,
mode and materialized size/SHA-256. `regular_payload_bytes` counts only regular
wire payloads; `independent_copy_bytes` counts all projected copies; their sum
is `materialized_regular_bytes`. No filesystem hardlink is created by inspection.

Limits remain 64 KiB chunks, 8 GiB per file, 32 GiB regular wire payload,
1,000,000 material members and 64:1 gzip expansion. Total materialization,
including every independent copy, is separately capped at 32 GiB. Raw headers
are capped at twice the member limit plus one. Root/control evidence, entries
and copy records are charged against the 64 MiB metadata budget before retention;
only one bounded long-name control may be pending. GNU start/end callbacks use
separate provisional types; controls emit no material-member callbacks. The
shared descriptor/gzip lifecycle returns a complete structural report only after
termination, CRC, topology and final input-identity checks. This parser adds no
extraction, provenance acceptance, metadata qualification or installation action.

### Metadata observation slice (unqualified)

`strixlab.rocm_metadata.observe_inode_metadata(parent_fd, name)` observes stored
xattr **names only** for one regular file, directory or symlink beneath a
caller-held directory descriptor. The leaf must encode as UTF-8 in at most 255
bytes, with no slash, NUL, or dot/dot-dot leaf. Its completed
`InodeMetadataObservationV1`
always reports `coverage: unknown`. It has no qualification, metadata-admission,
installation or activation API, and no existing lifecycle path calls it.
A completed probe, including an empty name list, does not establish absence in
all namespaces or satisfy the privileged metadata-audit requirements below.

The Linux native x86_64 adapter uses `listxattrat` syscall **465**, whose ABI is
`(int dirfd, const char *path, unsigned int flags, char *list, size_t size)`, with
`AT_SYMLINK_NOFOLLOW` (`0x100`). The ABI is recorded in upstream Linux
[v7.1.8 syscall table](https://github.com/gregkh/linux/blob/v7.1.8/arch/x86/entry/syscalls/syscall_64.tbl)
and [implementation](https://github.com/gregkh/linux/blob/v7.1.8/fs/xattr.c).
Native 64-bit pointer/long/size_t, platform and required-open-flag checks precede
calls. Other ABIs fail closed; a missing kernel syscall produces a distinct
`unsupported` observation with raw `ENOSYS`. No libc `listxattrat` symbol is
assumed. Symlink probes address a single leaf relative to the held parent;
`O_PATH` is used only to retain stat identity, never as an xattr I/O descriptor.

The probe duplicates the parent FD and brackets each observation with parent
and leaf stat identities, including the final no-follow leaf-name binding.
Observed changes raise a bounded error without returning a completed report.
The caller's parent descriptor is the authority: this function does not verify
its original pathname, ancestor containment, mount isolation or ownership
admission. A held descriptor does not freeze directory entries, and brackets
cannot exclude a hostile same-UID writer or changes after observation.

Name storage is capped at 64 KiB before buffer allocation, with at most one
retry for `ERANGE`. Reports distinguish observed names, malformed results,
resource limits, unsupported operations and errors; raw errno is retained.
`ENODATA`, access denial, unsupported operations and size failures never become
an empty observation. Names are UTF-8-independent bytes, sorted by bytes and
rendered as lowercase hex escapes; no xattr values are read. Synthetic tests
cover these semantics. Explicitly labeled live support tests may skip on older
kernels; neither their success nor a skip is host/backend qualification.

A future qualification requires separately reviewed seeded evidence and exact
backend, environment, namespace, credential and policy assumptions. Neither
`uname` nor empty user-visible lists suffice. This slice supplies no production
path to create or attach such a qualification and changes no activation gate.

### Prefix inventory slice (unqualified)

`strixlab.rocm_prefix.inspect_prefix(parent_fd, name)` inventories a quiescent,
caller-owned prefix using a parent directory descriptor and one UTF-8 leaf name.
The completed `PrefixInventoryV1` records the root at path `""`, sorted descendants,
mode/owner, regular-file lengths and SHA-256, literal symlink targets, inode identity
brackets, and the existing metadata observer result. Unsupported or failed xattr
probes retain their status and errno. `metadata_coverage` is always `unknown`;
`link_closure` remains `not-checked`. Completion is an observation, not admission,
qualification, permission to copy, or activation of ROCm 10.

One private Linux x86_64 `openat2` helper guards root, descent, payload, and final
binding opens with `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_XDEV` and
`O_NOFOLLOW | O_CLOEXEC`. Mount crossings, including same-device bind mounts, fail
closed; there is no `st_dev` fallback. The caller's parent descriptor is the
starting authority, without proof of its ancestors. Inode types are classified
with `O_PATH`; non-directories must have one link. Special mode bits are recorded,
not policy-approved. A regular-file read uses `O_NONBLOCK` and checks the opened
identity before reading; a raced device could have been opened before that check.
The metadata observer still uses named probes bracketed by identity and guarded
binding checks. These are not atomic guarantees against transient mounts, ABA
replacement, hostile same-UID writers, or mutation after the scan.

Collection bounds are one million descendants, 8 GiB per file, 32 GiB total file
payload, 256 path components, 256 owned descriptors, 4096 UTF-8 bytes per path or
link target, and 256 MiB of charged serialized evidence and enumeration buffers.
Enumeration is charged before retaining/sorting names; entry evidence is validated
and charged before hashing. File hashing uses 64 KiB reads and requires exact EOF.
An iterative traversal checks directory membership before and after descendants;
a final rewalk checks all bindings, identities, and metadata again. Any detected
drift, malformed record, unsupported guarded-open ABI, or resource failure yields
no completed inventory. These are implementation budgets, not a process RSS limit.

The inventory charge is a 4096-byte envelope plus both walks' complete per-entry
canonical JSON and indentation allowance, plus four charges for every descendant's
serialized basename (before/after enumeration in each walk). A diagnostic-only
measurement of the retained SDK tree estimated 177,169,168 bytes (168.9617 MiB):
`4096 + 2 * 87,438,136 + 4 * 572,200`. This exceeds 128 MiB and leaves 91,266,288
bytes below the fixed 256 MiB cap. The diagnostic used fixed-width zero digest
placeholders; it is not a payload hash, completed inventory or admission, and does
not permit reusing the failed quarantine. The increase preserves all evidence and
accounting. Pure link-map output retains its separate fixed 64 MiB cap.

`compare_prefixes(expected, observed)` validates both completed maps and compares
only recorded semantic fields, ignoring inode numbers, timestamps, directory
size/link count, and the root's external leaf name. Equal unknown metadata remains
unknown. It counts every differing path and returns at most 128 sample paths.
`resolve_inventory_links(inventory)` separately resolves the completed map without
filesystem access. It expands symlinks before later `..`, requires directories for
intermediate/trailing components, and reports absolute, escaping, dangling,
non-directory, cyclic, or expansion-limited targets. Root-directory resolution is
valid. Limits are 40 expansions per link, one million cumulative component-work
units, 64 KiB queued target components, and 64 MiB charged output evidence. The
separate link result cannot upgrade the inventory's metadata coverage or admission
state. This slice has no CLI, SDK execution, extraction, copy, or activation path.

### Synthetic structural quarantine slice (unqualified)

`strixlab.rocm_extract.extract_quarantine(archive_parent_fd, archive_name,
destination_grandparent_fd, destination_parent_name, quarantine_name)` duplicates
the caller's descriptors and completes the strict archive inspector before any
writes. It accepts no caller-supplied manifest. Preflight requires UTF-8 leaf names
of at most 255 bytes, paths of at most 4096 bytes and 240 components, owner-readable
final files, and owner-readable/searchable final directories. The selected
destination parent must be the operator's exact UID/GID and mode `0700`, opened
through the shared guarded `openat2` helper, with observed-empty metadata names.

The extractor exclusively creates a new mode-`0700` root, immediately captures its
identity through a guarded `O_PATH` descriptor, and verifies its readable descriptor
against that identity. It precreates explicit archive directories in parent order,
then uses the strict parser's provisional event consumer to write regular files in
bounded chunks. Headers and entries must match the first inspection, and the second
completed manifest must match canonically, including the compressed archive digest,
before symlink creation or final modes. Files are initially `0600`; directories are
initially `0700`. Loss of requested owner permissions at creation fails explicitly,
even when privileged access would succeed. Normalization and final modes use held
file descriptors, with directory final modes applied in postorder. The extractor
does not change the process umask or chmod by pathname.

All root/descent/reopen/binding opens reuse the prefix helper's fixed no-follow,
no-mount-crossing policy. Creation uses held parent descriptors and exclusive
no-follow file creation; symlinks are created last and never traversed. Created
inode numbers and devices are checked on every directory reopen and at final
projection. Directory access reopens one bounded chain at a time. Every new/final
inode must have observed-empty metadata names with matching held/named identities;
errors, unsupported probes, malformed responses, and known names prevent completion.
No metadata is stripped. Final prefix inspection must exactly match the archive's
member set, kinds, modes, operator ownership, file lengths/hashes, literal symlink
targets and their UTF-8 lengths, and single links for non-directories. The separate
root stays `0700`. The destination parent's binding, owner, and mode are rechecked
after writes while allowing the expected directory timestamp changes.

`QuarantineResultV1` contains the archive and inventory only after all checks, with
scope `structural-quarantine-only`, metadata coverage `unknown`, and link closure
`not-checked`. Observed-empty names never prove full metadata absence or admission.
`QuarantineError` reports a bounded reason, the retained leaf name only after mkdir
succeeds, and its captured device/inode when available. Every post-mkdir failure,
including cancellation, leaves an incomplete leaf; existing leaves are never
adopted, replaced, or cleaned up. Caller descriptors remain caller-owned.

This implementation is exercised only with synthetic archives. It does not authorize
SDK extraction, installation, privileged copying, GPU execution, or activation, and
makes no fsync/durability promise. Quiescent operator ownership remains a requirement;
named probes and binding checks do not provide atomic protection against hostile
same-UID writers, transient mount/ABA replacement, or post-observation mutation.

### GNU quarantine adapter (unqualified)

`extract_gnu_quarantine` takes the same five descriptor/name arguments as the
strict extractor and explicitly selects the reviewed GNU profile. The original
`extract_quarantine` default and `QuarantineResultV1` remain strict POSIX. Both
use the same guarded tree creation, provisional writer, finalization, projection
and retained-failure lifecycle. A separate internal deployment entry describes
filesystem expectations; GNU wire evidence is never converted into fabricated
POSIX V1 entries.

The first complete GNU inspection validates hardlink sources and total copy
budgets before any mkdir. Pass two writes only actual regular wire payloads and
compares original GNU headers/end entries. Exact second-manifest equality,
including compressed digest and copy projection, is required before any copy,
symlink or final mode. Root and long-name controls create no filesystem entries.

Each hardlink projection becomes a new exclusive `0600` regular file. The source
must match its captured created device/inode, operator ownership, regular type,
link count one, temporary `0600` mode and projected size. Guarded O_PATH and
read-only/nonblocking opens must agree. Stored metadata names must be observed
empty before and after copying, with stable held/named identities. Reads and
checked writes use at most 64 KiB at a time; exact size, EOF and SHA-256 must match
the copy projection. The destination has its own captured inode and must retain
link count one, observed-empty metadata and the exact written size. No `os.link`
operation, source link traversal, adoption or cleanup occurs.

Copies finish before symlinks and final modes. Final inventory explicitly maps
wire hardlinks to independent regular files with their projected size/hash and
mode; all existing ownership, metadata, binding and exact-member checks apply.
The root stays `0700`. The separate `GnuQuarantineResultV1` retains the original
GNU manifest and prefix inventory, fixes `materialization_policy` to
`independent-hardlink-copies-v1`, and preserves structural-only scope, unknown
metadata coverage and unchecked link closure. This adapter is validated with
synthetic archives; it adds no installation, metadata admission or SDK execution.

### Extraction and copy contract

The future, separately reviewed implementation must be descriptor anchored and
no-follow on both its source and destination sides:

1. Extraction may target only a newly created, operator-owned mode-`0700`
   staging root whose destination name was absent. It must create every listed
   component relative to a verified directory descriptor, refuse every
   preexisting entry, and use no-follow/exclusive semantics. A symlink may be
   created only after the accepted-member policy validates it and is never
   traversed during extraction.
2. Regular files and directories may initially use temporary safe owner-only
   writable modes. After a regular file is completely written and hashed, its
   final normalized mode must be applied through its verified descriptor. Final
   directory modes must be applied post-order after all descendants are complete,
   independently of the process umask. Re-inventory only after final modes apply.
3. Every staged descendant must be owned by the operator. A sufficiently
   privileged metadata audit must prove that capabilities, xattrs, and ACLs are
   absent; an unreadable metadata namespace is a failure, not evidence of
   absence. Every non-directory entry must have link count exactly one, so an
   external or newly introduced hard link fails closed. Staged entries must
   exactly match the accepted-member manifest, with no implicit or unexpected
   filesystem entry.
4. A later privileged copy remains unapproved. It may target only an absent
   `/opt/rocm-10`, create that root without replacement as UID/GID `0/0` mode
   `0755`, and create all descendants as UID/GID `0/0`. It must not restore
   archived ownership or metadata. It must not create or change `/opt/rocm`,
   alternatives, profile scripts, shell startup files, package repositories,
   packages, drivers, firmware, groups, the kernel, or global `PATH` or
   `LD_LIBRARY_PATH` settings.
5. The copy must also open the staging source through directory descriptors with
   no-follow semantics. It must compare each opened source entry with the
   approved staged evidence, validate type/inode/metadata before and after reads,
   refuse source drift, and exclusively create every destination entry without
   following or replacing a name.
6. After copying, completely re-inventory both roots. Prove that staging still
   equals its approved installation-source baseline and that `/opt/rocm-10`
   equals that same baseline within the defined equality scope. Both complete
   trees must still contain exactly the expected entries, and every non-directory
   entry must still have link count one.
7. Before approving the installed baseline, run the same sufficiently privileged,
   fail-closed capabilities/xattr/ACL audit on `/opt/rocm-10`; parent defaults or
   security labeling applied during creation are mismatches. Repeat that audit,
   the ownership/mode checks, and the link-count check during every later
   verification of the installed prefix.

No root extraction or copy commands are provided here. The later reviewed
implementation, isolation plan, and explicit human approval must define and
authorize those mutations.

### Required per-entry evidence and missing verifier

Retain and review a per-entry inventory for every filesystem entry strictly below
each root. Each record must contain the normalized relative path and its escaped
UTF-8 bytes, file type, normalized mode, symlink target when applicable, byte
length, link count for non-directories, and SHA-256 for regular files. Every
non-directory link count must equal one. Retain separate results for root
mode/ownership, descendant ownership, and the capabilities/xattr/ACL audit. Any
unexpected entry is a mismatch.

This branch does **not** define a canonical aggregate tree digest and does not
claim that ad hoc listings are reproducible. Per-file SHA-256 values are observed
local drift evidence, not vendor authentication. Activation remains blocked until
a separately reviewed, versioned implementation can produce and verify this
inventory while detecting filesystem drift, then prove:

1. the staged tree equals the accepted-member manifest and becomes the approved
   installation-source baseline;
2. `/opt/rocm-10` equals that staged baseline before becoming the approved
   installed ROCm 10 baseline;
3. `/opt/rocm` equals a separately captured and approved ROCm 7.2.4 control
   baseline; and
4. immediately before every later build or run, each selected prefix still equals
   its approved installed baseline, every non-directory link count remains one,
   and the required privileged metadata audit still passes.

If AMD publishes an authenticity anchor, bind it in this runbook and verify it
before extraction. If a human instead accepts the risk, record that decision and
the separately reviewed isolation plan. If the eventual installation layout
does not contain the absolute tools named by the profile, stop and amend the
reviewed profile; do not add ambient-path fallbacks.

## Planned preflight after installation

Do not run this section while the activation gate is blocked. These observations
do not prove runtime isolation, but would catch a wrong or incomplete prefix
before a build:

```bash
test "$(sed -n '1p' /opt/rocm/.info/version)" = "7.2.4"
test -x /opt/rocm/bin/amdclang++
test -x /opt/rocm-10/bin/amdclang++

/opt/rocm/bin/amdclang++ --version
/opt/rocm/bin/hipcc --version
/opt/rocm-10/bin/amdclang++ --version
/opt/rocm-10/bin/hipcc --version

git diff --exit-code -- configs/builds/hip-rocm-gfx1151.yaml
```

Review the output and the installation approval record. A profile ID containing
`rocm10` is not version evidence; the build's captured compiler output, tool
hashes, CMake-cache digest, artifacts, and resolved dynamic dependencies are
necessary evidence. They do not replace the blocked per-entry prefix verification
required above.

## Planned no-op paired validation

Do not run this section while the activation gate is blocked. After clearance,
use a new, dedicated `STRIXLAB_HOME`. Do not add a candidate patch. One source
preparation therefore pins both arms to the same clean source commit. These
commands do not run a model or benchmark. However, `build prepare` does execute
each newly built target with `--help` and `--version` while capturing artifact
capabilities. Those process launches may load the target's linked ROCm libraries;
they are part of this future, separately approved execution plan and must not be
described or approved as compile-only.

```bash
export STRIXLAB_HOME=/absolute/path/to/fresh/rocm10-bringup-state

uv run strixlab source prepare configs/sources/strix-llama.yaml
PREPARATION_ID='prep-strix-llama-...'

uv run strixlab source inspect "$PREPARATION_ID"

# BLOCKED: the separately reviewed versioned verifier must first prove that
# /opt/rocm equals its approved control baseline. No such verifier exists here.
uv run strixlab build prepare "$PREPARATION_ID" \
  configs/builds/hip-rocm-gfx1151.yaml
ROCM72_BUILD_ID='build-sha256:...'

# BLOCKED: the same verifier must first prove that /opt/rocm-10 equals its
# approved installed baseline. No such verifier exists here.
uv run strixlab build prepare "$PREPARATION_ID" \
  configs/builds/hip-rocm10-gfx1151.yaml
ROCM10_BUILD_ID='build-sha256:...'

mkdir -p "$STRIXLAB_HOME/review"
uv run strixlab build inspect "$ROCM72_BUILD_ID" \
  > "$STRIXLAB_HOME/review/rocm72-build-inspect.json"
uv run strixlab build inspect "$ROCM10_BUILD_ID" \
  > "$STRIXLAB_HOME/review/rocm10-build-inspect.json"
```

Before any optimization work, review both inspection documents and require:

1. `attested` is true and both canonical records name source `strix-llama`, base
   commit `ca94157f70a2776e8da6b6849b50b45a083d0478`, toolchain mode `rocm`, and
   gfx target `gfx1151`.
2. The build IDs and roots differ. The requested targets are the same three
   binaries and the non-prefix CMake choices match.
3. The control's compiler/tool paths and ROCm selections resolve under
   `/opt/rocm`; ROCm 10's resolve under `/opt/rocm-10`.
4. Canonical selections identify `/opt/rocm-10` and its compiler for ROCm 10 and
   `/opt/rocm` for the control; a captured CMake-cache digest is present for each.
5. Every dynamic target's `readelf`/`ldd` evidence is present. ROCm-linked
   dependencies for the ROCm 10 arm resolve under `/opt/rocm-10`, and the control
   arm's resolve under `/opt/rocm`. Any cross-prefix ROCm library is a blocker.
   Ordinary system libraries may resolve under the recorded system prefix.
6. Compiled target evidence contains `gfx1151`; no native or unintended gfx
   target silently replaced it.
7. The separately reviewed versioned verifier proved each prefix exactly equal to
   its approved installed baseline immediately before its `build prepare`.
   Because StrixLab records external ROCm libraries by resolved path rather than
   hashing their bytes, path isolation and the build attestation do not replace
   this check. This condition cannot pass with the current branch alone.

Only after the build pair passes may a human authorize the same registered smoke
model and suite under each attested build. Immediately before every future suite
or other target run, the versioned verifier must compare the selected prefix with
its approved installed baseline and fail on a mismatch. Those runs are
correctness/runtime validation and may exercise the GPU;
they are intentionally outside this no-GPU bring-up change. A later no-op
comparison should be reported as `inconclusive`, not as a win.
