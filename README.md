# Chestnut USB / RTX 3090 PR evidence

Supporting records for [zhentan/tinygrad PR #1](https://github.com/zhentan/tinygrad/pull/1), which remains a draft. This branch contains generated evidence only; it is separate from the code diff.

[Download the complete evidence archive](usb-pr-evidence.tar.gz?raw=true) and verify it against [SHA256SUMS](SHA256SUMS). The archive also contains checksums for all 377 payload entries. [Performance context and limitations](PERFORMANCE-CONTEXT.md) explains the measured rates and the firmware read path.

- Final branch head: `d3df666b84c1af840eaab042385212dae2626ac6`, rebased onto `b45cee5ccd127266a70a8fed14ebe43196f7d95d`.
- Full hardware gate at runtime/test commit `8eb65a7d2`: 520 passed, 21 skipped, 165 passing subtests; all nine USB checks, 64 MiB integrity, buffer retention and clean process/device shutdown.
- CPU gate: 636 passed, 44 skipped, 239 passing subtests. Mypy and Ruff pass.
- Final head adds only the ADR and passed one independently audited fresh-process run.
- The separate 100/100 fresh-process qualification belongs to pre-rebase `d72c079`. Every attempt and its audit are retained.
- Final warm transfer medians: 98.20 MB/s upload, 1.56 MB/s download. These are the baseline for further performance work, not a high-performance claim.

The archive contains the exact benchmark/runner sources, command settings, source/harness hashes, identity guards, raw logs and JUnit, all samples, first failures and fixes, range-diff mapping, ADR, and fork-push verification. The original workspace paths remain in raw records for provenance. Its README describes the sibling checkout/evidence layout required to rerun the harness. Qualification selected only the external Chestnut RTX 3090.
