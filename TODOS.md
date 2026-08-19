# TODOs

## Benchmark-Based Quantization Selection

- What: Benchmark fitting quantization candidates and recommend the fastest result.
- Why: The product may eventually need measured speed, not only estimated memory fit, to define “best quantization.”
- Pros: Gives users a performance-based recommendation and provides evidence for future policy changes.
- Cons: Requires extra conversions, a stable metric, benchmark fixtures, and more CI time.
- Context: Deferred from the approved decision-engine design. First prove pinned conversion, conversion envelopes, support fixtures, verification, and reports. Start by choosing whether prompt processing, generated tokens per second, peak memory, or a weighted score is the primary metric.
- Depends on / blocked by: Verified conversion pipeline, stable report schema, and supported Llama/Qwen fixtures.

## Reproducible Model Build Artifacts

- What: Add commands and report metadata for inspecting, comparing, checksumming, and reproducing prior conversions.
- Why: Contributors will need to compare model outputs and replay support or benchmark failures across machines.
- Pros: Makes generated models auditable and lowers the cost of diagnosing architecture and converter regressions.
- Cons: Requires artifact compatibility rules, checksum semantics, and storage decisions beyond the beginner workflow.
- Context: Deferred from the approved decision-engine design. Build on the pinned local snapshot, resolved revision, hardware snapshot, decision policy, schema-versioned report, and transactional output layout already planned for the first release.
- Depends on / blocked by: First-release report schema and transaction directory behavior.
