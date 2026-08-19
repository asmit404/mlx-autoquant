# Model Support Matrix

Support claims are backed by metadata checks and a pinned macOS conversion fixture. A family is not marked supported just because its `config.json` looks similar to another model.

| Family | Checkpoint fixture | Revision | Status | Notes |
| --- | --- | --- | --- | --- |
| Llama causal | `hf-internal-testing/tiny-random-LlamaForCausalLM` | CI resolves the tested revision | Supported | Conversion and generation verification run in macOS integration CI. |
| Qwen2 causal | `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | Supported | Small public fixture validates the Qwen config and conversion path. |
| Qwen3 causal | `Qwen/Qwen3-27B` | Not validated | Needs validation | The target validation returned HTTP 401 from Hugging Face on 2026-08-19. See `docs/validation/qwen3-27b-tthw.md`. |

Supported means the fixture passed preflight, conversion, model loading, and non-empty generation verification on macOS. It does not guarantee that every checkpoint in the family fits every Mac.
