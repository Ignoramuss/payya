# payya

Reinvent the AI wheel from scratch — in Rust.

A monorepo of **51 crates** that implement every major component of modern AI
infrastructure from first principles: autograd, transformers, training loops,
inference servers, vector databases, agents, and more.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full crate catalog, dependency
graph, and suggested implementation order.

---

## Prerequisites

- **Rust 1.75+** (2024 edition support) — install via [rustup](https://rustup.rs)
- **Cargo** (comes with Rust)
- Optional: CUDA toolkit 12+ (for GPU kernels in `payya-flash-attention`, `payya-matmul`)

## Build

```bash
# Build the entire workspace
cargo build

# Build a single crate
cargo build -p payya-transformer

# Build in release mode
cargo build --release
```

## Test

```bash
# Run all tests across the workspace
cargo test

# Test a single crate
cargo test -p payya-autograd

# Test with output visible
cargo test -p payya-tokenizer -- --nocapture
```

## Check & Lint

```bash
# Type-check everything (fast — no codegen)
cargo check

# Lint with clippy
cargo clippy --workspace --all-targets

# Format check
cargo fmt --check
```

## Run a Specific Binary

Some crates (like `payya-server`) will produce binaries:

```bash
cargo run -p payya-server
```

## Workspace Layout

```
crates/
├── core/       # Autograd, matmul, softmax, tokenizer, flash attention, logit processor
├── models/     # Transformer, ViT, SSM, MoE, CLIP, diffusion, AST, SLM, embedding
├── training/   # Distributed, LoRA, PEFT, RLHF, DPO, distillation, NAS
├── inference/  # Server, KV cache, speculative decoding, quantization, prompt cache
├── agent/      # Reasoner, agent loop, function calling, semantic router, code interpreter, structured output
├── data/       # Vector DB, RAG, graph RAG, knowledge graph, feature store, data curation, synthetic data, text-to-SQL, recommendation, DB driver
├── safety/     # Guardrails, eval harness, adversarial, interpretability
├── speech/     # ASR (Whisper-style), TTS
└── infra/      # AI gateway, model merger
```

## Developing a Single Crate

Each crate is self-contained. To work on one:

```bash
cd crates/core/payya-autograd
cargo test
cargo doc --open
```

Internal dependencies between crates are declared as workspace dependencies in
the root `Cargo.toml`. When you're ready to wire crates together, add the
dependency under `[dependencies]` in the crate's own `Cargo.toml`:

```toml
[dependencies]
payya-autograd.workspace = true
payya-matmul.workspace = true
```

## License

MIT
