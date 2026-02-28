# Payya Roadmap

> How to incrementally build a from-scratch AI stack in Rust — and why in this
> order.

This roadmap turns Payya's 51-crate scaffold into a sequence of **concrete,
testable milestones**. Each milestone delivers a working artifact — something
you can run, benchmark, or demonstrate — not just a compiled library. The
ordering is driven by two constraints: the dependency graph (lower layers
first) and the principle that **every milestone should be independently
useful**.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full crate catalog and
dependency DAG.

---

## How to Read This Document

- **Milestones** are numbered (M1, M2, …). Each produces a demonstrable
  result.
- **Deliverables** describe what you can *do* when the milestone is complete.
- **Crates** lists which workspace crates are implemented in that milestone.
- **Hard parts** flags the genuine technical risks — places where you'll spend
  most of your time.
- **Exit criteria** defines "done" for the milestone.
- **Stretch** marks optional work that's nice to have but shouldn't block
  progress.

---

## M0 — Project Hygiene

**Goal:** Make the repo trustworthy before writing any AI code.

**Deliverables:**
- CI pipeline (GitHub Actions) that runs `cargo check`, `cargo test`,
  `cargo clippy`, and `cargo fmt --check` on every PR.
- Layer-dependency linter: a script or CI job that verifies no crate depends
  on a higher layer.
- `cargo deny` or equivalent for license and vulnerability auditing.
- Pre-commit hooks for formatting.

**Exit criteria:** A green CI badge on `main`. Any PR that breaks the build,
lint, or layer rules is rejected automatically.

**Why first:** Everything that follows will produce code. Without CI, quality
erodes silently. This is the cheapest milestone and it pays compound interest
on every future one.

---

## M1 — Tensor Primitives & Autograd

**Goal:** A working automatic differentiation engine over N-dimensional
tensors, with efficient matrix multiplication.

**Crates:** `payya-autograd`, `payya-matmul`

**Deliverables:**
- A `Tensor` type backed by `ndarray` (or a custom contiguous buffer) that
  records operations on a tape.
- Forward and reverse-mode AD for a core op set: add, mul, matmul, relu,
  sigmoid, log, exp, sum, reshape, transpose.
- `payya-matmul` provides tiled, cache-friendly GEMM on CPU. SIMD
  (AVX2/NEON) kernels for `f32`.
- Demonstrate: train a 2-layer MLP on XOR or MNIST-tiny using only these
  crates. Print the loss curve.

**Hard parts:**
- Getting the tape-based AD right for arbitrary graphs (not just sequential).
  Handling in-place ops and shared references without `Rc<RefCell<>>` spaghetti
  is the Rust-specific challenge.
- Tiled matmul that actually beats naive nested loops requires careful
  cache-line reasoning. Don't reach for BLAS — the point is to understand it —
  but benchmark against `ndarray`'s BLAS backend to know where you stand.

**Exit criteria:**
- `payya-autograd` computes correct gradients for all supported ops (property
  tests against numerical finite-difference gradients).
- `payya-matmul` is within 5× of OpenBLAS on a single-threaded 1024×1024
  `f32` multiply (stretch: 2×).
- MLP demo trains to >95% accuracy on a toy dataset.

---

## M2 — Text Encoding

**Goal:** Tokenize text into IDs and back.

**Crates:** `payya-tokenizer`

**Deliverables:**
- Byte-Pair Encoding (BPE) tokenizer: train from a corpus, encode, decode.
- Support loading an existing vocabulary (e.g., GPT-2 merges file) so you
  can test against known-good tokenizations.
- CLI binary: `echo "hello world" | cargo run -p payya-tokenizer`.

**Hard parts:**
- BPE training performance. A naive merge loop is O(n² · vocab). Use a
  priority queue with lazy deletion.
- Unicode edge cases: grapheme clusters, whitespace normalization,
  pre-tokenization regex.

**Exit criteria:**
- Round-trip: `decode(encode(text)) == text` for a corpus of 10k sentences.
- Can load GPT-2 vocab and produce identical token IDs for a test suite of
  100 strings.

---

## M3 — Attention & Softmax

**Goal:** The core attention mechanism, numerically stable and reasonably
fast.

**Crates:** `payya-softmax`, `payya-flash-attention`

**Deliverables:**
- Online (streaming) softmax that avoids materializing the full logit vector.
- A Flash Attention implementation (tiled, IO-aware) on CPU. This won't
  match the GPU version's speedups, but it teaches the algorithm and
  prepares the interface for a future CUDA backend.
- Benchmark: attention over sequence lengths 128, 512, 2048 with head
  dimensions 64 and 128.

**Hard parts:**
- Flash Attention's tiling scheme is intricate. Start with the Algorithm 1
  from the Flash Attention paper, get correctness first, optimize second.
- Numerical stability across the tiled softmax reductions (the "online
  softmax trick").

**Exit criteria:**
- Softmax output sums to 1.0 (within f32 epsilon) for random inputs of
  length 1 to 65536.
- Flash Attention produces results within 1e-5 of naive attention for
  random Q/K/V of shape `(batch=4, heads=8, seq=512, dim=64)`.
- Gradients through attention are correct (validated against finite
  differences).

---

## M4 — The First Transformer

**Goal:** A from-scratch Transformer that can do forward passes and
backpropagation.

**Crates:** `payya-transformer`, `payya-logit-processor`

**Deliverables:**
- Multi-head self-attention, feed-forward network, layer norm, residual
  connections, positional encoding (sinusoidal + RoPE).
- `payya-logit-processor`: top-k, top-p (nucleus), temperature scaling,
  repetition penalty.
- A small Transformer (2–6 layers, 128–256 dim) that can overfit a tiny
  text corpus and generate (mostly garbage, but structurally valid) text.
- Demo: `cargo run -p payya-transformer -- --generate "The meaning of"`

**Hard parts:**
- Wiring autograd through multi-head attention, especially the
  reshape/transpose dance between heads.
- Getting layer norm gradients right (they're notoriously fiddly).

**Exit criteria:**
- The model overfits 1000 tokens of text to <0.1 perplexity (memorization
  test — proves the training loop works end to end).
- Generation produces valid token sequences (no panics, no NaNs) for
  sequences up to 512 tokens.
- All logit processor strategies produce valid probability distributions.

---

## M5 — Small Language Model (End-to-End)

**Goal:** A self-contained, trainable, small language model.

**Crates:** `payya-slm`, `payya-embedding`

**Deliverables:**
- `payya-slm` wires together the transformer, tokenizer, and embedding
  layers into a single trainable language model (think GPT-2 small scale:
  ~10M–125M parameters).
- `payya-embedding` provides sentence-level embeddings (mean pooling or
  [CLS] token) — useful for downstream retrieval tasks.
- Training script that can train the SLM on a small corpus (~10–100MB of
  text) on a single machine.
- Checkpoint save/load in a simple binary format.

**Hard parts:**
- Memory management for 100M+ parameters with gradient buffers. You'll
  need to think about memory layout and possibly gradient checkpointing
  early.
- Training stability: learning rate warmup, gradient clipping, weight
  decay. These are easy to implement but hard to tune.

**Exit criteria:**
- SLM trains on WikiText-2 and achieves a perplexity that improves
  monotonically over 1 epoch (proves learning is happening — absolute
  perplexity will be much worse than PyTorch baselines, and that's fine).
- Embedding model produces cosine similarities where "king - man + woman"
  is closer to "queen" than to random words (on pretrained weights or
  after sufficient training).
- Checkpoint round-trip: save, reload, continue training with no loss
  spike.

---

## M6 — Inference Server

**Goal:** Serve the SLM over HTTP with streaming responses.

**Crates:** `payya-kv-cache`, `payya-quantization`, `payya-server`,
`payya-prompt-cache`

**Deliverables:**
- `payya-kv-cache`: paged KV cache (vLLM-style block tables) so long
  sequences don't OOM.
- `payya-quantization`: post-training quantization to Int8 (stretch: FP4).
  Reduces memory footprint.
- `payya-server`: HTTP server (axum or actix-web) with an
  OpenAI-compatible `/v1/chat/completions` endpoint. Supports streaming
  via SSE.
- `payya-prompt-cache`: radix-tree prefix matching to reuse KV cache
  across requests with shared prefixes.
- Demo: `curl localhost:8080/v1/chat/completions -d '{"messages": [...]}'`

**Hard parts:**
- Paged attention is a systems-programming problem: block allocation,
  copy-on-write, defragmentation. This is where Rust shines but the
  bookkeeping is real.
- Continuous batching (serving multiple concurrent requests with different
  sequence lengths) is the difference between a toy server and a usable
  one.

**Exit criteria:**
- Server handles 10 concurrent requests without deadlocks or panics.
- Quantized model produces coherent text (no severe quality degradation
  from Int8).
- KV cache memory usage is bounded (doesn't grow without limit for long
  conversations).
- Prompt cache hits reduce time-to-first-token by >30% on repeated
  prefixes.

---

## M7 — Fine-Tuning & Alignment

**Goal:** Adapt pretrained models to new tasks without full retraining.

**Crates:** `payya-lora`, `payya-peft`, `payya-dpo`, `payya-rlhf`,
`payya-distillation`

**Deliverables:**
- `payya-lora`: freeze base weights, inject low-rank adapters, train only
  the adapters (~1% of parameters).
- `payya-peft`: orchestrate multiple PEFT methods (LoRA, prefix tuning).
- `payya-dpo`: Direct Preference Optimization — simpler than RLHF, no
  reward model needed. Implement the Bradley-Terry loss.
- `payya-rlhf`: PPO-based RLHF loop with a separate reward model. This is
  the full pipeline.
- `payya-distillation`: teacher-student knowledge distillation with KL
  divergence loss.

**Hard parts:**
- RLHF is genuinely difficult to stabilize. PPO hyperparameters,
  KL penalty coefficients, reward model quality — all interact. Start with
  DPO (M7 can be split: ship DPO first, RLHF after).
- LoRA's weight merging at inference time needs careful handling of
  precision and numerics.

**Exit criteria:**
- LoRA fine-tuning on a small instruction dataset produces measurably
  better instruction-following than the base model (evaluated by a simple
  prompt suite, not human eval).
- DPO training converges on a preference dataset of 1000 pairs.
- Distilled student model retains >90% of teacher accuracy at <50% of
  parameters.

---

## M8 — Retrieval & Vector Search

**Goal:** Search over embeddings. Build the RAG pipeline.

**Crates:** `payya-vector-db`, `payya-rag`, `payya-db-driver`,
`payya-data-curation`

**Deliverables:**
- `payya-vector-db`: HNSW index with configurable distance metrics
  (cosine, L2, dot product). Insert, search, delete.
- `payya-rag`: chunking strategies (fixed-size, sentence-based, recursive),
  retrieval, and context injection into prompts.
- `payya-db-driver`: persistence layer (memory-mapped files or simple
  binary format).
- `payya-data-curation`: MinHash deduplication and quality filtering for
  training data.
- Demo: index a set of documents, ask a question, get an answer grounded
  in the retrieved context.

**Hard parts:**
- HNSW construction performance and recall tuning (the `ef_construction`
  and `M` parameters have dramatic effects).
- Chunking quality drives RAG quality more than any other factor. Getting
  the chunk boundaries right (respecting sentence/paragraph structure) is
  deceptively important.

**Exit criteria:**
- HNSW achieves >95% recall@10 on 100k random vectors at <10ms query
  latency.
- RAG demo answers factual questions about an indexed document set more
  accurately than the base model alone (side-by-side comparison on 50
  questions).

---

## M9 — Agents & Tool Use

**Goal:** An LLM that can reason step-by-step and call external tools.

**Crates:** `payya-reasoner`, `payya-agent`, `payya-function-call`,
`payya-structured-output`, `payya-semantic-router`

**Deliverables:**
- `payya-reasoner`: chain-of-thought prompting, step decomposition.
- `payya-agent`: ReAct loop — observe, think, act, repeat.
- `payya-function-call`: JSON schema parsing, tool dispatch, argument
  validation.
- `payya-structured-output`: constrained decoding via context-free
  grammars (guarantee valid JSON, SQL, etc.).
- `payya-semantic-router`: embed user intent, route to the right tool or
  sub-agent by similarity threshold.
- Demo: an agent that can answer questions by searching a vector DB, do
  arithmetic via a calculator tool, and return structured JSON.

**Hard parts:**
- Constrained decoding (CFG-guided token masking) requires tight
  integration with the logit processor and careful handling of the grammar
  state machine.
- Agent loops can diverge. Implementing max-step limits, stuck-detection,
  and graceful fallbacks is essential.

**Exit criteria:**
- Agent correctly solves 5 multi-step tasks (each requiring 2+ tool calls)
  from a test suite.
- Structured output always produces valid JSON for a defined schema (fuzz
  test with 1000 random prompts, 0 parse failures).
- Semantic router correctly classifies >90% of a 200-query intent test
  set.

---

## M10 — Safety & Evaluation

**Goal:** Know how good (and how safe) the system is.

**Crates:** `payya-guardrails`, `payya-eval`, `payya-adversarial`,
`payya-interpretability`

**Deliverables:**
- `payya-guardrails`: input/output filtering (regex + classifier-based),
  PII redaction (regex patterns for emails, phone numbers, SSNs, etc.),
  toxicity scoring.
- `payya-eval`: evaluation harness that runs benchmarks (MMLU, HellaSwag,
  TruthfulQA subsets) and reports scores.
- `payya-adversarial`: prompt injection detection, jailbreak test suite.
- `payya-interpretability`: sparse autoencoders for feature extraction,
  activation visualization.

**Hard parts:**
- Evaluation is only meaningful if you have a model good enough to
  differentiate from random. This milestone depends heavily on model
  quality from M5–M7.
- Interpretability (SAEs) is active research — scope it tightly or it
  becomes unbounded.

**Exit criteria:**
- Guardrails correctly block >95% of a curated test set of harmful prompts
  while passing >95% of benign prompts.
- Eval harness produces reproducible scores across runs (variance <1%).
- Adversarial test suite identifies at least 3 distinct attack categories
  with detection rates reported.

---

## M11 — Distributed Training

**Goal:** Train on more than one machine.

**Crates:** `payya-distributed`, `payya-nas`

**Deliverables:**
- `payya-distributed`: all-reduce gradient synchronization (NCCL-style),
  FSDP (fully sharded data parallel), gradient accumulation.
- `payya-nas`: basic neural architecture search — random search over a
  defined search space with a performance predictor.

**Hard parts:**
- Distributed training in Rust lacks the mature ecosystem that PyTorch has
  (no `torch.distributed`). You'll need to build on raw TCP/gRPC or use
  MPI bindings.
- Fault tolerance: what happens when a node dies mid-step?
- NAS is almost a research project on its own. Keep scope tight — random
  search + supernet, not full DARTS.

**Exit criteria:**
- 2-node training produces the same converged loss (within noise) as
  single-node training, in less wall-clock time.
- Gradient synchronization overhead is <20% of step time on a fast
  network.

---

## M12 — Multimodal & Specialized Architectures

**Goal:** Move beyond text — vision, audio, mixture-of-experts.

**Crates:** `payya-vit`, `payya-clip`, `payya-ssm`, `payya-moe`,
`payya-diffusion`, `payya-audio-spectrogram`, `payya-asr`, `payya-tts`

**Deliverables:**
- `payya-vit`: Vision Transformer for image classification (ImageNet
  subset).
- `payya-clip`: contrastive image-text model — joint embedding space.
- `payya-ssm`: Mamba-style state space model as an alternative to
  attention.
- `payya-moe`: mixture-of-experts routing (top-k gating, load balancing).
- `payya-diffusion`: UNet + DDPM/DDIM scheduler for image generation.
- `payya-audio-spectrogram` + `payya-asr`: Whisper-style speech
  recognition.
- `payya-tts`: text-to-speech with a vocoder.

**Hard parts:**
- Each of these is a substantial project. Prioritize by interest and
  available compute. ViT and CLIP are the most tractable. Diffusion and
  TTS are the most compute-hungry.
- SSM (Mamba) requires a hardware-aware selective scan — the CPU version
  will be slow but educational.

**Exit criteria:**
- ViT classifies CIFAR-10 at >70% accuracy (from-scratch training).
- CLIP produces meaningful similarity scores between matched vs. unmatched
  image-text pairs.
- At least one speech crate (ASR or TTS) produces intelligible output on a
  test sample.

---

## M13 — Knowledge Graphs & Advanced Retrieval

**Goal:** Structured knowledge beyond flat vector search.

**Crates:** `payya-knowledge-graph`, `payya-graph-rag`,
`payya-text-to-sql`, `payya-recommendation`, `payya-feature-store`,
`payya-synthetic-data`

**Deliverables:**
- `payya-knowledge-graph`: entity/relation extraction, triple store with
  SPARQL-like queries.
- `payya-graph-rag`: knowledge-graph-augmented retrieval (subgraph
  extraction around query entities).
- `payya-text-to-sql`: schema-aware SQL generation from natural language.
- `payya-recommendation`: two-tower model for item recommendation.
- `payya-feature-store`: online/offline feature serving with
  point-in-time correctness.
- `payya-synthetic-data`: template-based and LLM-based data generation for
  training augmentation.

**Hard parts:**
- Knowledge graph extraction quality depends on NER/RE models — you may
  need to bootstrap with the SLM from M5.
- Text-to-SQL requires schema linking, which is an open research problem
  for complex schemas.

**Exit criteria:**
- Graph RAG improves answer accuracy over flat RAG on a multi-hop question
  set.
- Text-to-SQL generates valid SQL for >80% of a test suite against a known
  schema.

---

## M14 — Production Infrastructure

**Goal:** Make the stack production-grade.

**Crates:** `payya-gateway`, `payya-model-merger`, `payya-speculative`,
`payya-code-interpreter`

**Deliverables:**
- `payya-gateway`: load balancer, rate limiter, failover across multiple
  `payya-server` instances.
- `payya-model-merger`: SLERP interpolation between model checkpoints
  ("model soups").
- `payya-speculative`: speculative decoding with a small draft model for
  faster inference.
- `payya-code-interpreter`: sandboxed code execution (Wasm or subprocess
  isolation) for agent tool use.

**Hard parts:**
- Speculative decoding's acceptance/rejection logic must preserve the
  target model's distribution exactly. Subtle bugs here silently degrade
  quality.
- Sandboxed execution is a security boundary — get it wrong and you have
  RCE. Use `wasmtime` or strict seccomp-bpf.

**Exit criteria:**
- Gateway distributes load across 3 server instances with automatic
  failover when one goes down.
- Speculative decoding produces identical output distribution to standard
  decoding (statistical test on 1000 generations) while being >1.5×
  faster.
- Code interpreter executes Python snippets with no filesystem or network
  access.

---

## Milestone Dependency Graph

```
M0 (CI)
 │
M1 (Autograd + Matmul)
 │
 ├── M2 (Tokenizer)
 │
 ├── M3 (Softmax + Flash Attention)
 │
 └───┬──── M4 (Transformer + Logit Processing)
     │
     └── M5 (SLM + Embeddings)
          │
          ├── M6 (Inference Server)
          │    │
          │    └── M14 (Production Infra)
          │
          ├── M7 (Fine-Tuning & Alignment)
          │    │
          │    └── M11 (Distributed Training)
          │
          ├── M8 (Vector DB + RAG)
          │    │
          │    ├── M9 (Agents & Tool Use)
          │    │
          │    └── M13 (Knowledge Graphs & Advanced Retrieval)
          │
          ├── M10 (Safety & Eval)
          │
          └── M12 (Multimodal & Specialized Architectures)
```

Milestones at the same depth level can be worked on in parallel once their
parent is complete. M6–M12 are all independently pursuable after M5.

---

## Principles for Execution

1. **One crate at a time.** Finish a crate before starting the next. A half-
   implemented crate is worse than a missing one — it creates false
   confidence.

2. **Tests are not optional.** Every crate should have property-based tests
   (for numerical code, check against finite differences or reference
   implementations) and at least one integration test that runs end to end.

3. **Benchmark from day one.** `payya-matmul` without benchmarks is a
   science project. Use `criterion` for microbenchmarks. Track performance
   across commits.

4. **Document decisions, not just APIs.** When you choose an algorithm (e.g.,
   tiled GEMM over Strassen, HNSW over IVF), write a paragraph explaining
   why. Future-you will thank present-you.

5. **Don't gold-plate.** The first version of `payya-autograd` doesn't need
   GPU support. The first version of `payya-server` doesn't need continuous
   batching. Ship the simple version, prove it works, iterate.

6. **GPU is a feature, not a prerequisite.** Every crate should work on CPU
   first. Gate GPU kernels behind `#[cfg(feature = "cuda")]`. This keeps the
   build fast, the test matrix small, and the contributor barrier low.

7. **Steal the right ideas.** "From scratch" means writing the code yourself,
   not ignoring the literature. Read the papers, study reference
   implementations, then close the tab and write your own.

---

## What This Roadmap Does Not Cover

- **Specific timelines.** This is a learning-oriented project. Estimating
  timelines would be dishonest — M1 could take a week or a month depending
  on experience and available hours.
- **Funding or team structure.** The roadmap assumes a small team or solo
  developer. Adjust parallelism based on headcount.
- **GPU kernel optimization.** CUDA kernels are feature-gated and deferred.
  The roadmap focuses on correct CPU implementations first. A separate
  "GPU acceleration" track can run in parallel once M1–M4 are solid.
- **Model training at scale.** Training a competitive LLM requires thousands
  of GPU-hours and large datasets. This roadmap gets you to the point where
  you *could* train at scale — actually doing so is a resource question, not
  an engineering one.
