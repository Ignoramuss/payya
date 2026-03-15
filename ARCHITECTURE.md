# Payya Architecture

> Reinvent the AI wheel from scratch — in Rust.

This document describes the monorepo layout, the dependency graph between
crates, and the design philosophy behind **payya**.

---

## Design Principles

1. **From-scratch implementations.** Every component is built from first
   principles. No wrapping of Python libraries behind FFI — the goal is to
   understand *and* own every line.
2. **Layered dependency graph.** Crates are organised into layers.  Lower layers
   (core) have zero internal dependencies; higher layers compose them.
3. **Feature-gated heavyweights.** GPU/CUDA code, large model weights, and
   optional network I/O are behind Cargo features so `cargo check` stays fast.
4. **Workspace-level consistency.** Common dependency versions, edition, and
   lints are pinned once in the root `Cargo.toml`.

---

## Repository Layout

```
payya/
├── Cargo.toml              # Workspace root
├── ARCHITECTURE.md         # This file
├── README.md               # Quick-start guide
│
├── crates/
│   ├── core/               # Layer 0 — no internal deps
│   │   ├── payya-autograd/          # Autograd engine (Micrograd-style)
│   │   ├── payya-matmul/            # Matrix multiplication kernels
│   │   ├── payya-softmax/           # Softmax kernel optimizations
│   │   ├── payya-logit-processor/   # Logit processing & sampling
│   │   ├── payya-tokenizer/         # BPE tokenizer
│   │   └── payya-flash-attention/   # Flash Attention kernel
│   │
│   ├── models/             # Layer 1 — depends on core
│   │   ├── payya-transformer/       # Transformer (Attention Is All You Need)
│   │   ├── payya-vit/               # Vision Transformer
│   │   ├── payya-ssm/               # State Space Model (Mamba)
│   │   ├── payya-moe/               # Mixture of Experts routing
│   │   ├── payya-clip/              # CLIP multi-modal projector
│   │   ├── payya-diffusion/         # Diffusion model (UNet + Scheduler)
│   │   ├── payya-audio-spectrogram/ # Audio Spectrogram Transformer
│   │   ├── payya-slm/              # Small Language Model
│   │   └── payya-embedding/         # Embedding model
│   │
│   ├── training/           # Layer 2 — depends on core + models
│   │   ├── payya-distributed/       # Distributed training (FSDP/Tensor Parallel)
│   │   ├── payya-lora/              # LoRA trainer
│   │   ├── payya-peft/              # Parameter Efficient Fine-Tuning
│   │   ├── payya-rlhf/              # RLHF pipeline (PPO)
│   │   ├── payya-dpo/               # DPO loss function
│   │   ├── payya-distillation/      # Model distillation
│   │   └── payya-nas/               # Neural Architecture Search
│   │
│   ├── inference/          # Layer 2 — depends on core + models
│   │   ├── payya-server/            # Inference server
│   │   ├── payya-kv-cache/          # KV Cache paging (vLLM-style)
│   │   ├── payya-speculative/       # Speculative decoding
│   │   ├── payya-quantization/      # Quantization (Int8/FP4)
│   │   └── payya-prompt-cache/      # Prompt caching
│   │
│   ├── agent/              # Layer 3 — depends on inference + models
│   │   ├── payya-reasoner/          # Chain of Thought reasoner
│   │   ├── payya-agent/             # Agent loop (ReAct pattern)
│   │   ├── payya-function-call/     # Function calling router
│   │   ├── payya-semantic-router/   # Semantic router
│   │   ├── payya-code-interpreter/  # Code interpreter sandbox
│   │   └── payya-structured-output/ # Structured output parser (CFG)
│   │
│   ├── data/               # Layer 2 — depends on core
│   │   ├── payya-vector-db/         # Vector DB (HNSW index)
│   │   ├── payya-rag/               # RAG pipeline
│   │   ├── payya-graph-rag/         # Graph RAG system
│   │   ├── payya-knowledge-graph/   # Knowledge graph builder
│   │   ├── payya-feature-store/     # Feature store
│   │   ├── payya-data-curation/     # Data curation (MinHash/Dedup)
│   │   ├── payya-synthetic-data/    # Synthetic data generator
│   │   ├── payya-text-to-sql/       # Text-to-SQL engine
│   │   ├── payya-recommendation/    # Recommendation (Two-tower)
│   │   └── payya-db-driver/         # Database driver for vectors
│   │
│   ├── safety/             # Layer 3 — depends on models + inference
│   │   ├── payya-guardrails/        # Guardrails (I/O filtering)
│   │   ├── payya-eval/              # LLM eval harness
│   │   ├── payya-adversarial/       # Adversarial attack generator
│   │   └── payya-interpretability/  # Interpretability (SAE)
│   │
│   ├── speech/             # Layer 1 — depends on core
│   │   ├── payya-asr/               # Whisper-style ASR
│   │   └── payya-tts/               # Text-to-Speech pipeline
│   │
│   └── infra/              # Layer 4 — depends on everything
│       ├── payya-gateway/           # AI Gateway (LB/Failover)
│       └── payya-model-merger/      # Model merger (SLERP/Soups)
```

---

## Dependency Layers

The crate graph is organized into layers to prevent circular dependencies
and keep compilation parallel:

```
Layer 0  ─  core/
               │
Layer 1  ─  models/   speech/
               │
Layer 2  ─  training/  inference/  data/
               │
Layer 3  ─  agent/  safety/
               │
Layer 4  ─  infra/
```

**Rule:** A crate may only depend on crates in the same layer or lower.
This is enforced by convention (and can later be enforced by CI).

---

## Crate Catalog

### Core Primitives (`crates/core/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-autograd` | Automatic differentiation engine | Computational graphs, reverse-mode AD, gradient tape |
| `payya-matmul` | Matrix multiplication kernels | Tiling, SIMD, cache-oblivious algorithms |
| `payya-softmax` | Softmax kernel optimization | Numerical stability, online softmax, fused kernels |
| `payya-logit-processor` | Logit processing & sampling | Top-k, top-p (nucleus), temperature, repetition penalty |
| `payya-tokenizer` | BPE tokenizer | Byte-pair encoding, vocab merges, pre-tokenization |
| `payya-flash-attention` | Flash Attention kernel | Tiled attention, IO-aware, memory-efficient |

### Model Architectures (`crates/models/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-transformer` | Transformer from scratch | Multi-head attention, positional encoding, layer norm |
| `payya-vit` | Vision Transformer | Patch embedding, class token, image classification |
| `payya-ssm` | State Space Model (Mamba) | Selective scan, linear recurrence, hardware-aware scan |
| `payya-moe` | Mixture of Experts routing | Top-k gating, load balancing, expert parallelism |
| `payya-clip` | CLIP multi-modal projector | Contrastive learning, image-text alignment, projection heads |
| `payya-diffusion` | Diffusion model | UNet, noise schedules (DDPM/DDIM), denoising |
| `payya-audio-spectrogram` | Audio Spectrogram Transformer | Mel spectrograms, patch embedding for audio |
| `payya-slm` | Small Language Model | End-to-end LM: embeddings → transformer → LM head |
| `payya-embedding` | Embedding model | Sentence embeddings, contrastive training, pooling |

### Training (`crates/training/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-distributed` | Distributed training loop | FSDP, tensor parallelism, gradient accumulation, all-reduce |
| `payya-lora` | LoRA trainer | Low-rank weight decomposition, frozen base weights |
| `payya-peft` | PEFT library | Adapters, prefix tuning, LoRA orchestration |
| `payya-rlhf` | RLHF pipeline | PPO, reward model, KL divergence penalty |
| `payya-dpo` | DPO loss function | Direct preference optimization, Bradley-Terry model |
| `payya-distillation` | Model distillation | Teacher-student, KD loss, logit matching |
| `payya-nas` | Neural Architecture Search | Search spaces, supernets, performance predictors |

### Inference & Serving (`crates/inference/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-server` | Inference server | HTTP/gRPC, batching, streaming, OpenAI-compatible API |
| `payya-kv-cache` | KV Cache paging | Paged attention, block tables, memory management |
| `payya-speculative` | Speculative decoding | Draft model, verification, acceptance/rejection |
| `payya-quantization` | Quantization library | Int8/FP4, calibration, quantize-aware training |
| `payya-prompt-cache` | Prompt caching | Prefix matching, KV cache reuse, radix tree |

### Agent & Reasoning (`crates/agent/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-reasoner` | Chain of Thought reasoner | Thought decomposition, step-by-step prompting |
| `payya-agent` | Agent loop (ReAct) | Observation-thought-action cycle, tool use |
| `payya-function-call` | Function calling router | Schema parsing, tool dispatch, argument validation |
| `payya-semantic-router` | Semantic router | Embedding-based intent routing, threshold tuning |
| `payya-code-interpreter` | Code interpreter sandbox | Process isolation, sandboxing, I/O capture |
| `payya-structured-output` | Structured output parser | Context-free grammars, constrained decoding |

### Data & Retrieval (`crates/data/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-vector-db` | Vector database (HNSW) | HNSW index, approximate nearest neighbors, distance metrics |
| `payya-rag` | RAG pipeline | Chunking, retrieval, context injection |
| `payya-graph-rag` | Graph RAG system | Knowledge-graph-augmented retrieval, subgraph extraction |
| `payya-knowledge-graph` | Knowledge graph builder | Entity extraction, relation detection, triple stores |
| `payya-feature-store` | Feature store | Online/offline serving, point-in-time correctness |
| `payya-data-curation` | Data curation | MinHash, LSH, deduplication, quality filtering |
| `payya-synthetic-data` | Synthetic data generator | Template generation, LLM-based synthesis, diversity metrics |
| `payya-text-to-sql` | Text-to-SQL engine | Schema linking, SQL generation, query validation |
| `payya-recommendation` | Recommendation system | Two-tower architecture, approximate retrieval, ranking |
| `payya-db-driver` | Vector database driver | Connection pooling, serialization, query protocol |

### Safety & Evaluation (`crates/safety/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-guardrails` | Guardrails system | Input/output filters, toxicity detection, PII redaction |
| `payya-eval` | LLM eval harness | Benchmarks, metrics, automated scoring |
| `payya-adversarial` | Adversarial attacks | Prompt injection detection, jailbreak testing |
| `payya-interpretability` | Interpretability (SAE) | Sparse autoencoders, feature visualization, probing |

### Speech & Audio (`crates/speech/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-asr` | Whisper-style ASR | Encoder-decoder, mel spectrograms, CTC/attention |
| `payya-tts` | Text-to-Speech | Vocoder, alignment, mel generation |

### Infrastructure (`crates/infra/`)

| Crate | What You're Building | Key Concepts |
|---|---|---|
| `payya-gateway` | AI Gateway | Load balancing, failover, rate limiting, routing |
| `payya-model-merger` | Model merger | SLERP, model soups, weight interpolation |

---

## Intended Dependency Graph (per-crate)

Below are the planned internal dependencies once implementations are connected.
At scaffold time all crates are independent; wire them up as you implement each.

```
payya-transformer  →  payya-autograd, payya-matmul, payya-softmax,
                      payya-flash-attention, payya-tokenizer

payya-slm          →  payya-transformer, payya-tokenizer, payya-logit-processor

payya-server       →  payya-slm (or any model), payya-kv-cache,
                      payya-quantization, payya-prompt-cache

payya-agent        →  payya-reasoner, payya-function-call, payya-structured-output

payya-rag          →  payya-vector-db, payya-embedding, payya-tokenizer

payya-graph-rag    →  payya-rag, payya-knowledge-graph

payya-rlhf         →  payya-autograd, payya-transformer, payya-dpo

payya-peft         →  payya-lora, payya-autograd

payya-gateway      →  payya-server
```

---

## Implemented Feature Architectures

This section contains detailed architecture diagrams for every implemented
feature. Each diagram shows the internal structure, data flow, and key
invariants of the crate. Updated as each milestone lands.

---

### M1: `payya-matmul` — Tiled GEMM

Computes C = A × B for row-major `f32` matrices. Two execution paths,
selected deterministically by matrix dimensions (not by error/fallback):

```
         ┌─────────────────────────────────────┐
         │           matmul(a, b, m, k, n)      │
         │                                       │
         │  assert!(a.len() >= m*k)   ◄── invariant: no silent truncation
         │  assert!(b.len() >= k*n)              │
         │                                       │
         │  ┌──────────────────────┐             │
         │  │ m,k,n all <= TILE?   │             │
         │  └──────┬───────┬───────┘             │
         │     yes │       │ no                  │
         │         ▼       ▼                     │
         │   naive_matmul  tiled_matmul          │
         │   (ikj loops)   (blocked ikj)         │
         │         │       │                     │
         │         └───┬───┘                     │
         │             ▼                         │
         │     C[i*n+j] += A[i*k+p] * B[p*n+j]  │
         └─────────────────────────────────────┘

   Both paths compute the identical result (same algorithm, same
   accumulation order within each tile). The tiled path is not a
   "fallback" — it is the primary path for large matrices.
```

**Tiling scheme** — chosen so three tile blocks fit in L1 cache:

```
   A (m×k)              B (k×n)              C (m×n)
  ┌──┬──┬──┐          ┌──┬──┬──┐          ┌──┬──┬──┐
  │  │  │  │          │  │  │  │          │  │  │  │
  ├──┼──┼──┤          ├──┼──┼──┤          ├──┼──┼──┤
  │  │▓▓│  │ A tile   │  │  │  │          │  │  │▓▓│ C tile
  ├──┼──┼──┤ (i0..i1, ├──┼──┼──┤          ├──┼──┼──┤ (i0..i1,
  │  │  │  │  p0..p1) │▓▓│  │  │ B tile   │  │  │  │  j0..j1)
  └──┴──┴──┘          └──┴──┴──┘ (p0..p1, └──┴──┴──┘
                                   j0..j1)

   TILE = 32  →  3 tiles × 32×32 × 4 bytes = 12 KB (fits L1)

   Loop order: p-tiles outermost (reduction dimension),
   then i-tiles, then j-tiles. This maximises register reuse
   of the C-tile accumulator across p-iterations.
```

**Transposed variants** for autograd backward pass:

```
   matmul_at_b(A, B, C, m, k, n)     matmul_a_bt(A, B, C, m, k, n)
   ─────────────────────────────      ─────────────────────────────
   A stored as (k×m), read as Aᵀ     B stored as (n×k), read as Bᵀ
   C += Aᵀ × B                       C += A × Bᵀ

   Used by autograd backward:
     ∂L/∂A = ∂L/∂C × Bᵀ  (matmul_a_bt)
     ∂L/∂B = Aᵀ × ∂L/∂C  (matmul_at_b)
```

---

### M1: `payya-autograd` — Reverse-Mode Automatic Differentiation

Arena-based computation graph. Tensors are indices (`TensorId`) into a
`Vec<Node>` owned by the `Graph`. No `Rc<RefCell<>>`, no garbage collection.

```
   ┌──────────────────────────────────────────────────────┐
   │                      Graph                            │
   │                                                       │
   │  nodes: Vec<Node>                                     │
   │  ┌─────┬─────┬─────┬─────┬─────┬─────┐              │
   │  │  0  │  1  │  2  │  3  │  4  │  5  │  ...         │
   │  │Leaf │Leaf │ Add │MatMul│ReLU │ Sum │              │
   │  │(x)  │(W)  │(0,1)│(2,.) │(3)  │(4)  │              │
   │  └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘              │
   │     │     │     │     │     │     │                   │
   │  TensorId is just an index ───────────────────────►  │
   │  into this arena. Cheap to copy (usize wrapper).     │
   └──────────────────────────────────────────────────────┘
```

**Node structure:**

```
   Node {
       data: Vec<f32>         ◄── forward-pass result
       shape: Vec<usize>      ◄── dimensions (row-major)
       grad: Option<Vec<f32>> ◄── populated by backward()
       op: Op                 ◄── which operation + input TensorIds
       requires_grad: bool    ◄── param (true) vs constant (false)
   }
```

**Forward pass** — each operation reads its inputs, computes the result,
and appends a new `Node` to the arena. Because nodes are append-only,
the arena index IS the topological order:

```
   User code                          Graph arena
   ─────────                          ───────────
   let x = g.param(data, shape)  →   nodes[0] = Leaf(x)
   let w = g.param(data, shape)  →   nodes[1] = Leaf(w)
   let h = g.matmul(x, w)       →   nodes[2] = MatMul(0, 1)
   let h = g.relu(h)            →   nodes[3] = Relu(2)
   let loss = g.sum(h)          →   nodes[4] = Sum(3)

   Invariant: node i can only reference nodes j where j < i.
   This means the arena is always in valid topological order.
```

**Backward pass** — reverse iteration over the arena propagates
gradients. No explicit topological sort is needed:

```
   backward(root=4):

   idx=4  Sum(3)      grad[4] = [1.0]        (seed)
          ──────►     grad[3] += [1.0, 1.0, ...]   (broadcast)

   idx=3  Relu(2)     grad[3] = [1.0, 0.0, 1.0, ...]
          ──────►     grad[2] += grad[3] * (input > 0 ? 1 : 0)

   idx=2  MatMul(0,1) grad[2] = [...]
          ──────►     grad[0] += grad[2] × W^T    (matmul_a_bt)
                      grad[1] += X^T × grad[2]    (matmul_at_b)

   idx=1  Leaf(w)     grad[1] = accumulated       (done — read via g.grad(w))
   idx=0  Leaf(x)     grad[0] = accumulated       (done — read via g.grad(x))
```

**Broadcasting** — handled in both forward and backward for binary ops:

```
   Forward: (m,n) + (n,)  →  broadcast b across rows  →  (m,n)

   Backward of broadcast add:
     grad_a = grad_output                          (same shape)
     grad_b = sum_over_rows(grad_output)           (reduce back)

   Supported patterns:
     same shape    →  element-wise
     scalar (1,)   →  broadcast to any shape
     row (n,)      →  broadcast across rows of (m,n)
```

**Supported operations and their gradients:**

```
   ┌────────────┬───────────────────────┬────────────────────────────────┐
   │ Operation  │ Forward               │ Backward (∂L/∂input)          │
   ├────────────┼───────────────────────┼────────────────────────────────┤
   │ add(a, b)  │ a + b                 │ ∂L/∂a = grad, ∂L/∂b = grad   │
   │ sub(a, b)  │ a - b                 │ ∂L/∂a = grad, ∂L/∂b = -grad  │
   │ mul(a, b)  │ a * b                 │ ∂L/∂a = grad*b, ∂L/∂b = grad*a│
   │ matmul(a,b)│ a @ b                 │ ∂L/∂a = grad@bᵀ, ∂L/∂b = aᵀ@grad│
   │ relu(a)    │ max(0, a)             │ grad * (a > 0 ? 1 : 0)       │
   │ sigmoid(a) │ σ(a)                  │ grad * σ(a) * (1 - σ(a))     │
   │ log(a)     │ ln(a)                 │ grad / a                      │
   │ exp(a)     │ eᵃ                    │ grad * eᵃ                     │
   │ sum(a)     │ Σaᵢ                   │ broadcast grad to input shape │
   │ reshape(a) │ same data, new shape  │ same grad, original shape     │
   │ transpose(a)│ swap rows/cols       │ transpose(grad)               │
   └────────────┴───────────────────────┴────────────────────────────────┘

   All gradients validated against finite-difference numerical gradients.
```

**MLP training loop** (demonstrated in `examples/xor_mlp.rs`):

```
   ┌─────────────────────────────────────────────┐
   │              Training Epoch                   │
   │                                               │
   │  1. Build graph   x ─┬─► matmul ─► add ─► relu ─► matmul ─► add ─► sigmoid ─► sub ─► mul ─► sum
   │                   W₁─┘          b₁              W₂          b₂              target     (diff²) (loss)
   │                                               │
   │  2. Forward       Compute all node data       │
   │                   (happens during build)      │
   │                                               │
   │  3. Backward      g.backward(loss)            │
   │                   → gradients on W₁,b₁,W₂,b₂ │
   │                                               │
   │  4. SGD update    W -= lr * ∂L/∂W             │
   │                                               │
   │  5. New epoch     Rebuild graph from scratch   │
   │                   (old graph dropped)          │
   └─────────────────────────────────────────────┘
```

### M2: `payya-tokenizer` — Byte-Pair Encoding (BPE) Tokenizer

Trains, encodes, and decodes text using byte-pair encoding. Supports
loading GPT-2 vocabulary files for compatibility testing.

**Core data structures:**

```
   ┌─────────────────────────────────────────────────────────┐
   │                     Tokenizer                            │
   │                                                          │
   │  merges: Vec<MergeRule>     ◄── merge rules in priority  │
   │                                 order (index 0 = first)  │
   │                                                          │
   │  merge_map: HashMap<(TokenId, TokenId), TokenId>         │
   │                             ◄── O(1) pair→merged lookup  │
   │                                 (rebuilt from merges)    │
   │                                                          │
   │  vocab: HashMap<TokenId, Vec<u8>>                        │
   │                             ◄── token ID → byte sequence │
   │                                 for decoding             │
   └─────────────────────────────────────────────────────────┘

   MergeRule { left: TokenId, right: TokenId, merged: TokenId }
   TokenId = u32  (0–255 = raw bytes, ≥256 = learned merges)
```

**Training algorithm:**

```
   Input corpus (bytes)
   ┌──────────────────────────────────────────┐
   │  "aaabdaaabac"                            │
   │   → [97, 97, 97, 98, 100, 97, 97, 97,   │
   │       98, 97, 99]                         │
   └──────────────────┬───────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────┐
   │  While num_merges < target:              │
   │                                           │
   │  1. Count all adjacent pairs              │
   │     {(97,97): 4, (97,98): 2, ...}        │
   │                                           │
   │  2. Find most frequent pair               │
   │     → (97, 97) with count 4              │
   │                                           │
   │  3. Create merge rule:                    │
   │     MergeRule { left:97, right:97,        │
   │                 merged: 256 }             │
   │                                           │
   │  4. Apply merge in token sequence:        │
   │     [97,97,97,98,...] → [256,97,98,...]   │
   │                                           │
   │  5. Repeat with next_id++                 │
   └──────────────────────────────────────────┘

   Invariant: merges stop early if no pair appears ≥2 times
   (further merges would not compress).

   Determinism: ties broken by smallest (left, right) pair.
```

**Encoding (text → token IDs):**

```
   Input: "abc"
   ┌────────────────────────────────────────┐
   │  1. Byte-level tokenization:           │
   │     "abc" → [97, 98, 99]              │
   │                                        │
   │  2. Apply merges in priority order:    │
   │     merge 0: (97,98)→256              │
   │       [97, 98, 99] → [256, 99]        │
   │     merge 1: (256,99)→257             │
   │       [256, 99] → [257]               │
   │                                        │
   │  3. Output: [257]                      │
   └────────────────────────────────────────┘

   Each merge pass is a single linear scan: O(n) per merge.
   Total encoding: O(n × num_merges).
```

**Decoding (token IDs → text):**

```
   Input: [257]
   ┌────────────────────────────────────────┐
   │  1. Look up each ID in vocab:          │
   │     257 → [97, 98, 99]  (from merges) │
   │                                        │
   │  2. Concatenate byte sequences:        │
   │     [97, 98, 99]                       │
   │                                        │
   │  3. UTF-8 decode (lossy):              │
   │     → "abc"                            │
   └────────────────────────────────────────┘
```

**GPT-2 compatibility layer:**

```
   GPT-2 uses a byte-to-unicode mapping so that all tokens
   are printable strings. Non-printable bytes (0–32, 127–160,
   173) are mapped to Unicode code points starting at U+0100.

   Byte 0x00 → 'Ā' (U+0100)
   Byte 0x20 → 'Ġ' (U+0120)    ◄── space character
   Byte 0x41 → 'A' (U+0041)    ◄── printable: identity map

   Loading GPT-2 vocab:
   ┌────────────────────┐     ┌────────────────────┐
   │   vocab.json       │     │   merges.txt       │
   │   {"Ġ": 220,       │     │   #version: 0.2    │
   │    "Ġt": 256, ...} │     │   Ġ t              │
   └────────┬───────────┘     │   e r              │
            │                 │   ...               │
            ▼                 └────────┬───────────┘
   ┌─────────────────────────────────┐ │
   │  1. Parse vocab JSON → str→id  │ │
   │  2. Build byte-to-unicode map  │ │
   │  3. Convert token strings to   │◄┘
   │     byte sequences via map     │
   │  4. Parse merge lines into     │
   │     MergeRule { left, right,   │
   │     merged }                   │
   └─────────────────────────────────┘

   Invariant: byte-to-unicode map is bijective (256 unique
   chars for 256 bytes). Verified in tests.
```

**Key invariants:**

- `decode(encode(text)) == text` for all valid UTF-8 input.
- Token IDs 0–255 always represent raw bytes.
- Merge rules are topologically ordered: a merge at index i
  only references tokens that exist after merges 0..i-1.
- Training is deterministic: same corpus + vocab_size always
  produces the same merge rules.

---

### M3: `payya-softmax` — Numerically Stable Softmax

Provides both standard two-pass and single-pass online softmax.
The online variant uses the running-max correction trick to compute
softmax without a separate pass to find the maximum.

```
   ┌─────────────────────────────────────────────────────────┐
   │               softmax(logits)                            │
   │               Two-pass algorithm                         │
   │                                                          │
   │  Pass 1: max = max(logits)                              │
   │  Pass 2: out_i = exp(logits_i - max)                    │
   │          sum = Σ out_i                                   │
   │          out_i /= sum                                    │
   └─────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────┐
   │           softmax_online(logits)                         │
   │           Single-pass (Milakov & Gimelshein, 2018)       │
   │                                                          │
   │  Initialize: m = -∞, d = 0                              │
   │                                                          │
   │  For each x_i:                                           │
   │    m_prev = m                                            │
   │    m = max(m, x_i)                                       │
   │    d = d * exp(m_prev - m) + exp(x_i - m)  ◄── correct  │
   │                                                 previous │
   │                                                 sum for  │
   │                                                 new max  │
   │  Output: out_i = exp(x_i - m) / d                       │
   └─────────────────────────────────────────────────────────┘

   Both algorithms produce identical results in exact arithmetic.
   In f32, they agree within epsilon.
```

**Backward pass (Jacobian-vector product):**

```
   Given: s = softmax(x), upstream gradient g = ∂L/∂s

   ∂L/∂x_i = s_i * (g_i - dot(g, s))

   Derived from Jacobian: ∂s_j/∂x_i = s_j * (δ_ij - s_i)

   ┌────────────────────────────────────────────┐
   │  softmax_backward(s, grad_output)           │
   │                                             │
   │  dot = Σ (s_i * grad_output_i)              │
   │  grad_input_i = s_i * (grad_output_i - dot) │
   └────────────────────────────────────────────┘

   Row-wise variant: apply independently per row of (rows, cols).
```

**Key invariants:**

- `sum(softmax(x)) == 1.0` within f32 epsilon for any non-empty input.
- Numerically stable: no overflow for inputs up to ±1000.
- Online and standard variants produce matching results.
- All outputs are non-negative.
- Backward validated against finite-difference numerical gradients.

---

### M3: `payya-flash-attention` — Tiled, IO-Aware Attention

Implements Flash Attention (Dao et al., 2022) on CPU. The full N×N
attention matrix is never materialized. Instead, K/V are processed
in blocks of size B, maintaining running softmax statistics.

```
   ┌───────────────────────────────────────────────────────────┐
   │           flash_attention(Q, K, V, n, d)                   │
   │                                                            │
   │  assert!(q.len() == n*d)   ◄── shape invariants           │
   │  assert!(k.len() == n*d)                                  │
   │  assert!(v.len() == n*d)                                  │
   │                                                            │
   │  scale = 1/√d                                              │
   │  Initialize: O[n,d]=0, row_max[n]=-∞, row_sum[n]=0       │
   │                                                            │
   │  For each KV block j (size B=32):                          │
   │    ┌─────────────────────────────────────────────────┐     │
   │    │  For each query row i:                           │     │
   │    │    1. s = Q_i · K_j^T * scale    (B scores)     │     │
   │    │    2. block_max = max(s)                         │     │
   │    │    3. m_new = max(row_max[i], block_max)         │     │
   │    │    4. correction = exp(m_old - m_new)            │     │
   │    │    5. row_sum[i] *= correction                   │     │
   │    │       O[i,:] *= correction                       │     │
   │    │    6. w_j = exp(s_j - m_new)                     │     │
   │    │       row_sum[i] += Σ w_j                        │     │
   │    │       O[i,:] += Σ w_j * V_j                      │     │
   │    └─────────────────────────────────────────────────┘     │
   │                                                            │
   │  Final: O[i,:] /= row_sum[i]                              │
   └───────────────────────────────────────────────────────────┘

   Memory: O(n·d + B²) instead of O(n²)
   B = 32 chosen so block scores fit in cache.
```

**Batched attention** for multi-head, multi-batch:

```
   flash_attention_batched(Q, K, V, batch, heads, seq, dim)

   Input layout: (batch, heads, seq, dim) contiguous row-major
   Total elements: batch × heads × seq × dim

   ┌────────────────────────────────┐
   │  For each (b, h):              │
   │    offset = (b*heads + h) *    │
   │             seq * dim          │
   │    O[offset..] =               │
   │      flash_attention(          │
   │        Q[offset..],            │
   │        K[offset..],            │
   │        V[offset..],            │
   │        seq, dim)               │
   └────────────────────────────────┘
```

**Backward pass** (standard, non-tiled for correctness):

```
   Given: Q, K, V, grad_output (all shape n×d)

   Recompute:
     S = Q × K^T / √d        (n × n)
     P = softmax_rows(S)      (n × n)

   Gradients:
     ∂L/∂V = P^T × grad_out               (n × d)
     grad_P = grad_out × V^T              (n × n)
     grad_S = softmax_rows_backward(P, grad_P)  (n × n)
     ∂L/∂Q = grad_S × K / √d             (n × d)
     ∂L/∂K = grad_S^T × Q / √d           (n × d)

   All gradients validated against finite-difference oracles.
```

**Key invariants:**

- Flash attention output matches naive attention within 1e-5.
- All outputs are finite (no NaN/Inf).
- Batched version produces identical per-head results.
- Backward gradients validated against numerical finite differences.

---

### M4: `payya-logit-processor` — Logit Processing & Sampling

Provides composable logit processing strategies for controlling text
generation: temperature scaling, top-k filtering, nucleus (top-p) sampling,
and repetition penalty.

```
   ┌──────────────────────────────────────────────────────────┐
   │              LogitProcessor pipeline                      │
   │                                                           │
   │  Input: raw logits[vocab_size]                            │
   │                                                           │
   │  1. Repetition penalty (optional)                         │
   │     For each token in past_tokens:                        │
   │       logit > 0 → logit /= penalty                       │
   │       logit < 0 → logit *= penalty                        │
   │                                                           │
   │  2. Temperature scaling (optional)                        │
   │     logits[i] /= temperature                              │
   │     Higher T → more uniform, Lower T → more peaked        │
   │                                                           │
   │  3. Top-k filtering (optional)                            │
   │     Keep top-k logits, set rest to -∞                     │
   │     ┌────────────────────────────────┐                    │
   │     │  sorted = sort(logits, desc)   │                    │
   │     │  threshold = sorted[k-1]       │                    │
   │     │  if logit < threshold: -∞      │                    │
   │     └────────────────────────────────┘                    │
   │                                                           │
   │  4. Top-p (nucleus) filtering (optional)                  │
   │     ┌────────────────────────────────┐                    │
   │     │  probs = softmax(logits)       │                    │
   │     │  sort by prob descending       │                    │
   │     │  cumsum until >= p             │                    │
   │     │  mask out rest to -∞           │                    │
   │     └────────────────────────────────┘                    │
   │                                                           │
   │  5. Softmax → categorical sample                          │
   │     probs = softmax(logits)                               │
   │     return categorical_sample(probs)                      │
   └──────────────────────────────────────────────────────────┘
```

**Key invariants:**

- All strategies produce valid probability distributions (non-negative,
  sum to 1) after softmax.
- Temperature must be > 0. Penalty must be >= 1.0. k must be > 0.
  p must be in (0, 1].
- Processing order is deterministic: rep_penalty → temperature → top_k → top_p.

---

### M4: `payya-transformer` — Decoder-Only Transformer

A GPT-style decoder-only transformer with multi-head causal self-attention,
feed-forward networks, pre-norm layer normalization, and residual connections.
Supports sinusoidal and RoPE positional encodings.

```
   ┌──────────────────────────────────────────────────────────┐
   │              Transformer Forward Pass                     │
   │                                                           │
   │  tokens: [t₀, t₁, ..., tₙ₋₁]                            │
   │                                                           │
   │  1. Embedding lookup                                      │
   │     x = token_emb[tokens]         (seq, d_model)          │
   │                                                           │
   │  2. Positional encoding                                   │
   │     Sinusoidal: x += PE_table[:seq]                       │
   │     RoPE: applied to Q,K inside attention                 │
   │                                                           │
   │  3. N × Transformer Block                                 │
   │     ┌────────────────────────────────────────────┐        │
   │     │  residual = x                              │        │
   │     │  x = LayerNorm(x)                          │        │
   │     │  Q = x @ Wq + bq                           │        │
   │     │  K = x @ Wk + bk      ◄── linear projs     │        │
   │     │  V = x @ Wv + bv                           │        │
   │     │  [optional: apply RoPE to Q, K]            │        │
   │     │  x = MultiHeadAttention(Q, K, V, causal)   │        │
   │     │  x = x @ Wo + bo                           │        │
   │     │  x = residual + x     ◄── skip connection  │        │
   │     │                                             │        │
   │     │  residual = x                              │        │
   │     │  x = LayerNorm(x)                          │        │
   │     │  x = ReLU(x @ W1 + b1) @ W2 + b2  ◄ FFN  │        │
   │     │  x = residual + x     ◄── skip connection  │        │
   │     └────────────────────────────────────────────┘        │
   │                                                           │
   │  4. Final LayerNorm                                       │
   │  5. Output projection: logits = x @ Wout + bout           │
   │     logits shape: (seq, vocab_size)                       │
   └──────────────────────────────────────────────────────────┘
```

**Multi-head attention detail:**

```
   Input: Q, K, V  each (seq, d_model)
   d_head = d_model / num_heads

   For each head h ∈ [0, num_heads):
     Q_h = Q[:, h*d_head : (h+1)*d_head]   (seq, d_head)
     K_h = K[:, h*d_head : (h+1)*d_head]
     V_h = V[:, h*d_head : (h+1)*d_head]

     scores = Q_h @ K_h^T / √d_head        (seq, seq)

     Causal mask:
       scores[i][j] = -∞  for j > i        ◄── prevents
                                                attending to
                                                future tokens

     attn = softmax_rows(scores)            (seq, seq)
     out_h = attn @ V_h                     (seq, d_head)

   Output = concat(out_0, ..., out_{H-1})   (seq, d_model)
```

**Training loop (SGD):**

```
   ┌──────────────────────────────────────────────────┐
   │              Training Step                        │
   │                                                   │
   │  input  = tokens[:-1]                             │
   │  target = tokens[1:]    ◄── next-token prediction │
   │                                                   │
   │  1. Forward pass → logits (seq-1, vocab)          │
   │  2. Loss = CrossEntropy(logits, target)           │
   │  3. Backward → gradients for all parameters       │
   │  4. SGD: param -= lr × grad                       │
   │                                                   │
   │  Parameters per layer:                            │
   │    Wq, Wk, Wv, Wo: (d_model, d_model)  × 4      │
   │    bq, bk, bv, bo: (d_model,)           × 4      │
   │    W1: (d_model, d_ff), W2: (d_ff, d_model)      │
   │    b1: (d_ff,), b2: (d_model,)                    │
   │    LN1 γ,β: (d_model,) × 2                       │
   │    LN2 γ,β: (d_model,) × 2                       │
   │                                                   │
   │  Global: token_emb (vocab, d_model)               │
   │          final_LN γ,β, output W,b                 │
   └──────────────────────────────────────────────────┘
```

**Autograd extensions** (added to `payya-autograd` for M4):

```
   ┌──────────────┬─────────────────────┬──────────────────────────────┐
   │ Operation    │ Forward             │ Backward (∂L/∂input)         │
   ├──────────────┼─────────────────────┼──────────────────────────────┤
   │ softmax(x)   │ row-wise softmax    │ s_i*(g_i - dot(g,s))        │
   │ layer_norm   │ (x-μ)/σ * γ + β    │ standard LN grad formula    │
   │ embedding    │ gather rows by idx  │ scatter-add to table rows   │
   │ cross_entropy│ -mean(log_softmax)  │ (softmax - one_hot) / seq   │
   │ scaled_attn  │ multi-head QKV attn │ per-head score/softmax grad │
   └──────────────┴─────────────────────┴──────────────────────────────┘

   All gradients validated against finite-difference numerical gradients.
```

**Key invariants:**

- All outputs are finite (no NaN/Inf) for valid inputs.
- Causal masking ensures position i attends only to positions ≤ i.
- Layer norm output has approximately zero mean per row.
- Cross-entropy loss is non-negative.
- Training loss decreases monotonically on repeated data (overfitting test).
- Generation produces valid token indices (< vocab_size).

---

### M5: `payya-embedding` — Sentence Embedding Model

Wraps a transformer backbone to produce fixed-size vector representations
of token sequences. Supports mean pooling and first-token (CLS-style) pooling.

```
   ┌───────────────────────────────────────────────────────┐
   │              EmbeddingModel                            │
   │                                                        │
   │  tokens: [t₀, t₁, ..., tₙ₋₁]                         │
   │                                                        │
   │  1. Transformer forward_hidden(tokens)                 │
   │     ┌──────────────────────────────────────┐           │
   │     │  Token embedding + positional encoding│           │
   │     │  N × Transformer blocks               │           │
   │     │  Final layer norm                      │           │
   │     │  → hidden states (seq, d_model)        │           │
   │     └──────────────────────────────────────┘           │
   │                                                        │
   │  2. Pooling: hidden (seq, d_model) → embedding (d_model)│
   │     ┌────────────────────────────────────────────┐     │
   │     │  Mean:       emb = mean(h₀, h₁, ..., hₙ₋₁) │     │
   │     │              emb_j = (1/n) Σᵢ h_i,j         │     │
   │     │                                              │     │
   │     │  FirstToken: emb = h₀                        │     │
   │     │              (CLS-style, first position only) │     │
   │     └────────────────────────────────────────────┘     │
   │                                                        │
   │  Output: Vec<f32> of length d_model                    │
   └───────────────────────────────────────────────────────┘
```

**Similarity functions:**

```
   cosine_similarity(a, b) = dot(a, b) / (‖a‖ × ‖b‖)

   l2_normalize(v) = v / ‖v‖

   Typical usage:
     emb_a = model.embed(tokens_a)
     emb_b = model.embed(tokens_b)
     similarity = cosine_similarity(&emb_a, &emb_b)
     // → 1.0 if identical direction, 0.0 if orthogonal, -1.0 if opposite
```

**Key invariants:**

- Embedding dimension always equals `d_model`.
- All embedding values are finite.
- Mean pooling on a single token equals first-token pooling.
- `cosine_similarity(v, v) == 1.0` for any non-zero vector.
- `l2_normalize(v)` produces unit-length vectors.

---

### M5: `payya-slm` — Small Language Model (End-to-End)

Wires together `payya-transformer`, `payya-tokenizer`, and `payya-logit-processor`
into a single trainable language model with text-in/text-out training, generation,
and checkpoint save/load.

```
   ┌────────────────────────────────────────────────────────────┐
   │                          Slm                                │
   │                                                             │
   │  ┌────────────┐  ┌──────────────┐  ┌───────────────────┐   │
   │  │ Tokenizer  │  │ Transformer  │  │ LogitProcessor    │   │
   │  │ (optional) │  │              │  │ (for generation)  │   │
   │  └─────┬──────┘  └──────┬───────┘  └────────┬──────────┘   │
   │        │                │                    │              │
   │        │   encode()     │   forward()        │  sample()    │
   │        ▼                ▼                    ▼              │
   │  "the cat" ──► [t₀,t₁] ──► logits ──► next token          │
   └────────────────────────────────────────────────────────────┘
```

**Training pipeline:**

```
   ┌────────────────────────────────────────────────────────────┐
   │              train_step_ids(tokens, config)                  │
   │                                                             │
   │  input  = tokens[:-1]                                       │
   │  target = tokens[1:]      ◄── next-token prediction         │
   │                                                             │
   │  1. Forward pass → logits (seq-1, vocab)                    │
   │  2. Loss = CrossEntropy(logits, target)                     │
   │  3. Backward → gradients for all parameters                 │
   │                                                             │
   │  4. Learning rate with warmup:                              │
   │     ┌──────────────────────────────────────────┐            │
   │     │  if step < warmup_steps:                 │            │
   │     │    lr_eff = lr × (step+1) / warmup_steps │            │
   │     │  else:                                   │            │
   │     │    lr_eff = lr                            │            │
   │     └──────────────────────────────────────────┘            │
   │                                                             │
   │  5. Gradient clipping (global norm):                        │
   │     ┌──────────────────────────────────────────┐            │
   │     │  global_norm = √(Σ grad_i²)              │            │
   │     │  if global_norm > max_norm:               │            │
   │     │    scale = max_norm / global_norm          │            │
   │     │    grad_i *= scale                         │            │
   │     └──────────────────────────────────────────┘            │
   │                                                             │
   │  6. Weight decay (decoupled, AdamW-style):                  │
   │     param *= (1 - lr × weight_decay)                        │
   │                                                             │
   │  7. SGD update:                                             │
   │     param -= lr_eff × clipped_grad                          │
   └────────────────────────────────────────────────────────────┘
```

**Text training (train_text):**

```
   corpus: "the cat sat on the mat..."
       │
       ▼
   tokenizer.encode() → [t₀, t₁, t₂, ..., tₙ]
       │
       ▼
   Sliding window chunks (window_size tokens each):
   ┌──────────────────────┐
   │  [t₀ ... t_w]        │──► train_step_ids
   │  [t₃ ... t_{w+3}]    │──► train_step_ids
   │  [t₇ ... t_{w+7}]    │──► train_step_ids
   │  ...                  │
   └──────────────────────┘
   Stride through corpus deterministically: start = (i × 7) % max_start
```

**Checkpoint format (JSON):**

```
   ┌──────────────────────────────────────┐
   │            Checkpoint                  │
   │                                        │
   │  config: SlmConfig                     │
   │    ├── vocab_size, d_model, n_heads   │
   │    ├── n_layers, d_ff, max_seq_len    │
   │                                        │
   │  params: TransformerParams             │
   │    ├── token_emb: [f32; vocab×d]      │
   │    ├── layers: [LayerParams; n_layers] │
   │    ├── final_ln_gamma, final_ln_beta  │
   │    └── output_weight, output_bias     │
   │                                        │
   │  tokenizer: Option<String>  (JSON)     │
   │  step: usize                           │
   └──────────────────────────────────────┘

   Save:  slm.checkpoint().to_bytes() → Vec<u8>
   Load:  Slm::from_checkpoint(Checkpoint::from_bytes(&bytes))

   Invariant: save → load → continue training produces no loss spike.
```

**Generation pipeline:**

```
   generate_text(prompt, max_new_tokens, processor, rng)

   "the cat" ──► tokenizer.encode() ──► [t₀, t₁]
                                              │
                    ┌─────────────────────────┘
                    ▼
               ┌─────────────────────────────────────┐
               │  loop (max_new_tokens iterations):   │
               │    forward(tokens) → logits          │
               │    last_logits = logits[-1, :]        │
               │    processor.sample(last_logits)      │
               │    tokens.push(next_token)            │
               └─────────────────────────────────────┘
                    │
                    ▼
               tokenizer.decode(tokens) ──► "the cat sat on..."
```

**Key invariants:**

- All training losses are finite (no NaN/Inf).
- Training loss decreases on repeated data (overfitting test).
- Gradient clipping keeps updates bounded even with high learning rates.
- Weight decay monotonically shrinks parameter norms.
- Checkpoint round-trip preserves exact parameter values.
- Continuing training after checkpoint restore produces no loss spike.
- Generated tokens are valid indices (< vocab_size).
- Warmup linearly increases LR from 0 to `lr` over `warmup_steps`.

---

### M6: `payya-kv-cache` — Paged KV Cache

Paged block-based KV cache inspired by vLLM. Sequences don't own contiguous
memory; instead, a `BlockAllocator` manages a pool of fixed-size blocks, and
each sequence maintains a block table mapping logical positions to physical
blocks.

```
   ┌──────────────────────────────────────────────────────────┐
   │                    PagedKvCache                           │
   │                                                           │
   │  Config:                                                  │
   │    block_size: tokens per block (e.g. 16)                 │
   │    n_layers × n_heads × d_head: dims                      │
   │                                                           │
   │  ┌──────────────────────────────────────────────────┐     │
   │  │           BlockAllocator                          │     │
   │  │  total_blocks: N                                  │     │
   │  │  free_list: Vec<usize>  (stack-based allocation)  │     │
   │  │                                                   │     │
   │  │  alloc() → Option<usize>   (pop from free list)   │     │
   │  │  free(id)                  (push to free list)    │     │
   │  └──────────────────────────────────────────────────┘     │
   │                                                           │
   │  Storage: Vec<f32>  (total_blocks × floats_per_block)     │
   │                                                           │
   │  Block layout (per block):                                │
   │  ┌─────────────────────────────────────────────────┐      │
   │  │  K data: block_size × (n_layers × n_heads × d_head)│   │
   │  │  V data: block_size × (n_layers × n_heads × d_head)│   │
   │  └─────────────────────────────────────────────────┘      │
   │                                                           │
   │  Sequences: Vec<Option<SequenceState>>                    │
   │  ┌──────────────────────┐                                 │
   │  │  SequenceState       │                                 │
   │  │    block_table: [0, 3, 7]  ◄── logical→physical map   │
   │  │    len: 40                 ◄── tokens cached           │
   │  └──────────────────────┘                                 │
   └──────────────────────────────────────────────────────────┘
```

**Data flow (append):**

```
   append(seq_id, k_data, v_data)
   ┌────────────────────────────────────────────────────┐
   │  For each new token:                               │
   │    1. Compute block index = pos / block_size       │
   │    2. If new block needed → alloc() from pool      │
   │       Err(OutOfMemory) if free_list empty          │
   │    3. Copy K data to:                              │
   │       storage[phys_block * fpb + pos_in_block * kv]│
   │    4. Copy V data to:                              │
   │       storage[phys_block * fpb + BS*kv + pos*kv]   │
   │    5. len += 1                                     │
   └────────────────────────────────────────────────────┘
```

**Key invariants:**

- Block IDs are unique across all sequences (no aliasing).
- Removing a sequence frees all its blocks back to the allocator.
- Memory is bounded: total_blocks × floats_per_block × 4 bytes.
- Repeated alloc/free cycles don't leak blocks.

---

### M6: `payya-quantization` — Int8 Post-Training Quantization

Symmetric per-tensor quantization from f32 to i8, with quantized matrix
multiplication that accumulates in i32 to avoid overflow.

```
   ┌────────────────────────────────────────────────────────────┐
   │              Quantization Pipeline                          │
   │                                                             │
   │  f32 weights                                                │
   │  [0.5, -1.2, 0.8, ...]                                     │
   │       │                                                     │
   │       ▼                                                     │
   │  1. Compute scale:                                          │
   │     abs_max = max(|x_i|)                                    │
   │     scale = abs_max / 127                                   │
   │                                                             │
   │  2. Quantize:                                               │
   │     q_i = clamp(round(x_i / scale), -128, 127)             │
   │                                                             │
   │       ▼                                                     │
   │  QuantizedTensor                                            │
   │  ┌──────────────────────────────────────┐                   │
   │  │  data: Vec<i8>     ◄── 1 byte/elem   │                   │
   │  │  scale: f32        ◄── single value   │                   │
   │  │  shape: (rows, cols)                  │                   │
   │  └──────────────────────────────────────┘                   │
   │                                                             │
   │  3. Dequantize (when needed):                               │
   │     x_i ≈ q_i × scale                                      │
   │                                                             │
   │  Memory: 1 byte/element (vs 4 bytes for f32) → ~4× savings │
   └────────────────────────────────────────────────────────────┘
```

**Quantized matmul:**

```
   quantized_matmul(A_q, B_q) → C_f32

   A_q: (m, k) i8, scale_a
   B_q: (k, n) i8, scale_b

   ┌─────────────────────────────────────────────┐
   │  For each (i, j):                           │
   │    acc: i32 = Σ_p A_q[i,p] × B_q[p,j]      │
   │    C[i,j] = acc × (scale_a × scale_b)       │
   │                                              │
   │  i32 accumulator prevents overflow:          │
   │    max accumulation = 127 × 127 × k          │
   │    fits i32 for k up to ~133,000             │
   └─────────────────────────────────────────────┘
```

**Key invariants:**

- Dequantized values preserve the sign of the original.
- Max quantization error bounded by scale / 2.
- Quantized matmul matches f32 matmul within quantization tolerance.
- Zero inputs produce zero quantized values.

---

### M6: `payya-prompt-cache` — Radix-Tree Prefix Matching

A compressed trie (radix tree) over token sequences for finding the longest
cached prefix. When multiple requests share a system prompt, the KV cache
for that prefix can be reused, reducing time-to-first-token.

```
   ┌──────────────────────────────────────────────────────┐
   │                   RadixTree                           │
   │                                                       │
   │  root: Node                                           │
   │    ├── edge [100, 101, 102] → Node (cache_id: 0)     │
   │    │     ├── edge [200, 201] → Node (cache_id: 1)    │
   │    │     └── edge [300]      → Node (cache_id: 2)    │
   │    └── edge [400, 401] → Node (cache_id: 3)          │
   │                                                       │
   │  Node { children: HashMap<TokenId, Edge>,             │
   │         cache_id: Option<CacheId> }                   │
   │                                                       │
   │  Edge { label: Vec<TokenId>,                          │
   │         child: Node }                                 │
   └──────────────────────────────────────────────────────┘
```

**Lookup algorithm:**

```
   lookup([100, 101, 102, 200, 201, 999])

   1. Match edge [100, 101, 102] → node (cache_id: 0)  ✓ best=0, len=3
   2. Match edge [200, 201]      → node (cache_id: 1)  ✓ best=1, len=5
   3. No edge starting with 999  → stop
   4. Return PrefixMatch { cache_id: 1, matched_len: 5 }

   → Only 1 new token (999) needs KV computation.
```

**Edge splitting (insert diverging path):**

```
   Before: edge [1, 2, 3, 4, 5] → node A
   Insert: [1, 2, 3, 6, 7]

   After:  edge [1, 2, 3] → split_node
             ├── edge [4, 5] → node A
             └── edge [6, 7] → node B (new, cache_id assigned)
```

**Key invariants:**

- Lookup returns the longest matching cached prefix.
- Duplicate inserts return the same cache ID.
- Removing a cache entry preserves sibling entries.
- The tree structure is always a valid radix tree (no empty edges).

---

### M6: `payya-server` — HTTP Inference Server

An axum-based HTTP server exposing an OpenAI-compatible `/v1/chat/completions`
endpoint. Supports both non-streaming JSON responses and SSE streaming.

```
   ┌────────────────────────────────────────────────────────────┐
   │                     payya-server                            │
   │                                                             │
   │  ┌──────────────────────────────────────────────────────┐   │
   │  │                    axum Router                        │   │
   │  │                                                       │   │
   │  │  GET  /health             → HealthResponse            │   │
   │  │  POST /v1/chat/completions → ChatCompletionResponse   │   │
   │  │                              or SSE stream             │   │
   │  └──────────────────────┬────────────────────────────────┘   │
   │                         │                                    │
   │  ┌──────────────────────▼────────────────────────────────┐   │
   │  │               AppState (shared)                        │   │
   │  │                                                        │   │
   │  │  engine: Mutex<InferenceEngine>                        │   │
   │  │  semaphore: Semaphore(max_concurrent)                  │   │
   │  │  model_name: String                                    │   │
   │  └──────────────────────┬────────────────────────────────┘   │
   │                         │                                    │
   │  ┌──────────────────────▼────────────────────────────────┐   │
   │  │             InferenceEngine                            │   │
   │  │                                                        │   │
   │  │  slm: Slm              ◄── language model              │   │
   │  │  prompt_cache: RadixTree ◄── prefix matching           │   │
   │  │  seed: u64              ◄── deterministic RNG          │   │
   │  └────────────────────────────────────────────────────────┘   │
   └────────────────────────────────────────────────────────────┘
```

**Request flow (non-streaming):**

```
   POST /v1/chat/completions
   { "messages": [...], "max_tokens": 128, "temperature": 0.7 }
        │
        ▼
   1. Acquire semaphore permit (bounded concurrency)
   2. Validate request (messages non-empty)
   3. Format messages → prompt string
   4. Check prompt cache → prefix hit?
        │
        ▼
   5. engine.generate(messages, max_tokens, temperature, top_p)
        │ tokenize → forward → sample → decode
        ▼
   6. Return ChatCompletionResponse {
        id: "chatcmpl-<uuid>",
        choices: [{ message: { role: "assistant", content: "..." } }],
        usage: { prompt_tokens, completion_tokens, total_tokens }
      }
```

**SSE streaming flow:**

```
   POST /v1/chat/completions  (stream: true)
        │
        ▼
   Generate full response, then stream word-by-word:

   data: {"choices":[{"delta":{"role":"assistant"}}]}
   data: {"choices":[{"delta":{"content":"Hello "}}]}
   data: {"choices":[{"delta":{"content":"world"}}]}
   data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
   data: [DONE]
```

**Key invariants:**

- Concurrent requests bounded by semaphore (default: 10).
- Empty messages rejected with 400 Bad Request.
- All responses include valid JSON with required OpenAI fields.
- SSE streams always end with `[DONE]` sentinel.
- Prompt cache accumulates entries for prefix reuse.

---

## Suggested Implementation Order

Work bottom-up through the layers for the smoothest experience:

### Phase 1 — Foundations
1. `payya-autograd` — everything else needs gradients
2. `payya-matmul` — core compute primitive
3. `payya-softmax` — used by every attention layer
4. `payya-tokenizer` — needed before any text model runs

### Phase 2 — First Model
5. `payya-flash-attention` — efficient attention
6. `payya-logit-processor` — sampling strategies
7. `payya-transformer` — compose the above into a real model
8. `payya-embedding` — useful standalone and as a building block

### Phase 3 — Make It Run
9. `payya-kv-cache` — fast autoregressive inference
10. `payya-quantization` — fit models in memory
11. `payya-server` — serve the model over HTTP

### Phase 4 — Train It
12. `payya-lora` / `payya-peft` — lightweight fine-tuning
13. `payya-dpo` — preference alignment
14. `payya-rlhf` — full RLHF loop
15. `payya-distributed` — scale training

### Phase 5 — Retrieval & Agents
16. `payya-vector-db` — similarity search
17. `payya-rag` — retrieval-augmented generation
18. `payya-reasoner` + `payya-agent` — agentic reasoning
19. `payya-function-call` + `payya-structured-output` — tool use

### Phase 6 — Expand
20. Remaining model architectures (ViT, SSM, MoE, CLIP, Diffusion, etc.)
21. Safety & eval crates
22. Speech crates
23. Infrastructure & advanced data crates
