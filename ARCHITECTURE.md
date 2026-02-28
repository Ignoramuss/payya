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
