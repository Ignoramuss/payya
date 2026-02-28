# Payya — Project Instructions

These instructions govern how code is written, reviewed, and maintained in this
repository. They apply to all contributors — human and AI.

---

## Core Engineering Principles

### 1. Invariants Over Fallbacks

**Enforce correctness through invariants, not through best-effort recovery.**

When a function receives invalid input, it must fail immediately and loudly —
via `assert!`, `panic!`, or a typed `Result::Err` — not silently produce a
"reasonable" answer. A panic during development is a gift: it tells you exactly
where the contract was violated. A silent fallback hides the bug and lets it
compound downstream.

```rust
// WRONG: silent fallback creates bimodal behavior
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    if a.len() < m * k {
        return vec![0.0; m * n]; // "safe" default — actually hides a bug
    }
    // ...
}

// RIGHT: invariant enforced at the boundary
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert!(a.len() >= m * k, "a.len()={} but need m*k={}", a.len(), m * k);
    assert!(b.len() >= k * n, "b.len()={} but need k*n={}", b.len(), k * n);
    // ...
}
```

Use `Result` when the caller can meaningfully handle the error (I/O, parsing
user input, network). Use `assert!`/`panic!` when the error means the program
has a bug (shape mismatches, index out of bounds, violated preconditions).

### 2. No Bimodal Behavior

**Every code path must be the intended code path.** If a function has two
execution modes — a "normal" path and a "fallback" path — then in production
one of them is untested. When the fallback finally activates under real load,
it will behave differently from what was tested, and the difference will be a
bug.

```rust
// WRONG: bimodal — one path is never exercised in happy-path testing
fn compute(x: &Tensor) -> Tensor {
    match try_fast_path(x) {
        Ok(result) => result,
        Err(_) => slow_fallback(x), // different algorithm, different numerics
    }
}

// RIGHT: single path, single behavior
fn compute(x: &Tensor) -> Tensor {
    fast_path(x) // panics if preconditions aren't met — fix the caller
}
```

Acceptable exceptions to this rule (and how to handle them):

- **Performance tiers** (e.g., tiled vs. naive matmul) are fine IF both paths
  produce bit-identical results and the selection is deterministic based on
  input dimensions — not on whether one path "failed." Both paths must be
  tested in CI.
- **Graceful degradation in infrastructure code** (e.g., a gateway retrying a
  different backend) is fine IF each tier is independently tested and monitored.
- **Feature gates** (`#[cfg(feature = "cuda")]`) are fine because the selection
  is compile-time, not runtime branching on failure.

### 3. No Shortcuts

**Write the real implementation, not a placeholder that "works for now."**

This project exists to understand every component from first principles.
Cutting corners defeats the purpose. Specifically:

- Do not wrap external libraries behind a thin facade and call it "from
  scratch." Depend on foundational crates (`ndarray`, `serde`, `tokio`) for
  infrastructure, but implement the algorithms yourself.
- Do not skip edge cases "because they won't happen in practice." If the type
  signature admits an input, handle it or constrain the type so it can't occur.
- Do not leave `todo!()` or `unimplemented!()` in merged code. Scaffold crates
  may have stub tests (`fn it_works() {}`), but any crate with real
  functionality must have real tests.
- Do not add `#[allow(unused)]` to suppress warnings about code that should
  exist but doesn't yet. Either implement it or remove the dead code.

### 4. Robustness Through the Type System

**Use Rust's type system to make illegal states unrepresentable.**

Prefer compile-time guarantees over runtime checks wherever practical:

- Use newtypes to distinguish things that are the same underlying type but
  have different semantics (e.g., `TensorId(usize)` instead of raw `usize`).
- Use enums to represent closed sets of options (e.g., `Op` variants instead
  of string tags).
- Use `assert!` at construction boundaries so that once an object exists, its
  invariants are guaranteed. Internal code can then skip redundant checks.

### 5. Test Against Ground Truth

**Every numerical algorithm must have a correctness oracle.**

- Gradient computations: validate against finite-difference numerical
  gradients. Every op's backward pass must be tested this way.
- Matrix operations: validate tiled/optimized implementations against a
  naive reference implementation on randomized inputs.
- Models: validate against known-good outputs from reference implementations
  (e.g., matching GPT-2 tokenization, matching PyTorch attention output on
  identical inputs).

Do not rely solely on "the loss goes down" as a correctness signal. Loss
going down can mask compensating bugs.

---

## Code Style

### Assertions and Panics

- **Public API boundaries**: use `assert!` with descriptive messages for
  precondition checks on every public function that takes dimensional
  parameters (shapes, sizes, indices).
- **Internal code**: omit redundant assertions if the invariant is already
  enforced at the public boundary. Trust the internal contract.
- **Error messages**: always include the actual vs. expected values.
  `assert_eq!(a.len(), numel, "data length must match shape")` — not just
  `assert!(a.len() == numel)`.

### Module Structure

- One crate per concept. A crate should do one thing well.
- Prefer a single `lib.rs` for small crates. Split into modules only when a
  single file exceeds ~500 lines or has clearly separable concerns.
- Keep `pub` surface area small. Expose the types and functions users need;
  keep internals private.

### Testing

- Every crate must have unit tests.
- Every crate with numerical algorithms must have property-based or
  oracle-based tests (finite differences, reference implementations).
- Integration tests (in `tests/`) for end-to-end behavior (e.g., "train an
  MLP to convergence").
- Tests must be deterministic. Use seeded RNGs for randomized tests.

### Dependencies

- Shared dependency versions are pinned in the workspace root `Cargo.toml`.
- Internal crate dependencies use `{ workspace = true }`.
- Only depend on what you need. Do not add a dependency for a single utility
  function you can write in 10 lines.
- All dependencies must pass `cargo deny` (license and vulnerability audit).

---

## Architecture Documentation

Every implemented feature must have an architecture diagram in
`ARCHITECTURE.md` under the "Implemented Feature Architectures" section.
The diagram should show:

1. **Data structures** — what the key types are and how they relate.
2. **Data flow** — how data moves through the system in the forward/hot path.
3. **Invariants** — what properties are guaranteed and where they are enforced.
4. **Backward/gradient flow** (for differentiable components) — how gradients
   propagate.

Use ASCII art for diagrams. They render everywhere, diff cleanly, and don't
require external tools.

---

## Workflow

- Run `cargo test -p <crate>` before committing changes to a crate.
- Run `cargo clippy -p <crate> -- -D warnings` — zero warnings policy.
- Run `cargo fmt --check` — formatting is enforced by pre-commit hook and CI.
- The layer-dependency linter (`scripts/check-layers.sh`) runs in CI. A crate
  may only depend on crates in the same layer or lower.
- See `ROADMAP.md` for milestone ordering and exit criteria.
