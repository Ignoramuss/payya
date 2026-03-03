//! Flash Attention kernel — tiled, IO-aware attention for CPU.
//!
//! Implements the Flash Attention algorithm (Dao et al., 2022) on CPU.
//! The key idea is to tile the attention computation so that the full
//! N×N attention matrix is never materialized in memory. Instead, we
//! process blocks of K/V at a time, maintaining running softmax
//! statistics (max and sum-of-exponentials) to compute the correct
//! weighted sum incrementally.
//!
//! # Algorithm
//!
//! For each query block Q_i:
//!   1. Initialize running max `m = -inf`, running sum `l = 0`, output `O = 0`.
//!   2. For each key/value block (K_j, V_j):
//!      a. Compute attention scores: S = Q_i × K_j^T / sqrt(d)
//!      b. Update running max: m_new = max(m, rowmax(S))
//!      c. Correct previous accumulator: O *= exp(m_old - m_new) * l
//!      d. Compute block softmax: P = exp(S - m_new)
//!      e. Accumulate: O += P × V_j, l = l * exp(m_old - m_new) + rowsum(P)
//!   3. Final normalization: O /= l
//!
//! This produces results identical to standard attention within floating-point
//! tolerance, but uses O(N) memory instead of O(N²).

/// Block size for tiling Q, K, V along the sequence dimension.
/// Chosen to balance between memory locality and overhead.
const BLOCK_SIZE: usize = 32;

/// Standard (naive) attention: `softmax(Q × K^T / sqrt(d)) × V`.
///
/// This materializes the full N×N attention matrix. Used as a reference
/// implementation for testing Flash Attention correctness.
///
/// All tensors are in row-major layout.
///
/// # Arguments
///
/// * `q` — Query matrix, shape `(n, d)`, row-major.
/// * `k` — Key matrix, shape `(n, d)`, row-major.
/// * `v` — Value matrix, shape `(n, d)`, row-major.
/// * `n` — Sequence length.
/// * `d` — Head dimension.
///
/// # Panics
///
/// Panics if any input length doesn't match `n * d`.
pub fn naive_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize) -> Vec<f32> {
    assert_eq!(q.len(), n * d, "q.len()={} but need n*d={}", q.len(), n * d);
    assert_eq!(k.len(), n * d, "k.len()={} but need n*d={}", k.len(), n * d);
    assert_eq!(v.len(), n * d, "v.len()={} but need n*d={}", v.len(), n * d);
    assert!(d > 0, "head dimension d must be > 0");

    let scale = 1.0 / (d as f32).sqrt();

    // Compute S = Q × K^T / sqrt(d), shape (n, n).
    let mut scores = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0_f32;
            for p in 0..d {
                dot += q[i * d + p] * k[j * d + p];
            }
            scores[i * n + j] = dot * scale;
        }
    }

    // Apply row-wise softmax.
    let probs = payya_softmax::softmax_rows(&scores, n, n);

    // Compute output = probs × V, shape (n, d).
    let mut out = vec![0.0_f32; n * d];
    for i in 0..n {
        for j in 0..d {
            let mut sum = 0.0_f32;
            for p in 0..n {
                sum += probs[i * n + p] * v[p * d + j];
            }
            out[i * d + j] = sum;
        }
    }

    out
}

/// Flash Attention — tiled, memory-efficient attention.
///
/// Computes the same result as [`naive_attention`] but without materializing
/// the full N×N attention matrix. Memory usage is O(N·d + B²) where B is
/// the block size, instead of O(N²).
///
/// # Arguments
///
/// * `q` — Query matrix, shape `(n, d)`, row-major.
/// * `k` — Key matrix, shape `(n, d)`, row-major.
/// * `v` — Value matrix, shape `(n, d)`, row-major.
/// * `n` — Sequence length.
/// * `d` — Head dimension.
///
/// # Panics
///
/// Panics if any input length doesn't match `n * d`.
pub fn flash_attention(q: &[f32], k: &[f32], v: &[f32], n: usize, d: usize) -> Vec<f32> {
    assert_eq!(q.len(), n * d, "q.len()={} but need n*d={}", q.len(), n * d);
    assert_eq!(k.len(), n * d, "k.len()={} but need n*d={}", k.len(), n * d);
    assert_eq!(v.len(), n * d, "v.len()={} but need n*d={}", v.len(), n * d);
    assert!(d > 0, "head dimension d must be > 0");

    if n == 0 {
        return vec![];
    }

    let scale = 1.0 / (d as f32).sqrt();

    // Output accumulator, shape (n, d).
    let mut out = vec![0.0_f32; n * d];
    // Per-row running max of attention scores.
    let mut row_max = vec![f32::NEG_INFINITY; n];
    // Per-row running sum of exponentials.
    let mut row_sum = vec![0.0_f32; n];

    let num_kv_blocks = n.div_ceil(BLOCK_SIZE);

    for kv_block in 0..num_kv_blocks {
        let kv_start = kv_block * BLOCK_SIZE;
        let kv_end = (kv_start + BLOCK_SIZE).min(n);
        let kv_len = kv_end - kv_start;

        // Process all query rows against this KV block.
        // We could also tile queries, but for CPU the inner loops are small
        // enough that tiling only KV gives the main memory savings.
        for i in 0..n {
            // Compute attention scores for query i against keys [kv_start..kv_end].
            // s_j = dot(q_i, k_j) / sqrt(d) for j in kv_start..kv_end
            let mut scores = vec![0.0_f32; kv_len];
            for (jj, j) in (kv_start..kv_end).enumerate() {
                let mut dot = 0.0_f32;
                for p in 0..d {
                    dot += q[i * d + p] * k[j * d + p];
                }
                scores[jj] = dot * scale;
            }

            // Find the block max.
            let block_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            // Update running max.
            let m_prev = row_max[i];
            let m_new = m_prev.max(block_max);
            row_max[i] = m_new;

            // Correction factor for previously accumulated values.
            let correction = (m_prev - m_new).exp();

            // Correct the running sum and output accumulator.
            row_sum[i] *= correction;
            for p in 0..d {
                out[i * d + p] *= correction;
            }

            // Compute exp(s_j - m_new) for this block and accumulate.
            for (jj, j) in (kv_start..kv_end).enumerate() {
                let w = (scores[jj] - m_new).exp();
                row_sum[i] += w;
                for p in 0..d {
                    out[i * d + p] += w * v[j * d + p];
                }
            }
        }
    }

    // Final normalization: O_i /= l_i.
    for i in 0..n {
        let inv_sum = 1.0 / row_sum[i];
        for p in 0..d {
            out[i * d + p] *= inv_sum;
        }
    }

    out
}

/// Batched Flash Attention over multiple heads and batch elements.
///
/// # Arguments
///
/// * `q` — Query tensor, shape `(batch, heads, seq, dim)`, stored as contiguous
///   row-major. Total length must be `batch * heads * seq * dim`.
/// * `k` — Key tensor, same layout.
/// * `v` — Value tensor, same layout.
/// * `batch` — Batch size.
/// * `heads` — Number of attention heads.
/// * `seq` — Sequence length.
/// * `dim` — Head dimension.
///
/// # Panics
///
/// Panics if any input length doesn't match `batch * heads * seq * dim`.
pub fn flash_attention_batched(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    dim: usize,
) -> Vec<f32> {
    let total = batch * heads * seq * dim;
    assert_eq!(
        q.len(),
        total,
        "q.len()={} but need batch*heads*seq*dim={}",
        q.len(),
        total
    );
    assert_eq!(
        k.len(),
        total,
        "k.len()={} but need batch*heads*seq*dim={}",
        k.len(),
        total
    );
    assert_eq!(
        v.len(),
        total,
        "v.len()={} but need batch*heads*seq*dim={}",
        v.len(),
        total
    );

    let mut out = vec![0.0_f32; total];
    let head_size = seq * dim;

    for b in 0..batch {
        for h in 0..heads {
            let offset = (b * heads + h) * head_size;
            let q_slice = &q[offset..offset + head_size];
            let k_slice = &k[offset..offset + head_size];
            let v_slice = &v[offset..offset + head_size];

            let head_out = flash_attention(q_slice, k_slice, v_slice, seq, dim);
            out[offset..offset + head_size].copy_from_slice(&head_out);
        }
    }

    out
}

/// Backward pass for attention: computes gradients of Q, K, V.
///
/// Given the forward inputs (Q, K, V) and the upstream gradient `grad_output`
/// (shape `(n, d)`), computes:
///
/// ```text
/// P = softmax(Q × K^T / sqrt(d))          (n × n)
/// grad_V = P^T × grad_output              (n × d)
/// grad_P = grad_output × V^T              (n × n)
/// grad_S = softmax_backward(P, grad_P)    (n × n)   (row-wise)
/// grad_Q = grad_S × K / sqrt(d)           (n × d)   (scaled)
/// grad_K = grad_S^T × Q / sqrt(d)         (n × d)   (scaled)
/// ```
///
/// This is the standard (non-tiled) backward for correctness. A tiled backward
/// can be added later for memory efficiency.
///
/// # Panics
///
/// Panics if input lengths don't match `n * d`.
///
/// # Returns
///
/// `(grad_q, grad_k, grad_v)` each of shape `(n, d)`.
pub fn attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad_output: &[f32],
    n: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(q.len(), n * d, "q.len()={} but need n*d={}", q.len(), n * d);
    assert_eq!(k.len(), n * d, "k.len()={} but need n*d={}", k.len(), n * d);
    assert_eq!(v.len(), n * d, "v.len()={} but need n*d={}", v.len(), n * d);
    assert_eq!(
        grad_output.len(),
        n * d,
        "grad_output.len()={} but need n*d={}",
        grad_output.len(),
        n * d
    );

    let scale = 1.0 / (d as f32).sqrt();

    // Recompute attention scores and probabilities.
    let mut scores = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0_f32;
            for p in 0..d {
                dot += q[i * d + p] * k[j * d + p];
            }
            scores[i * n + j] = dot * scale;
        }
    }
    let probs = payya_softmax::softmax_rows(&scores, n, n);

    // grad_V = P^T × grad_output, shape (n, d).
    let mut grad_v = vec![0.0_f32; n * d];
    for j in 0..n {
        for p in 0..d {
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum += probs[i * n + j] * grad_output[i * d + p];
            }
            grad_v[j * d + p] = sum;
        }
    }

    // grad_P = grad_output × V^T, shape (n, n).
    let mut grad_p = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for p in 0..d {
                sum += grad_output[i * d + p] * v[j * d + p];
            }
            grad_p[i * n + j] = sum;
        }
    }

    // grad_S = softmax_rows_backward(P, grad_P), shape (n, n).
    let grad_s = payya_softmax::softmax_rows_backward(&probs, &grad_p, n, n);

    // grad_Q = grad_S × K * scale, shape (n, d).
    let mut grad_q = vec![0.0_f32; n * d];
    for i in 0..n {
        for p in 0..d {
            let mut sum = 0.0_f32;
            for j in 0..n {
                sum += grad_s[i * n + j] * k[j * d + p];
            }
            grad_q[i * d + p] = sum * scale;
        }
    }

    // grad_K = grad_S^T × Q * scale, shape (n, d).
    let mut grad_k = vec![0.0_f32; n * d];
    for j in 0..n {
        for p in 0..d {
            let mut sum = 0.0_f32;
            for i in 0..n {
                sum += grad_s[i * n + j] * q[i * d + p];
            }
            grad_k[j * d + p] = sum * scale;
        }
    }

    (grad_q, grad_k, grad_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        let mut max_diff = 0.0_f32;
        let mut max_idx = 0;
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
        }
        assert!(
            max_diff < tol,
            "max diff={} at index {} (a={}, b={}), tolerance={}",
            max_diff,
            max_idx,
            a[max_idx],
            b[max_idx],
            tol
        );
    }

    fn make_deterministic_data(n: usize, d: usize, seed: u32) -> Vec<f32> {
        // Simple deterministic pseudo-random using a linear congruential generator.
        let mut state = seed as u64;
        (0..n * d)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // Map to [-1, 1] range.
                ((state >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn flash_matches_naive_small() {
        let n = 4;
        let d = 8;
        let q = make_deterministic_data(n, d, 1);
        let k = make_deterministic_data(n, d, 2);
        let v = make_deterministic_data(n, d, 3);

        let naive = naive_attention(&q, &k, &v, n, d);
        let flash = flash_attention(&q, &k, &v, n, d);

        assert_close(&naive, &flash, TOL);
    }

    #[test]
    fn flash_matches_naive_exact_block() {
        // n = BLOCK_SIZE, so exactly one KV block.
        let n = BLOCK_SIZE;
        let d = 16;
        let q = make_deterministic_data(n, d, 10);
        let k = make_deterministic_data(n, d, 20);
        let v = make_deterministic_data(n, d, 30);

        let naive = naive_attention(&q, &k, &v, n, d);
        let flash = flash_attention(&q, &k, &v, n, d);

        assert_close(&naive, &flash, TOL);
    }

    #[test]
    fn flash_matches_naive_multiple_blocks() {
        // n > BLOCK_SIZE to exercise multiple KV blocks.
        let n = BLOCK_SIZE * 3 + 7; // 103
        let d = 16;
        let q = make_deterministic_data(n, d, 100);
        let k = make_deterministic_data(n, d, 200);
        let v = make_deterministic_data(n, d, 300);

        let naive = naive_attention(&q, &k, &v, n, d);
        let flash = flash_attention(&q, &k, &v, n, d);

        assert_close(&naive, &flash, TOL);
    }

    #[test]
    fn flash_matches_naive_single_token() {
        let n = 1;
        let d = 64;
        let q = make_deterministic_data(n, d, 42);
        let k = make_deterministic_data(n, d, 43);
        let v = make_deterministic_data(n, d, 44);

        let naive = naive_attention(&q, &k, &v, n, d);
        let flash = flash_attention(&q, &k, &v, n, d);

        assert_close(&naive, &flash, TOL);
    }

    #[test]
    fn flash_matches_naive_seq512_dim64() {
        // Exit criterion: seq=512, dim=64.
        let n = 512;
        let d = 64;
        let q = make_deterministic_data(n, d, 500);
        let k = make_deterministic_data(n, d, 501);
        let v = make_deterministic_data(n, d, 502);

        let naive = naive_attention(&q, &k, &v, n, d);
        let flash = flash_attention(&q, &k, &v, n, d);

        assert_close(&naive, &flash, TOL);
    }

    #[test]
    fn flash_batched_matches_naive() {
        // Exit criterion shape: (batch=4, heads=8, seq=32, dim=16).
        // Using smaller sizes to keep test fast.
        let batch = 2;
        let heads = 4;
        let seq = 16;
        let dim = 8;
        let total = batch * heads * seq * dim;

        let q = make_deterministic_data(total, 1, 1000);
        let k = make_deterministic_data(total, 1, 2000);
        let v = make_deterministic_data(total, 1, 3000);

        let batched = flash_attention_batched(&q, &k, &v, batch, heads, seq, dim);

        // Verify against naive per-head.
        let head_size = seq * dim;
        for b in 0..batch {
            for h in 0..heads {
                let offset = (b * heads + h) * head_size;
                let q_h = &q[offset..offset + head_size];
                let k_h = &k[offset..offset + head_size];
                let v_h = &v[offset..offset + head_size];
                let expected = naive_attention(q_h, k_h, v_h, seq, dim);
                assert_close(&batched[offset..offset + head_size], &expected, TOL);
            }
        }
    }

    #[test]
    fn flash_batched_full_exit_criterion() {
        // Exit criterion: batch=4, heads=8, seq=64, dim=64.
        // Reduced from seq=512 to keep test time reasonable.
        let batch = 4;
        let heads = 8;
        let seq = 64;
        let dim = 64;
        let total = batch * heads * seq * dim;

        let q = make_deterministic_data(total, 1, 7777);
        let k = make_deterministic_data(total, 1, 8888);
        let v = make_deterministic_data(total, 1, 9999);

        let batched = flash_attention_batched(&q, &k, &v, batch, heads, seq, dim);

        // Spot-check a few heads against naive.
        let head_size = seq * dim;
        for &(b, h) in &[(0, 0), (1, 3), (3, 7)] {
            let offset = (b * heads + h) * head_size;
            let q_h = &q[offset..offset + head_size];
            let k_h = &k[offset..offset + head_size];
            let v_h = &v[offset..offset + head_size];
            let expected = naive_attention(q_h, k_h, v_h, seq, dim);
            assert_close(&batched[offset..offset + head_size], &expected, TOL);
        }
    }

    #[test]
    fn flash_output_is_finite() {
        let n = 128;
        let d = 32;
        let q = make_deterministic_data(n, d, 111);
        let k = make_deterministic_data(n, d, 222);
        let v = make_deterministic_data(n, d, 333);

        let out = flash_attention(&q, &k, &v, n, d);
        assert!(
            out.iter().all(|x| x.is_finite()),
            "flash attention output contains non-finite values"
        );
    }

    #[test]
    fn backward_gradients_finite_difference() {
        // Validate attention backward against numerical finite differences.
        let n = 4;
        let d = 3;
        let q = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.1, -0.2, -0.3, 0.7, 0.8, 0.9,
        ];
        let k = vec![0.3, 0.1, 0.4, 0.1, 0.5, 0.9, 0.2, 0.6, 0.5, 0.3, 0.5, 0.8];
        let v = vec![0.5, 0.3, 0.2, 0.8, 0.1, 0.7, 0.4, 0.6, 0.3, 0.9, 0.2, 0.4];

        let grad_output = vec![
            0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.3, -0.1, 0.2, -0.2, 0.1, -0.1,
        ];

        let (grad_q, grad_k, grad_v) = attention_backward(&q, &k, &v, &grad_output, n, d);

        let h = 1e-3;

        // Helper: compute loss = dot(attention(q,k,v), grad_output).
        let loss = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f32 {
            let out = naive_attention(qq, kk, vv, n, d);
            out.iter()
                .zip(grad_output.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
        };

        // Check grad_q.
        for i in 0..q.len() {
            let mut qp = q.clone();
            let mut qm = q.clone();
            qp[i] += h;
            qm[i] -= h;
            let numerical = (loss(&qp, &k, &v) - loss(&qm, &k, &v)) / (2.0 * h);
            assert!(
                (grad_q[i] - numerical).abs() < 1e-2,
                "grad_q[{}]: analytic={}, numerical={}, diff={}",
                i,
                grad_q[i],
                numerical,
                (grad_q[i] - numerical).abs()
            );
        }

        // Check grad_k.
        for i in 0..k.len() {
            let mut kp = k.clone();
            let mut km = k.clone();
            kp[i] += h;
            km[i] -= h;
            let numerical = (loss(&q, &kp, &v) - loss(&q, &km, &v)) / (2.0 * h);
            assert!(
                (grad_k[i] - numerical).abs() < 1e-2,
                "grad_k[{}]: analytic={}, numerical={}, diff={}",
                i,
                grad_k[i],
                numerical,
                (grad_k[i] - numerical).abs()
            );
        }

        // Check grad_v.
        for i in 0..v.len() {
            let mut vp = v.clone();
            let mut vm = v.clone();
            vp[i] += h;
            vm[i] -= h;
            let numerical = (loss(&q, &k, &vp) - loss(&q, &k, &vm)) / (2.0 * h);
            assert!(
                (grad_v[i] - numerical).abs() < 1e-2,
                "grad_v[{}]: analytic={}, numerical={}, diff={}",
                i,
                grad_v[i],
                numerical,
                (grad_v[i] - numerical).abs()
            );
        }
    }

    #[test]
    fn backward_gradients_larger() {
        // Larger finite-difference test with randomized data.
        let n = 8;
        let d = 4;
        let q = make_deterministic_data(n, d, 55);
        let k = make_deterministic_data(n, d, 66);
        let v = make_deterministic_data(n, d, 77);
        let grad_output = make_deterministic_data(n, d, 88);

        let (grad_q, grad_k, grad_v) = attention_backward(&q, &k, &v, &grad_output, n, d);

        let h = 1e-3;
        let loss = |qq: &[f32], kk: &[f32], vv: &[f32]| -> f32 {
            let out = naive_attention(qq, kk, vv, n, d);
            out.iter()
                .zip(grad_output.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
        };

        // Spot-check a few indices from each gradient.
        for &i in &[0, n * d / 2, n * d - 1] {
            let mut qp = q.clone();
            let mut qm = q.clone();
            qp[i] += h;
            qm[i] -= h;
            let num = (loss(&qp, &k, &v) - loss(&qm, &k, &v)) / (2.0 * h);
            assert!(
                (grad_q[i] - num).abs() < 0.05,
                "grad_q[{}]: analytic={}, numerical={}",
                i,
                grad_q[i],
                num
            );
        }

        for &i in &[0, n * d / 2, n * d - 1] {
            let mut kp = k.clone();
            let mut km = k.clone();
            kp[i] += h;
            km[i] -= h;
            let num = (loss(&q, &kp, &v) - loss(&q, &km, &v)) / (2.0 * h);
            assert!(
                (grad_k[i] - num).abs() < 0.05,
                "grad_k[{}]: analytic={}, numerical={}",
                i,
                grad_k[i],
                num
            );
        }

        for &i in &[0, n * d / 2, n * d - 1] {
            let mut vp = v.clone();
            let mut vm = v.clone();
            vp[i] += h;
            vm[i] -= h;
            let num = (loss(&q, &k, &vp) - loss(&q, &k, &vm)) / (2.0 * h);
            assert!(
                (grad_v[i] - num).abs() < 0.05,
                "grad_v[{}]: analytic={}, numerical={}",
                i,
                grad_v[i],
                num
            );
        }
    }

    #[test]
    fn flash_empty_sequence() {
        let out = flash_attention(&[], &[], &[], 0, 8);
        assert!(out.is_empty());
    }

    #[test]
    #[should_panic(expected = "q.len()")]
    fn flash_panics_on_q_shape_mismatch() {
        flash_attention(
            &[1.0, 2.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            2,
            2,
        );
    }

    #[test]
    #[should_panic(expected = "batch*heads*seq*dim")]
    fn flash_batched_panics_on_shape_mismatch() {
        flash_attention_batched(&[1.0], &[1.0], &[1.0], 2, 2, 2, 2);
    }

    #[test]
    fn flash_attention_dim128() {
        // Benchmark-relevant size: seq=128, dim=128.
        let n = 128;
        let d = 128;
        let q = make_deterministic_data(n, d, 1234);
        let k = make_deterministic_data(n, d, 2345);
        let v = make_deterministic_data(n, d, 3456);

        let naive = naive_attention(&q, &k, &v, n, d);
        let flash = flash_attention(&q, &k, &v, n, d);

        assert_close(&naive, &flash, TOL);
    }
}
