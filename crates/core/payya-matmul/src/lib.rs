//! High-performance matrix multiplication kernels.
//!
//! Provides tiled, cache-friendly GEMM (General Matrix Multiply) for `f32`.
//! The core operation computes C = A * B where:
//! - A is (m × k)
//! - B is (k × n)
//! - C is (m × n)
//!
//! All matrices are stored in row-major order as flat slices.

/// Tile size for the cache-friendly tiled GEMM.
/// Chosen to fit three tiles (A, B, C) in L1 cache (~32KB).
/// 3 * 32 * 32 * 4 bytes = 12KB, well within L1.
const TILE: usize = 32;

/// Multiply two matrices: C = A * B.
///
/// - `a`: row-major slice of shape (m × k)
/// - `b`: row-major slice of shape (k × n)
/// - `m`: number of rows in A (and C)
/// - `k`: shared dimension (cols of A, rows of B)
/// - `n`: number of cols in B (and C)
///
/// # Panics
///
/// Panics if `a.len() < m * k` or `b.len() < k * n`.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert!(
        a.len() >= m * k,
        "a.len()={} but need m*k={}",
        a.len(),
        m * k
    );
    assert!(
        b.len() >= k * n,
        "b.len()={} but need k*n={}",
        b.len(),
        k * n
    );

    let mut c = vec![0.0f32; m * n];
    matmul_into(a, b, &mut c, m, k, n);
    c
}

/// Multiply two matrices, accumulating into an existing buffer: C += A * B.
///
/// Same layout as [`matmul`] but writes into a pre-allocated output.
///
/// # Panics
///
/// Panics if any buffer is too small.
pub fn matmul_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    assert!(a.len() >= m * k);
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);

    // For very small matrices, use the naive kernel to avoid tiling overhead.
    if m <= TILE && k <= TILE && n <= TILE {
        naive_matmul(a, b, c, m, k, n);
        return;
    }

    tiled_matmul(a, b, c, m, k, n);
}

/// Naive triple-loop matmul. Used as fallback for small matrices
/// and as the inner kernel within each tile.
#[inline]
fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
}

/// Tiled (blocked) GEMM for cache efficiency.
///
/// Tiles over all three dimensions (i, j, p) so that the working set
/// of each inner kernel fits in L1 cache. The `p` (reduction) dimension
/// is tiled in the outermost loop to maximize reuse of C-tile values
/// in registers.
fn tiled_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // Tile over the reduction dimension outermost for C-tile register reuse.
    for p0 in (0..k).step_by(TILE) {
        let p_end = (p0 + TILE).min(k);
        for i0 in (0..m).step_by(TILE) {
            let i_end = (i0 + TILE).min(m);
            for j0 in (0..n).step_by(TILE) {
                let j_end = (j0 + TILE).min(n);
                // Inner micro-kernel over the tile.
                for i in i0..i_end {
                    for p in p0..p_end {
                        let a_ip = a[i * k + p];
                        for j in j0..j_end {
                            c[i * n + j] += a_ip * b[p * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// Compute the transpose of a row-major matrix.
///
/// Input: `a` of shape (rows × cols), output: shape (cols × rows).
pub fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert!(a.len() >= rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

/// Matmul with A transposed: C += A^T * B.
///
/// `a` is stored as (k × m) row-major, but treated as (m × k).
pub fn matmul_at_b(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // a is (k × m) row-major, we want A^T which is (m × k)
    assert!(a.len() >= k * m);
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);

    for p in 0..k {
        for i in 0..m {
            let a_val = a[p * m + i]; // a[p][i] in (k×m) layout = A^T[i][p]
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

/// Matmul with B transposed: C += A * B^T.
///
/// `b` is stored as (n × k) row-major, but treated as (k × n).
pub fn matmul_a_bt(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    // b is (n × k) row-major, we want B^T which is (k × n)
    assert!(a.len() >= m * k);
    assert!(b.len() >= n * k);
    assert!(c.len() >= m * n);

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[j * k + p]; // b[j][p] in (n×k) layout = B^T[p][j]
            }
            c[i * n + j] += sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matmul() {
        // 2×2 identity * 2×2 matrix = same matrix
        let eye = [1.0, 0.0, 0.0, 1.0];
        let a = [1.0, 2.0, 3.0, 4.0];
        let c = matmul(&eye, &a, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_small_matmul() {
        // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
        // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let c = matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_rectangular_matmul() {
        // (2×3) * (3×2)
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = matmul(&a, &b, 2, 3, 2);
        // row 0: 1*7+2*9+3*11=7+18+33=58, 1*8+2*10+3*12=8+20+36=64
        // row 1: 4*7+5*9+6*11=28+45+66=139, 4*8+5*10+6*12=32+50+72=154
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_tiled_large_matmul() {
        // Test with a matrix larger than TILE to exercise tiled path.
        let n = 65;
        let a: Vec<f32> = (0..n * n).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i % 5) as f32 - 2.0).collect();

        // Compare tiled against naive
        let mut c_naive = vec![0.0f32; n * n];
        naive_matmul(&a, &b, &mut c_naive, n, n, n);

        let c_tiled = matmul(&a, &b, n, n, n);

        for (i, (a_val, b_val)) in c_naive.iter().zip(c_tiled.iter()).enumerate() {
            assert!(
                (a_val - b_val).abs() < 1e-3,
                "mismatch at {i}: naive={a_val}, tiled={b_val}"
            );
        }
    }

    #[test]
    fn test_transpose() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
        let t = transpose(&a, 2, 3); // 3×2
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matmul_at_b() {
        // A^T * B where A is (3×2) stored, so A^T is (2×3)
        // B is (3×2)
        let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // (3×2) row-major
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // (3×2) row-major
        let mut c = vec![0.0f32; 4]; // (2×2)
        matmul_at_b(&a, &b, &mut c, 2, 3, 2);
        // A^T = [1 2 3; 4 5 6], B = [7 8; 9 10; 11 12]
        // A^T * B = [1*7+2*9+3*11  1*8+2*10+3*12] = [58 64]
        //           [4*7+5*9+6*11  4*8+5*10+6*12]   [139 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_a_bt() {
        // A * B^T where A is (2×3), B stored as (2×3) so B^T is (3×2)
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2×3)
        let b = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0]; // (2×3), B^T = [7 8; 9 10; 11 12]
        let mut c = vec![0.0f32; 4]; // (2×2)
        matmul_a_bt(&a, &b, &mut c, 2, 3, 2);
        // A * B^T = A * [7 8; 9 10; 11 12]
        // [1*7+2*9+3*11  1*8+2*10+3*12] = [58 64]
        // [4*7+5*9+6*11  4*8+5*10+6*12]   [139 154]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_vector_matmul() {
        // (1×3) * (3×1) = (1×1)
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let c = matmul(&a, &b, 1, 3, 1);
        assert_eq!(c, vec![32.0]); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_large_random_matmul_consistency() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let m = 128;
        let k = 96;
        let n = 64;

        let a: Vec<f32> = (0..m * k).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let mut c_naive = vec![0.0f32; m * n];
        naive_matmul(&a, &b, &mut c_naive, m, k, n);

        let c_tiled = matmul(&a, &b, m, k, n);

        for (i, (nv, tv)) in c_naive.iter().zip(c_tiled.iter()).enumerate() {
            assert!(
                (nv - tv).abs() < 1e-2,
                "mismatch at {i}: naive={nv}, tiled={tv}"
            );
        }
    }
}
