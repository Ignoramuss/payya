//! Optimized softmax kernel implementations.
//!
//! Provides numerically stable softmax computation using the online softmax
//! trick (Milakov & Gimelshein, 2018). The key insight is that the max and
//! sum-of-exponentials can be computed in a single pass by maintaining a
//! running correction factor, avoiding a separate pass to find the maximum.
//!
//! # Algorithms
//!
//! - [`softmax`] — standard two-pass softmax: find max, then compute exp/sum.
//! - [`softmax_online`] — single-pass online softmax using running max correction.
//! - [`softmax_rows`] — apply softmax independently to each row of a matrix.
//! - [`softmax_rows_online`] — online variant for row-wise softmax.
//! - [`softmax_backward`] — gradient of softmax for reverse-mode AD.

/// Two-pass numerically stable softmax: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`.
///
/// # Panics
///
/// Panics if `logits` is empty.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    assert!(!logits.is_empty(), "softmax: input must be non-empty");

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut out = vec![0.0_f32; logits.len()];
    let mut sum = 0.0_f32;
    for (o, &x) in out.iter_mut().zip(logits.iter()) {
        let e = (x - max).exp();
        *o = e;
        sum += e;
    }

    let inv_sum = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv_sum;
    }

    out
}

/// Single-pass online softmax using the running-max correction trick.
///
/// Computes softmax without a separate pass to find the maximum. Maintains
/// a running maximum `m` and a running sum `d` of exponentials. When a new
/// maximum is encountered, the accumulated sum is corrected by multiplying
/// by `exp(old_max - new_max)`.
///
/// This produces bit-identical results to [`softmax`] in exact arithmetic.
/// In floating-point, results match within f32 epsilon.
///
/// # Panics
///
/// Panics if `logits` is empty.
pub fn softmax_online(logits: &[f32]) -> Vec<f32> {
    assert!(
        !logits.is_empty(),
        "softmax_online: input must be non-empty"
    );

    // Pass 1 (online): compute max and sum-of-exponentials in one scan.
    let mut m = f32::NEG_INFINITY; // running max
    let mut d = 0.0_f32; // running sum of exp(x_i - m)

    for &x in logits {
        let m_prev = m;
        m = m.max(x);
        // Correct the accumulated sum for the new max.
        d = d * (m_prev - m).exp() + (x - m).exp();
    }

    // Pass 2: compute output probabilities using the final max and sum.
    let inv_d = 1.0 / d;
    logits.iter().map(|&x| (x - m).exp() * inv_d).collect()
}

/// Apply softmax independently to each row of a row-major matrix.
///
/// # Panics
///
/// - Panics if `logits.len()` is not equal to `rows * cols`.
/// - Panics if `cols` is zero.
pub fn softmax_rows(logits: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        logits.len(),
        rows * cols,
        "softmax_rows: logits.len()={} but need rows*cols={}",
        logits.len(),
        rows * cols,
    );
    assert!(cols > 0, "softmax_rows: cols must be > 0");

    let mut out = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        let row = &logits[i * cols..(i + 1) * cols];
        let row_out = softmax(row);
        out[i * cols..(i + 1) * cols].copy_from_slice(&row_out);
    }
    out
}

/// Apply online softmax independently to each row of a row-major matrix.
///
/// # Panics
///
/// - Panics if `logits.len()` is not equal to `rows * cols`.
/// - Panics if `cols` is zero.
pub fn softmax_rows_online(logits: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(
        logits.len(),
        rows * cols,
        "softmax_rows_online: logits.len()={} but need rows*cols={}",
        logits.len(),
        rows * cols,
    );
    assert!(cols > 0, "softmax_rows_online: cols must be > 0");

    let mut out = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        let row = &logits[i * cols..(i + 1) * cols];
        let row_out = softmax_online(row);
        out[i * cols..(i + 1) * cols].copy_from_slice(&row_out);
    }
    out
}

/// In-place softmax over a mutable slice.
///
/// # Panics
///
/// Panics if `logits` is empty.
pub fn softmax_inplace(logits: &mut [f32]) {
    assert!(
        !logits.is_empty(),
        "softmax_inplace: input must be non-empty"
    );

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0_f32;
    for x in logits.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }

    let inv_sum = 1.0 / sum;
    for x in logits.iter_mut() {
        *x *= inv_sum;
    }
}

/// Compute the backward pass (Jacobian-vector product) of softmax.
///
/// Given the softmax output `s` and the upstream gradient `grad_output`,
/// computes the gradient with respect to the softmax input:
///
/// ```text
/// grad_input_i = s_i * (grad_output_i - dot(grad_output, s))
/// ```
///
/// This is the standard softmax backward formula derived from:
/// `dL/dx_i = sum_j (dL/ds_j * ds_j/dx_i)` where the Jacobian is
/// `ds_j/dx_i = s_j * (delta_ij - s_i)`.
///
/// # Panics
///
/// - Panics if `softmax_output` and `grad_output` have different lengths.
/// - Panics if either is empty.
pub fn softmax_backward(softmax_output: &[f32], grad_output: &[f32]) -> Vec<f32> {
    assert_eq!(
        softmax_output.len(),
        grad_output.len(),
        "softmax_backward: softmax_output.len()={} != grad_output.len()={}",
        softmax_output.len(),
        grad_output.len(),
    );
    assert!(
        !softmax_output.is_empty(),
        "softmax_backward: inputs must be non-empty"
    );

    // dot = sum(grad_output_i * s_i)
    let dot: f32 = softmax_output
        .iter()
        .zip(grad_output.iter())
        .map(|(&s, &g)| s * g)
        .sum();

    // grad_input_i = s_i * (grad_output_i - dot)
    softmax_output
        .iter()
        .zip(grad_output.iter())
        .map(|(&s, &g)| s * (g - dot))
        .collect()
}

/// Compute the backward pass of row-wise softmax.
///
/// # Panics
///
/// - Panics if `softmax_output` and `grad_output` have different lengths.
/// - Panics if lengths are not equal to `rows * cols`.
/// - Panics if `cols` is zero.
pub fn softmax_rows_backward(
    softmax_output: &[f32],
    grad_output: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    assert_eq!(
        softmax_output.len(),
        grad_output.len(),
        "softmax_rows_backward: output.len()={} != grad.len()={}",
        softmax_output.len(),
        grad_output.len(),
    );
    assert_eq!(
        softmax_output.len(),
        rows * cols,
        "softmax_rows_backward: output.len()={} but need rows*cols={}",
        softmax_output.len(),
        rows * cols,
    );
    assert!(cols > 0, "softmax_rows_backward: cols must be > 0");

    let mut grad_input = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        let s = &softmax_output[i * cols..(i + 1) * cols];
        let g = &grad_output[i * cols..(i + 1) * cols];
        let row_grad = softmax_backward(s, g);
        grad_input[i * cols..(i + 1) * cols].copy_from_slice(&row_grad);
    }
    grad_input
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "mismatch at index {}: {} vs {} (diff={})",
                i,
                x,
                y,
                (x - y).abs()
            );
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let s = softmax(&logits);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "softmax sum={}, expected 1.0", sum);
    }

    #[test]
    fn softmax_all_equal() {
        let logits = vec![5.0; 4];
        let s = softmax(&logits);
        for &p in &s {
            assert!((p - 0.25).abs() < EPS, "expected uniform 0.25, got {}", p);
        }
    }

    #[test]
    fn softmax_single_element() {
        let s = softmax(&[42.0]);
        assert!((s[0] - 1.0).abs() < EPS);
    }

    #[test]
    fn softmax_numerical_stability_large_values() {
        // Values that would cause overflow without max subtraction.
        let logits = vec![1000.0, 1001.0, 1002.0];
        let s = softmax(&logits);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "sum={}", sum);
        assert!(s.iter().all(|&p| p.is_finite()), "non-finite output");
    }

    #[test]
    fn softmax_numerical_stability_negative_values() {
        let logits = vec![-1000.0, -999.0, -998.0];
        let s = softmax(&logits);
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < EPS, "sum={}", sum);
        assert!(s.iter().all(|&p| p.is_finite()), "non-finite output");
    }

    #[test]
    fn online_matches_standard() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let standard = softmax(&logits);
        let online = softmax_online(&logits);
        assert_close(&standard, &online, EPS);
    }

    #[test]
    fn online_matches_standard_large_input() {
        // Test with a larger input to exercise the running-max correction.
        use std::f32::consts::PI;
        let logits: Vec<f32> = (0..1000)
            .map(|i| (i as f32 * PI * 0.1).sin() * 10.0)
            .collect();
        let standard = softmax(&logits);
        let online = softmax_online(&logits);
        assert_close(&standard, &online, 1e-5);
    }

    #[test]
    fn online_sums_to_one_large() {
        // Verify sum=1 for a large vector (up to 65536 as per exit criteria).
        let logits: Vec<f32> = (0..65536).map(|i| (i as f32) * 0.001 - 32.0).collect();
        let s = softmax_online(&logits);
        let sum: f32 = s.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "sum={} for 65536-element softmax",
            sum
        );
    }

    #[test]
    fn softmax_rows_correctness() {
        let logits = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        let out = softmax_rows(&logits, 2, 3);
        let row0 = softmax(&logits[0..3]);
        let row1 = softmax(&logits[3..6]);
        assert_close(&out[0..3], &row0, EPS);
        assert_close(&out[3..6], &row1, EPS);
    }

    #[test]
    fn softmax_rows_online_matches_standard() {
        let logits = vec![
            10.0, 20.0, 30.0, 40.0, // row 0
            -1.0, -2.0, -3.0, -4.0, // row 1
            0.0, 0.0, 0.0, 0.0, // row 2
        ];
        let standard = softmax_rows(&logits, 3, 4);
        let online = softmax_rows_online(&logits, 3, 4);
        assert_close(&standard, &online, EPS);
    }

    #[test]
    fn softmax_inplace_matches_functional() {
        let logits = vec![1.0, 3.0, 5.0, 7.0];
        let expected = softmax(&logits);
        let mut inplace = logits.clone();
        softmax_inplace(&mut inplace);
        assert_close(&inplace, &expected, EPS);
    }

    #[test]
    fn backward_finite_difference() {
        // Validate softmax backward against numerical finite differences.
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let s = softmax(&logits);

        // Upstream gradient: arbitrary non-uniform vector.
        let grad_output = vec![0.1, -0.2, 0.3, -0.4];
        let analytic = softmax_backward(&s, &grad_output);

        // Numerical gradient via finite differences.
        let h = 1e-4;
        let mut numerical = vec![0.0_f32; logits.len()];
        for i in 0..logits.len() {
            let mut logits_plus = logits.clone();
            let mut logits_minus = logits.clone();
            logits_plus[i] += h;
            logits_minus[i] -= h;

            let s_plus = softmax(&logits_plus);
            let s_minus = softmax(&logits_minus);

            // Compute the loss change: L = dot(grad_output, softmax(logits))
            let l_plus: f32 = s_plus
                .iter()
                .zip(grad_output.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let l_minus: f32 = s_minus
                .iter()
                .zip(grad_output.iter())
                .map(|(&a, &b)| a * b)
                .sum();

            numerical[i] = (l_plus - l_minus) / (2.0 * h);
        }

        assert_close(&analytic, &numerical, 1e-3);
    }

    #[test]
    fn backward_rows_finite_difference() {
        let logits = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
        ];
        let rows = 2;
        let cols = 3;
        let s = softmax_rows(&logits, rows, cols);

        let grad_output = vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3];
        let analytic = softmax_rows_backward(&s, &grad_output, rows, cols);

        let h = 1e-4;
        let mut numerical = vec![0.0_f32; logits.len()];
        for i in 0..logits.len() {
            let mut lp = logits.clone();
            let mut lm = logits.clone();
            lp[i] += h;
            lm[i] -= h;

            let sp = softmax_rows(&lp, rows, cols);
            let sm = softmax_rows(&lm, rows, cols);

            let lp_val: f32 = sp
                .iter()
                .zip(grad_output.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let lm_val: f32 = sm
                .iter()
                .zip(grad_output.iter())
                .map(|(&a, &b)| a * b)
                .sum();

            numerical[i] = (lp_val - lm_val) / (2.0 * h);
        }

        assert_close(&analytic, &numerical, 1e-3);
    }

    #[test]
    fn softmax_monotonicity() {
        // Larger input logit should produce larger probability.
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let s = softmax(&logits);
        // Sorted logits indices: 0(1.0) < 2(2.0) < 1(3.0) < 4(4.0) < 3(5.0)
        assert!(s[0] < s[2], "s[0]={} should be < s[2]={}", s[0], s[2]);
        assert!(s[2] < s[1], "s[2]={} should be < s[1]={}", s[2], s[1]);
        assert!(s[1] < s[4], "s[1]={} should be < s[4]={}", s[1], s[4]);
        assert!(s[4] < s[3], "s[4]={} should be < s[3]={}", s[4], s[3]);
    }

    #[test]
    fn softmax_all_outputs_non_negative() {
        let logits: Vec<f32> = (-50..50).map(|i| i as f32).collect();
        let s = softmax(&logits);
        assert!(
            s.iter().all(|&p| p >= 0.0),
            "all probabilities must be >= 0"
        );
    }

    #[test]
    #[should_panic(expected = "non-empty")]
    fn softmax_panics_on_empty() {
        softmax(&[]);
    }

    #[test]
    #[should_panic(expected = "non-empty")]
    fn softmax_online_panics_on_empty() {
        softmax_online(&[]);
    }

    #[test]
    #[should_panic(expected = "rows*cols")]
    fn softmax_rows_panics_on_shape_mismatch() {
        softmax_rows(&[1.0, 2.0, 3.0], 2, 3);
    }

    #[test]
    fn softmax_sums_to_one_various_lengths() {
        // Exit criterion: sum=1 for lengths 1 to 65536.
        for &n in &[1, 2, 3, 10, 100, 1000, 10000, 65536] {
            let logits: Vec<f32> = (0..n)
                .map(|i| (i as f32) * 0.01 - (n as f32) * 0.005)
                .collect();
            let s = softmax(&logits);
            let sum: f32 = s.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "n={}: sum={}, expected 1.0",
                n,
                sum
            );
        }
    }
}
