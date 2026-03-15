//! Post-training quantization for reduced memory and faster inference.
//!
//! Implements **symmetric per-tensor Int8 quantization**:
//!
//! ```text
//!   scale = max(|x|) / 127
//!   quantized = round(x / scale)          ← f32 → i8
//!   dequantized = quantized * scale       ← i8 → f32
//! ```
//!
//! For matrix multiplication, the quantized matmul accumulates in i32 to avoid
//! overflow, then rescales to f32 at the end. This gives ~4× memory reduction
//! and faster inner loops on integer hardware.
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────┐   quantize   ┌───────────────────────┐
//!  │ f32 data │ ───────────► │ QuantizedTensor       │
//!  │          │              │  data: Vec<i8>         │
//!  └──────────┘              │  scale: f32            │
//!       ▲                    │  shape: (rows, cols)   │
//!       │ dequantize         └───────────────────────┘
//!       └────────────────────────────┘
//! ```

// ── Quantized tensor ────────────────────────────────────────────────

/// A tensor quantized to Int8 with a per-tensor scale factor.
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Int8 quantized values.
    data: Vec<i8>,
    /// Scale factor: original_value ≈ quantized_value * scale.
    scale: f32,
    /// Shape as (rows, cols) for 2D tensors, or (1, len) for 1D.
    shape: (usize, usize),
}

impl QuantizedTensor {
    /// Quantize an f32 slice using symmetric per-tensor quantization.
    ///
    /// `shape` is `(rows, cols)` — the product must equal `data.len()`.
    pub fn quantize(data: &[f32], shape: (usize, usize)) -> Self {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "data.len()={} != rows*cols={}",
            data.len(),
            shape.0 * shape.1
        );

        let scale = compute_scale(data);
        let quantized = quantize_symmetric(data, scale);

        Self {
            data: quantized,
            scale,
            shape,
        }
    }

    /// Dequantize back to f32.
    pub fn dequantize(&self) -> Vec<f32> {
        dequantize_symmetric(&self.data, self.scale)
    }

    /// Access the raw i8 data.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// The per-tensor scale factor.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Shape as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Memory footprint in bytes (i8 data only, excluding metadata).
    pub fn memory_bytes(&self) -> usize {
        self.data.len() // 1 byte per element
    }
}

// ── Core quantization functions ─────────────────────────────────────

/// Compute the symmetric scale factor: max(|x|) / 127.
///
/// Returns a small epsilon if all values are zero to avoid division by zero.
fn compute_scale(data: &[f32]) -> f32 {
    let abs_max = data.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if abs_max == 0.0 {
        1e-10 // avoid div-by-zero
    } else {
        abs_max / 127.0
    }
}

/// Quantize f32 values to i8 using a given scale.
fn quantize_symmetric(data: &[f32], scale: f32) -> Vec<i8> {
    let inv_scale = 1.0 / scale;
    data.iter()
        .map(|&x| {
            let q = (x * inv_scale).round();
            q.clamp(-128.0, 127.0) as i8
        })
        .collect()
}

/// Dequantize i8 values back to f32.
fn dequantize_symmetric(data: &[i8], scale: f32) -> Vec<f32> {
    data.iter().map(|&q| q as f32 * scale).collect()
}

// ── Quantized matrix multiplication ─────────────────────────────────

/// Multiply two quantized matrices: A (m, k) × B (k, n) → C (m, n) in f32.
///
/// Accumulates in i32 to avoid overflow, then rescales to f32.
pub fn quantized_matmul(a: &QuantizedTensor, b: &QuantizedTensor) -> Vec<f32> {
    let (m, k_a) = a.shape;
    let (k_b, n) = b.shape;
    assert_eq!(
        k_a, k_b,
        "inner dimensions must match: a.cols={k_a} != b.rows={k_b}"
    );
    let k = k_a;

    let combined_scale = a.scale * b.scale;
    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for p in 0..k {
                acc += a.data[i * k + p] as i32 * b.data[p * n + j] as i32;
            }
            c[i * n + j] = acc as f32 * combined_scale;
        }
    }

    c
}

// ── Batch quantization utilities ────────────────────────────────────

/// Quantize a list of weight matrices (e.g., all layers of a model).
///
/// Each entry is `(data, rows, cols)`.
pub fn quantize_weights(weights: &[(&[f32], usize, usize)]) -> Vec<QuantizedTensor> {
    weights
        .iter()
        .map(|&(data, rows, cols)| QuantizedTensor::quantize(data, (rows, cols)))
        .collect()
}

/// Compute the maximum absolute error between original and dequantized values.
pub fn max_quantization_error(original: &[f32], dequantized: &[f32]) -> f32 {
    assert_eq!(
        original.len(),
        dequantized.len(),
        "length mismatch: {} vs {}",
        original.len(),
        dequantized.len()
    );
    original
        .iter()
        .zip(dequantized.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Compute memory savings ratio: original_bytes / quantized_bytes.
pub fn compression_ratio(num_elements: usize) -> f32 {
    let original = num_elements * 4; // f32 = 4 bytes
    let quantized = num_elements + 4; // i8 per element + f32 scale
    original as f32 / quantized as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_roundtrip_small_values() {
        let data = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let qt = QuantizedTensor::quantize(&data, (1, 5));
        let deq = qt.dequantize();
        let err = max_quantization_error(&data, &deq);
        // For values in [-1, 1], quantization error should be small.
        assert!(err < 0.01, "quantization error too large: {err}");
    }

    #[test]
    fn quantize_roundtrip_large_values() {
        let data: Vec<f32> = (-50..50).map(|i| i as f32 * 0.7).collect();
        let qt = QuantizedTensor::quantize(&data, (10, 10));
        let deq = qt.dequantize();
        let err = max_quantization_error(&data, &deq);
        // Absolute max is 34.3, scale = 34.3/127 ≈ 0.27.
        // Max error should be < scale/2 ≈ 0.135.
        assert!(err < 0.2, "quantization error too large: {err}");
    }

    #[test]
    fn quantize_zeros() {
        let data = vec![0.0; 16];
        let qt = QuantizedTensor::quantize(&data, (4, 4));
        assert!(qt.data.iter().all(|&x| x == 0));
        let deq = qt.dequantize();
        assert!(deq.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn quantize_memory_savings() {
        let n = 1_000_000;
        let ratio = compression_ratio(n);
        // Should be close to 4x.
        assert!(
            ratio > 3.9 && ratio < 4.01,
            "expected ~4x compression, got {ratio}"
        );
    }

    #[test]
    fn quantized_tensor_memory_bytes() {
        let data = vec![1.0f32; 1024];
        let qt = QuantizedTensor::quantize(&data, (32, 32));
        assert_eq!(qt.memory_bytes(), 1024);
    }

    #[test]
    fn quantized_matmul_identity() {
        // A = I (4x4), B = some values.
        let mut eye = vec![0.0f32; 16];
        for i in 0..4 {
            eye[i * 4 + i] = 1.0;
        }
        let b_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();

        let qa = QuantizedTensor::quantize(&eye, (4, 4));
        let qb = QuantizedTensor::quantize(&b_data, (4, 4));
        let result = quantized_matmul(&qa, &qb);

        // I * B ≈ B (with quantization error).
        for (i, (&r, &b)) in result.iter().zip(b_data.iter()).enumerate() {
            assert!(
                (r - b).abs() < 0.05,
                "element {i}: result={r}, expected≈{b}"
            );
        }
    }

    #[test]
    fn quantized_matmul_correctness() {
        // Compare quantized matmul against naive f32 matmul.
        let m = 4;
        let k = 8;
        let n = 3;

        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) - 16.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) - 12.0) * 0.15).collect();

        // Naive f32 matmul.
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    expected[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }

        let qa = QuantizedTensor::quantize(&a, (m, k));
        let qb = QuantizedTensor::quantize(&b, (k, n));
        let result = quantized_matmul(&qa, &qb);

        // Quantized result should be close to exact.
        let err = max_quantization_error(&expected, &result);
        assert!(err < 0.5, "quantized matmul error too large: {err}");
    }

    #[test]
    fn quantize_preserves_sign() {
        let data = vec![-5.0, -2.5, 0.0, 2.5, 5.0];
        let qt = QuantizedTensor::quantize(&data, (1, 5));
        let deq = qt.dequantize();
        for (orig, deq_val) in data.iter().zip(deq.iter()) {
            assert_eq!(
                orig.signum() as i32,
                deq_val.signum() as i32,
                "sign mismatch: orig={orig}, deq={deq_val}"
            );
        }
    }

    #[test]
    fn quantize_clamp_extreme() {
        // Values that exceed the i8 range after scaling should be clamped.
        let data = vec![-200.0, -100.0, 0.0, 100.0, 200.0];
        let qt = QuantizedTensor::quantize(&data, (1, 5));
        // Max magnitude = 200, scale = 200/127 ≈ 1.575
        // 200 / 1.575 ≈ 127 → should clamp to 127
        assert_eq!(*qt.data.last().unwrap(), 127);
        assert_eq!(qt.data[0], -127); // -200 / 1.575 ≈ -127
    }

    #[test]
    fn batch_quantize() {
        let w1 = vec![1.0f32; 12];
        let w2 = vec![0.5f32; 8];
        let quantized = quantize_weights(&[(&w1, 3, 4), (&w2, 2, 4)]);
        assert_eq!(quantized.len(), 2);
        assert_eq!(quantized[0].shape(), (3, 4));
        assert_eq!(quantized[1].shape(), (2, 4));
    }

    #[test]
    fn scale_computation() {
        let data = vec![-3.0, 1.0, 2.0, -0.5];
        let scale = compute_scale(&data);
        assert!((scale - 3.0 / 127.0).abs() < 1e-7);
    }
}
