//! Automatic differentiation engine for N-dimensional tensors.
//!
//! Provides a [`Graph`]-based tape-recording system for reverse-mode AD.
//! Tensors are handles ([`TensorId`]) into a graph that records operations.
//! Calling [`Graph::backward`] propagates gradients through the recorded ops.
//!
//! # Supported operations
//!
//! Element-wise: add, sub, mul, relu, sigmoid, log, exp
//! Linear algebra: matmul
//! Reductions: sum (all elements)
//! Shape: reshape, transpose
//! Neural network: softmax (row-wise), layer_norm, embedding, cross_entropy,
//!   scaled_attention (multi-head, optional causal mask)
//!
//! # Example
//!
//! ```
//! use payya_autograd::Graph;
//!
//! let mut g = Graph::new();
//! let x = g.param(&[2.0, 3.0], &[2]);
//! let y = g.param(&[4.0, 5.0], &[2]);
//! let z = g.mul(x, y);       // [8.0, 15.0]
//! let loss = g.sum(z);       // 23.0
//! g.backward(loss);
//! assert_eq!(g.grad(x), &[4.0, 5.0]); // dz/dx = y
//! assert_eq!(g.grad(y), &[2.0, 3.0]); // dz/dy = x
//! ```

// ── TensorId ────────────────────────────────────────────────────────────

/// A handle to a tensor within a [`Graph`]. Cheap to copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(usize);

impl TensorId {
    /// Create a TensorId from a raw index. Used for iterating over graph nodes.
    pub fn from_raw(index: usize) -> Self {
        Self(index)
    }

    /// Get the raw index.
    pub fn raw(self) -> usize {
        self.0
    }
}

// ── Op ──────────────────────────────────────────────────────────────────

/// Records which operation produced a tensor, and its inputs.
#[derive(Debug, Clone)]
enum Op {
    /// Leaf node — created by the user, not computed.
    Leaf,
    Add(TensorId, TensorId),
    Sub(TensorId, TensorId),
    Mul(TensorId, TensorId),
    MatMul(TensorId, TensorId),
    Relu(TensorId),
    Sigmoid(TensorId),
    Log(TensorId),
    Exp(TensorId),
    Sum(TensorId),
    Reshape(TensorId),
    Transpose(TensorId),
    /// Row-wise softmax on 2D input.
    Softmax(TensorId),
    /// Layer normalization: input (rows, cols), gamma (cols,), beta (cols,).
    LayerNorm {
        input: TensorId,
        gamma: TensorId,
        beta: TensorId,
        eps: f32,
    },
    /// Gather rows from a table by index: table (vocab, dim), indices stored here.
    Embedding {
        table: TensorId,
        indices: Vec<usize>,
    },
    /// Cross-entropy loss: logits (seq, vocab), target indices stored here → scalar.
    CrossEntropy {
        logits: TensorId,
        targets: Vec<usize>,
    },
    /// Multi-head scaled dot-product attention with optional causal mask.
    /// q, k, v are (seq, d_model). Internally splits into heads.
    ScaledAttention {
        q: TensorId,
        k: TensorId,
        v: TensorId,
        num_heads: usize,
        causal: bool,
    },
}

// ── Node ────────────────────────────────────────────────────────────────

/// A single tensor node in the computation graph.
#[derive(Debug, Clone)]
struct Node {
    data: Vec<f32>,
    shape: Vec<usize>,
    grad: Option<Vec<f32>>,
    op: Op,
    #[allow(dead_code)]
    requires_grad: bool,
}

impl Node {
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ── Graph ───────────────────────────────────────────────────────────────

/// The computation graph. All tensors live here.
///
/// Tensors are created via [`Graph::tensor`] (constant) or [`Graph::param`]
/// (trainable), and combined with operations like [`Graph::add`],
/// [`Graph::matmul`], etc. Each operation records itself on the tape.
/// [`Graph::backward`] performs reverse-mode AD.
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    /// Create an empty computation graph.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    // ── Tensor creation ─────────────────────────────────────────────

    fn push_node(&mut self, node: Node) -> TensorId {
        let id = TensorId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Create a constant tensor (does not accumulate gradients).
    pub fn tensor(&mut self, data: &[f32], shape: &[usize]) -> TensorId {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length must match shape");
        self.push_node(Node {
            data: data.to_vec(),
            shape: shape.to_vec(),
            grad: None,
            op: Op::Leaf,
            requires_grad: false,
        })
    }

    /// Create a trainable parameter (accumulates gradients).
    pub fn param(&mut self, data: &[f32], shape: &[usize]) -> TensorId {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length must match shape");
        self.push_node(Node {
            data: data.to_vec(),
            shape: shape.to_vec(),
            grad: None,
            op: Op::Leaf,
            requires_grad: true,
        })
    }

    // ── Accessors ───────────────────────────────────────────────────

    /// Get the data of a tensor.
    pub fn data(&self, id: TensorId) -> &[f32] {
        &self.nodes[id.0].data
    }

    /// Get a mutable reference to the data (for parameter updates).
    pub fn data_mut(&mut self, id: TensorId) -> &mut Vec<f32> {
        &mut self.nodes[id.0].data
    }

    /// Get the shape of a tensor.
    pub fn shape(&self, id: TensorId) -> &[usize] {
        &self.nodes[id.0].shape
    }

    /// Get the gradient of a tensor. Panics if backward hasn't been called
    /// or this tensor didn't receive gradients.
    pub fn grad(&self, id: TensorId) -> &[f32] {
        self.nodes[id.0]
            .grad
            .as_ref()
            .expect("no gradient — call backward() first")
    }

    /// Check if a gradient exists for this tensor.
    pub fn has_grad(&self, id: TensorId) -> bool {
        self.nodes[id.0].grad.is_some()
    }

    /// Return the number of nodes in the graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Check if a tensor is a trainable parameter (leaf with requires_grad=true).
    pub fn is_param(&self, id: TensorId) -> bool {
        let node = &self.nodes[id.0];
        matches!(node.op, Op::Leaf) && node.requires_grad
    }

    /// Clear all gradients (call before a new forward/backward pass).
    pub fn zero_grad(&mut self) {
        for node in &mut self.nodes {
            node.grad = None;
        }
    }

    /// Reset the graph, removing all computed (non-leaf) nodes.
    /// Returns tensor IDs for parameters that still exist (remapped).
    /// For simplicity, this just clears gradients — call between epochs.
    pub fn clear_intermediates(&mut self) {
        // Remove all non-leaf nodes (those produced by ops).
        // But this would invalidate TensorIds... so instead, just clear grads
        // and let the user rebuild the forward pass.
        // Actually, for simplicity, let's just zero the grads.
        self.zero_grad();
    }

    // ── Forward operations ──────────────────────────────────────────

    /// Element-wise addition with broadcasting.
    ///
    /// Supports: same shape, scalar broadcast, row broadcast (1D + 2D).
    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (data, shape) = self.broadcast_binary_op(a, b, |x, y| x + y);
        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::Add(a, b),
            requires_grad: true,
        })
    }

    /// Element-wise subtraction with broadcasting.
    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (data, shape) = self.broadcast_binary_op(a, b, |x, y| x - y);
        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::Sub(a, b),
            requires_grad: true,
        })
    }

    /// Element-wise multiplication with broadcasting.
    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let (data, shape) = self.broadcast_binary_op(a, b, |x, y| x * y);
        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::Mul(a, b),
            requires_grad: true,
        })
    }

    /// Matrix multiplication: (m,k) @ (k,n) → (m,n).
    ///
    /// Both inputs must be 2-dimensional.
    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let a_shape = self.nodes[a.0].shape.clone();
        let b_shape = self.nodes[b.0].shape.clone();
        assert_eq!(a_shape.len(), 2, "matmul: a must be 2D, got {:?}", a_shape);
        assert_eq!(b_shape.len(), 2, "matmul: b must be 2D, got {:?}", b_shape);
        assert_eq!(
            a_shape[1], b_shape[0],
            "matmul: shape mismatch ({:?} vs {:?})",
            a_shape, b_shape
        );

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let data = payya_matmul::matmul(&self.nodes[a.0].data, &self.nodes[b.0].data, m, k, n);

        self.push_node(Node {
            data,
            shape: vec![m, n],
            grad: None,
            op: Op::MatMul(a, b),
            requires_grad: true,
        })
    }

    /// Element-wise ReLU: max(0, x).
    pub fn relu(&mut self, a: TensorId) -> TensorId {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| x.max(0.0)).collect();
        let shape = self.nodes[a.0].shape.clone();
        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::Relu(a),
            requires_grad: true,
        })
    }

    /// Element-wise sigmoid: 1 / (1 + exp(-x)).
    pub fn sigmoid(&mut self, a: TensorId) -> TensorId {
        let data: Vec<f32> = self.nodes[a.0]
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let shape = self.nodes[a.0].shape.clone();
        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::Sigmoid(a),
            requires_grad: true,
        })
    }

    /// Element-wise natural log: ln(x).
    pub fn log(&mut self, a: TensorId) -> TensorId {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| x.ln()).collect();
        let shape = self.nodes[a.0].shape.clone();
        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::Log(a),
            requires_grad: true,
        })
    }

    /// Element-wise exponential: exp(x).
    pub fn exp(&mut self, a: TensorId) -> TensorId {
        let data: Vec<f32> = self.nodes[a.0].data.iter().map(|&x| x.exp()).collect();
        let shape = self.nodes[a.0].shape.clone();
        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::Exp(a),
            requires_grad: true,
        })
    }

    /// Sum all elements to a scalar.
    pub fn sum(&mut self, a: TensorId) -> TensorId {
        let total: f32 = self.nodes[a.0].data.iter().sum();
        self.push_node(Node {
            data: vec![total],
            shape: vec![1],
            grad: None,
            op: Op::Sum(a),
            requires_grad: true,
        })
    }

    /// Reshape a tensor. Total number of elements must remain the same.
    pub fn reshape(&mut self, a: TensorId, new_shape: &[usize]) -> TensorId {
        let numel = self.nodes[a.0].numel();
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            numel, new_numel,
            "reshape: element count mismatch ({numel} vs {new_numel})"
        );
        let data = self.nodes[a.0].data.clone();
        self.push_node(Node {
            data,
            shape: new_shape.to_vec(),
            grad: None,
            op: Op::Reshape(a),
            requires_grad: true,
        })
    }

    /// Transpose a 2D matrix.
    pub fn transpose(&mut self, a: TensorId) -> TensorId {
        let shape = &self.nodes[a.0].shape;
        assert_eq!(shape.len(), 2, "transpose: must be 2D, got {:?}", shape);
        let rows = shape[0];
        let cols = shape[1];
        let data = payya_matmul::transpose(&self.nodes[a.0].data, rows, cols);
        self.push_node(Node {
            data,
            shape: vec![cols, rows],
            grad: None,
            op: Op::Transpose(a),
            requires_grad: true,
        })
    }

    // ── Neural network operations ─────────────────────────────────

    /// Row-wise softmax on a 2D tensor: each row sums to 1.
    ///
    /// Input must be 2D (rows, cols). Output has the same shape.
    pub fn softmax(&mut self, a: TensorId) -> TensorId {
        let shape = self.nodes[a.0].shape.clone();
        assert_eq!(shape.len(), 2, "softmax: input must be 2D, got {:?}", shape);
        let rows = shape[0];
        let cols = shape[1];
        let data = payya_softmax::softmax_rows(&self.nodes[a.0].data, rows, cols);
        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::Softmax(a),
            requires_grad: true,
        })
    }

    /// Layer normalization over the last dimension.
    ///
    /// - `input`: 2D (rows, cols)
    /// - `gamma`: 1D (cols,) — scale parameter
    /// - `beta`: 1D (cols,) — shift parameter
    /// - `eps`: small constant for numerical stability
    pub fn layer_norm(
        &mut self,
        input: TensorId,
        gamma: TensorId,
        beta: TensorId,
        eps: f32,
    ) -> TensorId {
        let shape = self.nodes[input.0].shape.clone();
        assert_eq!(
            shape.len(),
            2,
            "layer_norm: input must be 2D, got {:?}",
            shape
        );
        let rows = shape[0];
        let cols = shape[1];
        assert_eq!(
            self.nodes[gamma.0].shape,
            [cols],
            "layer_norm: gamma shape must be [{cols}]"
        );
        assert_eq!(
            self.nodes[beta.0].shape,
            [cols],
            "layer_norm: beta shape must be [{cols}]"
        );

        let input_data = &self.nodes[input.0].data;
        let gamma_data = &self.nodes[gamma.0].data;
        let beta_data = &self.nodes[beta.0].data;

        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            let row = &input_data[i * cols..(i + 1) * cols];
            let mean: f32 = row.iter().sum::<f32>() / cols as f32;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
            let std = (var + eps).sqrt();
            for j in 0..cols {
                let normalized = (row[j] - mean) / std;
                data[i * cols + j] = gamma_data[j] * normalized + beta_data[j];
            }
        }

        self.push_node(Node {
            data,
            shape,
            grad: None,
            op: Op::LayerNorm {
                input,
                gamma,
                beta,
                eps,
            },
            requires_grad: true,
        })
    }

    /// Embedding lookup: gather rows from a 2D table by index.
    ///
    /// - `table`: 2D (vocab_size, dim) parameter
    /// - `indices`: which rows to select
    ///
    /// Returns a 2D tensor (indices.len(), dim).
    pub fn embedding(&mut self, table: TensorId, indices: &[usize]) -> TensorId {
        let shape = self.nodes[table.0].shape.clone();
        assert_eq!(
            shape.len(),
            2,
            "embedding: table must be 2D, got {:?}",
            shape
        );
        let vocab = shape[0];
        let dim = shape[1];
        for &idx in indices {
            assert!(
                idx < vocab,
                "embedding: index {idx} out of bounds for vocab size {vocab}"
            );
        }

        let table_data = &self.nodes[table.0].data;
        let seq_len = indices.len();
        let mut data = vec![0.0f32; seq_len * dim];
        for (i, &idx) in indices.iter().enumerate() {
            data[i * dim..(i + 1) * dim].copy_from_slice(&table_data[idx * dim..(idx + 1) * dim]);
        }

        self.push_node(Node {
            data,
            shape: vec![seq_len, dim],
            grad: None,
            op: Op::Embedding {
                table,
                indices: indices.to_vec(),
            },
            requires_grad: true,
        })
    }

    /// Cross-entropy loss: logits (seq, vocab), targets (seq indices) → scalar.
    ///
    /// Computes: -mean(log(softmax(logits)[i, target[i]]) for i in 0..seq).
    pub fn cross_entropy(&mut self, logits: TensorId, targets: &[usize]) -> TensorId {
        let shape = self.nodes[logits.0].shape.clone();
        assert_eq!(
            shape.len(),
            2,
            "cross_entropy: logits must be 2D, got {:?}",
            shape
        );
        let seq = shape[0];
        let vocab = shape[1];
        assert_eq!(
            targets.len(),
            seq,
            "cross_entropy: targets.len()={} must match seq={}",
            targets.len(),
            seq
        );
        for &t in targets {
            assert!(
                t < vocab,
                "cross_entropy: target {t} out of bounds for vocab {vocab}"
            );
        }

        let logits_data = &self.nodes[logits.0].data;
        let mut total_loss = 0.0f32;
        for i in 0..seq {
            let row = &logits_data[i * vocab..(i + 1) * vocab];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_softmax_target = (row[targets[i]] - max_val) - sum_exp.ln();
            total_loss -= log_softmax_target;
        }
        total_loss /= seq as f32;

        self.push_node(Node {
            data: vec![total_loss],
            shape: vec![1],
            grad: None,
            op: Op::CrossEntropy {
                logits,
                targets: targets.to_vec(),
            },
            requires_grad: true,
        })
    }

    /// Multi-head scaled dot-product attention.
    ///
    /// - `q`, `k`, `v`: 2D (seq, d_model) where d_model = num_heads * head_dim
    /// - `num_heads`: number of attention heads
    /// - `causal`: if true, apply causal mask (position i attends only to j <= i)
    ///
    /// Returns (seq, d_model).
    pub fn scaled_attention(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        num_heads: usize,
        causal: bool,
    ) -> TensorId {
        let q_shape = self.nodes[q.0].shape.clone();
        let k_shape = self.nodes[k.0].shape.clone();
        let v_shape = self.nodes[v.0].shape.clone();
        assert_eq!(q_shape.len(), 2, "attention: Q must be 2D");
        assert_eq!(k_shape.len(), 2, "attention: K must be 2D");
        assert_eq!(v_shape.len(), 2, "attention: V must be 2D");
        let seq = q_shape[0];
        let d_model = q_shape[1];
        assert_eq!(k_shape, [seq, d_model], "attention: K shape mismatch");
        assert_eq!(v_shape, [seq, d_model], "attention: V shape mismatch");
        assert!(
            d_model.is_multiple_of(num_heads),
            "attention: d_model={d_model} not divisible by num_heads={num_heads}"
        );
        let d_head = d_model / num_heads;
        let scale = 1.0 / (d_head as f32).sqrt();

        let q_data = &self.nodes[q.0].data;
        let k_data = &self.nodes[k.0].data;
        let v_data = &self.nodes[v.0].data;

        let mut output = vec![0.0f32; seq * d_model];

        for h in 0..num_heads {
            let col_start = h * d_head;
            // Extract per-head Q, K, V (seq, d_head) from (seq, d_model).
            let mut qh = vec![0.0f32; seq * d_head];
            let mut kh = vec![0.0f32; seq * d_head];
            let mut vh = vec![0.0f32; seq * d_head];
            for i in 0..seq {
                for j in 0..d_head {
                    qh[i * d_head + j] = q_data[i * d_model + col_start + j];
                    kh[i * d_head + j] = k_data[i * d_model + col_start + j];
                    vh[i * d_head + j] = v_data[i * d_model + col_start + j];
                }
            }
            // scores = Q_h @ K_h^T * scale → (seq, seq)
            let mut scores = vec![0.0f32; seq * seq];
            payya_matmul::matmul_a_bt(&qh, &kh, &mut scores, seq, d_head, seq);
            for s in scores.iter_mut() {
                *s *= scale;
            }
            // Apply causal mask.
            if causal {
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[i * seq + j] = f32::NEG_INFINITY;
                    }
                }
            }
            // Softmax per row.
            let attn = payya_softmax::softmax_rows(&scores, seq, seq);
            // out_h = attn @ V_h → (seq, d_head)
            let out_h = payya_matmul::matmul(&attn, &vh, seq, seq, d_head);
            // Scatter back to output columns.
            for i in 0..seq {
                for j in 0..d_head {
                    output[i * d_model + col_start + j] = out_h[i * d_head + j];
                }
            }
        }

        self.push_node(Node {
            data: output,
            shape: vec![seq, d_model],
            grad: None,
            op: Op::ScaledAttention {
                q,
                k,
                v,
                num_heads,
                causal,
            },
            requires_grad: true,
        })
    }

    // ── Broadcasting helpers ────────────────────────────────────────

    /// Apply a binary op with broadcasting support.
    /// Returns (result_data, result_shape).
    fn broadcast_binary_op(
        &self,
        a: TensorId,
        b: TensorId,
        op: impl Fn(f32, f32) -> f32,
    ) -> (Vec<f32>, Vec<usize>) {
        let a_node = &self.nodes[a.0];
        let b_node = &self.nodes[b.0];

        if a_node.shape == b_node.shape {
            // Same shape — simple element-wise.
            let data: Vec<f32> = a_node
                .data
                .iter()
                .zip(b_node.data.iter())
                .map(|(&x, &y)| op(x, y))
                .collect();
            return (data, a_node.shape.clone());
        }

        // Scalar broadcast: one side has exactly 1 element.
        if b_node.numel() == 1 {
            let scalar = b_node.data[0];
            let data: Vec<f32> = a_node.data.iter().map(|&x| op(x, scalar)).collect();
            return (data, a_node.shape.clone());
        }
        if a_node.numel() == 1 {
            let scalar = a_node.data[0];
            let data: Vec<f32> = b_node.data.iter().map(|&y| op(scalar, y)).collect();
            return (data, b_node.shape.clone());
        }

        // Row broadcast: (m, n) op (n,) or (m, n) op (1, n).
        if a_node.shape.len() == 2 {
            let cols = a_node.shape[1];
            if (b_node.shape == [cols]) || (b_node.shape == [1, cols]) {
                let rows = a_node.shape[0];
                let mut data = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    for j in 0..cols {
                        data.push(op(a_node.data[i * cols + j], b_node.data[j]));
                    }
                }
                return (data, a_node.shape.clone());
            }
        }
        if b_node.shape.len() == 2 {
            let cols = b_node.shape[1];
            if (a_node.shape == [cols]) || (a_node.shape == [1, cols]) {
                let rows = b_node.shape[0];
                let mut data = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    for j in 0..cols {
                        data.push(op(a_node.data[j], b_node.data[i * cols + j]));
                    }
                }
                return (data, b_node.shape.clone());
            }
        }

        panic!(
            "broadcast: incompatible shapes {:?} and {:?}",
            a_node.shape, b_node.shape
        );
    }

    // ── Backward pass ───────────────────────────────────────────────

    /// Compute gradients for all tensors reachable from `root` via
    /// reverse-mode automatic differentiation.
    ///
    /// `root` is typically a scalar loss tensor.
    pub fn backward(&mut self, root: TensorId) {
        let n = root.0 + 1;

        // Topological order is simply 0..n because we append in order.
        // Seed the root gradient.
        let root_numel = self.nodes[root.0].numel();
        self.nodes[root.0].grad = Some(vec![1.0; root_numel]);

        // Traverse in reverse topological order.
        for idx in (0..n).rev() {
            let grad = match self.nodes[idx].grad.clone() {
                Some(g) => g,
                None => continue,
            };

            let op = self.nodes[idx].op.clone();
            match op {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    self.backward_add(a, b, &grad, &self.nodes[idx].shape.clone());
                }
                Op::Sub(a, b) => {
                    self.backward_sub(a, b, &grad, &self.nodes[idx].shape.clone());
                }
                Op::Mul(a, b) => {
                    self.backward_mul(a, b, &grad, &self.nodes[idx].shape.clone());
                }
                Op::MatMul(a, b) => {
                    self.backward_matmul(a, b, &grad);
                }
                Op::Relu(a) => {
                    self.backward_relu(a, &grad);
                }
                Op::Sigmoid(a) => {
                    self.backward_sigmoid(idx, a, &grad);
                }
                Op::Log(a) => {
                    self.backward_log(a, &grad);
                }
                Op::Exp(_a) => {
                    self.backward_exp(idx, &grad);
                }
                Op::Sum(a) => {
                    self.backward_sum(a, &grad);
                }
                Op::Reshape(a) => {
                    self.backward_reshape(a, &grad);
                }
                Op::Transpose(a) => {
                    self.backward_transpose(a, &grad);
                }
                Op::Softmax(a) => {
                    self.backward_softmax(idx, a, &grad);
                }
                Op::LayerNorm {
                    input,
                    gamma,
                    beta,
                    eps,
                } => {
                    self.backward_layer_norm(input, gamma, beta, eps, &grad);
                }
                Op::Embedding { table, ref indices } => {
                    let indices = indices.clone();
                    self.backward_embedding(table, &indices, &grad);
                }
                Op::CrossEntropy {
                    logits,
                    ref targets,
                } => {
                    let targets = targets.clone();
                    self.backward_cross_entropy(logits, &targets, &grad);
                }
                Op::ScaledAttention {
                    q,
                    k,
                    v,
                    num_heads,
                    causal,
                } => {
                    self.backward_scaled_attention(q, k, v, num_heads, causal, &grad);
                }
            }
        }
    }

    /// Accumulate gradient into a node.
    fn accum_grad(&mut self, id: TensorId, grad: &[f32]) {
        let node = &mut self.nodes[id.0];
        match &mut node.grad {
            Some(existing) => {
                for (e, g) in existing.iter_mut().zip(grad.iter()) {
                    *e += g;
                }
            }
            None => {
                node.grad = Some(grad.to_vec());
            }
        }
    }

    /// Reduce a gradient from output_shape back to input_shape by summing
    /// over broadcast dimensions.
    fn reduce_grad_for_broadcast(
        &self,
        input_id: TensorId,
        grad: &[f32],
        output_shape: &[usize],
    ) -> Vec<f32> {
        let input_shape = &self.nodes[input_id.0].shape;
        let input_numel = self.nodes[input_id.0].numel();

        if input_shape == output_shape {
            return grad.to_vec();
        }

        // Scalar case: sum all gradients.
        if input_numel == 1 {
            return vec![grad.iter().sum()];
        }

        // Row broadcast: input is (n,) or (1,n), output is (m,n).
        // Sum grad across rows.
        if output_shape.len() == 2 {
            let rows = output_shape[0];
            let cols = output_shape[1];
            if input_shape == &[cols] || input_shape == &[1, cols] {
                let mut reduced = vec![0.0f32; cols];
                for i in 0..rows {
                    for j in 0..cols {
                        reduced[j] += grad[i * cols + j];
                    }
                }
                return reduced;
            }
        }

        grad.to_vec()
    }

    fn backward_add(&mut self, a: TensorId, b: TensorId, grad: &[f32], out_shape: &[usize]) {
        let grad_a = self.reduce_grad_for_broadcast(a, grad, out_shape);
        let grad_b = self.reduce_grad_for_broadcast(b, grad, out_shape);
        self.accum_grad(a, &grad_a);
        self.accum_grad(b, &grad_b);
    }

    fn backward_sub(&mut self, a: TensorId, b: TensorId, grad: &[f32], out_shape: &[usize]) {
        let grad_a = self.reduce_grad_for_broadcast(a, grad, out_shape);
        let neg_grad: Vec<f32> = grad.iter().map(|g| -g).collect();
        let grad_b = self.reduce_grad_for_broadcast(b, &neg_grad, out_shape);
        self.accum_grad(a, &grad_a);
        self.accum_grad(b, &grad_b);
    }

    fn backward_mul(&mut self, a: TensorId, b: TensorId, grad: &[f32], out_shape: &[usize]) {
        // d(a*b)/da = b, d(a*b)/db = a
        // But with broadcasting, we need to expand then reduce.
        let a_data = self.nodes[a.0].data.clone();
        let b_data = self.nodes[b.0].data.clone();
        let a_shape = self.nodes[a.0].shape.clone();
        let b_shape = self.nodes[b.0].shape.clone();
        let a_numel = self.nodes[a.0].numel();
        let b_numel = self.nodes[b.0].numel();

        // Compute full-size gradients, then reduce for broadcast.
        let out_numel: usize = out_shape.iter().product();

        // grad_a_full[i] = grad[i] * b_broadcast[i]
        let mut grad_a_full = vec![0.0f32; out_numel];
        let mut grad_b_full = vec![0.0f32; out_numel];

        if a_shape == b_shape {
            for i in 0..out_numel {
                grad_a_full[i] = grad[i] * b_data[i];
                grad_b_full[i] = grad[i] * a_data[i];
            }
        } else if b_numel == 1 {
            let bv = b_data[0];
            for i in 0..out_numel {
                grad_a_full[i] = grad[i] * bv;
                grad_b_full[i] = grad[i] * a_data[i];
            }
        } else if a_numel == 1 {
            let av = a_data[0];
            for i in 0..out_numel {
                grad_a_full[i] = grad[i] * b_data[i];
                grad_b_full[i] = grad[i] * av;
            }
        } else if out_shape.len() == 2 {
            let rows = out_shape[0];
            let cols = out_shape[1];
            // One of them is (cols,) broadcast to (rows, cols)
            if b_shape == [cols] || b_shape == [1, cols] {
                for i in 0..rows {
                    for (j, bj) in b_data.iter().enumerate().take(cols) {
                        let idx = i * cols + j;
                        grad_a_full[idx] = grad[idx] * bj;
                        grad_b_full[idx] = grad[idx] * a_data[idx];
                    }
                }
            } else if a_shape == [cols] || a_shape == [1, cols] {
                for i in 0..rows {
                    for (j, aj) in a_data.iter().enumerate().take(cols) {
                        let idx = i * cols + j;
                        grad_a_full[idx] = grad[idx] * b_data[idx];
                        grad_b_full[idx] = grad[idx] * aj;
                    }
                }
            }
        }

        let grad_a = self.reduce_grad_for_broadcast(a, &grad_a_full, out_shape);
        let grad_b = self.reduce_grad_for_broadcast(b, &grad_b_full, out_shape);
        self.accum_grad(a, &grad_a);
        self.accum_grad(b, &grad_b);
    }

    fn backward_matmul(&mut self, a: TensorId, b: TensorId, grad: &[f32]) {
        let a_shape = self.nodes[a.0].shape.clone();
        let b_shape = self.nodes[b.0].shape.clone();
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        // grad_a = grad_c @ B^T  (m,n) @ (n,k) → (m,k)
        let b_data = self.nodes[b.0].data.clone();
        let mut grad_a = vec![0.0f32; m * k];
        payya_matmul::matmul_a_bt(grad, &b_data, &mut grad_a, m, n, k);
        self.accum_grad(a, &grad_a);

        // grad_b = A^T @ grad_c  (k,m) @ (m,n) → (k,n)
        let a_data = self.nodes[a.0].data.clone();
        let mut grad_b = vec![0.0f32; k * n];
        payya_matmul::matmul_at_b(&a_data, grad, &mut grad_b, k, m, n);
        self.accum_grad(b, &grad_b);
    }

    fn backward_relu(&mut self, a: TensorId, grad: &[f32]) {
        let a_data = &self.nodes[a.0].data;
        let grad_a: Vec<f32> = a_data
            .iter()
            .zip(grad.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect();
        self.accum_grad(a, &grad_a);
    }

    fn backward_sigmoid(&mut self, out_idx: usize, a: TensorId, grad: &[f32]) {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) = output * (1 - output)
        let out_data = self.nodes[out_idx].data.clone();
        let grad_a: Vec<f32> = out_data
            .iter()
            .zip(grad.iter())
            .map(|(&s, &g)| g * s * (1.0 - s))
            .collect();
        self.accum_grad(a, &grad_a);
    }

    fn backward_log(&mut self, a: TensorId, grad: &[f32]) {
        let a_data = &self.nodes[a.0].data;
        let grad_a: Vec<f32> = a_data
            .iter()
            .zip(grad.iter())
            .map(|(&x, &g)| g / x)
            .collect();
        self.accum_grad(a, &grad_a);
    }

    fn backward_exp(&mut self, out_idx: usize, grad: &[f32]) {
        // exp'(x) = exp(x) = output
        let out_data = &self.nodes[out_idx].data;
        let op = self.nodes[out_idx].op.clone();
        if let Op::Exp(a) = op {
            let grad_a: Vec<f32> = out_data
                .iter()
                .zip(grad.iter())
                .map(|(&e, &g)| g * e)
                .collect();
            self.accum_grad(a, &grad_a);
        }
    }

    fn backward_sum(&mut self, a: TensorId, grad: &[f32]) {
        // d(sum(x))/dx_i = 1 for all i
        let numel = self.nodes[a.0].numel();
        let grad_a = vec![grad[0]; numel];
        self.accum_grad(a, &grad_a);
    }

    fn backward_reshape(&mut self, a: TensorId, grad: &[f32]) {
        // Gradient just gets reshaped back — data is the same.
        self.accum_grad(a, grad);
    }

    fn backward_transpose(&mut self, a: TensorId, grad: &[f32]) {
        let a_shape = &self.nodes[a.0].shape;
        let rows = a_shape[0];
        let cols = a_shape[1];
        // grad is (cols, rows) shaped (the transposed shape). Transpose it back.
        let grad_a = payya_matmul::transpose(grad, cols, rows);
        self.accum_grad(a, &grad_a);
    }

    fn backward_softmax(&mut self, out_idx: usize, a: TensorId, grad: &[f32]) {
        let shape = self.nodes[out_idx].shape.clone();
        let rows = shape[0];
        let cols = shape[1];
        let softmax_out = self.nodes[out_idx].data.clone();
        let grad_a = payya_softmax::softmax_rows_backward(&softmax_out, grad, rows, cols);
        self.accum_grad(a, &grad_a);
    }

    fn backward_layer_norm(
        &mut self,
        input: TensorId,
        gamma: TensorId,
        beta: TensorId,
        eps: f32,
        grad: &[f32],
    ) {
        let input_data = self.nodes[input.0].data.clone();
        let gamma_data = self.nodes[gamma.0].data.clone();
        let shape = self.nodes[input.0].shape.clone();
        let rows = shape[0];
        let cols = shape[1];
        let n = cols as f32;

        let mut grad_input = vec![0.0f32; rows * cols];
        let mut grad_gamma = vec![0.0f32; cols];
        let mut grad_beta = vec![0.0f32; cols];

        for i in 0..rows {
            let row = &input_data[i * cols..(i + 1) * cols];
            let g = &grad[i * cols..(i + 1) * cols];

            let mean: f32 = row.iter().sum::<f32>() / n;
            let var: f32 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
            let std = (var + eps).sqrt();
            let inv_std = 1.0 / std;

            // x_hat = (x - mean) / std
            let x_hat: Vec<f32> = row.iter().map(|&x| (x - mean) * inv_std).collect();

            // grad_beta += g, grad_gamma += g * x_hat
            for j in 0..cols {
                grad_beta[j] += g[j];
                grad_gamma[j] += g[j] * x_hat[j];
            }

            // grad_input: dx_hat = g * gamma
            let dx_hat: Vec<f32> = (0..cols).map(|j| g[j] * gamma_data[j]).collect();

            // dvar = sum(dx_hat * (x - mean)) * (-0.5) * (var + eps)^(-1.5)
            let dvar: f32 = dx_hat
                .iter()
                .zip(row.iter())
                .map(|(&dxh, &x)| dxh * (x - mean))
                .sum::<f32>()
                * (-0.5)
                * (var + eps).powf(-1.5);

            // dmean = sum(-dx_hat * inv_std) + dvar * sum(-2*(x-mean)) / n
            let dmean: f32 = dx_hat.iter().map(|&dxh| -dxh * inv_std).sum::<f32>()
                + dvar * row.iter().map(|&x| -2.0 * (x - mean)).sum::<f32>() / n;

            // dx = dx_hat * inv_std + dvar * 2*(x-mean)/n + dmean/n
            for j in 0..cols {
                grad_input[i * cols + j] =
                    dx_hat[j] * inv_std + dvar * 2.0 * (row[j] - mean) / n + dmean / n;
            }
        }

        self.accum_grad(input, &grad_input);
        self.accum_grad(gamma, &grad_gamma);
        self.accum_grad(beta, &grad_beta);
    }

    fn backward_embedding(&mut self, table: TensorId, indices: &[usize], grad: &[f32]) {
        let table_shape = self.nodes[table.0].shape.clone();
        let vocab = table_shape[0];
        let dim = table_shape[1];
        let mut grad_table = vec![0.0f32; vocab * dim];

        // Scatter-add gradients back to the embedding table rows.
        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..dim {
                grad_table[idx * dim + j] += grad[i * dim + j];
            }
        }
        self.accum_grad(table, &grad_table);
    }

    fn backward_cross_entropy(&mut self, logits: TensorId, targets: &[usize], grad: &[f32]) {
        let shape = self.nodes[logits.0].shape.clone();
        let seq = shape[0];
        let vocab = shape[1];
        let logits_data = self.nodes[logits.0].data.clone();

        // grad_logits = (softmax(logits) - one_hot(targets)) / seq * upstream_grad
        let upstream = grad[0]; // scalar loss
        let mut grad_logits = vec![0.0f32; seq * vocab];
        for i in 0..seq {
            let row = &logits_data[i * vocab..(i + 1) * vocab];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            for j in 0..vocab {
                let softmax_val = exps[j] / sum_exp;
                let target_indicator = if j == targets[i] { 1.0 } else { 0.0 };
                grad_logits[i * vocab + j] =
                    (softmax_val - target_indicator) / seq as f32 * upstream;
            }
        }
        self.accum_grad(logits, &grad_logits);
    }

    fn backward_scaled_attention(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        num_heads: usize,
        causal: bool,
        grad: &[f32],
    ) {
        let q_data = self.nodes[q.0].data.clone();
        let k_data = self.nodes[k.0].data.clone();
        let v_data = self.nodes[v.0].data.clone();
        let shape = self.nodes[q.0].shape.clone();
        let seq = shape[0];
        let d_model = shape[1];
        let d_head = d_model / num_heads;
        let scale = 1.0 / (d_head as f32).sqrt();

        let mut grad_q = vec![0.0f32; seq * d_model];
        let mut grad_k = vec![0.0f32; seq * d_model];
        let mut grad_v = vec![0.0f32; seq * d_model];

        for h in 0..num_heads {
            let col_start = h * d_head;

            // Extract per-head data.
            let mut qh = vec![0.0f32; seq * d_head];
            let mut kh = vec![0.0f32; seq * d_head];
            let mut vh = vec![0.0f32; seq * d_head];
            let mut grad_out_h = vec![0.0f32; seq * d_head];
            for i in 0..seq {
                for j in 0..d_head {
                    qh[i * d_head + j] = q_data[i * d_model + col_start + j];
                    kh[i * d_head + j] = k_data[i * d_model + col_start + j];
                    vh[i * d_head + j] = v_data[i * d_model + col_start + j];
                    grad_out_h[i * d_head + j] = grad[i * d_model + col_start + j];
                }
            }

            // Recompute forward: scores = Q_h @ K_h^T * scale
            let mut scores = vec![0.0f32; seq * seq];
            payya_matmul::matmul_a_bt(&qh, &kh, &mut scores, seq, d_head, seq);
            for s in scores.iter_mut() {
                *s *= scale;
            }
            if causal {
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        scores[i * seq + j] = f32::NEG_INFINITY;
                    }
                }
            }
            let attn = payya_softmax::softmax_rows(&scores, seq, seq);

            // grad_V_h = attn^T @ grad_out_h (seq, seq)^T @ (seq, d_head) → (seq, d_head)
            let attn_t = payya_matmul::transpose(&attn, seq, seq);
            let gv_h = payya_matmul::matmul(&attn_t, &grad_out_h, seq, seq, d_head);

            // grad_attn = grad_out_h @ V_h^T (seq, d_head) @ (d_head, seq) → (seq, seq)
            let mut grad_attn = vec![0.0f32; seq * seq];
            payya_matmul::matmul_a_bt(&grad_out_h, &vh, &mut grad_attn, seq, d_head, seq);

            // grad_scores = softmax_backward(attn, grad_attn) per row
            let grad_scores = payya_softmax::softmax_rows_backward(&attn, &grad_attn, seq, seq);

            // Apply causal mask to gradient (masked positions get zero gradient).
            let mut grad_scores_masked = grad_scores;
            if causal {
                for i in 0..seq {
                    for j in (i + 1)..seq {
                        grad_scores_masked[i * seq + j] = 0.0;
                    }
                }
            }

            // grad_Q_h = grad_scores @ K_h * scale (seq, seq) @ (seq, d_head) → (seq, d_head)
            let gq_h = payya_matmul::matmul(&grad_scores_masked, &kh, seq, seq, d_head);

            // grad_K_h = grad_scores^T @ Q_h * scale (seq, seq)^T @ (seq, d_head) → (seq, d_head)
            let grad_scores_t = payya_matmul::transpose(&grad_scores_masked, seq, seq);
            let gk_h = payya_matmul::matmul(&grad_scores_t, &qh, seq, seq, d_head);

            // Scatter per-head gradients back into full d_model gradients.
            for i in 0..seq {
                for j in 0..d_head {
                    grad_q[i * d_model + col_start + j] += gq_h[i * d_head + j] * scale;
                    grad_k[i * d_model + col_start + j] += gk_h[i * d_head + j] * scale;
                    grad_v[i * d_model + col_start + j] += gv_h[i * d_head + j];
                }
            }
        }

        self.accum_grad(q, &grad_q);
        self.accum_grad(k, &grad_k);
        self.accum_grad(v, &grad_v);
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerical gradient via finite differences.
    fn numerical_grad(
        build_loss: impl Fn(&mut Graph, TensorId) -> TensorId,
        data: &[f32],
        shape: &[usize],
    ) -> Vec<f32> {
        let eps = 1e-3; // Larger eps for f32 to reduce cancellation error
        let mut grads = vec![0.0f32; data.len()];

        for i in 0..data.len() {
            let mut data_plus = data.to_vec();
            data_plus[i] += eps;
            let mut g1 = Graph::new();
            let x = g1.param(&data_plus, shape);
            let loss = build_loss(&mut g1, x);
            let l1 = g1.data(loss)[0];

            let mut data_minus = data.to_vec();
            data_minus[i] -= eps;
            let mut g2 = Graph::new();
            let x = g2.param(&data_minus, shape);
            let loss = build_loss(&mut g2, x);
            let l2 = g2.data(loss)[0];

            grads[i] = (l1 - l2) / (2.0 * eps);
        }
        grads
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, msg: &str) {
        assert_eq!(a.len(), b.len(), "{msg}: length mismatch");
        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            // Use both absolute and relative tolerance for f32 robustness.
            let abs_diff = (av - bv).abs();
            let scale = av.abs().max(bv.abs()).max(1e-8);
            let rel_diff = abs_diff / scale;
            assert!(
                abs_diff < tol || rel_diff < tol,
                "{msg}: mismatch at [{i}]: analytic={av}, numerical={bv}, \
                 abs_diff={abs_diff}, rel_diff={rel_diff}"
            );
        }
    }

    // ── Basic forward tests ─────────────────────────────────────────

    #[test]
    fn test_add_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[1.0, 2.0, 3.0], &[3]);
        let b = g.tensor(&[4.0, 5.0, 6.0], &[3]);
        let c = g.add(a, b);
        assert_eq!(g.data(c), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[2.0, 3.0], &[2]);
        let b = g.tensor(&[4.0, 5.0], &[2]);
        let c = g.mul(a, b);
        assert_eq!(g.data(c), &[8.0, 15.0]);
    }

    #[test]
    fn test_matmul_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = g.tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = g.matmul(a, b);
        assert_eq!(g.data(c), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_relu_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[-1.0, 0.0, 1.0, 2.0], &[4]);
        let b = g.relu(a);
        assert_eq!(g.data(b), &[0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[0.0], &[1]);
        let b = g.sigmoid(a);
        assert!((g.data(b)[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sum_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let s = g.sum(a);
        assert_eq!(g.data(s), &[10.0]);
    }

    #[test]
    fn test_reshape_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = g.reshape(a, &[3, 2]);
        assert_eq!(g.shape(b), &[3, 2]);
        assert_eq!(g.data(b), g.data(a));
    }

    #[test]
    fn test_transpose_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = g.transpose(a);
        assert_eq!(g.shape(b), &[3, 2]);
        assert_eq!(g.data(b), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // ── Gradient tests (against finite differences) ─────────────────

    #[test]
    fn test_grad_add() {
        let data = [1.0f32, 2.0, 3.0];
        let build = |g: &mut Graph, x: TensorId| {
            let c = g.tensor(&[10.0, 20.0, 30.0], &[3]);
            let s = g.add(x, c);
            g.sum(s)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[3]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[3]);
        assert_close(&analytic, &numerical, 0.02, "add grad");
    }

    #[test]
    fn test_grad_mul() {
        let data = [2.0f32, 3.0, 4.0];
        let build = |g: &mut Graph, x: TensorId| {
            let c = g.tensor(&[5.0, 6.0, 7.0], &[3]);
            let s = g.mul(x, c);
            g.sum(s)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[3]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[3]);
        assert_close(&analytic, &numerical, 0.02, "mul grad");
    }

    #[test]
    fn test_grad_matmul() {
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let build = |g: &mut Graph, a: TensorId| {
            let b = g.tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
            let c = g.matmul(a, b);
            g.sum(c)
        };

        let mut g = Graph::new();
        let a = g.param(&a_data, &[2, 2]);
        let loss = build(&mut g, a);
        g.backward(loss);

        let analytic = g.grad(a).to_vec();
        let numerical = numerical_grad(build, &a_data, &[2, 2]);
        assert_close(&analytic, &numerical, 0.02, "matmul grad (A)");
    }

    #[test]
    fn test_grad_matmul_b() {
        let b_data = [5.0f32, 6.0, 7.0, 8.0];
        let build = |g: &mut Graph, b: TensorId| {
            let a = g.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            let c = g.matmul(a, b);
            g.sum(c)
        };

        let mut g = Graph::new();
        let b = g.param(&b_data, &[2, 2]);
        let loss = build(&mut g, b);
        g.backward(loss);

        let analytic = g.grad(b).to_vec();
        let numerical = numerical_grad(build, &b_data, &[2, 2]);
        assert_close(&analytic, &numerical, 0.02, "matmul grad (B)");
    }

    #[test]
    fn test_grad_relu() {
        let data = [-1.0f32, 0.5, -0.3, 2.0];
        let build = |g: &mut Graph, x: TensorId| {
            let r = g.relu(x);
            g.sum(r)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[4]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[4]);
        assert_close(&analytic, &numerical, 0.02, "relu grad");
    }

    #[test]
    fn test_grad_sigmoid() {
        let data = [-1.0f32, 0.0, 1.0, 2.0];
        let build = |g: &mut Graph, x: TensorId| {
            let s = g.sigmoid(x);
            g.sum(s)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[4]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[4]);
        assert_close(&analytic, &numerical, 0.02, "sigmoid grad");
    }

    #[test]
    fn test_grad_log() {
        let data = [1.0f32, 2.0, 3.0, 0.5];
        let build = |g: &mut Graph, x: TensorId| {
            let l = g.log(x);
            g.sum(l)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[4]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[4]);
        assert_close(&analytic, &numerical, 0.02, "log grad");
    }

    #[test]
    fn test_grad_exp() {
        let data = [0.0f32, 0.5, -1.0, 1.0];
        let build = |g: &mut Graph, x: TensorId| {
            let e = g.exp(x);
            g.sum(e)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[4]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[4]);
        assert_close(&analytic, &numerical, 0.02, "exp grad");
    }

    #[test]
    fn test_grad_transpose() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let build = |g: &mut Graph, x: TensorId| {
            let t = g.transpose(x);
            g.sum(t)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[2, 3]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[2, 3]);
        assert_close(&analytic, &numerical, 0.02, "transpose grad");
    }

    #[test]
    fn test_grad_reshape() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let build = |g: &mut Graph, x: TensorId| {
            let r = g.reshape(x, &[3, 2]);
            g.sum(r)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[2, 3]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[2, 3]);
        assert_close(&analytic, &numerical, 0.02, "reshape grad");
    }

    #[test]
    fn test_grad_sub() {
        let data = [1.0f32, 2.0, 3.0];
        let build = |g: &mut Graph, x: TensorId| {
            let c = g.tensor(&[10.0, 20.0, 30.0], &[3]);
            let s = g.sub(c, x);
            g.sum(s)
        };

        let mut g = Graph::new();
        let x = g.param(&data, &[3]);
        let loss = build(&mut g, x);
        g.backward(loss);

        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[3]);
        assert_close(&analytic, &numerical, 0.02, "sub grad");
    }

    // ── Broadcast gradient tests ────────────────────────────────────

    #[test]
    fn test_grad_add_broadcast_row() {
        let data = [1.0f32, 2.0, 3.0];
        let build = |g: &mut Graph, b: TensorId| {
            let a = g.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let c = g.add(a, b);
            g.sum(c)
        };

        let mut g = Graph::new();
        let b = g.param(&data, &[3]);
        let loss = build(&mut g, b);
        g.backward(loss);

        let analytic = g.grad(b).to_vec();
        let numerical = numerical_grad(build, &data, &[3]);
        assert_close(&analytic, &numerical, 0.02, "add broadcast grad");
    }

    #[test]
    fn test_grad_mul_broadcast_scalar() {
        let data = [3.0f32];
        let build = |g: &mut Graph, s: TensorId| {
            let a = g.tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
            let c = g.mul(a, s);
            g.sum(c)
        };

        let mut g = Graph::new();
        let s = g.param(&data, &[1]);
        let loss = build(&mut g, s);
        g.backward(loss);

        let analytic = g.grad(s).to_vec();
        let numerical = numerical_grad(build, &data, &[1]);
        assert_close(&analytic, &numerical, 0.02, "mul scalar broadcast grad");
    }

    // ── Composite gradient tests ────────────────────────────────────

    #[test]
    fn test_grad_chain_relu_matmul() {
        let w_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let build = |g: &mut Graph, w: TensorId| {
            let x = g.tensor(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.5], &[2, 3]);
            let h = g.matmul(x, w);
            let h = g.relu(h);
            g.sum(h)
        };

        let mut g = Graph::new();
        let w = g.param(&w_data, &[3, 2]);
        let loss = build(&mut g, w);
        g.backward(loss);

        let analytic = g.grad(w).to_vec();
        let numerical = numerical_grad(build, &w_data, &[3, 2]);
        assert_close(&analytic, &numerical, 0.02, "chain relu+matmul grad");
    }

    #[test]
    fn test_grad_mlp_pattern() {
        // Mini MLP: relu(x @ W + b) summed — tests matmul + broadcast add + relu
        let w_data = [0.1f32, -0.2, 0.3, 0.4, -0.1, 0.5];
        let build = |g: &mut Graph, w: TensorId| {
            let x = g.tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
            let h = g.matmul(x, w); // (2,2) @ (2,3) = (2,3)
            let b = g.tensor(&[0.1, 0.2, 0.3], &[3]);
            let h = g.add(h, b);
            let h = g.relu(h);
            g.sum(h)
        };

        let mut g = Graph::new();
        let w = g.param(&w_data, &[2, 3]);
        let loss = build(&mut g, w);
        g.backward(loss);

        let analytic = g.grad(w).to_vec();
        let numerical = numerical_grad(build, &w_data, &[2, 3]);
        assert_close(&analytic, &numerical, 0.02, "MLP pattern grad");
    }

    // ── New op tests ─────────────────────────────────────────────

    #[test]
    fn test_softmax_forward() {
        let mut g = Graph::new();
        let a = g.tensor(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
        let s = g.softmax(a);
        let data = g.data(s);
        // Each row should sum to 1.
        let row1_sum: f32 = data[0..3].iter().sum();
        let row2_sum: f32 = data[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-5);
        assert!((row2_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_grad_softmax() {
        let data = [1.0f32, 2.0, 0.5, -1.0, 0.0, 1.0];
        let build = |g: &mut Graph, x: TensorId| {
            let s = g.softmax(x);
            g.sum(s)
        };
        let mut g = Graph::new();
        let x = g.param(&data, &[2, 3]);
        let loss = build(&mut g, x);
        g.backward(loss);
        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[2, 3]);
        assert_close(&analytic, &numerical, 0.02, "softmax grad");
    }

    #[test]
    fn test_layer_norm_forward() {
        let mut g = Graph::new();
        let input = g.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let gamma = g.tensor(&[1.0, 1.0, 1.0], &[3]);
        let beta = g.tensor(&[0.0, 0.0, 0.0], &[3]);
        let out = g.layer_norm(input, gamma, beta, 1e-5);
        let data = g.data(out);
        // With gamma=1, beta=0: each row should have mean≈0, std≈1.
        for i in 0..2 {
            let row = &data[i * 3..(i + 1) * 3];
            let mean: f32 = row.iter().sum::<f32>() / 3.0;
            assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
        }
    }

    #[test]
    fn test_grad_layer_norm_input() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let build = |g: &mut Graph, x: TensorId| {
            let gamma = g.tensor(&[0.5, 1.0, 1.5], &[3]);
            let beta = g.tensor(&[0.1, 0.2, 0.3], &[3]);
            let out = g.layer_norm(x, gamma, beta, 1e-5);
            g.sum(out)
        };
        let mut g = Graph::new();
        let x = g.param(&data, &[2, 3]);
        let loss = build(&mut g, x);
        g.backward(loss);
        let analytic = g.grad(x).to_vec();
        let numerical = numerical_grad(build, &data, &[2, 3]);
        assert_close(&analytic, &numerical, 0.05, "layer_norm input grad");
    }

    #[test]
    fn test_grad_layer_norm_gamma() {
        let gamma_data = [0.5f32, 1.0, 1.5];
        let build = |g: &mut Graph, gamma: TensorId| {
            let x = g.tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
            let beta = g.tensor(&[0.1, 0.2, 0.3], &[3]);
            let out = g.layer_norm(x, gamma, beta, 1e-5);
            g.sum(out)
        };
        let mut g = Graph::new();
        let gamma = g.param(&gamma_data, &[3]);
        let loss = build(&mut g, gamma);
        g.backward(loss);
        let analytic = g.grad(gamma).to_vec();
        let numerical = numerical_grad(build, &gamma_data, &[3]);
        assert_close(&analytic, &numerical, 0.05, "layer_norm gamma grad");
    }

    #[test]
    fn test_embedding_forward() {
        let mut g = Graph::new();
        // 4-word vocab, dim=3
        let table = g.tensor(
            &[
                0.1, 0.2, 0.3, // word 0
                0.4, 0.5, 0.6, // word 1
                0.7, 0.8, 0.9, // word 2
                1.0, 1.1, 1.2, // word 3
            ],
            &[4, 3],
        );
        let out = g.embedding(table, &[2, 0, 3]);
        assert_eq!(g.shape(out), &[3, 3]);
        assert_eq!(g.data(out), &[0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2]);
    }

    #[test]
    fn test_grad_embedding() {
        let table_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let build = |g: &mut Graph, table: TensorId| {
            let emb = g.embedding(table, &[1, 3, 1]);
            g.sum(emb)
        };
        let mut g = Graph::new();
        let table = g.param(&table_data, &[4, 3]);
        let loss = build(&mut g, table);
        g.backward(loss);
        let analytic = g.grad(table).to_vec();
        let numerical = numerical_grad(build, &table_data, &[4, 3]);
        assert_close(&analytic, &numerical, 0.02, "embedding grad");
    }

    #[test]
    fn test_cross_entropy_forward() {
        let mut g = Graph::new();
        // 2 positions, 3 classes. Target: [1, 0].
        let logits = g.tensor(&[1.0, 2.0, 0.5, 3.0, 1.0, 0.0], &[2, 3]);
        let loss = g.cross_entropy(logits, &[1, 0]);
        let val = g.data(loss)[0];
        // Should be a positive loss value.
        assert!(val > 0.0, "cross-entropy loss must be positive, got {val}");
    }

    #[test]
    fn test_grad_cross_entropy() {
        let data = [1.0f32, 2.0, 0.5, 3.0, 1.0, 0.0, -1.0, 0.5, 2.0];
        let build = |g: &mut Graph, logits: TensorId| g.cross_entropy(logits, &[1, 0, 2]);
        let mut g = Graph::new();
        let logits = g.param(&data, &[3, 3]);
        let loss = build(&mut g, logits);
        g.backward(loss);
        let analytic = g.grad(logits).to_vec();
        let numerical = numerical_grad(build, &data, &[3, 3]);
        assert_close(&analytic, &numerical, 0.02, "cross_entropy grad");
    }

    #[test]
    fn test_scaled_attention_single_head() {
        let mut g = Graph::new();
        // seq=3, d_model=4, 1 head
        let q = g.tensor(
            &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            &[3, 4],
        );
        let k = g.tensor(
            &[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            &[3, 4],
        );
        let v = g.tensor(
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
            &[3, 4],
        );
        let out = g.scaled_attention(q, k, v, 1, false);
        assert_eq!(g.shape(out), &[3, 4]);
        // Output should be finite.
        for &val in g.data(out) {
            assert!(val.is_finite(), "attention output must be finite");
        }
    }

    #[test]
    fn test_scaled_attention_causal() {
        let mut g = Graph::new();
        // With causal masking, first position can only attend to itself.
        let q = g.tensor(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let k = g.tensor(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let v = g.tensor(&[1.0, 0.0, 2.0, 0.0, 3.0, 0.0], &[3, 2]);
        let out = g.scaled_attention(q, k, v, 1, true);
        // First row should be exactly v[0] since it only attends to itself.
        let data = g.data(out);
        assert!(
            (data[0] - 1.0).abs() < 1e-5 && (data[1] - 0.0).abs() < 1e-5,
            "first position with causal mask should attend only to itself: [{}, {}]",
            data[0],
            data[1]
        );
    }

    #[test]
    fn test_grad_scaled_attention_q() {
        let q_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let build = |g: &mut Graph, q: TensorId| {
            let k = g.tensor(&[0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7], &[4, 2]);
            let v = g.tensor(&[0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8], &[4, 2]);
            let out = g.scaled_attention(q, k, v, 1, false);
            g.sum(out)
        };
        let mut g = Graph::new();
        let q = g.param(&q_data, &[4, 2]);
        let loss = build(&mut g, q);
        g.backward(loss);
        let analytic = g.grad(q).to_vec();
        let numerical = numerical_grad(build, &q_data, &[4, 2]);
        assert_close(&analytic, &numerical, 0.05, "attention grad Q");
    }

    #[test]
    fn test_grad_scaled_attention_v() {
        let v_data = [0.1f32, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8];
        let build = |g: &mut Graph, v: TensorId| {
            let q = g.tensor(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], &[4, 2]);
            let k = g.tensor(&[0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7], &[4, 2]);
            let out = g.scaled_attention(q, k, v, 1, false);
            g.sum(out)
        };
        let mut g = Graph::new();
        let v = g.param(&v_data, &[4, 2]);
        let loss = build(&mut g, v);
        g.backward(loss);
        let analytic = g.grad(v).to_vec();
        let numerical = numerical_grad(build, &v_data, &[4, 2]);
        assert_close(&analytic, &numerical, 0.05, "attention grad V");
    }

    #[test]
    fn test_grad_scaled_attention_causal() {
        let q_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let build = |g: &mut Graph, q: TensorId| {
            let k = g.tensor(&[0.2, 0.1, 0.4, 0.3, 0.6, 0.5], &[3, 2]);
            let v = g.tensor(&[0.1, 0.3, 0.5, 0.7, 0.2, 0.4], &[3, 2]);
            let out = g.scaled_attention(q, k, v, 1, true);
            g.sum(out)
        };
        let mut g = Graph::new();
        let q = g.param(&q_data, &[3, 2]);
        let loss = build(&mut g, q);
        g.backward(loss);
        let analytic = g.grad(q).to_vec();
        let numerical = numerical_grad(build, &q_data, &[3, 2]);
        assert_close(&analytic, &numerical, 0.05, "causal attention grad Q");
    }

    #[test]
    fn test_grad_rectangular_matmul() {
        // (3,2) @ (2,4) = (3,4), test grad on both sides
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

        // Grad w.r.t. A
        let build_a = |g: &mut Graph, a: TensorId| {
            let b = g.tensor(&b_data, &[2, 4]);
            let c = g.matmul(a, b);
            g.sum(c)
        };
        let mut g = Graph::new();
        let a = g.param(&a_data, &[3, 2]);
        let loss = build_a(&mut g, a);
        g.backward(loss);
        let analytic_a = g.grad(a).to_vec();
        let numerical_a = numerical_grad(build_a, &a_data, &[3, 2]);
        assert_close(&analytic_a, &numerical_a, 0.02, "rect matmul grad A");

        // Grad w.r.t. B
        let build_b = |g: &mut Graph, b: TensorId| {
            let a = g.tensor(&a_data, &[3, 2]);
            let c = g.matmul(a, b);
            g.sum(c)
        };
        let mut g = Graph::new();
        let b = g.param(&b_data, &[2, 4]);
        let loss = build_b(&mut g, b);
        g.backward(loss);
        let analytic_b = g.grad(b).to_vec();
        let numerical_b = numerical_grad(build_b, &b_data, &[2, 4]);
        assert_close(&analytic_b, &numerical_b, 0.02, "rect matmul grad B");
    }
}
