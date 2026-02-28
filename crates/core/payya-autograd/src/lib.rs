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
