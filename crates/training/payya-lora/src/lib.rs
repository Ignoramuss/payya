//! LoRA (Low-Rank Adaptation) for transformer fine-tuning.
//!
//! Freezes the base model weights and injects trainable low-rank adapter
//! matrices (A, B) into specified projection layers. For a frozen weight W
//! of shape (d_in, d_out), the adapter computes: output = x @ W + (alpha/r) * x @ B @ A
//! where B is (d_in, r) and A is (r, d_out).
//!
//! # Example
//!
//! ```
//! use payya_lora::{LoraConfig, LoraModel, LoraTarget};
//! use payya_transformer::{Transformer, TransformerConfig, PosEncoding};
//! use rand::SeedableRng;
//!
//! let config = TransformerConfig {
//!     vocab_size: 32, d_model: 16, n_heads: 2,
//!     n_layers: 2, d_ff: 32, max_seq_len: 64,
//!     pos_encoding: PosEncoding::Sinusoidal,
//! };
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//! let base = Transformer::new(config, &mut rng);
//! let lora_config = LoraConfig { rank: 4, alpha: 8.0, targets: vec![LoraTarget::Wq, LoraTarget::Wv] };
//! let mut model = LoraModel::new(base, lora_config, &mut rng);
//! let loss = model.train_step(&[0, 1, 2, 3], 0.01);
//! assert!(loss.is_finite());
//! ```

use payya_autograd::{Graph, TensorId};
use payya_transformer::{LayerParams, PosEncoding, Transformer};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ── Configuration ────────────────────────────────────────────────────

/// Which weight matrices to apply LoRA adapters to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoraTarget {
    /// Query projection (d_model × d_model).
    Wq,
    /// Key projection (d_model × d_model).
    Wk,
    /// Value projection (d_model × d_model).
    Wv,
    /// Output projection (d_model × d_model).
    Wo,
    /// FFN first layer (d_model × d_ff).
    W1,
    /// FFN second layer (d_ff × d_model).
    W2,
}

/// LoRA hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    /// Rank of the low-rank adapters.
    pub rank: usize,
    /// Scaling factor. The adapter contribution is scaled by alpha / rank.
    pub alpha: f32,
    /// Which weight matrices to adapt.
    pub targets: Vec<LoraTarget>,
}

impl LoraConfig {
    /// The scaling factor applied to adapter outputs.
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }
}

// ── Adapter parameters ──────────────────────────────────────────────

/// Low-rank adapter matrices for a single target weight.
/// For weight W of shape (d_in, d_out):
///   B is (d_in, rank), A is (rank, d_out).
///   delta = B @ A, scaled by alpha / rank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraAdapter {
    /// Down projection: (d_in, rank). Initialized with Kaiming uniform.
    pub b_matrix: Vec<f32>,
    /// Up projection: (rank, d_out). Initialized to zeros.
    pub a_matrix: Vec<f32>,
    pub d_in: usize,
    pub d_out: usize,
    pub rank: usize,
}

impl LoraAdapter {
    /// Create a new adapter with Kaiming-style initialization for B, zeros for A.
    fn new(d_in: usize, d_out: usize, rank: usize, rng: &mut impl Rng) -> Self {
        assert!(rank > 0, "LoRA rank must be > 0, got {rank}");
        assert!(
            rank <= d_in.min(d_out),
            "LoRA rank={rank} must be <= min(d_in={d_in}, d_out={d_out})"
        );
        // B: Kaiming uniform for fan_in = d_in
        let std = (1.0 / d_in as f32).sqrt();
        let b_matrix: Vec<f32> = (0..d_in * rank).map(|_| rng.gen_range(-std..std)).collect();
        // A: zeros so the initial adapter contribution is zero.
        let a_matrix = vec![0.0f32; rank * d_out];
        Self {
            b_matrix,
            a_matrix,
            d_in,
            d_out,
            rank,
        }
    }

    /// Number of trainable parameters in this adapter.
    pub fn num_params(&self) -> usize {
        self.d_in * self.rank + self.rank * self.d_out
    }
}

/// All LoRA adapters for one transformer layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAdapters {
    pub wq: Option<LoraAdapter>,
    pub wk: Option<LoraAdapter>,
    pub wv: Option<LoraAdapter>,
    pub wo: Option<LoraAdapter>,
    pub w1: Option<LoraAdapter>,
    pub w2: Option<LoraAdapter>,
}

/// Serializable LoRA checkpoint: config + adapter weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraCheckpoint {
    pub lora_config: LoraConfig,
    pub layer_adapters: Vec<LayerAdapters>,
}

impl LoraCheckpoint {
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("LoRA checkpoint serialization failed")
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        serde_json::from_slice(bytes).expect("LoRA checkpoint deserialization failed")
    }
}

// ── LoRA Model ──────────────────────────────────────────────────────

/// A transformer with frozen base weights and trainable LoRA adapters.
pub struct LoraModel {
    /// The base transformer (weights are frozen during LoRA training).
    base: Transformer,
    /// LoRA configuration.
    pub config: LoraConfig,
    /// Per-layer adapter parameters.
    pub layer_adapters: Vec<LayerAdapters>,
}

impl LoraModel {
    /// Create a LoRA model from a base transformer.
    pub fn new(base: Transformer, config: LoraConfig, rng: &mut impl Rng) -> Self {
        assert!(!config.targets.is_empty(), "LoRA targets must not be empty");
        assert!(config.rank > 0, "LoRA rank must be > 0");

        let d = base.config.d_model;
        let ff = base.config.d_ff;
        let n_layers = base.config.n_layers;

        let mut layer_adapters = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let mut make = |d_in: usize, d_out: usize, target: LoraTarget| -> Option<LoraAdapter> {
                if config.targets.contains(&target) {
                    Some(LoraAdapter::new(d_in, d_out, config.rank, rng))
                } else {
                    None
                }
            };
            layer_adapters.push(LayerAdapters {
                wq: make(d, d, LoraTarget::Wq),
                wk: make(d, d, LoraTarget::Wk),
                wv: make(d, d, LoraTarget::Wv),
                wo: make(d, d, LoraTarget::Wo),
                w1: make(d, ff, LoraTarget::W1),
                w2: make(ff, d, LoraTarget::W2),
            });
        }

        Self {
            base,
            config,
            layer_adapters,
        }
    }

    /// Create a LoRA model from a base transformer and a checkpoint.
    pub fn from_checkpoint(base: Transformer, checkpoint: LoraCheckpoint) -> Self {
        assert_eq!(
            checkpoint.layer_adapters.len(),
            base.config.n_layers,
            "checkpoint layer count {} != model layer count {}",
            checkpoint.layer_adapters.len(),
            base.config.n_layers
        );
        Self {
            base,
            config: checkpoint.lora_config,
            layer_adapters: checkpoint.layer_adapters,
        }
    }

    /// Save adapter weights to a checkpoint (does not include base model).
    pub fn checkpoint(&self) -> LoraCheckpoint {
        LoraCheckpoint {
            lora_config: self.config.clone(),
            layer_adapters: self.layer_adapters.clone(),
        }
    }

    /// Total number of trainable parameters across all adapters.
    pub fn num_trainable_params(&self) -> usize {
        self.layer_adapters
            .iter()
            .map(|la| {
                [&la.wq, &la.wk, &la.wv, &la.wo, &la.w1, &la.w2]
                    .iter()
                    .filter_map(|a| a.as_ref())
                    .map(|a| a.num_params())
                    .sum::<usize>()
            })
            .sum()
    }

    /// Total number of base model parameters.
    pub fn num_base_params(&self) -> usize {
        let cfg = &self.base.config;
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let v = cfg.vocab_size;
        // token_emb + per-layer + final_ln + output
        v * d
            + cfg.n_layers * (4 * d * d + 4 * d + 2 * d + 2 * d + d * ff + ff * d + ff + d + 2 * d)
            + 2 * d
            + d * v
            + v
    }

    /// Access the base transformer.
    pub fn base(&self) -> &Transformer {
        &self.base
    }

    /// Run a forward pass through the LoRA-adapted model.
    /// Base weights are registered as constants (no gradients).
    /// Adapter weights are registered as parameters (trainable).
    /// Returns (graph, logits_id) where logits is (seq, vocab_size).
    pub fn forward(&self, tokens: &[usize]) -> (Graph, TensorId) {
        let cfg = &self.base.config;
        let seq = tokens.len();
        assert!(
            seq <= cfg.max_seq_len,
            "sequence length {seq} exceeds max_seq_len {}",
            cfg.max_seq_len
        );
        assert!(seq > 0, "tokens must not be empty");
        let d = cfg.d_model;

        let mut g = Graph::new();

        // Token embeddings (frozen).
        let emb_table = g.tensor(&self.base.params.token_emb, &[cfg.vocab_size, d]);
        let mut x = g.embedding(emb_table, tokens);

        // Positional encoding.
        if cfg.pos_encoding == PosEncoding::Sinusoidal {
            let pe = sinusoidal_encoding(seq, d);
            let pe_id = g.tensor(&pe, &[seq, d]);
            x = g.add(x, pe_id);
        }

        // Transformer layers with LoRA.
        for (layer_idx, layer) in self.base.params.layers.iter().enumerate() {
            x = self.forward_layer(&mut g, x, layer, &self.layer_adapters[layer_idx], seq);
        }

        // Final layer norm (frozen).
        let final_gamma = g.tensor(&self.base.params.final_ln_gamma, &[d]);
        let final_beta = g.tensor(&self.base.params.final_ln_beta, &[d]);
        x = g.layer_norm(x, final_gamma, final_beta, 1e-5);

        // Output projection (frozen).
        let out_w = g.tensor(&self.base.params.output_weight, &[d, cfg.vocab_size]);
        let out_b = g.tensor(&self.base.params.output_bias, &[cfg.vocab_size]);
        let logits = g.matmul(x, out_w);
        let logits = g.add(logits, out_b);

        (g, logits)
    }

    /// Forward through one transformer layer with LoRA adapters.
    fn forward_layer(
        &self,
        g: &mut Graph,
        x: TensorId,
        layer: &LayerParams,
        adapters: &LayerAdapters,
        _seq: usize,
    ) -> TensorId {
        let d = self.base.config.d_model;
        let ff = self.base.config.d_ff;
        let scale = self.config.scale();

        // Pre-norm for attention (frozen).
        let ln1_gamma = g.tensor(&layer.ln1_gamma, &[d]);
        let ln1_beta = g.tensor(&layer.ln1_beta, &[d]);
        let normed = g.layer_norm(x, ln1_gamma, ln1_beta, 1e-5);

        // Q, K, V projections with LoRA.
        let q = self.linear_with_lora(g, normed, &layer.wq, &layer.bq, d, d, &adapters.wq, scale);
        let k = self.linear_with_lora(g, normed, &layer.wk, &layer.bk, d, d, &adapters.wk, scale);
        let v = self.linear_with_lora(g, normed, &layer.wv, &layer.bv, d, d, &adapters.wv, scale);

        // Multi-head attention with causal mask.
        let wo_base = g.tensor(&layer.wo, &[d, d]);
        let bo = g.tensor(&layer.bo, &[d]);
        let attn_out = g.scaled_attention(q, k, v, self.base.config.n_heads, true);

        // Output projection with LoRA.
        let base_proj = g.matmul(attn_out, wo_base);
        let mut out = g.add(base_proj, bo);

        if let Some(adapter) = &adapters.wo {
            let lora_delta = self.lora_matmul(g, attn_out, adapter, scale);
            out = g.add(out, lora_delta);
        }

        // Residual connection.
        let x = g.add(x, out);

        // Pre-norm for FFN (frozen).
        let ln2_gamma = g.tensor(&layer.ln2_gamma, &[d]);
        let ln2_beta = g.tensor(&layer.ln2_beta, &[d]);
        let normed2 = g.layer_norm(x, ln2_gamma, ln2_beta, 1e-5);

        // FFN with LoRA on W1 and W2.
        let w1_base = g.tensor(&layer.w1, &[d, ff]);
        let b1 = g.tensor(&layer.b1, &[ff]);
        let mut h = g.matmul(normed2, w1_base);
        h = g.add(h, b1);
        if let Some(adapter) = &adapters.w1 {
            let lora_delta = self.lora_matmul(g, normed2, adapter, scale);
            h = g.add(h, lora_delta);
        }
        h = g.relu(h);

        let w2_base = g.tensor(&layer.w2, &[ff, d]);
        let b2 = g.tensor(&layer.b2, &[d]);
        let mut ffn_out = g.matmul(h, w2_base);
        ffn_out = g.add(ffn_out, b2);
        if let Some(adapter) = &adapters.w2 {
            let lora_delta = self.lora_matmul(g, h, adapter, scale);
            ffn_out = g.add(ffn_out, lora_delta);
        }

        // Residual connection.
        g.add(x, ffn_out)
    }

    /// Compute x @ W + bias, optionally adding LoRA delta.
    #[allow(clippy::too_many_arguments)]
    fn linear_with_lora(
        &self,
        g: &mut Graph,
        x: TensorId,
        weight: &[f32],
        bias: &[f32],
        d_in: usize,
        d_out: usize,
        adapter: &Option<LoraAdapter>,
        scale: f32,
    ) -> TensorId {
        let w = g.tensor(weight, &[d_in, d_out]);
        let b = g.tensor(bias, &[d_out]);
        let mut out = g.matmul(x, w);
        out = g.add(out, b);

        if let Some(adapter) = adapter {
            let lora_delta = self.lora_matmul(g, x, adapter, scale);
            out = g.add(out, lora_delta);
        }

        out
    }

    /// Compute x @ B @ A * scale. B and A are trainable parameters.
    fn lora_matmul(
        &self,
        g: &mut Graph,
        x: TensorId,
        adapter: &LoraAdapter,
        scale: f32,
    ) -> TensorId {
        let r = adapter.rank;
        let b = g.param(&adapter.b_matrix, &[adapter.d_in, r]);
        let a = g.param(&adapter.a_matrix, &[r, adapter.d_out]);

        // x @ B @ A * scale
        let xb = g.matmul(x, b);
        let xba = g.matmul(xb, a);

        // Scale by alpha / rank.
        let scale_data = vec![scale];
        let scale_id = g.tensor(&scale_data, &[1]);
        g.mul(xba, scale_id)
    }

    /// Run one training step. Only adapter parameters receive gradients.
    /// Returns the loss value.
    pub fn train_step(&mut self, tokens: &[usize], lr: f32) -> f32 {
        self.train_step_with_lr(tokens, lr)
    }

    /// Collect mutable references to all adapter parameter vectors in graph order.
    fn collect_adapter_refs_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut refs: Vec<&mut Vec<f32>> = Vec::new();
        for la in &mut self.layer_adapters {
            // Order must match forward_layer: wq, wk, wv adapters first,
            // then wo, then w1, w2. Within each adapter: B then A.
            for a in [&mut la.wq, &mut la.wk, &mut la.wv].into_iter().flatten() {
                refs.push(&mut a.b_matrix);
                refs.push(&mut a.a_matrix);
            }
            if let Some(a) = &mut la.wo {
                refs.push(&mut a.b_matrix);
                refs.push(&mut a.a_matrix);
            }
            if let Some(a) = &mut la.w1 {
                refs.push(&mut a.b_matrix);
                refs.push(&mut a.a_matrix);
            }
            if let Some(a) = &mut la.w2 {
                refs.push(&mut a.b_matrix);
                refs.push(&mut a.a_matrix);
            }
        }
        refs
    }

    /// Merge LoRA adapters back into the base model weights.
    /// After merging, the model behaves identically but no longer needs adapters.
    /// Consumes self and returns the modified base transformer.
    pub fn merge(mut self) -> Transformer {
        let scale = self.config.scale();

        for (layer_idx, adapters) in self.layer_adapters.iter().enumerate() {
            let layer = &mut self.base.params.layers[layer_idx];

            if let Some(a) = &adapters.wq {
                merge_adapter_into(&mut layer.wq, a, scale);
            }
            if let Some(a) = &adapters.wk {
                merge_adapter_into(&mut layer.wk, a, scale);
            }
            if let Some(a) = &adapters.wv {
                merge_adapter_into(&mut layer.wv, a, scale);
            }
            if let Some(a) = &adapters.wo {
                merge_adapter_into(&mut layer.wo, a, scale);
            }
            if let Some(a) = &adapters.w1 {
                merge_adapter_into(&mut layer.w1, a, scale);
            }
            if let Some(a) = &adapters.w2 {
                merge_adapter_into(&mut layer.w2, a, scale);
            }
        }

        self.base
    }

    /// Run one training step with a specified learning rate.
    /// Returns the loss value.
    pub fn train_step_with_lr(&mut self, tokens: &[usize], lr: f32) -> f32 {
        assert!(
            tokens.len() >= 2,
            "need at least 2 tokens for next-token prediction"
        );
        let input = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        let (mut g, logits) = self.forward(input);
        let loss = g.cross_entropy(logits, targets);
        let loss_val = g.data(loss)[0];
        g.backward(loss);

        self.update_adapters_with_lr(&g, lr);

        loss_val
    }

    /// Extract gradients and apply SGD with explicit learning rate.
    fn update_adapters_with_lr(&mut self, g: &Graph, lr: f32) {
        let mut adapter_refs = self.collect_adapter_refs_mut();
        let mut adapter_idx = 0;

        let num_nodes = g.num_nodes();
        for node_id in 0..num_nodes {
            let tid = TensorId::from_raw(node_id);
            if !g.is_param(tid) {
                continue;
            }
            if adapter_idx >= adapter_refs.len() {
                break;
            }
            if !g.has_grad(tid) {
                adapter_idx += 1;
                if adapter_idx >= adapter_refs.len() {
                    break;
                }
                continue;
            }
            let grad = g.grad(tid);
            let params = &mut adapter_refs[adapter_idx];
            assert_eq!(
                params.len(),
                grad.len(),
                "adapter param/grad size mismatch at idx={adapter_idx}"
            );
            for (p, &gr) in params.iter_mut().zip(grad.iter()) {
                *p -= lr * gr;
            }
            adapter_idx += 1;
        }
    }
}

/// Merge adapter delta (B @ A * scale) into the base weight matrix.
fn merge_adapter_into(weight: &mut [f32], adapter: &LoraAdapter, scale: f32) {
    let d_in = adapter.d_in;
    let d_out = adapter.d_out;
    let r = adapter.rank;
    assert_eq!(
        weight.len(),
        d_in * d_out,
        "weight size {} != d_in*d_out={}",
        weight.len(),
        d_in * d_out
    );

    // Compute B @ A: (d_in, r) @ (r, d_out) → (d_in, d_out).
    for i in 0..d_in {
        for j in 0..d_out {
            let mut sum = 0.0f32;
            for k in 0..r {
                sum += adapter.b_matrix[i * r + k] * adapter.a_matrix[k * d_out + j];
            }
            weight[i * d_out + j] += sum * scale;
        }
    }
}

/// Compute sinusoidal positional encoding table: (seq_len, d_model).
fn sinusoidal_encoding(seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; seq_len * d_model];
    for pos in 0..seq_len {
        for i in 0..d_model {
            let angle = pos as f32 / 10000.0f32.powf((2 * (i / 2)) as f32 / d_model as f32);
            pe[pos * d_model + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    pe
}

#[cfg(test)]
mod tests {
    use super::*;
    use payya_transformer::TransformerConfig;
    use rand::SeedableRng;

    fn test_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 2,
            d_ff: 32,
            max_seq_len: 64,
            pos_encoding: PosEncoding::Sinusoidal,
        }
    }

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn forward_produces_logits() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let lora_config = LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        };
        let model = LoraModel::new(base, lora_config, &mut rng);
        let (g, logits) = model.forward(&[0, 1, 2, 3, 4]);
        assert_eq!(g.shape(logits), &[5, 32]);
        for &val in g.data(logits) {
            assert!(val.is_finite(), "logit must be finite, got {val}");
        }
    }

    #[test]
    fn initial_lora_matches_base() {
        // Since A is initialized to zeros, the initial LoRA model output
        // should match the base model output.
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let base_clone = base.clone();

        let lora_config = LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        };
        let model = LoraModel::new(base, lora_config, &mut rng);

        let tokens = vec![0, 1, 2, 3];
        let (g_base, logits_base) = base_clone.forward(&tokens);
        let (g_lora, logits_lora) = model.forward(&tokens);

        let base_data = g_base.data(logits_base);
        let lora_data = g_lora.data(logits_lora);
        assert_eq!(base_data.len(), lora_data.len());
        for (a, b) in base_data.iter().zip(lora_data.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "initial LoRA should match base: {a} vs {b}"
            );
        }
    }

    #[test]
    fn training_loss_decreases() {
        let mut rng = seeded_rng();
        let config = TransformerConfig {
            vocab_size: 16,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 32,
            pos_encoding: PosEncoding::Sinusoidal,
        };
        let base = Transformer::new(config, &mut rng);
        let lora_config = LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        };
        let mut model = LoraModel::new(base, lora_config, &mut rng);

        let pattern: Vec<usize> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let mut losses = Vec::new();
        for _ in 0..30 {
            let loss = model.train_step_with_lr(&pattern, 0.01);
            assert!(loss.is_finite(), "loss must be finite");
            losses.push(loss);
        }

        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn trainable_params_much_smaller_than_base() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let lora_config = LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        };
        let model = LoraModel::new(base, lora_config, &mut rng);

        let trainable = model.num_trainable_params();
        let base_total = model.num_base_params();
        assert!(
            trainable < base_total / 5,
            "trainable params ({trainable}) should be much less than base ({base_total})"
        );
    }

    #[test]
    fn merge_produces_valid_model() {
        let mut rng = seeded_rng();
        let config = TransformerConfig {
            vocab_size: 16,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 32,
            pos_encoding: PosEncoding::Sinusoidal,
        };
        let base = Transformer::new(config, &mut rng);
        let lora_config = LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        };
        let mut model = LoraModel::new(base, lora_config, &mut rng);

        // Train a few steps to make adapters non-zero.
        let pattern: Vec<usize> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        for _ in 0..5 {
            model.train_step_with_lr(&pattern, 0.01);
        }

        // Get LoRA forward output before merge.
        let tokens = vec![0, 1, 2, 3];
        let (g_lora, logits_lora) = model.forward(&tokens);
        let lora_output: Vec<f32> = g_lora.data(logits_lora).to_vec();

        // Merge and check the merged model produces same output.
        let merged = model.merge();
        let (g_merged, logits_merged) = merged.forward(&tokens);
        let merged_output = g_merged.data(logits_merged);

        assert_eq!(lora_output.len(), merged_output.len());
        for (a, b) in lora_output.iter().zip(merged_output.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "merged model should match LoRA output: {a} vs {b}"
            );
        }
    }

    #[test]
    fn checkpoint_roundtrip() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let lora_config = LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        };
        let mut model = LoraModel::new(base.clone(), lora_config, &mut rng);

        // Train a bit.
        let pattern: Vec<usize> = vec![0, 1, 2, 3, 4, 5];
        for _ in 0..3 {
            model.train_step_with_lr(&pattern, 0.01);
        }

        // Save and restore.
        let ckpt = model.checkpoint();
        let bytes = ckpt.to_bytes();
        let restored_ckpt = LoraCheckpoint::from_bytes(&bytes);
        let restored = LoraModel::from_checkpoint(base, restored_ckpt);

        // Forward should produce identical output.
        let tokens = vec![0, 1, 2];
        let (g1, l1) = model.forward(&tokens);
        let (g2, l2) = restored.forward(&tokens);
        let d1 = g1.data(l1);
        let d2 = g2.data(l2);
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "checkpoint roundtrip mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn all_targets() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let lora_config = LoraConfig {
            rank: 2,
            alpha: 4.0,
            targets: vec![
                LoraTarget::Wq,
                LoraTarget::Wk,
                LoraTarget::Wv,
                LoraTarget::Wo,
                LoraTarget::W1,
                LoraTarget::W2,
            ],
        };
        let model = LoraModel::new(base, lora_config, &mut rng);
        let (g, logits) = model.forward(&[0, 1, 2]);
        for &val in g.data(logits) {
            assert!(val.is_finite());
        }
    }
}
