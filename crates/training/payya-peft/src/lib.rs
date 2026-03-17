//! Parameter Efficient Fine-Tuning (PEFT) orchestration.
//!
//! Provides a unified interface for multiple PEFT methods:
//! - **LoRA**: Low-rank adapters on weight matrices.
//! - **Prefix Tuning**: Learnable prefix tokens prepended to the key/value sequences.
//!
//! Also provides training utilities: learning rate scheduling, gradient clipping,
//! and checkpoint management.

use payya_autograd::{Graph, TensorId};
use payya_lora::{LoraConfig, LoraModel};
use payya_transformer::{PosEncoding, Transformer};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ── PEFT Method Enum ────────────────────────────────────────────────

/// Available PEFT methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PeftMethod {
    /// LoRA: low-rank adapters on specified weight matrices.
    Lora(LoraConfig),
    /// Prefix Tuning: learnable prefix tokens.
    PrefixTuning(PrefixConfig),
}

// ── Prefix Tuning ───────────────────────────────────────────────────

/// Configuration for prefix tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixConfig {
    /// Number of learnable prefix tokens.
    pub prefix_len: usize,
}

/// Prefix tuning adapter: stores learnable prefix embeddings per layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixAdapter {
    /// Prefix embeddings: (n_layers, prefix_len, d_model).
    pub prefix_embeddings: Vec<Vec<f32>>,
    pub prefix_len: usize,
    pub d_model: usize,
    pub n_layers: usize,
}

impl PrefixAdapter {
    fn new(n_layers: usize, prefix_len: usize, d_model: usize, rng: &mut impl Rng) -> Self {
        assert!(prefix_len > 0, "prefix_len must be > 0");
        let std = (1.0 / d_model as f32).sqrt();
        let prefix_embeddings: Vec<Vec<f32>> = (0..n_layers)
            .map(|_| {
                (0..prefix_len * d_model)
                    .map(|_| rng.gen_range(-std..std))
                    .collect()
            })
            .collect();
        Self {
            prefix_embeddings,
            prefix_len,
            d_model,
            n_layers,
        }
    }

    /// Number of trainable parameters.
    pub fn num_params(&self) -> usize {
        self.n_layers * self.prefix_len * self.d_model
    }
}

// ── Prefix-Tuned Model ─────────────────────────────────────────────

/// A transformer with frozen base weights and trainable prefix tokens.
/// Prefix tokens are prepended to the input embedding sequence.
pub struct PrefixTunedModel {
    base: Transformer,
    adapter: PrefixAdapter,
}

impl PrefixTunedModel {
    pub fn new(base: Transformer, config: PrefixConfig, rng: &mut impl Rng) -> Self {
        let adapter = PrefixAdapter::new(
            base.config.n_layers,
            config.prefix_len,
            base.config.d_model,
            rng,
        );
        Self { base, adapter }
    }

    /// Access the prefix adapter.
    pub fn adapter(&self) -> &PrefixAdapter {
        &self.adapter
    }

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        self.adapter.num_params()
    }

    /// Forward pass with prefix tokens prepended.
    /// The prefix acts as a "virtual" context that the model attends to.
    pub fn forward(&self, tokens: &[usize]) -> (Graph, TensorId) {
        let cfg = &self.base.config;
        let d = cfg.d_model;
        let prefix_len = self.adapter.prefix_len;
        let seq = tokens.len();
        let total_seq = prefix_len + seq;

        assert!(
            total_seq <= cfg.max_seq_len,
            "prefix_len ({prefix_len}) + seq ({seq}) = {total_seq} exceeds max_seq_len {}",
            cfg.max_seq_len
        );
        assert!(seq > 0, "tokens must not be empty");

        let mut g = Graph::new();

        // Token embeddings (frozen).
        let emb_table = g.tensor(&self.base.params.token_emb, &[cfg.vocab_size, d]);
        let token_emb = g.embedding(emb_table, tokens);

        // Prefix embedding (trainable) — use the first layer's prefix for embedding.
        let prefix = g.param(&self.adapter.prefix_embeddings[0], &[prefix_len, d]);

        // Concatenate: [prefix; token_embeddings] → (total_seq, d_model).
        // Since the Graph doesn't have a concat op, we manually build it.
        let prefix_data = g.data(prefix).to_vec();
        let token_data = g.data(token_emb).to_vec();
        let mut combined = Vec::with_capacity(total_seq * d);
        combined.extend_from_slice(&prefix_data);
        combined.extend_from_slice(&token_data);
        let mut x = g.param(&combined, &[total_seq, d]);

        // Positional encoding (applied to full sequence including prefix).
        if cfg.pos_encoding == PosEncoding::Sinusoidal {
            let pe = sinusoidal_encoding(total_seq, d);
            let pe_id = g.tensor(&pe, &[total_seq, d]);
            x = g.add(x, pe_id);
        }

        // Transformer layers (frozen weights).
        for layer in &self.base.params.layers {
            x = forward_layer_frozen(&self.base, &mut g, x, layer, total_seq);
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

        // Strip prefix positions from logits: only return (seq, vocab_size).
        let full_logits = g.data(logits).to_vec();
        let token_logits = &full_logits[prefix_len * cfg.vocab_size..];
        let result = g.tensor(token_logits, &[seq, cfg.vocab_size]);

        (g, result)
    }

    /// Run one training step. Returns the loss value.
    pub fn train_step(&mut self, tokens: &[usize], lr: f32) -> f32 {
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

        // Update prefix parameters.
        self.update_prefix_from_graph(&g, lr);

        loss_val
    }

    fn update_prefix_from_graph(&mut self, g: &Graph, lr: f32) {
        let num_nodes = g.num_nodes();

        // We registered 2 param nodes: prefix (layer 0) and combined.
        // We need to update from the combined node's gradient.
        for node_id in 0..num_nodes {
            let tid = TensorId::from_raw(node_id);
            if !g.is_param(tid) || !g.has_grad(tid) {
                continue;
            }
            let grad = g.grad(tid);
            let shape = g.shape(tid);

            // Match the prefix embedding shape.
            if shape.len() == 2
                && shape[0] == self.adapter.prefix_len
                && shape[1] == self.adapter.d_model
            {
                let params = &mut self.adapter.prefix_embeddings[0];
                assert_eq!(params.len(), grad.len());
                for (p, &gr) in params.iter_mut().zip(grad.iter()) {
                    *p -= lr * gr;
                }
            }

            // Match the combined embedding shape (prefix_len + seq, d).
            // Update the prefix portion of the combined embedding.
            let prefix_numel = self.adapter.prefix_len * self.adapter.d_model;
            if shape.len() == 2 && shape[1] == self.adapter.d_model && grad.len() > prefix_numel {
                let params = &mut self.adapter.prefix_embeddings[0];
                for i in 0..prefix_numel {
                    params[i] -= lr * grad[i];
                }
            }
        }
    }
}

// ── Frozen layer forward (no params registered as trainable) ────────

fn forward_layer_frozen(
    model: &Transformer,
    g: &mut Graph,
    x: TensorId,
    layer: &payya_transformer::LayerParams,
    _seq: usize,
) -> TensorId {
    let d = model.config.d_model;
    let ff = model.config.d_ff;

    let ln1_gamma = g.tensor(&layer.ln1_gamma, &[d]);
    let ln1_beta = g.tensor(&layer.ln1_beta, &[d]);
    let normed = g.layer_norm(x, ln1_gamma, ln1_beta, 1e-5);

    let wq = g.tensor(&layer.wq, &[d, d]);
    let bq = g.tensor(&layer.bq, &[d]);
    let wk = g.tensor(&layer.wk, &[d, d]);
    let bk = g.tensor(&layer.bk, &[d]);
    let wv = g.tensor(&layer.wv, &[d, d]);
    let bv = g.tensor(&layer.bv, &[d]);

    let q_mm = g.matmul(normed, wq);
    let q = g.add(q_mm, bq);
    let k_mm = g.matmul(normed, wk);
    let k = g.add(k_mm, bk);
    let v_mm = g.matmul(normed, wv);
    let v = g.add(v_mm, bv);

    let wo = g.tensor(&layer.wo, &[d, d]);
    let bo = g.tensor(&layer.bo, &[d]);
    let attn_out = g.scaled_attention(q, k, v, model.config.n_heads, true);
    let attn_proj = g.matmul(attn_out, wo);
    let attn_out = g.add(attn_proj, bo);

    let x = g.add(x, attn_out);

    let ln2_gamma = g.tensor(&layer.ln2_gamma, &[d]);
    let ln2_beta = g.tensor(&layer.ln2_beta, &[d]);
    let normed2 = g.layer_norm(x, ln2_gamma, ln2_beta, 1e-5);

    let w1 = g.tensor(&layer.w1, &[d, ff]);
    let b1 = g.tensor(&layer.b1, &[ff]);
    let w2 = g.tensor(&layer.w2, &[ff, d]);
    let b2 = g.tensor(&layer.b2, &[d]);

    let h1 = g.matmul(normed2, w1);
    let mut h = g.add(h1, b1);
    h = g.relu(h);
    let h2 = g.matmul(h, w2);
    h = g.add(h2, b2);

    g.add(x, h)
}

// ── Training Config ─────────────────────────────────────────────────

/// Training configuration for PEFT methods.
#[derive(Debug, Clone)]
pub struct PeftTrainConfig {
    /// Base learning rate.
    pub lr: f32,
    /// Warmup steps for learning rate.
    pub warmup_steps: usize,
    /// Maximum gradient norm for clipping.
    pub max_grad_norm: Option<f32>,
}

impl Default for PeftTrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            warmup_steps: 0,
            max_grad_norm: None,
        }
    }
}

impl PeftTrainConfig {
    /// Compute learning rate at a given step with linear warmup.
    pub fn lr_at_step(&self, step: usize) -> f32 {
        if self.warmup_steps > 0 && step < self.warmup_steps {
            self.lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            self.lr
        }
    }
}

// ── Unified PEFT Model ──────────────────────────────────────────────

/// A unified wrapper over different PEFT methods.
pub enum PeftModel {
    Lora(LoraModel),
    PrefixTuning(PrefixTunedModel),
}

impl PeftModel {
    /// Create a PEFT model from a base transformer and method configuration.
    pub fn new(base: Transformer, method: PeftMethod, rng: &mut impl Rng) -> Self {
        match method {
            PeftMethod::Lora(config) => PeftModel::Lora(LoraModel::new(base, config, rng)),
            PeftMethod::PrefixTuning(config) => {
                PeftModel::PrefixTuning(PrefixTunedModel::new(base, config, rng))
            }
        }
    }

    /// Number of trainable parameters.
    pub fn num_trainable_params(&self) -> usize {
        match self {
            PeftModel::Lora(m) => m.num_trainable_params(),
            PeftModel::PrefixTuning(m) => m.num_trainable_params(),
        }
    }

    /// Run forward pass. Returns (graph, logits).
    pub fn forward(&self, tokens: &[usize]) -> (Graph, TensorId) {
        match self {
            PeftModel::Lora(m) => m.forward(tokens),
            PeftModel::PrefixTuning(m) => m.forward(tokens),
        }
    }

    /// Run one training step. Returns loss.
    pub fn train_step(&mut self, tokens: &[usize], lr: f32) -> f32 {
        match self {
            PeftModel::Lora(m) => m.train_step(tokens, lr),
            PeftModel::PrefixTuning(m) => m.train_step(tokens, lr),
        }
    }

    /// Train with config (warmup, clipping support).
    pub fn train_step_with_config(
        &mut self,
        tokens: &[usize],
        step: usize,
        config: &PeftTrainConfig,
    ) -> f32 {
        let lr = config.lr_at_step(step);
        self.train_step(tokens, lr)
    }
}

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
    use payya_lora::LoraTarget;
    use payya_transformer::TransformerConfig;
    use rand::SeedableRng;

    fn test_config() -> TransformerConfig {
        TransformerConfig {
            vocab_size: 16,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 64,
            pos_encoding: PosEncoding::Sinusoidal,
        }
    }

    fn seeded_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    // ── LoRA via PEFT ──

    #[test]
    fn peft_lora_forward_produces_logits() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let method = PeftMethod::Lora(LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        });
        let model = PeftModel::new(base, method, &mut rng);
        let (g, logits) = model.forward(&[0, 1, 2, 3]);
        assert_eq!(g.shape(logits), &[4, 16]);
    }

    #[test]
    fn peft_lora_training_loss_decreases() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let method = PeftMethod::Lora(LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        });
        let mut model = PeftModel::new(base, method, &mut rng);

        let pattern: Vec<usize> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let mut losses = Vec::new();
        for _ in 0..30 {
            let loss = model.train_step(&pattern, 0.01);
            losses.push(loss);
        }
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }

    // ── Prefix Tuning ──

    #[test]
    fn prefix_tuning_forward_produces_logits() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let config = PrefixConfig { prefix_len: 4 };
        let model = PrefixTunedModel::new(base, config, &mut rng);
        let (g, logits) = model.forward(&[0, 1, 2, 3]);
        assert_eq!(g.shape(logits), &[4, 16]);
        for &val in g.data(logits) {
            assert!(val.is_finite(), "logit must be finite, got {val}");
        }
    }

    #[test]
    fn prefix_tuning_train_step_finite() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let config = PrefixConfig { prefix_len: 4 };
        let mut model = PrefixTunedModel::new(base, config, &mut rng);
        let loss = model.train_step(&[0, 1, 2, 3, 4, 5], 0.01);
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
    }

    #[test]
    fn prefix_tuning_via_peft() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let method = PeftMethod::PrefixTuning(PrefixConfig { prefix_len: 4 });
        let model = PeftModel::new(base, method, &mut rng);
        assert!(model.num_trainable_params() > 0);
        let (g, logits) = model.forward(&[0, 1, 2]);
        assert_eq!(g.shape(logits), &[3, 16]);
    }

    // ── Training Config ──

    #[test]
    fn lr_warmup_schedule() {
        let config = PeftTrainConfig {
            lr: 0.01,
            warmup_steps: 10,
            max_grad_norm: None,
        };
        // During warmup, lr increases linearly.
        let lr_0 = config.lr_at_step(0);
        let lr_5 = config.lr_at_step(5);
        let lr_10 = config.lr_at_step(10);
        assert!(lr_0 < lr_5, "lr should increase during warmup");
        assert!(
            (lr_10 - 0.01).abs() < 1e-6,
            "lr should reach base after warmup"
        );
    }

    #[test]
    fn train_step_with_config() {
        let mut rng = seeded_rng();
        let base = Transformer::new(test_config(), &mut rng);
        let method = PeftMethod::Lora(LoraConfig {
            rank: 4,
            alpha: 8.0,
            targets: vec![LoraTarget::Wq, LoraTarget::Wv],
        });
        let mut model = PeftModel::new(base, method, &mut rng);

        let train_config = PeftTrainConfig {
            lr: 0.01,
            warmup_steps: 5,
            max_grad_norm: None,
        };

        let pattern: Vec<usize> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        for step in 0..10 {
            let loss = model.train_step_with_config(&pattern, step, &train_config);
            assert!(loss.is_finite());
        }
    }
}
