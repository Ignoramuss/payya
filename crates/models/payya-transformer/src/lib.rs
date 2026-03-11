//! Decoder-only Transformer from scratch.
//!
//! Implements multi-head self-attention, feed-forward network, layer norm,
//! residual connections, and positional encoding (sinusoidal + RoPE).
//! Built on top of `payya-autograd` for automatic differentiation.

use payya_autograd::{Graph, TensorId};
use rand::Rng;
use serde::{Deserialize, Serialize};

// ── Configuration ────────────────────────────────────────────────────

/// Positional encoding strategy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PosEncoding {
    Sinusoidal,
    RoPE,
}

/// Transformer hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub pos_encoding: PosEncoding,
}

impl TransformerConfig {
    pub fn d_head(&self) -> usize {
        assert!(
            self.d_model.is_multiple_of(self.n_heads),
            "d_model={} must be divisible by n_heads={}",
            self.d_model,
            self.n_heads
        );
        self.d_model / self.n_heads
    }
}

// ── Parameter storage ────────────────────────────────────────────────

/// All learnable parameters for one transformer layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParams {
    /// Attention Q, K, V projection weights: (d_model, d_model) each.
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    /// Attention output projection: (d_model, d_model).
    pub wo: Vec<f32>,
    /// Attention biases: (d_model,) each.
    pub bq: Vec<f32>,
    pub bk: Vec<f32>,
    pub bv: Vec<f32>,
    pub bo: Vec<f32>,
    /// Layer norm 1 (pre-attention): gamma, beta (d_model,).
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    /// FFN weights: W1 (d_model, d_ff), W2 (d_ff, d_model).
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    /// FFN biases: b1 (d_ff,), b2 (d_model,).
    pub b1: Vec<f32>,
    pub b2: Vec<f32>,
    /// Layer norm 2 (pre-FFN): gamma, beta (d_model,).
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
}

/// All learnable parameters for the full transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerParams {
    /// Token embedding table: (vocab_size, d_model).
    pub token_emb: Vec<f32>,
    /// Layers.
    pub layers: Vec<LayerParams>,
    /// Final layer norm: gamma, beta (d_model,).
    pub final_ln_gamma: Vec<f32>,
    pub final_ln_beta: Vec<f32>,
    /// Output head (tied or separate): (d_model, vocab_size).
    pub output_weight: Vec<f32>,
    /// Output bias: (vocab_size,).
    pub output_bias: Vec<f32>,
}

/// Initialize a parameter vector with Xavier/Glorot uniform distribution.
fn xavier_init(fan_in: usize, fan_out: usize, rng: &mut impl Rng) -> Vec<f32> {
    let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
    (0..fan_in * fan_out)
        .map(|_| rng.gen_range(-limit..limit))
        .collect()
}

fn zeros(n: usize) -> Vec<f32> {
    vec![0.0; n]
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

impl TransformerParams {
    /// Initialize all parameters with Xavier initialization.
    pub fn init(config: &TransformerConfig, rng: &mut impl Rng) -> Self {
        let d = config.d_model;
        let ff = config.d_ff;
        let v = config.vocab_size;

        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(LayerParams {
                wq: xavier_init(d, d, rng),
                wk: xavier_init(d, d, rng),
                wv: xavier_init(d, d, rng),
                wo: xavier_init(d, d, rng),
                bq: zeros(d),
                bk: zeros(d),
                bv: zeros(d),
                bo: zeros(d),
                ln1_gamma: ones(d),
                ln1_beta: zeros(d),
                w1: xavier_init(d, ff, rng),
                w2: xavier_init(ff, d, rng),
                b1: zeros(ff),
                b2: zeros(d),
                ln2_gamma: ones(d),
                ln2_beta: zeros(d),
            });
        }

        Self {
            token_emb: xavier_init(v, d, rng),
            layers,
            final_ln_gamma: ones(d),
            final_ln_beta: zeros(d),
            output_weight: xavier_init(d, v, rng),
            output_bias: zeros(v),
        }
    }
}

// ── Positional Encoding ──────────────────────────────────────────────

/// Compute sinusoidal positional encoding table: (max_seq_len, d_model).
fn sinusoidal_encoding(max_seq_len: usize, d_model: usize) -> Vec<f32> {
    let mut pe = vec![0.0f32; max_seq_len * d_model];
    for pos in 0..max_seq_len {
        for i in 0..d_model {
            let angle = pos as f32 / 10000.0f32.powf((2 * (i / 2)) as f32 / d_model as f32);
            pe[pos * d_model + i] = if i.is_multiple_of(2) {
                angle.sin()
            } else {
                angle.cos()
            };
        }
    }
    pe
}

/// Apply Rotary Position Embeddings (RoPE) to a (seq, dim) tensor in-place.
/// Expects dim to be even. Rotates pairs of dimensions by position-dependent angles.
fn apply_rope(data: &mut [f32], seq: usize, dim: usize) {
    assert!(
        dim.is_multiple_of(2),
        "RoPE requires even dimension, got {dim}"
    );
    for pos in 0..seq {
        for i in (0..dim).step_by(2) {
            let freq = 1.0 / 10000.0f32.powf(i as f32 / dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let idx = pos * dim + i;
            let x0 = data[idx];
            let x1 = data[idx + 1];
            data[idx] = x0 * cos_a - x1 * sin_a;
            data[idx + 1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/// Apply inverse RoPE rotation (for backward pass through RoPE).
pub fn apply_rope_inverse(data: &mut [f32], seq: usize, dim: usize) {
    assert!(
        dim.is_multiple_of(2),
        "RoPE requires even dimension, got {dim}"
    );
    for pos in 0..seq {
        for i in (0..dim).step_by(2) {
            let freq = 1.0 / 10000.0f32.powf(i as f32 / dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let idx = pos * dim + i;
            let x0 = data[idx];
            let x1 = data[idx + 1];
            // Inverse rotation: negate sin.
            data[idx] = x0 * cos_a + x1 * sin_a;
            data[idx + 1] = -x0 * sin_a + x1 * cos_a;
        }
    }
}

// ── Transformer model ────────────────────────────────────────────────

/// A decoder-only Transformer language model.
#[derive(Clone)]
pub struct Transformer {
    pub config: TransformerConfig,
    pub params: TransformerParams,
    /// Precomputed sinusoidal PE table (only used if pos_encoding == Sinusoidal).
    pe_table: Vec<f32>,
}

impl Transformer {
    /// Create a new transformer with random parameters.
    pub fn new(config: TransformerConfig, rng: &mut impl Rng) -> Self {
        let pe_table = if config.pos_encoding == PosEncoding::Sinusoidal {
            sinusoidal_encoding(config.max_seq_len, config.d_model)
        } else {
            Vec::new()
        };
        let params = TransformerParams::init(&config, rng);
        Self {
            config,
            params,
            pe_table,
        }
    }

    /// Create a transformer from an existing config and params.
    pub fn from_params(config: TransformerConfig, params: TransformerParams) -> Self {
        let pe_table = if config.pos_encoding == PosEncoding::Sinusoidal {
            sinusoidal_encoding(config.max_seq_len, config.d_model)
        } else {
            Vec::new()
        };
        Self {
            config,
            params,
            pe_table,
        }
    }

    /// Run forward pass returning hidden states (seq, d_model) after final layer norm,
    /// before the output projection. Useful for building embedding models.
    pub fn forward_hidden(&self, tokens: &[usize]) -> (Graph, TensorId) {
        let cfg = &self.config;
        let seq = tokens.len();
        assert!(
            seq <= cfg.max_seq_len,
            "sequence length {seq} exceeds max_seq_len {}",
            cfg.max_seq_len
        );
        assert!(seq > 0, "tokens must not be empty");
        let d = cfg.d_model;

        let mut g = Graph::new();

        let emb_table = g.param(&self.params.token_emb, &[cfg.vocab_size, d]);
        let mut x = g.embedding(emb_table, tokens);

        match cfg.pos_encoding {
            PosEncoding::Sinusoidal => {
                let pe_data: Vec<f32> = self.pe_table[..seq * d].to_vec();
                let pe = g.tensor(&pe_data, &[seq, d]);
                x = g.add(x, pe);
            }
            PosEncoding::RoPE => {}
        }

        for layer in &self.params.layers {
            x = self.forward_layer(&mut g, x, layer, seq);
        }

        let final_gamma = g.param(&self.params.final_ln_gamma, &[d]);
        let final_beta = g.param(&self.params.final_ln_beta, &[d]);
        x = g.layer_norm(x, final_gamma, final_beta, 1e-5);

        (g, x)
    }

    /// Run the forward pass, returning logits (seq, vocab_size).
    ///
    /// Builds a computation graph, runs forward, and returns (graph, logits_id).
    /// The caller can then call `g.backward(loss)` for training.
    pub fn forward(&self, tokens: &[usize]) -> (Graph, TensorId) {
        let cfg = &self.config;
        let seq = tokens.len();
        assert!(
            seq <= cfg.max_seq_len,
            "sequence length {seq} exceeds max_seq_len {}",
            cfg.max_seq_len
        );
        assert!(seq > 0, "tokens must not be empty");
        let d = cfg.d_model;

        let mut g = Graph::new();

        // Token embeddings: (seq, d_model).
        let emb_table = g.param(&self.params.token_emb, &[cfg.vocab_size, d]);
        let mut x = g.embedding(emb_table, tokens);

        // Positional encoding.
        match cfg.pos_encoding {
            PosEncoding::Sinusoidal => {
                let pe_data: Vec<f32> = self.pe_table[..seq * d].to_vec();
                let pe = g.tensor(&pe_data, &[seq, d]);
                x = g.add(x, pe);
            }
            PosEncoding::RoPE => {
                // RoPE is applied to Q and K inside attention, not to embeddings.
                // No-op here.
            }
        }

        // Transformer layers.
        for layer in &self.params.layers {
            x = self.forward_layer(&mut g, x, layer, seq);
        }

        // Final layer norm.
        let final_gamma = g.param(&self.params.final_ln_gamma, &[d]);
        let final_beta = g.param(&self.params.final_ln_beta, &[d]);
        x = g.layer_norm(x, final_gamma, final_beta, 1e-5);

        // Output projection: (seq, d_model) @ (d_model, vocab) → (seq, vocab).
        let out_w = g.param(&self.params.output_weight, &[d, cfg.vocab_size]);
        let out_b = g.param(&self.params.output_bias, &[cfg.vocab_size]);
        let logits = g.matmul(x, out_w);
        let logits = g.add(logits, out_b);

        (g, logits)
    }

    /// Forward through one transformer layer.
    fn forward_layer(
        &self,
        g: &mut Graph,
        x: TensorId,
        layer: &LayerParams,
        seq: usize,
    ) -> TensorId {
        let d = self.config.d_model;
        let ff = self.config.d_ff;

        // Pre-norm for attention.
        let ln1_gamma = g.param(&layer.ln1_gamma, &[d]);
        let ln1_beta = g.param(&layer.ln1_beta, &[d]);
        let normed = g.layer_norm(x, ln1_gamma, ln1_beta, 1e-5);

        // Q, K, V projections.
        let wq = g.param(&layer.wq, &[d, d]);
        let bq = g.param(&layer.bq, &[d]);
        let wk = g.param(&layer.wk, &[d, d]);
        let bk = g.param(&layer.bk, &[d]);
        let wv = g.param(&layer.wv, &[d, d]);
        let bv = g.param(&layer.bv, &[d]);

        let mut q = g.matmul(normed, wq);
        q = g.add(q, bq);
        let mut k = g.matmul(normed, wk);
        k = g.add(k, bk);
        let mut v = g.matmul(normed, wv);
        v = g.add(v, bv);

        // Apply RoPE to Q and K if configured.
        if self.config.pos_encoding == PosEncoding::RoPE {
            let mut q_data = g.data(q).to_vec();
            apply_rope(&mut q_data, seq, d);
            let q_rope = g.param(&q_data, &[seq, d]);
            q = q_rope;

            let mut k_data = g.data(k).to_vec();
            apply_rope(&mut k_data, seq, d);
            let k_rope = g.param(&k_data, &[seq, d]);
            k = k_rope;
        }

        // Multi-head attention with causal mask.
        let wo = g.param(&layer.wo, &[d, d]);
        let bo = g.param(&layer.bo, &[d]);
        let mut attn_out = g.scaled_attention(q, k, v, self.config.n_heads, true);
        attn_out = g.matmul(attn_out, wo);
        attn_out = g.add(attn_out, bo);

        // Residual connection.
        let x = g.add(x, attn_out);

        // Pre-norm for FFN.
        let ln2_gamma = g.param(&layer.ln2_gamma, &[d]);
        let ln2_beta = g.param(&layer.ln2_beta, &[d]);
        let normed2 = g.layer_norm(x, ln2_gamma, ln2_beta, 1e-5);

        // Feed-forward network: relu(x @ W1 + b1) @ W2 + b2.
        let w1 = g.param(&layer.w1, &[d, ff]);
        let b1 = g.param(&layer.b1, &[ff]);
        let w2 = g.param(&layer.w2, &[ff, d]);
        let b2 = g.param(&layer.b2, &[d]);

        let mut h = g.matmul(normed2, w1);
        h = g.add(h, b1);
        h = g.relu(h);
        h = g.matmul(h, w2);
        h = g.add(h, b2);

        // Residual connection.
        g.add(x, h)
    }

    /// Run one training step with SGD. Returns the loss value.
    ///
    /// `tokens` is the full sequence; targets are `tokens[1..]`.
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

        // SGD update: walk all parameter nodes and update.
        self.update_params_from_graph(&g, lr);

        loss_val
    }

    /// Extract gradients from the graph and apply SGD updates to parameters.
    fn update_params_from_graph(&mut self, g: &Graph, lr: f32) {
        // We need to map graph TensorIds back to our parameter vectors.
        // Since we build the graph deterministically in forward(), the param
        // order is: token_emb, then per-layer params, then final_ln, output.
        //
        // Instead of tracking IDs, we iterate over all nodes in the graph
        // and update params by matching shapes and order.
        //
        // Simpler approach: rebuild the graph to know the param IDs.
        // Actually, we can use a simpler technique: collect all param refs
        // and their gradients by building a fresh graph with the same structure.
        //
        // The cleanest approach: store TensorIds during forward and use them.
        // But our forward rebuilds the graph each time.
        //
        // Practical approach: the graph nodes are in order. Params are the leaf
        // nodes. We iterate and match them to our stored parameter vectors.

        let mut param_idx = 0;
        let mut param_vecs = self.collect_param_refs_mut();

        // Walk graph nodes in order. Each Leaf with requires_grad=true is a param.
        // We skip non-leaf or non-grad nodes.
        let num_nodes = g.num_nodes();
        for node_id in 0..num_nodes {
            let tid = payya_autograd::TensorId::from_raw(node_id);
            if !g.is_param(tid) {
                continue;
            }
            if !g.has_grad(tid) {
                param_idx += 1;
                if param_idx >= param_vecs.len() {
                    break;
                }
                continue;
            }
            if param_idx >= param_vecs.len() {
                break;
            }
            let grad = g.grad(tid);
            let params = &mut param_vecs[param_idx];
            assert_eq!(
                params.len(),
                grad.len(),
                "param/grad size mismatch at param_idx={param_idx}"
            );
            for (p, &gr) in params.iter_mut().zip(grad.iter()) {
                *p -= lr * gr;
            }
            param_idx += 1;
        }
    }

    /// Collect mutable references to all parameter vectors in graph construction order.
    fn collect_param_refs_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut refs: Vec<&mut Vec<f32>> = Vec::new();

        refs.push(&mut self.params.token_emb);

        // Per-layer params in the same order as forward_layer.
        for layer in &mut self.params.layers {
            refs.push(&mut layer.ln1_gamma);
            refs.push(&mut layer.ln1_beta);
            refs.push(&mut layer.wq);
            refs.push(&mut layer.bq);
            refs.push(&mut layer.wk);
            refs.push(&mut layer.bk);
            refs.push(&mut layer.wv);
            refs.push(&mut layer.bv);
            // RoPE creates extra params for Q and K — skip those since they're
            // derived (not stored). They are ephemeral.
            refs.push(&mut layer.wo);
            refs.push(&mut layer.bo);
            refs.push(&mut layer.ln2_gamma);
            refs.push(&mut layer.ln2_beta);
            refs.push(&mut layer.w1);
            refs.push(&mut layer.b1);
            refs.push(&mut layer.w2);
            refs.push(&mut layer.b2);
        }

        refs.push(&mut self.params.final_ln_gamma);
        refs.push(&mut self.params.final_ln_beta);
        refs.push(&mut self.params.output_weight);
        refs.push(&mut self.params.output_bias);

        refs
    }

    /// Generate text autoregressively.
    ///
    /// Takes a prompt (token IDs), generates up to `max_new_tokens` additional tokens.
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new_tokens: usize,
        processor: &payya_logit_processor::LogitProcessor,
        rng: &mut impl Rng,
    ) -> Vec<usize> {
        let mut tokens = prompt.to_vec();

        for _ in 0..max_new_tokens {
            if tokens.len() >= self.config.max_seq_len {
                break;
            }
            let (g, logits_id) = self.forward(&tokens);
            let logits_data = g.data(logits_id);
            let vocab = self.config.vocab_size;
            // Take logits for the last position.
            let last_row = &logits_data[(tokens.len() - 1) * vocab..tokens.len() * vocab];
            let mut last_logits = last_row.to_vec();

            let past: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let next_token = processor.sample(&mut last_logits, &past, rng);
            tokens.push(next_token);
        }

        tokens
    }
}

// ── Public helpers for Graph introspection needed by update_params ──

// We need a few extra accessors on Graph that aren't in the original API.
// Add them via a module extension.

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_forward_produces_logits() {
        let mut rng = seeded_rng();
        let config = test_config();
        let model = Transformer::new(config.clone(), &mut rng);
        let tokens = vec![0, 1, 2, 3, 4];
        let (g, logits) = model.forward(&tokens);
        let shape = g.shape(logits);
        assert_eq!(shape, &[5, 32], "logits shape should be (seq, vocab)");
        // All logits should be finite.
        for &val in g.data(logits) {
            assert!(val.is_finite(), "logit must be finite, got {val}");
        }
    }

    #[test]
    fn test_forward_rope() {
        let mut rng = seeded_rng();
        let config = TransformerConfig {
            vocab_size: 32,
            d_model: 16,
            n_heads: 2,
            n_layers: 1,
            d_ff: 32,
            max_seq_len: 64,
            pos_encoding: PosEncoding::RoPE,
        };
        let model = Transformer::new(config, &mut rng);
        let tokens = vec![0, 1, 2];
        let (g, logits) = model.forward(&tokens);
        for &val in g.data(logits) {
            assert!(val.is_finite(), "RoPE logit must be finite, got {val}");
        }
    }

    #[test]
    fn test_training_loss_decreases() {
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
        let mut model = Transformer::new(config, &mut rng);

        // Train on a tiny repeating pattern.
        let pattern: Vec<usize> = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
        let mut losses = Vec::new();
        for _ in 0..30 {
            let loss = model.train_step(&pattern, 0.01);
            losses.push(loss);
        }

        // Loss should decrease: first loss > last loss.
        let first = losses[0];
        let last = *losses.last().unwrap();
        assert!(
            last < first,
            "loss should decrease: first={first}, last={last}"
        );
    }

    #[test]
    fn test_generation_produces_valid_tokens() {
        let mut rng = seeded_rng();
        let config = test_config();
        let model = Transformer::new(config.clone(), &mut rng);
        let prompt = vec![0, 1, 2];
        let processor = payya_logit_processor::LogitProcessor::new().with_temperature(1.0);
        let generated = model.generate(&prompt, 10, &processor, &mut rng);
        assert_eq!(generated.len(), 13); // 3 prompt + 10 new
        for &tok in &generated {
            assert!(
                tok < config.vocab_size,
                "generated token {tok} out of vocab range"
            );
        }
    }

    #[test]
    fn test_generation_no_nans() {
        let mut rng = seeded_rng();
        let config = test_config();
        let model = Transformer::new(config, &mut rng);
        let prompt = vec![0];
        let processor = payya_logit_processor::LogitProcessor::new()
            .with_temperature(0.8)
            .with_top_k(10);
        let generated = model.generate(&prompt, 20, &processor, &mut rng);
        assert!(generated.len() > 1, "should generate at least one token");
    }

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let pe = sinusoidal_encoding(10, 8);
        assert_eq!(pe.len(), 80);
        // Position 0 should have sin(0)=0 for even dims.
        assert!((pe[0]).abs() < 1e-5);
    }

    #[test]
    fn test_rope_roundtrip() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = data.clone();
        apply_rope(&mut data, 2, 4);
        // Data should be different after RoPE.
        assert_ne!(data, original);
        // Inverse should restore original.
        apply_rope_inverse(&mut data, 2, 4);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "RoPE roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn test_overfit_tiny_corpus() {
        // The model should be able to memorize a tiny sequence.
        let mut rng = seeded_rng();
        let config = TransformerConfig {
            vocab_size: 8,
            d_model: 32,
            n_heads: 2,
            n_layers: 2,
            d_ff: 64,
            max_seq_len: 32,
            pos_encoding: PosEncoding::Sinusoidal,
        };
        let mut model = Transformer::new(config, &mut rng);
        let data: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];

        let mut last_loss = f32::INFINITY;
        for epoch in 0..200 {
            let loss = model.train_step(&data, 0.005);
            if epoch % 50 == 0 {
                eprintln!("epoch {epoch}: loss={loss:.4}");
            }
            last_loss = loss;
        }
        // Should achieve reasonably low loss on memorization.
        assert!(
            last_loss < 1.0,
            "overfit loss should be < 1.0, got {last_loss}"
        );
    }
}
